import unittest

import cv2
import numpy as np

from disappear_in_style.background import BackgroundPlate, match_plate
from disappear_in_style.config import parse_args
from disappear_in_style.pipeline import Pipeline
from disappear_in_style.state import State

W, H = 160, 90


def gradient(base: int) -> np.ndarray:
    x = np.linspace(0, 40, W, dtype=np.float32)[None, :, None]
    return np.clip(base + x + np.zeros((H, W, 3), np.float32), 0, 255).astype(np.uint8)


def soft_mask() -> np.ndarray:
    m = np.zeros((H, W), np.float32)
    m[10:80, 50:110] = 1
    return cv2.GaussianBlur(m, (0, 0), 3)  # soft edges, like the real segmenter


def run_to_gone(plate_img, live_bg, mask, **kw):
    cfg = parse_args(["--seed", "1", "--no-virtualcam", "--no-preview", "--no-auto-background",
                      "--width", str(W), "--height", str(H), "--max-particles", "5000",
                      "--disintegrate-seconds", "1", "--reintegrate-seconds", "1"])
    pipe = Pipeline(cfg, lambda f, p: mask)
    if plate_img is not None:
        pipe.plate.set(plate_img)
    live = live_bg.copy()
    live[mask > 0.3] = (15, 15, 20)  # dark clothes / hair; the soft mask edge reaches past the person
    pipe.process(live, 0.0, 1 / 30)
    pipe.process(live, 0.1, 1 / 30, manual_toggle=True)
    for i in range(60):
        out = pipe.process(live, 0.2 + i / 30, 1 / 30)
    assert pipe.sm.state is State.GONE
    return out


class GoneTest(unittest.TestCase):
    def test_soft_edges_match_plate(self):
        bg = gradient(100)
        mask = soft_mask()
        out = run_to_gone(bg, bg, mask)
        err = np.abs(out.astype(int) - bg.astype(int)).max(axis=2)
        self.assertLessEqual(err[mask > 0.3].max(), 3)
        self.assertLessEqual(err.max(), 3)

    def test_no_dark_halo_when_plate_is_darker_than_room(self):
        live_bg = gradient(120)
        plate = (live_bg.astype(int) - 45).clip(0, 255).astype(np.uint8)  # captured while exposure ramped up
        mask = soft_mask()
        out = run_to_gone(plate, live_bg, mask)
        inside = mask > 0.3
        diff = out.astype(int) - live_bg.astype(int)
        self.assertGreater(diff[inside].min(), -8)  # no silhouette darker than the room
        self.assertLess(np.abs(diff).max(), 8)

    def test_match_plate_keeps_matching_plate(self):
        bg = gradient(100)
        np.testing.assert_array_equal(match_plate(bg, bg, soft_mask()), bg)

    def test_fallback_plate_has_no_black(self):
        bg = gradient(120)
        mask = soft_mask()
        out = run_to_gone(None, bg, mask)
        self.assertGreater(out[mask > 0.3].min(), 90)

    def test_default_mask_grow_is_adequate(self):
        self.assertGreaterEqual(parse_args([]).mask_grow, 5)


if __name__ == "__main__":
    unittest.main()
