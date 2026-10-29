import unittest

import numpy as np

from disappear_in_style.config import parse_args
from disappear_in_style.pipeline import Pipeline
from disappear_in_style.state import State
from tests.helpers import hand

W, H = 160, 90


def make(**kw):
    cfg = parse_args(["--seed", "1", "--no-virtualcam", "--no-preview", "--no-auto-background",
                      "--width", str(W), "--height", str(H), "--max-particles", "5000",
                      "--disintegrate-seconds", "1", "--reintegrate-seconds", "1"]
                     + [a for k, v in kw.items() for a in (f"--{k.replace('_', '-')}", str(v))])
    mask = np.zeros((H, W), np.float32)
    mask[10:80, 50:110] = 1
    pipe = Pipeline(cfg, lambda f, p: mask)
    plate = np.full((H, W, 3), 40, np.uint8)
    pipe.plate.set(plate)
    live = plate.copy()
    live[mask > 0] = (200, 120, 60)
    return pipe, live, plate, mask


class PipelineTest(unittest.TestCase):
    def step(self, pipe, live, t, **kw):
        return pipe.process(live, t, 1 / 30, **kw)

    def test_full_cycle_with_manual_toggle(self):
        pipe, live, plate, mask = make()
        out = self.step(pipe, live, 0.0)
        np.testing.assert_array_equal(out, live)
        self.step(pipe, live, 0.1, manual_toggle=True)
        self.assertIs(pipe.sm.state, State.DISINTEGRATING)
        for _ in range(15):
            mid = self.step(pipe, live, 0.2)
        self.assertFalse(np.array_equal(mid, live))
        for i in range(40):
            gone = self.step(pipe, live, 0.3 + i / 30)
        self.assertIs(pipe.sm.state, State.GONE)
        np.testing.assert_array_equal(gone, plate)  # empty background, not black
        self.step(pipe, live, 5.0, manual_toggle=True)
        self.assertIs(pipe.sm.state, State.REINTEGRATING)
        for i in range(40):
            back = self.step(pipe, live, 5.1 + i / 30)
        self.assertIs(pipe.sm.state, State.VISIBLE)
        np.testing.assert_array_equal(back, live)

    def test_audio_event_triggers(self):
        pipe, live, *_ = make()
        self.step(pipe, live, 1.0, audio_events=[1.0])
        self.assertIs(pipe.sm.state, State.DISINTEGRATING)

    def test_gesture_triggers_via_pipeline(self):
        pipe, live, *_ = make()
        for i, r in enumerate([1.0, 0.1, 0.1, 0.1, 1.0]):
            self.step(pipe, live, i / 30, hands=[hand(r)])
        self.assertIs(pipe.sm.state, State.DISINTEGRATING)

    def test_audio_mode_ignores_gesture(self):
        pipe, live, *_ = make(mode="audio")
        for i, r in enumerate([1.0, 0.1, 0.1, 0.1, 1.0]):
            self.step(pipe, live, i / 30, hands=[hand(r)])
        self.assertIs(pipe.sm.state, State.VISIBLE)

    def test_disintegration_wave_goes_left_to_right(self):
        pipe, live, plate, mask = make(noise_mix=0.0)
        self.step(pipe, live, 0.0, manual_toggle=True)
        for _ in range(10):  # ~1/3 through
            out = self.step(pipe, live, 0.1)
        person = mask > 0
        left_gone = (out[:, 50:70][person[:, 50:70]] == plate[0, 0]).all(axis=1).mean()
        right_gone = (out[:, 90:110][person[:, 90:110]] == plate[0, 0]).all(axis=1).mean()
        self.assertGreater(left_gone, right_gone)

    def test_fallback_background_when_no_plate(self):
        pipe, live, plate, mask = make()
        pipe.plate.image = None
        self.step(pipe, live, 0.0, manual_toggle=True)
        for i in range(40):
            out = self.step(pipe, live, 0.1 + i / 30)
        self.assertIs(pipe.sm.state, State.GONE)
        self.assertGreater(out.mean(), 1)  # not black

    def test_background_capture_countdown(self):
        pipe, live, *_ = make(bg_delay=1.0, bg_frames=3)
        pipe.plate.image = None
        pipe.request_background(0.0)
        self.step(pipe, live, 0.5)
        self.assertIsNotNone(pipe.countdown)
        for t in (1.1, 1.2, 1.3):
            self.step(pipe, live, t)
        self.assertTrue(pipe.plate.available)


class ConfigTest(unittest.TestCase):
    def test_default_mode_is_any(self):
        self.assertEqual(parse_args([]).mode, "any")

    def test_flags_and_bad_mode(self):
        cfg = parse_args(["--mode", "confirm", "--cooldown", "3", "--audio-threshold", "0.2"])
        self.assertEqual((cfg.mode, cfg.cooldown, cfg.audio_threshold), ("confirm", 3.0, 0.2))
        with self.assertRaises(SystemExit):
            parse_args(["--mode", "bogus"])

    def test_json_config_overridden_by_flags(self):
        import json, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"mode": "gesture", "cooldown": 2.0}, f)
        cfg = parse_args(["--config", f.name, "--cooldown", "4"])
        self.assertEqual((cfg.mode, cfg.cooldown), ("gesture", 4.0))


if __name__ == "__main__":
    unittest.main()
