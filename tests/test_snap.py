import unittest

import numpy as np

from disappear_in_style.snap import AudioSnapDetector, GestureSnapDetector, SnapTrigger, pinch_ratio
from tests.helpers import hand

DT = 1 / 30


def run(det, ratios, t0=0.0, dt=DT):
    """Feed a sequence of pinch ratios (None = no hand); return indices where it fired."""
    fired = []
    for i, r in enumerate(ratios):
        hands = [] if r is None else [hand(r)]
        if det.update(t0 + i * dt, hands):
            fired.append(i)
    return fired


class PinchRatioTest(unittest.TestCase):
    def test_ratio(self):
        self.assertAlmostEqual(pinch_ratio(hand(0.5)), 0.5, places=5)

    def test_degenerate_palm(self):
        self.assertEqual(pinch_ratio(np.zeros((21, 2))), float("inf"))


class GestureTest(unittest.TestCase):
    def test_fast_flick_fires_once(self):
        self.assertEqual(run(GestureSnapDetector(), [1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 1.0]), [5])

    def test_flick_through_hysteresis_frame(self):
        self.assertEqual(run(GestureSnapDetector(), [1.0, 0.1, 0.1, 0.45, 1.0]), [4])

    def test_slow_release_does_not_fire(self):
        seq = [0.1, 0.1, 0.1] + [0.45] * 10 + [1.0]
        self.assertEqual(run(GestureSnapDetector(), seq), [])

    def test_single_pinch_frame_does_not_fire(self):
        self.assertEqual(run(GestureSnapDetector(), [1.0, 0.1, 1.0]), [])

    def test_long_hold_does_not_fire(self):
        seq = [0.1] * 60 + [1.0]  # 2 seconds
        self.assertEqual(run(GestureSnapDetector(), seq), [])

    def test_hand_disappearing_does_not_fire(self):
        self.assertEqual(run(GestureSnapDetector(), [0.1, 0.1, 0.1, None, None, 1.0]), [])

    def test_open_hand_never_fires(self):
        self.assertEqual(run(GestureSnapDetector(), [1.0] * 30), [])

    def test_thresholds_are_tunable(self):
        seq = [0.4, 0.4, 0.4, 1.0]
        self.assertEqual(run(GestureSnapDetector(), seq), [])
        self.assertEqual(run(GestureSnapDetector(pinch_threshold=0.5, open_threshold=0.8), seq), [3])


def click(n=512, pos=200, amp=0.8):
    x = np.zeros(n, dtype=np.float32)
    x[pos : pos + 6] = amp * np.array([1, -0.8, 0.6, -0.4, 0.2, -0.1])
    return x


class AudioTest(unittest.TestCase):
    def quiet(self, rng, n=512, level=0.003):
        return (rng.standard_normal(n) * level).astype(np.float32)

    def test_click_after_noise_fires(self):
        rng = np.random.default_rng(0)
        det = AudioSnapDetector()
        for _ in range(50):
            self.assertFalse(det.process(self.quiet(rng)))
        self.assertTrue(det.process(self.quiet(rng) + click()))

    def test_click_only_fires_on_onset(self):
        rng = np.random.default_rng(1)
        det = AudioSnapDetector()
        for _ in range(20):
            det.process(self.quiet(rng))
        self.assertTrue(det.process(click()))
        self.assertFalse(det.process(click()))  # tail / repeat not an onset

    def test_quiet_click_below_threshold(self):
        det = AudioSnapDetector()
        self.assertFalse(det.process(click(amp=0.02)))

    def test_sustained_tone_never_fires(self):
        det = AudioSnapDetector()
        t = np.arange(512 * 100) / 44100
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        self.assertFalse(any(det.process(tone[i : i + 512]) for i in range(0, len(tone), 512)))

    def test_loud_speechlike_noise_does_not_fire(self):
        rng = np.random.default_rng(2)
        det = AudioSnapDetector()
        fired = [det.process((rng.standard_normal(512) * 0.1).astype(np.float32)) for _ in range(100)]
        self.assertFalse(any(fired))

    def test_threshold_tunable(self):
        det = AudioSnapDetector(threshold=0.005, ratio=2.0)
        self.assertTrue(det.process(click(amp=0.05)))


class TriggerTest(unittest.TestCase):
    def test_any(self):
        tr = SnapTrigger("any", cooldown=1.0)
        self.assertTrue(tr.event("audio", 0.0))
        self.assertFalse(tr.event("gesture", 0.5))  # cooldown
        self.assertTrue(tr.event("gesture", 1.5))

    def test_gesture_only(self):
        tr = SnapTrigger("gesture")
        self.assertFalse(tr.event("audio", 0))
        self.assertTrue(tr.event("gesture", 1))

    def test_audio_only(self):
        tr = SnapTrigger("audio")
        self.assertFalse(tr.event("gesture", 0))
        self.assertTrue(tr.event("audio", 1))

    def test_confirm_needs_both_in_window(self):
        tr = SnapTrigger("confirm", confirm_window=0.4)
        self.assertFalse(tr.event("audio", 0.0))
        self.assertTrue(tr.event("gesture", 0.3))

    def test_confirm_either_order(self):
        tr = SnapTrigger("confirm", confirm_window=0.4)
        self.assertFalse(tr.event("gesture", 0.0))
        self.assertTrue(tr.event("audio", 0.2))

    def test_confirm_outside_window(self):
        tr = SnapTrigger("confirm", confirm_window=0.4)
        self.assertFalse(tr.event("audio", 0.0))
        self.assertFalse(tr.event("gesture", 1.0))

    def test_confirm_same_source_twice_not_enough(self):
        tr = SnapTrigger("confirm")
        self.assertFalse(tr.event("audio", 0.0))
        self.assertFalse(tr.event("audio", 0.1))

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            SnapTrigger("nope")


if __name__ == "__main__":
    unittest.main()
