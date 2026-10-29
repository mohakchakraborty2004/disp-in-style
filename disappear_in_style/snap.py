"""Snap detection: hand-gesture detector, audio click detector, and a trigger that combines them.

All detectors are pure logic fed with explicit timestamps so they can be unit tested
with synthetic input.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# MediaPipe hand landmark indices
WRIST, THUMB_TIP, MIDDLE_MCP, MIDDLE_TIP = 0, 4, 9, 12


def pinch_ratio(landmarks: np.ndarray) -> float:
    """Thumb-tip to middle-fingertip distance, normalised by palm size (wrist to middle MCP).

    `landmarks` is (21, >=2); x/y must be in the same units (scale x by aspect ratio first).
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    palm = np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2])
    if palm < 1e-6:
        return float("inf")
    return float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[MIDDLE_TIP, :2]) / palm)


class _HandSlot:
    __slots__ = ("pinched", "frames", "start", "last_pinched")

    def __init__(self):
        self.reset()

    def reset(self):
        self.pinched = False
        self.frames = 0
        self.start = 0.0
        self.last_pinched = 0.0


class GestureSnapDetector:
    """Detects "thumb and middle finger pinched, then flicked open fast".

    Per hand: IDLE -> PINCHED (ratio < pinch_threshold) -> fire when the ratio goes above
    open_threshold within `max_release_time` of the last pinched frame, after at least
    `min_pinch_frames` pinched frames and no longer than `max_pinch_hold`.
    Slow releases, long holds (e.g. an OK sign) and hands that vanish never fire.
    """

    def __init__(
        self,
        pinch_threshold: float = 0.30,
        open_threshold: float = 0.60,
        min_pinch_frames: int = 2,
        max_release_time: float = 0.20,
        max_pinch_hold: float = 1.5,
    ):
        self.pinch_threshold = pinch_threshold
        self.open_threshold = open_threshold
        self.min_pinch_frames = min_pinch_frames
        self.max_release_time = max_release_time
        self.max_pinch_hold = max_pinch_hold
        self._slots: Dict[int, _HandSlot] = {}

    def update(self, t: float, hands: Optional[Sequence[np.ndarray]]) -> bool:
        """Feed one frame of hand landmarks (possibly empty/None). Returns True on a snap."""
        hands = hands or []
        fired = False
        for idx in list(self._slots):
            if idx >= len(hands):
                del self._slots[idx]
        for idx, lm in enumerate(hands):
            slot = self._slots.setdefault(idx, _HandSlot())
            if self._step(slot, t, pinch_ratio(lm)):
                fired = True
        if fired:
            self._slots.clear()
        return fired

    def _step(self, s: _HandSlot, t: float, r: float) -> bool:
        if r < self.pinch_threshold:
            if not s.pinched:
                s.pinched, s.frames, s.start = True, 0, t
            s.frames += 1
            s.last_pinched = t
            return False
        if not s.pinched:
            return False
        if r > self.open_threshold:
            ok = (
                s.frames >= self.min_pinch_frames
                and t - s.last_pinched <= self.max_release_time
                and s.last_pinched - s.start <= self.max_pinch_hold
            )
            s.reset()
            return ok
        # in the hysteresis band: give up if the release is too slow to be a flick
        if t - s.last_pinched > self.max_release_time:
            s.reset()
        return False


class AudioSnapDetector:
    """Click/transient detector for mono float audio blocks in [-1, 1].

    The block is high-passed (first difference) so speech and hum matter less; a snap fires
    when the high-passed peak is above an absolute threshold, well above the running noise
    floor, spiky (high crest factor) and an onset (previous block was much quieter).
    """

    def __init__(
        self,
        threshold: float = 0.08,
        ratio: float = 6.0,
        crest_min: float = 3.0,
        floor_decay: float = 0.98,
    ):
        self.threshold = threshold
        self.ratio = ratio
        self.crest_min = crest_min
        self.floor_decay = floor_decay
        self.floor = 1e-3
        self._prev_peak = 0.0
        self._last_sample = 0.0

    def process(self, block: np.ndarray) -> bool:
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return False
        hp = np.diff(x, prepend=np.float32(self._last_sample)) * 0.5
        self._last_sample = float(x[-1])
        a = np.abs(hp)
        peak = float(a.max())
        rms = float(np.sqrt(np.mean(hp * hp))) + 1e-9
        fired = (
            peak > self.threshold
            and peak > self.ratio * self.floor
            and peak / rms > self.crest_min
            and self._prev_peak < 0.5 * peak
        )
        if not fired:
            # learn the floor only from non-events so a click doesn't desensitise us
            self.floor = self.floor_decay * self.floor + (1 - self.floor_decay) * peak
        self._prev_peak = peak
        return fired


class SnapTrigger:
    """Combines gesture/audio events according to the mode, with a cooldown.

    modes: "gesture", "audio" (only that source counts), "any" (either fires),
    "confirm" (both sources within `confirm_window` seconds).
    """

    def __init__(self, mode: str = "any", cooldown: float = 1.5, confirm_window: float = 0.4):
        if mode not in ("gesture", "audio", "any", "confirm"):
            raise ValueError(f"bad mode {mode!r}")
        self.mode = mode
        self.cooldown = cooldown
        self.confirm_window = confirm_window
        self._last_fire = -float("inf")
        self._last: Dict[str, float] = {}

    def event(self, source: str, t: float) -> bool:
        """Report a raw detection from `source` ("gesture" or "audio") at time t.

        Returns True if this event triggers a snap.
        """
        if source not in ("gesture", "audio"):
            raise ValueError(f"bad source {source!r}")
        if self.mode in ("gesture", "audio") and source != self.mode:
            return False
        if t - self._last_fire < self.cooldown:
            return False
        if self.mode == "confirm":
            self._last[source] = t
            other = "audio" if source == "gesture" else "gesture"
            if other in self._last and abs(t - self._last[other]) <= self.confirm_window:
                return self._fire(t)
            return False
        return self._fire(t)

    def _fire(self, t: float) -> bool:
        self._last_fire = t
        self._last.clear()
        return True
