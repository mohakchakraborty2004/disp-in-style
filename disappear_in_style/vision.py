"""MediaPipe wrappers: person segmentation and hand tracking (imported lazily)."""

from __future__ import annotations

import sys
from typing import List, Optional

import cv2
import numpy as np

_mp = None


def _mediapipe():
    global _mp
    if _mp is None:
        try:
            import mediapipe as mp  # noqa: WPS433

            mp.solutions.hands  # legacy "solutions" API
            _mp = mp
        except Exception as exc:  # ImportError or missing solutions API
            raise RuntimeError(f"MediaPipe (solutions API) unavailable: {exc}") from exc
    return _mp


class Segmenter:
    """Soft person mask (float32 HxW at full resolution, 1 = person).

    Uses MediaPipe selfie segmentation; if MediaPipe is missing it falls back to differencing
    against the background plate (needs a captured plate).
    """

    SMALL_W = 256

    def __init__(self, threshold: float = 0.5, grow: int = 5, smoothing: float = 0.5):
        self.threshold = threshold
        self.grow = grow
        self.smoothing = smoothing
        self._prev: Optional[np.ndarray] = None
        self._seg = None
        try:
            self._seg = _mediapipe().solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except RuntimeError as exc:
            print(f"[warn] {exc}\n[warn] falling back to background-difference segmentation "
                  "(capture a clean background with 'b').", file=sys.stderr)

    @property
    def uses_mediapipe(self) -> bool:
        return self._seg is not None

    def segment(self, frame_bgr: np.ndarray, plate_bgr: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        sw = self.SMALL_W
        sh = max(1, round(h * sw / w))
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        if self._seg is not None:
            raw = self._seg.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).segmentation_mask
            raw = np.asarray(raw, dtype=np.float32)
        elif plate_bgr is not None:
            ps = cv2.resize(plate_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            diff = np.abs(small.astype(np.int16) - ps.astype(np.int16)).max(axis=2)
            raw = np.clip((diff.astype(np.float32) - 20.0) / 30.0, 0.0, 1.0)
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        else:
            return np.zeros((h, w), np.float32)
        return self._finish(raw, (w, h))

    def _finish(self, raw: np.ndarray, size) -> np.ndarray:
        if self._prev is not None and self._prev.shape == raw.shape:
            raw = self.smoothing * self._prev + (1.0 - self.smoothing) * raw
        self._prev = raw
        t = self.threshold
        m = np.clip((raw - (t - 0.15)) / 0.3, 0.0, 1.0)  # soft threshold around t
        if self.grow > 0:
            k = 2 * self.grow + 1
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        return cv2.resize(m, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)


class HandTracker:
    """Returns a list of (21, 2) landmark arrays, x scaled by aspect so distances are isotropic."""

    SMALL_W = 480

    def __init__(self):
        self._hands = _mediapipe().solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )

    def process(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        sw = min(self.SMALL_W, w)
        small = cv2.resize(frame_bgr, (sw, round(h * sw / w)), interpolation=cv2.INTER_AREA)
        res = self._hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        out = []
        for hand in res.multi_hand_landmarks or []:
            pts = np.array([[lm.x * w / h, lm.y] for lm in hand.landmark], dtype=np.float32)
            out.append(pts)
        return out
