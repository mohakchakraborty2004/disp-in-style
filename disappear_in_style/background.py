"""Clean background plate: capture, averaging, persistence and an inpainting fallback."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class BackgroundPlate:
    def __init__(self):
        self.image: Optional[np.ndarray] = None  # uint8 BGR
        self._acc: Optional[np.ndarray] = None
        self._need = 0

    @property
    def available(self) -> bool:
        return self.image is not None

    @property
    def capturing(self) -> bool:
        return self._need > 0

    def set(self, image: np.ndarray) -> None:
        self.image = np.ascontiguousarray(image, dtype=np.uint8)

    def load(self, path: str, size: Optional[tuple] = None) -> bool:
        img = cv2.imread(path)
        if img is None:
            return False
        if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
            img = cv2.resize(img, tuple(size))
        self.set(img)
        return True

    def save(self, path: str) -> None:
        if self.image is not None:
            cv2.imwrite(path, self.image)

    def begin_capture(self, frames: int = 15) -> None:
        self._acc = None
        self._need = max(1, int(frames))
        self._n = 0

    def add_frame(self, frame: np.ndarray) -> bool:
        """Accumulate a frame while capturing; returns True when the plate was finalised."""
        if self._need <= 0:
            return False
        f = frame.astype(np.float32)
        self._acc = f if self._acc is None else self._acc + f
        self._n += 1
        if self._n >= self._need:
            self.set(np.clip(self._acc / self._n, 0, 255).astype(np.uint8))
            self._acc, self._need = None, 0
            return True
        return False

    @staticmethod
    def fallback(frame: np.ndarray, mask: np.ndarray, scale: int = 4) -> np.ndarray:
        """Fill in the person region by inpainting + heavy blur (used when no plate was captured)."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        m = cv2.resize((mask > 0.5).astype(np.uint8) * 255, (w // scale, h // scale))
        m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=2)
        if m.any():
            small = cv2.inpaint(small, m, 5, cv2.INPAINT_TELEA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def match_plate(plate: np.ndarray, live: np.ndarray, mask: np.ndarray, scale: int = 8) -> np.ndarray:
    """Re-light the plate to the live frame so the patch where the person was does not stand out.

    Webcams drift (auto exposure / white balance, and the first frames after start are often dark),
    so a stored plate can be darker than the room now looks: pasting it over the person leaves a dark
    silhouette. The live-minus-plate difference is measured where there is no person, smoothly
    extrapolated into the person region (normalised blur) and added to the plate.
    """
    h, w = plate.shape[:2]
    sw, sh = max(1, w // scale), max(1, h // scale)
    ls = cv2.resize(live, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    ps = cv2.resize(plate, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    ms = cv2.resize((mask > 0.02).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
    ms = cv2.dilate(ms, np.ones((5, 5), np.uint8))  # keep the person's rim / shadow out of the sample
    wgt = (ms == 0).astype(np.float32)
    if wgt.sum() < 16:
        return plate
    diff = np.clip(ls - ps, -80.0, 80.0)
    sigma = max(2.0, sw / 10.0)
    num = cv2.GaussianBlur(diff * wgt[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(wgt, (0, 0), sigma)[..., None]
    mean = (diff * wgt[..., None]).sum(axis=(0, 1)) / wgt.sum()
    field = np.where(den > 1e-3, num / np.maximum(den, 1e-3), mean)
    # blend towards the global mean where there is little support nearby
    conf = np.clip(den * 4.0, 0.0, 1.0)
    field = conf * field + (1.0 - conf) * mean
    field = cv2.resize(field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(plate.astype(np.float32) + field, 0, 255).astype(np.uint8)


class AutoCapture:
    """Fires once the person has been out of frame for `stable_frames` consecutive frames."""

    def __init__(self, stable_frames: int = 20, max_person_fraction: float = 0.005):
        self.stable_frames = stable_frames
        self.max_person_fraction = max_person_fraction
        self._count = 0

    def update(self, person_fraction: float) -> bool:
        if person_fraction <= self.max_person_fraction:
            self._count += 1
        else:
            self._count = 0
        if self._count >= self.stable_frames:
            self._count = 0
            return True
        return False
