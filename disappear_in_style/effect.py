"""Dust effect: the person turns to particles in a left-to-right, noise-driven wave.

Model: every pixel has a release threshold T in [0, 1] (x position mixed with smooth noise).
For effect progress p in [0, 1] the wave front is at thr = p * (1 + W). A pixel is "released"
once thr > T, and its particle age is a = clip((thr - T) / W, 0, 1): it drifts up and sideways
and fades as a goes 0 -> 1. p = 1 means everything is dust that has fully faded.

Disintegration plays p = 0 -> 1. Reintegration is the same function played backwards
(p = 1 -> 0) with a mirrored wave so the person reforms from the left, dust flowing back in.

The still-attached part of the person comes from the live frame; the particles are drawn from a
snapshot taken when the effect starts. All per-frame work is numpy/cv2 vectorised.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class DustEffect:
    COVER_GAIN = 4.0

    def __init__(
        self,
        width: int,
        height: int,
        max_particles: int = 150000,
        particle_size: int = 2,
        wave_width: float = 0.6,
        noise_mix: float = 0.25,
        wind: float = 0.35,
        rise: float = 0.3,
        turbulence: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.w, self.h = width, height
        self.max_particles = max_particles
        self.particle_size = particle_size
        self.W = max(1e-3, wave_width)
        self.wind, self.rise, self.turb = wind, rise, turbulence
        self.rng = np.random.default_rng(seed)

        # smooth noise at low res, upscaled
        lw, lh = max(4, width // 16), max(4, height // 16)
        noise = self.rng.random((lh, lw)).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        noise = cv2.normalize(noise, None, 0.0, 1.0, cv2.NORM_MINMAX)
        fine = self.rng.random((height, width)).astype(np.float32)
        noise = 0.8 * noise + 0.2 * fine
        x = np.broadcast_to(np.linspace(0, 1, width, dtype=np.float32), (height, width))
        m = float(np.clip(noise_mix, 0, 1))
        self._t_fwd = np.ascontiguousarray((1 - m) * x + m * noise)
        self._t_rev = np.ascontiguousarray((1 - m) * (1 - x) + m * noise)

        self._dust = np.zeros((height + 2, width + 2, 3), np.uint8)
        self._amap = np.zeros((height + 2, width + 2), np.uint8)
        self._reverse = False
        self._n = 0
        self._ox = self._oy = self._col = self._t = None

    # ------------------------------------------------------------------ setup
    def start(self, frame: np.ndarray, mask: np.ndarray, reverse: bool = False) -> None:
        """Snapshot the person's pixels as particles. `mask` is a float/bool HxW array."""
        self._reverse = reverse
        tmap = self._t_rev if reverse else self._t_fwd
        ys, xs = np.nonzero(mask > 0.5)
        if len(xs) > self.max_particles:
            pick = self.rng.choice(len(xs), self.max_particles, replace=False)
            ys, xs = ys[pick], xs[pick]
        n = len(xs)
        self._n = n
        self._ox = xs.astype(np.float32)
        self._oy = ys.astype(np.float32)
        self._col = np.ascontiguousarray(frame[ys, xs])
        self._t = tmap[ys, xs]
        r = self.rng
        # per-particle motion, as multiples of frame size
        self._vx = (self.wind * (0.3 + 1.2 * r.random(n)) * self.w).astype(np.float32)
        self._vy = (-self.rise * (0.2 + 1.3 * r.random(n)) * self.h).astype(np.float32)
        self._phase = (r.random(n) * 6.2832).astype(np.float32)
        self._freq = (3.0 + 5.0 * r.random(n)).astype(np.float32)
        self._amp = (self.turb * self.w * (0.3 + r.random(n))).astype(np.float32)

    # ----------------------------------------------------------------- render
    def render(
        self,
        live: np.ndarray,
        mask: np.ndarray,
        plate: np.ndarray,
        p: float,
    ) -> np.ndarray:
        """Compose the output frame for effect progress p (0 = person intact, 1 = fully gone).

        `mask` is the live person mask as float32 HxW in [0, 1] (soft edges are fine).
        """
        p = float(np.clip(p, 0.0, 1.0))
        if p <= 0.0:
            return live
        tmap = self._t_rev if self._reverse else self._t_fwd
        thr = p * (1.0 + self.W)
        # hidden = person pixels already released from the wave
        # soft mask edges would let a faint copy of the person through; cover anything above ~25 %
        hidden = np.clip(mask * self.COVER_GAIN, 0.0, 1.0) * (tmap <= thr)
        out = cv2.blendLinear(plate, live, hidden, 1.0 - hidden)
        if p < 1.0 and self._n:
            self._draw_particles(out, thr)
        return out

    def _draw_particles(self, out: np.ndarray, thr: float) -> None:
        age = (thr - self._t) * (1.0 / self.W)
        active = np.flatnonzero((age > 0.0) & (age < 1.0))
        if active.size == 0:
            return
        a = age[active]
        ease = a ** 1.4
        x = self._ox[active] + self._vx[active] * ease
        x += self._amp[active] * np.sin(self._freq[active] * a + self._phase[active])
        y = self._oy[active] + self._vy[active] * ease
        # clip into a 1px guard border: off-screen dust piles up there and is cropped away
        xi = np.clip(x, -1.0, self.w).astype(np.int32)
        yi = np.clip(y, -1.0, self.h).astype(np.int32)
        xi += 1
        yi += 1
        alpha = (255.0 * (1.0 - a) ** 0.8).astype(np.uint8)

        # scatter-write (no gather) into a dust layer, then composite once
        dust, amap = self._dust, self._amap
        dust[:] = 0
        amap[:] = 0
        dust[yi, xi] = self._col[active]
        amap[yi, xi] = alpha
        dust = dust[1:-1, 1:-1]
        amap = amap[1:-1, 1:-1]
        if self.particle_size == 2:
            k = np.ones((2, 2), np.uint8)
            dust = cv2.dilate(dust, k)
            amap = cv2.dilate(amap, k)
        w = amap.astype(np.float32) * (1.0 / 255.0)
        out[:] = cv2.blendLinear(out, np.ascontiguousarray(dust), 1.0 - w, w)
