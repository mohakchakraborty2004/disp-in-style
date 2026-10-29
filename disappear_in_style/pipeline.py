"""Per-frame processing, free of any camera / window / audio I/O so it can run headless."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from .background import AutoCapture, BackgroundPlate, match_plate
from .config import Config
from .effect import DustEffect
from .snap import GestureSnapDetector, SnapTrigger
from .state import State, StateMachine


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        segment: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        plate: Optional[BackgroundPlate] = None,
    ):
        self.cfg = cfg
        self._segment = segment
        self.plate = plate or BackgroundPlate()
        self.sm = StateMachine(cfg.disintegrate_seconds, cfg.reintegrate_seconds)
        self.effect = DustEffect(
            cfg.width, cfg.height, cfg.max_particles, cfg.particle_size, cfg.wave_width,
            cfg.noise_mix, cfg.wind, cfg.rise, cfg.turbulence, cfg.seed,
        )
        self.gesture = GestureSnapDetector(
            cfg.pinch_threshold, cfg.open_threshold, cfg.min_pinch_frames,
            cfg.max_release_time, cfg.max_pinch_hold,
        )
        self.trigger = SnapTrigger(cfg.mode, cfg.cooldown, cfg.confirm_window)
        self.auto = AutoCapture() if cfg.auto_background and not self.plate.available else None
        self._countdown_until: Optional[float] = None
        self._countdown_left: Optional[float] = None
        self._fallback: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.capturing_background = False

    # -------------------------------------------------------------- controls
    def request_background(self, now: float) -> None:
        """Start the 'step out of frame' countdown, then average a few clean frames."""
        self._countdown_until = now + self.cfg.bg_delay
        self.auto = None

    @property
    def countdown(self) -> Optional[float]:
        return self._countdown_left

    def toggle(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
        """Manual snap (also used by the trigger). Returns True if a transition started."""
        if mask is None:
            mask = self._segment(frame, self.plate.image)
        if not self.sm.snap():
            return False
        reverse = self.sm.state is State.REINTEGRATING
        if self.sm.state is State.DISINTEGRATING and not self.plate.available:
            self._fallback = BackgroundPlate.fallback(frame, mask)
        self.effect.start(frame, mask, reverse=reverse)
        return True

    # ------------------------------------------------------------------ frame
    def process(
        self,
        frame: np.ndarray,
        now: float,
        dt: float,
        hands: Optional[Sequence[np.ndarray]] = None,
        audio_events: Sequence[float] = (),
        manual_toggle: bool = False,
    ) -> np.ndarray:
        cfg = self.cfg
        self._countdown_left = None

        # --- background capture flow: pass the live frame through while sampling it
        if self._countdown_until is not None:
            left = self._countdown_until - now
            if left > 0:
                self._countdown_left = left
                return frame
            self._countdown_until = None
            self.plate.begin_capture(cfg.bg_frames)
        if self.plate.capturing:
            if self.plate.add_frame(frame) and cfg.save_background:
                self.plate.save(cfg.save_background)
            return frame

        mask = self._segment(frame, self.plate.image)
        self.last_mask = mask

        if self.auto is not None and self.sm.state is State.VISIBLE:
            if self.auto.update(float((mask > 0.5).mean())):
                self.plate.begin_capture(cfg.bg_frames)
                self.auto = None
                return frame

        # --- snap detection
        events = [(t, "audio") for t in audio_events]
        if cfg.mode != "audio" and self.gesture.update(now, hands):
            events.append((now, "gesture"))
        fired = manual_toggle
        for t, src in sorted(events):
            fired = self.trigger.event(src, t) or fired
        if fired and not self.sm.in_transition:
            self.toggle(frame, mask)

        self.sm.update(dt)
        st = self.sm.state
        if st is State.VISIBLE:
            return frame
        plate = self.plate.image if self.plate.available else self._fallback
        if plate is None:  # e.g. started GONE somehow; blur-fill from this frame
            plate = self._fallback = BackgroundPlate.fallback(frame, mask)
        if st is State.DISINTEGRATING:
            p = self.sm.progress
        elif st is State.GONE:
            p = 1.0
        else:
            p = 1.0 - self.sm.progress
        return self.effect.render(frame, mask, match_plate(plate, frame, mask), p)
