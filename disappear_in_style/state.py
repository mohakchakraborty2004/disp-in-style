"""VISIBLE -> DISINTEGRATING -> GONE -> REINTEGRATING -> VISIBLE."""

from __future__ import annotations

from enum import Enum


class State(Enum):
    VISIBLE = "visible"
    DISINTEGRATING = "disintegrating"
    GONE = "gone"
    REINTEGRATING = "reintegrating"


class StateMachine:
    """Pure logic, no clocks: the caller supplies dt in `update`.

    A snap in a stable state starts the next transition; snaps during a
    transition are ignored so the effect always plays out in full.
    """

    def __init__(self, disintegrate_seconds: float = 3.0, reintegrate_seconds: float = 2.5):
        if disintegrate_seconds <= 0 or reintegrate_seconds <= 0:
            raise ValueError("durations must be positive")
        self.disintegrate_seconds = disintegrate_seconds
        self.reintegrate_seconds = reintegrate_seconds
        self.state = State.VISIBLE
        self.elapsed = 0.0

    @property
    def progress(self) -> float:
        """0..1 through the current transition; 0 when VISIBLE, 1 when GONE."""
        if self.state is State.VISIBLE:
            return 0.0
        if self.state is State.GONE:
            return 1.0
        total = (
            self.disintegrate_seconds
            if self.state is State.DISINTEGRATING
            else self.reintegrate_seconds
        )
        return min(1.0, self.elapsed / total)

    @property
    def in_transition(self) -> bool:
        return self.state in (State.DISINTEGRATING, State.REINTEGRATING)

    def snap(self) -> bool:
        """Returns True if the snap started a transition."""
        if self.state is State.VISIBLE:
            self.state = State.DISINTEGRATING
        elif self.state is State.GONE:
            self.state = State.REINTEGRATING
        else:
            return False
        self.elapsed = 0.0
        return True

    def update(self, dt: float) -> State:
        if self.in_transition:
            self.elapsed += max(0.0, dt)
            if self.progress >= 1.0:
                self.state = State.GONE if self.state is State.DISINTEGRATING else State.VISIBLE
                self.elapsed = 0.0
        return self.state
