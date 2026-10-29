"""Configuration: one dataclass, exposed as CLI flags and an optional JSON file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields
from typing import Optional, Sequence

MODES = ("gesture", "audio", "any", "confirm")


def _f(default, help_text):
    return field(default=default, metadata={"help": help_text})


@dataclass
class Config:
    # --- video ---
    camera: int = _f(0, "OpenCV index of the real webcam")
    width: int = _f(1280, "capture / output width")
    height: int = _f(720, "capture / output height")
    fps: int = _f(30, "target frame rate")
    mirror: bool = _f(False, "mirror the output (the preview is NOT mirrored by default)")
    virtualcam: bool = _f(True, "send frames to the virtual camera (--no-virtualcam = preview only)")
    preview: bool = _f(True, "show the local preview window (--no-preview for headless)")

    # --- snap detection ---
    mode: str = _f("any", "snap trigger mode: gesture | audio | any (either fires) | confirm (both must fire)")
    cooldown: float = _f(1.5, "seconds after a snap during which further snaps are ignored")
    confirm_window: float = _f(0.4, "confirm mode: max seconds between gesture and audio events")

    # gesture (MediaPipe hands)
    pinch_threshold: float = _f(0.30, "thumb-middle distance / palm size below which fingers count as pinched")
    open_threshold: float = _f(0.60, "thumb-middle distance / palm size above which fingers count as released")
    min_pinch_frames: int = _f(2, "frames the pinch must be seen before a release counts")
    max_release_time: float = _f(0.20, "max seconds between last pinched frame and the open frame")
    max_pinch_hold: float = _f(1.5, "pinches held longer than this are not snaps")

    # audio (sounddevice)
    audio_device: Optional[int] = _f(None, "sounddevice input device index (default: system default)")
    audio_samplerate: int = _f(44100, "microphone sample rate")
    audio_block: int = _f(512, "microphone block size in samples")
    audio_threshold: float = _f(0.08, "absolute minimum high-passed peak (0..1) for a click")
    audio_ratio: float = _f(6.0, "click peak must exceed this multiple of the running noise floor")

    # --- state machine ---
    disintegrate_seconds: float = _f(3.0, "duration of the dust-away effect")
    reintegrate_seconds: float = _f(2.5, "duration of the reform effect")

    # --- effect ---
    max_particles: int = _f(150000, "max dust particles (lower = faster)")
    particle_size: int = _f(2, "particle size in px (1 or 2)")
    wave_width: float = _f(0.6, "how long each particle takes to fly, as a fraction of the effect")
    noise_mix: float = _f(0.25, "0 = clean left-to-right wipe, 1 = pure noise-driven dissolve")
    wind: float = _f(0.35, "sideways drift distance as fraction of frame width")
    rise: float = _f(0.3, "upward drift distance as fraction of frame height")
    turbulence: float = _f(0.02, "swirl amplitude as fraction of frame width")

    # --- segmentation / background ---
    seg_threshold: float = _f(0.5, "person mask threshold (0..1)")
    mask_grow: int = _f(3, "grow the person mask by this many (low-res) pixels to hide halos")
    background: Optional[str] = _f(None, "path to a saved background image to use instead of capturing")
    auto_background: bool = _f(True, "auto-capture the background at startup once you step out of frame")
    bg_delay: float = _f(3.0, "seconds countdown after pressing 'b' before the background is captured")
    bg_frames: int = _f(15, "frames averaged when capturing the background")
    save_background: Optional[str] = _f(None, "save the captured background to this path")

    seed: Optional[int] = _f(None, "random seed (for reproducible dust)")

    def validate(self) -> "Config":
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.open_threshold <= self.pinch_threshold:
            raise ValueError("open_threshold must be greater than pinch_threshold")
        if self.particle_size not in (1, 2):
            raise ValueError("particle_size must be 1 or 2")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("width/height/fps must be positive")
        return self


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="disappear-in-style",
        description="Snap your fingers and vanish from your video call, Thanos-style.",
    )
    p.add_argument("--config", help="JSON file whose keys are the option names below (CLI flags override)")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        help_text = f.metadata.get("help", "") + f" [default: {f.default}]"
        if f.name == "mode":
            p.add_argument(flag, choices=MODES, help=help_text)
        elif isinstance(f.default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, help=help_text)
        elif f.default is None:
            typ = int if f.name in ("audio_device", "seed") else str
            p.add_argument(flag, type=typ, help=help_text)
        else:
            p.add_argument(flag, type=type(f.default), help=help_text)
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = build_parser()
    ns = parser.parse_args(argv)
    values = {}
    if ns.config:
        with open(ns.config) as fh:
            values.update(json.load(fh))
    known = {f.name for f in fields(Config)}
    unknown = set(values) - known
    if unknown:
        parser.error(f"unknown keys in config file: {sorted(unknown)}")
    for name in known:
        v = getattr(ns, name)
        if v is not None:
            values[name] = v
    return Config(**values).validate()
