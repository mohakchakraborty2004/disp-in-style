"""Entry point: webcam -> pipeline -> virtual camera + preview window."""

from __future__ import annotations

import sys
import time

import cv2

from .audio import AudioListener
from .config import Config, parse_args
from .pipeline import Pipeline
from .background import BackgroundPlate
from .snap import AudioSnapDetector
from .state import State


def open_camera(cfg: Config):
    api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.camera, api)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {cfg.camera}. On macOS grant your terminal camera access.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    return cap


def open_virtualcam(cfg: Config):
    import pyvirtualcam

    try:
        return pyvirtualcam.Camera(
            width=cfg.width, height=cfg.height, fps=cfg.fps, fmt=pyvirtualcam.PixelFormat.BGR
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not open the virtual camera: {exc}\n"
            "Install OBS, run it once, click 'Start Virtual Camera' once, then quit OBS "
            "(see README), or run with --no-virtualcam for preview only."
        )


def hud(img, pipe: Pipeline, cfg: Config, fps: float):
    lines = [f"{pipe.sm.state.value}  mode={cfg.mode}  {fps:4.1f} fps"]
    if pipe.countdown is not None:
        lines.append(f"STEP OUT OF FRAME: capturing background in {pipe.countdown:.0f}s")
    elif pipe.plate.capturing:
        lines.append("capturing background...")
    elif not pipe.plate.available:
        lines.append("no background yet (using fallback fill) - press b")
    lines.append("q quit | space toggle | b recapture bg")
    for i, text in enumerate(lines):
        cv2.putText(img, text, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main(argv=None) -> int:
    cfg = parse_args(argv)
    from .vision import HandTracker, Segmenter

    segmenter = Segmenter(cfg.seg_threshold, cfg.mask_grow)
    tracker = HandTracker() if cfg.mode in ("gesture", "any", "confirm") else None
    plate = BackgroundPlate()
    if cfg.background and not plate.load(cfg.background, (cfg.width, cfg.height)):
        print(f"[warn] could not read background image {cfg.background}", file=sys.stderr)
    pipe = Pipeline(cfg, segmenter.segment, plate)

    audio = None
    if cfg.mode in ("audio", "any", "confirm"):
        audio = AudioListener(
            AudioSnapDetector(cfg.audio_threshold, cfg.audio_ratio),
            cfg.audio_device, cfg.audio_samplerate, cfg.audio_block,
        )
        audio.start()

    cap = open_camera(cfg)
    cam = open_virtualcam(cfg) if cfg.virtualcam else None
    if cam:
        print(f"Virtual camera running: {cam.device}. Select it in Discord (Settings > Voice & Video).")
    if not plate.available and cfg.auto_background:
        print("Step out of frame for a second and the background will be captured automatically "
              "(or press 'b').")

    last, fps = time.monotonic(), float(cfg.fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] camera read failed", file=sys.stderr)
                time.sleep(0.05)
                continue
            if frame.shape[1] != cfg.width or frame.shape[0] != cfg.height:
                frame = cv2.resize(frame, (cfg.width, cfg.height))
            if cfg.mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            dt, last = now - last, now
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3)

            hands = tracker.process(frame) if tracker and pipe.sm.state in (State.VISIBLE, State.GONE) else None
            events = audio.drain() if audio else []
            key = cv2.waitKey(1) & 0xFF if cfg.preview else -1
            out = pipe.process(frame, now, dt, hands, events, manual_toggle=(key == ord(" ")))
            if key == ord("q"):
                break
            if key == ord("b"):
                pipe.request_background(now)

            if cam:
                cam.send(out)
            if cfg.preview:
                shown = out.copy()
                hud(shown, pipe, cfg, fps)
                cv2.imshow("disappear-in-style (preview)", shown)
            if cam:
                cam.sleep_until_next_frame()
    except KeyboardInterrupt:
        pass
    finally:
        if audio:
            audio.stop()
        cap.release()
        if cam:
            cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
