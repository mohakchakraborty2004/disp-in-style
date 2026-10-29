# disappear-in-style

Snap your fingers and **vanish from your Discord video call like the Thanos snap** in *Avengers: Infinity War*.
You turn to dust in a left-to-right, noise-driven wave, the dust drifts up and sideways and fades,
and your call shows the **empty room behind you**. Snap again and the dust flows back and reforms you.

```
webcam ──► OpenCV ──► person segmentation (MediaPipe) ──► dust effect ──► pyvirtualcam ──► OBS Virtual Camera ──► Discord
                 └──► hand tracking (MediaPipe) ┐
 microphone ─► click detector (sounddevice) ────┴─► snap trigger ─► VISIBLE → DISINTEGRATING → GONE → REINTEGRATING → VISIBLE
```

## Setup (macOS)

1. **Install OBS Studio** (provides the virtual camera driver that pyvirtualcam feeds): <https://obsproject.com> or `brew install --cask obs`.
2. **Register the virtual camera once:** open OBS, click **Start Virtual Camera**, then click **Stop Virtual Camera** and **quit OBS**.
   (OBS must not be running while this app uses the virtual camera. On recent macOS you may need to approve the
   camera extension in *System Settings → Privacy & Security* and restart.)
3. **Python 3.9–3.12** environment and dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Permissions:** the first run asks for **Camera** and **Microphone** access for your terminal app
   (*System Settings → Privacy & Security*). Without the microphone, only the gesture works.

## Run

```bash
python -m disappear_in_style
```

A preview window opens and the virtual camera starts ("OBS Virtual Camera"). Then in **Discord**:
*User Settings → Voice & Video → Camera → select **OBS Virtual Camera*** (and turn on your camera).

Try it without a call: `python -m disappear_in_style --no-virtualcam` (preview only).

### Capturing the empty background

The "gone" frame shows a clean background plate, so the app needs one picture of your room without you in it:

* **Auto:** at startup, step out of frame for about a second and the plate is captured automatically.
* **Manual:** press **b** in the preview window. A countdown (`--bg-delay`, default 3 s) lets you step out of frame, then
  a few frames are averaged. Press `b` again any time (e.g. if the lighting changes). Keep the camera still.
* **From file:** `--background room.png` (and `--save-background room.png` to save a capture for reuse).
* **Fallback:** with no plate, your silhouette is inpainted/blurred from the surrounding room. It works but looks smeared, so capture a real plate.

### Preview keys

| key | action |
|-----|--------|
| `q` | quit |
| `space` | manual snap (test the effect without snapping) |
| `b` | recapture the background |

## Snapping

Snap with a finger snap held in view of the camera and/or an audible snap. `--mode` picks the logic:

| mode | triggers when |
|------|---------------|
| `gesture` | only the camera sees a snap |
| `audio` | only the microphone hears a snap |
| `any` (default) | either fires |
| `confirm` | both fire within `--confirm-window` seconds (fewest false positives) |

* **Gesture:** MediaPipe Hands; fires when thumb + middle fingertip are pinched together and then flick open fast
  (slow releases and long holds such as an "OK" sign are ignored). Tune with `--pinch-threshold`, `--open-threshold`,
  `--min-pinch-frames`, `--max-release-time`, `--max-pinch-hold`. Keep your hand reasonably close and facing the camera; fast
  snaps blur at 30 fps, so loosen `--max-release-time`/`--open-threshold` if it misses.
* **Audio:** a transient detector on the high-passed mic signal (peak above `--audio-threshold`, above `--audio-ratio` × the
  running noise floor, spiky, and an onset). Raise the threshold/ratio in noisy rooms, lower them if snaps are missed.
  Pick a mic with `--audio-device N` (see `python -m sounddevice`).
* `--cooldown` (default 1.5 s) ignores snaps right after one. Snaps during an effect are ignored.

All options: `python -m disappear_in_style --help`. They can also be stored in a JSON file whose keys are the option names
with underscores, e.g. `{"mode": "confirm", "audio_threshold": 0.15}`, passed as `--config my.json` (CLI flags win).

Effect tuning: `--disintegrate-seconds`, `--reintegrate-seconds`, `--wave-width`, `--noise-mix`, `--wind`, `--rise`,
`--turbulence`. Performance: lower `--max-particles` (default 150000) or use `--particle-size 1`, or `--width 960 --height 540`.

## How it works

* `state.py`: `VISIBLE → DISINTEGRATING → GONE → REINTEGRATING → VISIBLE`.
* `effect.py`: every pixel gets a release time (x position mixed with smooth noise). A wave sweeps through; released person
  pixels become particles (from a snapshot) that drift up/sideways with fade. The reintegration plays the same function
  in reverse with a mirrored wave. All numpy/OpenCV vectorised (≈10–20 ms/frame for the effect at 720p on a modest CPU).
* `snap.py`: gesture detector, audio detector and mode/cooldown trigger: pure logic, unit tested.
* `vision.py`: MediaPipe selfie segmentation + hands. If MediaPipe is missing it falls back to differencing against the background plate.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest        # or: python -m unittest discover -s tests -t .
```

Tests cover the state machine, gesture/audio/trigger logic with synthetic input, the config, and the end-to-end frame pipeline with a fake segmenter (no camera needed).

## Troubleshooting

* *"Could not open the virtual camera"*: do the OBS step above, and make sure OBS itself is closed.
* *Discord doesn't list "OBS Virtual Camera"*: restart Discord after the first registration.
* *Feed is mirrored in the preview?* The preview is not mirrored; use `--mirror` to flip the output too.
* *Ghost of you remains / halo*: raise `--mask-grow`, recapture the background, keep lighting stable.
