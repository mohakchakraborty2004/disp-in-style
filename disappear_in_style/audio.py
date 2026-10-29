"""Microphone listener: runs AudioSnapDetector on a sounddevice stream."""

from __future__ import annotations

import queue
import sys
import time
from typing import List

from .snap import AudioSnapDetector


class AudioListener:
    def __init__(self, detector: AudioSnapDetector, device=None, samplerate=44100, block=512):
        self.detector = detector
        self.device, self.samplerate, self.block = device, samplerate, block
        self._events: "queue.SimpleQueue[float]" = queue.SimpleQueue()
        self._stream = None

    def start(self) -> bool:
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.block,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            return True
        except Exception as exc:
            print(f"[warn] microphone unavailable ({exc}); audio snap detection disabled. "
                  "Grant your terminal microphone access in System Settings > Privacy & Security.",
                  file=sys.stderr)
            self._stream = None
            return False

    def _callback(self, indata, frames, time_info, status):  # audio thread
        if self.detector.process(indata[:, 0]):
            self._events.put(time.monotonic())

    def drain(self) -> List[float]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
