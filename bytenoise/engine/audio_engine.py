"""Audio playback engine.

Wraps ``sounddevice.OutputStream`` and exposes a small API:

    engine = AudioEngine()
    engine.set_buffer(buffer, sample_rate)
    engine.play(); engine.pause(); engine.stop()
    engine.set_volume(0.8)

The engine reads from a NumPy buffer that callers can replace at any time
(e.g. when synthesis parameters change). Playback is wrap-around looping so
the user always has audio while tweaking knobs.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

try:  # pragma: no cover - sounddevice may be missing on headless CI.
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None


class AudioEngine:
    """A buffer-backed playback engine with master volume."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._sample_rate: int = 44100
        self._position: int = 0
        self._volume: float = 0.8
        self._stream: Optional["sd.OutputStream"] = None
        self._is_playing: bool = False

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def set_buffer(self, buffer: np.ndarray, sample_rate: int) -> None:
        """Replace the current playback buffer.

        Restarts the underlying stream if the sample rate changed. Playback
        position is preserved as a fractional offset so live re-renders feel
        continuous.
        """
        with self._lock:
            new_buf = np.ascontiguousarray(buffer, dtype=np.float32)
            if new_buf.size == 0:
                self._buffer = np.zeros(1, dtype=np.float32)
            else:
                self._buffer = new_buf

            if sample_rate != self._sample_rate:
                self._sample_rate = int(sample_rate)
                # Need to recreate the stream at the new rate.
                self._teardown_stream_locked()

            # Clamp position into the new buffer.
            if self._buffer.size > 0:
                self._position %= self._buffer.size

    def get_buffer(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()

    def get_sample_rate(self) -> int:
        return self._sample_rate

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = float(np.clip(volume, 0.0, 1.0))

    def get_volume(self) -> float:
        return self._volume

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def play(self) -> None:
        if sd is None:
            return
        with self._lock:
            if self._stream is None:
                self._create_stream_locked()
            if self._stream is not None and not self._is_playing:
                self._stream.start()
                self._is_playing = True

    def pause(self) -> None:
        if sd is None:
            return
        with self._lock:
            if self._stream is not None and self._is_playing:
                self._stream.stop()
                self._is_playing = False

    def stop(self) -> None:
        if sd is None:
            return
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
            self._is_playing = False
            self._position = 0

    def shutdown(self) -> None:
        if sd is None:
            return
        with self._lock:
            self._teardown_stream_locked()

    def is_playing(self) -> bool:
        return self._is_playing

    # ------------------------------------------------------------------
    # Internal stream wiring
    # ------------------------------------------------------------------

    def _create_stream_locked(self) -> None:
        if sd is None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
                blocksize=1024,
            )
        except Exception:
            # Audio device might not be available; fail soft so the GUI still
            # comes up and the user can still export.
            self._stream = None

    def _teardown_stream_locked(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:  # pragma: no cover
        # Called from sounddevice's audio thread; keep this minimal.
        with self._lock:
            buf = self._buffer
            n = buf.size
            vol = self._volume
            pos = self._position
            if n == 0:
                outdata[:] = 0
                return
            # Wrap-around fill.
            indices = (np.arange(frames) + pos) % n
            outdata[:, 0] = buf[indices] * vol
            self._position = int((pos + frames) % n)
