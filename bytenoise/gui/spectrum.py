"""Spectrum analyser widget.

Renders a log-frequency, log-magnitude FFT view of the current audio buffer.
Like the waveform widget it's purely passive: ``set_buffer()`` recomputes
the spectrum and triggers a repaint. To stay fast on long buffers we only
analyse a single window from the start of the buffer (large enough to give
decent low-frequency resolution) rather than averaging the whole thing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget


# Window size used for the FFT. 8192 samples = ~5 Hz resolution at 44.1 kHz.
FFT_SIZE = 8192


class SpectrumView(QWidget):
    """Log-log spectrum analyser display."""

    BG_COLOR = QColor(20, 22, 28)
    GRID_COLOR = QColor(40, 44, 52)
    BAR_COLOR = QColor(140, 200, 240)
    LABEL_COLOR = QColor(120, 130, 140)

    # Display range in dB.
    DB_FLOOR = -90.0
    DB_CEIL = 0.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._magnitudes_db: np.ndarray = np.zeros(0, dtype=np.float32)
        self._sample_rate: int = 44100
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_buffer(self, buffer: np.ndarray, sample_rate: int) -> None:
        """Recompute the spectrum from a fresh buffer."""
        self._sample_rate = int(sample_rate)
        if buffer is None or buffer.size < 4:
            self._magnitudes_db = np.zeros(0, dtype=np.float32)
            self.update()
            return

        n = min(FFT_SIZE, buffer.size)
        # Take a Hann-windowed slice from somewhere near the middle of the
        # buffer to avoid the start/end fades pulling the spectrum down.
        start = max(0, (buffer.size - n) // 2)
        slice_ = buffer[start : start + n].astype(np.float32, copy=False)
        window = np.hanning(n).astype(np.float32)
        spec = np.fft.rfft(slice_ * window)
        mag = np.abs(spec).astype(np.float32) / (n * 0.5)
        # Convert to dBFS, clamp the floor.
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(np.maximum(mag, 1e-9))
        self._magnitudes_db = db.astype(np.float32)
        self.update()

    def clear(self) -> None:
        self._magnitudes_db = np.zeros(0, dtype=np.float32)
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect()
        w = rect.width()
        h = rect.height()

        painter.fillRect(rect, self.BG_COLOR)

        if self._magnitudes_db.size == 0 or w <= 0:
            painter.setPen(QPen(self.LABEL_COLOR))
            painter.drawText(rect, Qt.AlignCenter, "No spectrum data")
            return

        # Frequency axis: log-spaced from 20 Hz to Nyquist.
        nyq = self._sample_rate * 0.5
        f_min = 20.0
        f_max = nyq
        bin_freqs = np.linspace(0.0, nyq, self._magnitudes_db.size, dtype=np.float32)

        # Build column->bin mapping by computing the target frequency for each
        # pixel column on a log scale, then finding the nearest FFT bin.
        cols = w
        log_min = np.log10(f_min)
        log_max = np.log10(f_max)
        col_freqs = np.logspace(log_min, log_max, cols)
        # For each column, average all FFT bins that fall inside that pixel.
        bin_indices = np.searchsorted(bin_freqs, col_freqs).clip(0, self._magnitudes_db.size - 1)

        col_db = self._magnitudes_db[bin_indices]

        # Map dB to pixel y.
        usable_h = h - 4
        y_floor = h - 2
        clipped = np.clip(col_db, self.DB_FLOOR, self.DB_CEIL)
        norm = (clipped - self.DB_FLOOR) / (self.DB_CEIL - self.DB_FLOOR)
        bar_heights = (norm * usable_h).astype(np.int32)

        # Grid lines at -60, -40, -20 dB.
        painter.setPen(QPen(self.GRID_COLOR, 1, Qt.DotLine))
        for db_line in (-60.0, -40.0, -20.0):
            n_line = (db_line - self.DB_FLOOR) / (self.DB_CEIL - self.DB_FLOOR)
            y = int(y_floor - n_line * usable_h)
            painter.drawLine(0, y, w, y)

        painter.setPen(QPen(self.BAR_COLOR, 1))
        for x in range(cols):
            top = y_floor - int(bar_heights[x])
            painter.drawLine(x, y_floor, x, top)
