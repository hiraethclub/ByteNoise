"""Waveform display widget.

Renders a downsampled view of the current audio buffer using QPainter. The
display is updated whenever the engine produces a new buffer (via
``set_buffer``). To stay responsive on large buffers we precompute a
min/max envelope at a fixed number of horizontal pixels rather than drawing
every sample.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget


class WaveformView(QWidget):
    """A simple buffer-backed waveform display."""

    BG_COLOR = QColor(20, 22, 28)
    GRID_COLOR = QColor(40, 44, 52)
    WAVE_COLOR = QColor(110, 220, 180)
    CENTER_COLOR = QColor(70, 80, 90)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_buffer(self, buffer: np.ndarray) -> None:
        if buffer is None:
            self._buffer = np.zeros(0, dtype=np.float32)
        else:
            self._buffer = np.asarray(buffer, dtype=np.float32)
        self.update()

    def clear(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
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

        # Background
        painter.fillRect(rect, self.BG_COLOR)

        # Center line
        center_y = h // 2
        painter.setPen(QPen(self.CENTER_COLOR, 1, Qt.DashLine))
        painter.drawLine(0, center_y, w, center_y)

        if self._buffer.size == 0 or w <= 0:
            painter.setPen(QPen(QColor(120, 130, 140)))
            painter.drawText(rect, Qt.AlignCenter, "No audio buffer loaded")
            return

        # Build min/max envelope at one pair of values per pixel column.
        n = self._buffer.size
        cols = max(1, w)
        # Bin sizes - use at least 1 sample per bin.
        if n >= cols:
            # Reshape into (cols, ~) where possible.
            samples_per_col = n // cols
            usable = samples_per_col * cols
            view = self._buffer[:usable].reshape(cols, samples_per_col)
            mins = view.min(axis=1)
            maxs = view.max(axis=1)
        else:
            # Stretch the smaller buffer across the columns.
            idx = np.linspace(0, n - 1, cols).astype(np.int64)
            mins = self._buffer[idx]
            maxs = mins

        # Map values from -1..1 to pixel y coordinates.
        amp = (h * 0.5) - 4
        y_max = (center_y - maxs * amp).astype(np.int32)
        y_min = (center_y - mins * amp).astype(np.int32)

        painter.setPen(QPen(self.WAVE_COLOR, 1))
        for x in range(cols):
            painter.drawLine(x, int(y_min[x]), x, int(y_max[x]))
