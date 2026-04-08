"""Tiny horizontal level meter for per-effect activity indicators.

Each effect group in the effects panel embeds one of these. The main window
calls ``set_peak()`` after each render with the peak the chain recorded for
that effect, so the meter reflects what's actually flowing through the chain
right now (not just whether the effect is enabled).

Peak values are smoothed and held briefly so the bar doesn't visually
flicker between renders. The meter colours are green up to ~70%, yellow up
to ~90%, red above that.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QSizePolicy, QWidget


class LevelMeter(QWidget):
    """Compact horizontal peak meter with peak-hold."""

    BG_COLOR = QColor(20, 22, 28)
    BORDER_COLOR = QColor(50, 55, 64)
    GREEN = QColor(110, 220, 140)
    YELLOW = QColor(230, 220, 90)
    RED = QColor(230, 90, 80)
    HOLD_COLOR = QColor(220, 230, 240)

    PEAK_HOLD_SECS = 1.2

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._level: float = 0.0
        self._hold: float = 0.0
        self._hold_t: float = 0.0
        self.setFixedHeight(8)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_peak(self, value: float) -> None:
        v = max(0.0, min(1.0, float(value)))
        self._level = v
        now = time.monotonic()
        if v >= self._hold or (now - self._hold_t) > self.PEAK_HOLD_SECS:
            self._hold = v
            self._hold_t = now
        self.update()

    def reset(self) -> None:
        self._level = 0.0
        self._hold = 0.0
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        rect = self.rect()
        w = rect.width()
        h = rect.height()

        painter.fillRect(rect, self.BG_COLOR)

        if w <= 2 or h <= 2:
            return

        # Coloured fill: split into green/yellow/red zones based on level.
        bar_w = int(w * self._level)
        if bar_w > 0:
            green_end = int(w * 0.7)
            yellow_end = int(w * 0.9)

            x = 0
            if bar_w > 0:
                seg_end = min(bar_w, green_end)
                if seg_end > x:
                    painter.fillRect(x, 1, seg_end - x, h - 2, self.GREEN)
                    x = seg_end
            if bar_w > green_end:
                seg_end = min(bar_w, yellow_end)
                if seg_end > x:
                    painter.fillRect(x, 1, seg_end - x, h - 2, self.YELLOW)
                    x = seg_end
            if bar_w > yellow_end:
                painter.fillRect(x, 1, bar_w - x, h - 2, self.RED)

        # Peak hold marker
        hold_x = int(w * self._hold)
        if hold_x > 0 and hold_x < w:
            painter.fillRect(hold_x - 1, 0, 2, h, self.HOLD_COLOR)
