"""Transport bar: play / pause / stop / volume / export."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)


class TransportBar(QWidget):
    """Bottom bar with playback controls and the export button."""

    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    volume_changed = pyqtSignal(float)  # 0..1

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.export_btn = QPushButton("Export WAV...")

        self.play_btn.clicked.connect(self.play_clicked.emit)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.export_btn.clicked.connect(self.export_clicked.emit)

        layout.addWidget(self.play_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.stop_btn)

        layout.addSpacing(20)
        layout.addWidget(QLabel("Volume"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(80)
        self.volume_slider.setMaximumWidth(220)
        self.volume_slider.valueChanged.connect(self._on_volume)
        layout.addWidget(self.volume_slider)
        self.volume_label = QLabel("80%")
        self.volume_label.setMinimumWidth(40)
        layout.addWidget(self.volume_label)

        layout.addStretch()
        layout.addWidget(self.export_btn)

    def _on_volume(self, value: int) -> None:
        self.volume_label.setText(f"{value}%")
        self.volume_changed.emit(value / 100.0)
