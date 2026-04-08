"""Layers panel: list of loaded files with per-layer gain and remove.

Owns nothing - the model lives on the main window as a list of
``LoadedFile`` instances. The panel just displays them, lets the user
adjust each layer's gain, and emits signals when something needs to
happen (the user removed a layer, or changed a gain).

The main window calls ``set_layers()`` whenever the underlying list
changes (after add / remove / load preset etc) and the panel rebuilds
its rows. Gain edits emit ``gain_changed`` and don't trigger a rebuild.
"""

from __future__ import annotations

from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..engine.file_reader import LoadedFile


class LayersPanel(QWidget):
    """A vertical list of layer rows shown above the synth panel."""

    layer_removed = pyqtSignal(int)   # emitted with the layer's index
    gain_changed = pyqtSignal()       # emitted on any gain slider movement

    def __init__(self) -> None:
        super().__init__()
        self._layers: List[LoadedFile] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(4)

        self._group = QGroupBox("Layers")
        outer.addWidget(self._group)

        self._rows_layout = QVBoxLayout(self._group)
        self._rows_layout.setContentsMargins(8, 14, 8, 8)
        self._rows_layout.setSpacing(4)

        self._empty_label = QLabel("No files loaded.\nClick \u201cLoad file\u201d above.")
        self._empty_label.setStyleSheet("color: #888;")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._rows_layout.addWidget(self._empty_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_layers(self, layers: List[LoadedFile]) -> None:
        """Rebuild the row list from the given layers."""
        self._layers = list(layers)

        # Drop existing rows.
        while self._rows_layout.count() > 0:
            item = self._rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        if not self._layers:
            self._rows_layout.addWidget(self._empty_label)
            self._empty_label.show()
            return

        for i, layer in enumerate(self._layers):
            self._rows_layout.addWidget(self._build_row(i, layer))

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def _build_row(self, index: int, layer: LoadedFile) -> QWidget:
        row = QWidget()
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        name = QLabel(layer.name)
        name.setStyleSheet("font-weight: bold;")
        name.setToolTip(layer.path)
        header.addWidget(name, 1)

        size_label = QLabel(_short_size(layer.size))
        size_label.setStyleSheet("color: #888;")
        header.addWidget(size_label)

        remove_btn = QPushButton("\u2715")
        remove_btn.setFixedWidth(24)
        remove_btn.setToolTip("Remove this layer")
        remove_btn.clicked.connect(lambda _=False, i=index: self.layer_removed.emit(i))
        header.addWidget(remove_btn)
        v.addLayout(header)

        gain_row = QHBoxLayout()
        gain_row.setContentsMargins(0, 0, 0, 0)
        gain_row.addWidget(QLabel("Gain"))
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(200)
        slider.setValue(int(round(layer.gain * 100)))
        gain_label = QLabel(f"{int(round(layer.gain * 100))}%")
        gain_label.setMinimumWidth(40)
        gain_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def on_change(value: int, lyr: LoadedFile = layer, lbl: QLabel = gain_label) -> None:
            lyr.gain = value / 100.0
            lbl.setText(f"{value}%")
            self.gain_changed.emit()

        slider.valueChanged.connect(on_change)
        gain_row.addWidget(slider, 1)
        gain_row.addWidget(gain_label)
        v.addLayout(gain_row)

        return row


def _short_size(size: int) -> str:
    val = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024.0:
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} TB"
