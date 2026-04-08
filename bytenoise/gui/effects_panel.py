"""Effects chain controls panel.

Each effect has a collapsible group containing:
    * an "enabled" checkbox (the QGroupBox itself is checkable)
    * its parameter sliders
    * a wet/dry mix slider where applicable

The panel emits ``parameters_changed`` whenever any control updates and
exposes ``refresh()`` so undo/redo and randomise can push state back into the
controls without firing change signals.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..engine.effects import EffectsChain


class _ParamRow(QWidget):
    """Slider row with name + value labels and a silent ``set_value()``."""

    def __init__(
        self,
        name: str,
        minimum: int,
        maximum: int,
        initial: int,
        formatter: Callable[[int], str],
        on_change: Callable[[int], None],
    ) -> None:
        super().__init__()
        self._formatter = formatter
        self._on_change = on_change

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(name), 0, 0)
        self.value_label = QLabel(formatter(initial))
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setMinimumWidth(80)
        layout.addWidget(self.value_label, 0, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(initial)
        self.slider.valueChanged.connect(self._handle)
        layout.addWidget(self.slider, 1, 0, 1, 2)

    def _handle(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))
        self._on_change(value)

    def set_value(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.slider.blockSignals(False)
        self.value_label.setText(self._formatter(int(value)))


class EffectsPanel(QWidget):
    """Build all effect groups and route their changes to ``parameters_changed``."""

    parameters_changed = pyqtSignal()

    def __init__(self, effects: EffectsChain) -> None:
        super().__init__()
        self.effects = effects

        # Refreshable widget registries.
        self._sliders: List[Tuple[_ParamRow, Callable[[], int]]] = []
        self._groups: List[Tuple[QGroupBox, Callable[[], bool]]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Effects Chain")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._build_lowpass())
        layout.addWidget(self._build_highpass())
        layout.addWidget(self._build_distortion())
        layout.addWidget(self._build_bitcrusher())
        layout.addWidget(self._build_delay())
        layout.addWidget(self._build_reverb())
        layout.addStretch()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_slider(
        self,
        layout,
        name: str,
        minimum: int,
        maximum: int,
        getter: Callable[[], int],
        setter: Callable[[int], None],
        formatter: Callable[[int], str],
    ) -> _ParamRow:
        def on_change(value: int) -> None:
            setter(value)
            self.parameters_changed.emit()

        row = _ParamRow(name, minimum, maximum, getter(), formatter, on_change)
        layout.addWidget(row)
        self._sliders.append((row, getter))
        return row

    def _make_group(
        self,
        title: str,
        getter: Callable[[], bool],
        setter: Callable[[bool], None],
    ) -> QGroupBox:
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(getter())

        def on_toggle(checked: bool) -> None:
            setter(bool(checked))
            self.parameters_changed.emit()

        group.toggled.connect(on_toggle)
        self._groups.append((group, getter))
        return group

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_lowpass(self) -> QGroupBox:
        fx = self.effects.low_pass
        group = self._make_group(
            "Low-pass filter",
            getter=lambda: fx.enabled,
            setter=lambda v: setattr(fx, "enabled", v),
        )
        layout = QVBoxLayout(group)
        self._add_slider(
            layout, "Cutoff", 50, 18000,
            getter=lambda: int(fx.cutoff),
            setter=lambda v: setattr(fx, "cutoff", float(v)),
            formatter=lambda v: f"{v} Hz",
        )
        self._add_slider(
            layout, "Resonance", 50, 500,
            getter=lambda: int(fx.resonance * 100),
            setter=lambda v: setattr(fx, "resonance", v / 100.0),
            formatter=lambda v: f"{v / 100.0:.2f}",
        )
        self._add_slider(
            layout, "Mix", 0, 100,
            getter=lambda: int(fx.mix * 100),
            setter=lambda v: setattr(fx, "mix", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        return group

    def _build_highpass(self) -> QGroupBox:
        fx = self.effects.high_pass
        group = self._make_group(
            "High-pass filter",
            getter=lambda: fx.enabled,
            setter=lambda v: setattr(fx, "enabled", v),
        )
        layout = QVBoxLayout(group)
        self._add_slider(
            layout, "Cutoff", 20, 12000,
            getter=lambda: int(fx.cutoff),
            setter=lambda v: setattr(fx, "cutoff", float(v)),
            formatter=lambda v: f"{v} Hz",
        )
        self._add_slider(
            layout, "Resonance", 50, 500,
            getter=lambda: int(fx.resonance * 100),
            setter=lambda v: setattr(fx, "resonance", v / 100.0),
            formatter=lambda v: f"{v / 100.0:.2f}",
        )
        self._add_slider(
            layout, "Mix", 0, 100,
            getter=lambda: int(fx.mix * 100),
            setter=lambda v: setattr(fx, "mix", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        return group

    def _build_distortion(self) -> QGroupBox:
        fx = self.effects.distortion
        group = self._make_group(
            "Distortion / Overdrive",
            getter=lambda: fx.enabled,
            setter=lambda v: setattr(fx, "enabled", v),
        )
        layout = QVBoxLayout(group)
        self._add_slider(
            layout, "Drive", 10, 200,
            getter=lambda: int(fx.drive * 10),
            setter=lambda v: setattr(fx, "drive", v / 10.0),
            formatter=lambda v: f"{v / 10.0:.1f}",
        )
        self._add_slider(
            layout, "Tone", 0, 100,
            getter=lambda: int(fx.tone * 100),
            setter=lambda v: setattr(fx, "tone", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        self._add_slider(
            layout, "Mix", 0, 100,
            getter=lambda: int(fx.mix * 100),
            setter=lambda v: setattr(fx, "mix", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        return group

    def _build_bitcrusher(self) -> QGroupBox:
        fx = self.effects.bit_crusher
        group = self._make_group(
            "Bit crusher",
            getter=lambda: fx.enabled,
            setter=lambda v: setattr(fx, "enabled", v),
        )
        layout = QVBoxLayout(group)
        self._add_slider(
            layout, "Bit depth", 2, 16,
            getter=lambda: int(fx.bit_depth),
            setter=lambda v: setattr(fx, "bit_depth", int(v)),
            formatter=lambda v: f"{v}-bit",
        )
        self._add_slider(
            layout, "Rate reduction", 1, 32,
            getter=lambda: int(fx.rate_reduction),
            setter=lambda v: setattr(fx, "rate_reduction", int(v)),
            formatter=lambda v: f"1/{v}",
        )
        self._add_slider(
            layout, "Mix", 0, 100,
            getter=lambda: int(fx.mix * 100),
            setter=lambda v: setattr(fx, "mix", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        return group

    def _build_delay(self) -> QGroupBox:
        fx = self.effects.delay
        group = self._make_group(
            "Delay",
            getter=lambda: fx.enabled,
            setter=lambda v: setattr(fx, "enabled", v),
        )
        layout = QVBoxLayout(group)
        self._add_slider(
            layout, "Time", 10, 1500,
            getter=lambda: int(fx.time_ms),
            setter=lambda v: setattr(fx, "time_ms", float(v)),
            formatter=lambda v: f"{v} ms",
        )
        self._add_slider(
            layout, "Feedback", 0, 95,
            getter=lambda: int(fx.feedback * 100),
            setter=lambda v: setattr(fx, "feedback", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        self._add_slider(
            layout, "Mix", 0, 100,
            getter=lambda: int(fx.mix * 100),
            setter=lambda v: setattr(fx, "mix", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        return group

    def _build_reverb(self) -> QGroupBox:
        fx = self.effects.reverb
        group = self._make_group(
            "Reverb",
            getter=lambda: fx.enabled,
            setter=lambda v: setattr(fx, "enabled", v),
        )
        layout = QVBoxLayout(group)
        self._add_slider(
            layout, "Room size", 0, 100,
            getter=lambda: int(fx.room_size * 100),
            setter=lambda v: setattr(fx, "room_size", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        self._add_slider(
            layout, "Damping", 0, 100,
            getter=lambda: int(fx.damping * 100),
            setter=lambda v: setattr(fx, "damping", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        self._add_slider(
            layout, "Mix", 0, 100,
            getter=lambda: int(fx.mix * 100),
            setter=lambda v: setattr(fx, "mix", v / 100.0),
            formatter=lambda v: f"{v}%",
        )
        return group

    # ------------------------------------------------------------------
    # Public refresh API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read every control from ``self.effects`` without firing changes."""
        for slider, getter in self._sliders:
            try:
                slider.set_value(getter())
            except Exception:
                pass
        for group, getter in self._groups:
            try:
                group.blockSignals(True)
                group.setChecked(bool(getter()))
                group.blockSignals(False)
            except Exception:
                group.blockSignals(False)
