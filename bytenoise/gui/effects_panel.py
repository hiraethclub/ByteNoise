"""Effects chain controls panel.

Each effect has a collapsible group containing:
    * an "enabled" checkbox
    * its parameter sliders
    * a wet/dry mix slider where applicable

The panel emits ``parameters_changed`` whenever any control updates.
"""

from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..engine.effects import EffectsChain


def _make_slider(
    minimum: int,
    maximum: int,
    initial: int,
    on_change: Callable[[int], None],
) -> QSlider:
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(initial)
    slider.valueChanged.connect(on_change)
    return slider


class _ParamRow(QWidget):
    """A row containing a name label, a value label, and a slider."""

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

        self.slider = _make_slider(minimum, maximum, initial, self._handle)
        layout.addWidget(self.slider, 1, 0, 1, 2)

    def _handle(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))
        self._on_change(value)


class EffectsPanel(QWidget):
    """Build all effect groups and route their changes to ``parameters_changed``."""

    parameters_changed = pyqtSignal()

    def __init__(self, effects: EffectsChain) -> None:
        super().__init__()
        self.effects = effects

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Effects Chain")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        outer.addWidget(title)

        # Use a scroll area so the panel still fits on smaller screens.
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
    # Group builders
    # ------------------------------------------------------------------

    def _make_group(self, title: str, enabled: bool, on_toggle: Callable[[bool], None]) -> QGroupBox:
        group = QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(enabled)
        group.toggled.connect(on_toggle)
        return group

    def _build_lowpass(self) -> QGroupBox:
        fx = self.effects.low_pass

        def on_toggle(checked: bool) -> None:
            fx.enabled = bool(checked)
            self.parameters_changed.emit()

        group = self._make_group("Low-pass filter", fx.enabled, on_toggle)
        layout = QVBoxLayout(group)

        def set_cutoff(v: int) -> None:
            fx.cutoff = float(v)
            self.parameters_changed.emit()

        def set_res(v: int) -> None:
            fx.resonance = v / 100.0
            self.parameters_changed.emit()

        def set_mix(v: int) -> None:
            fx.mix = v / 100.0
            self.parameters_changed.emit()

        layout.addWidget(_ParamRow("Cutoff", 50, 18000, int(fx.cutoff),
                                   lambda v: f"{v} Hz", set_cutoff))
        layout.addWidget(_ParamRow("Resonance", 50, 500, int(fx.resonance * 100),
                                   lambda v: f"{v / 100.0:.2f}", set_res))
        layout.addWidget(_ParamRow("Mix", 0, 100, int(fx.mix * 100),
                                   lambda v: f"{v}%", set_mix))
        return group

    def _build_highpass(self) -> QGroupBox:
        fx = self.effects.high_pass

        def on_toggle(checked: bool) -> None:
            fx.enabled = bool(checked)
            self.parameters_changed.emit()

        group = self._make_group("High-pass filter", fx.enabled, on_toggle)
        layout = QVBoxLayout(group)

        def set_cutoff(v: int) -> None:
            fx.cutoff = float(v)
            self.parameters_changed.emit()

        def set_res(v: int) -> None:
            fx.resonance = v / 100.0
            self.parameters_changed.emit()

        def set_mix(v: int) -> None:
            fx.mix = v / 100.0
            self.parameters_changed.emit()

        layout.addWidget(_ParamRow("Cutoff", 20, 12000, int(fx.cutoff),
                                   lambda v: f"{v} Hz", set_cutoff))
        layout.addWidget(_ParamRow("Resonance", 50, 500, int(fx.resonance * 100),
                                   lambda v: f"{v / 100.0:.2f}", set_res))
        layout.addWidget(_ParamRow("Mix", 0, 100, int(fx.mix * 100),
                                   lambda v: f"{v}%", set_mix))
        return group

    def _build_distortion(self) -> QGroupBox:
        fx = self.effects.distortion

        def on_toggle(checked: bool) -> None:
            fx.enabled = bool(checked)
            self.parameters_changed.emit()

        group = self._make_group("Distortion / Overdrive", fx.enabled, on_toggle)
        layout = QVBoxLayout(group)

        def set_drive(v: int) -> None:
            fx.drive = float(v) / 10.0
            self.parameters_changed.emit()

        def set_tone(v: int) -> None:
            fx.tone = v / 100.0
            self.parameters_changed.emit()

        def set_mix(v: int) -> None:
            fx.mix = v / 100.0
            self.parameters_changed.emit()

        layout.addWidget(_ParamRow("Drive", 10, 200, int(fx.drive * 10),
                                   lambda v: f"{v / 10.0:.1f}", set_drive))
        layout.addWidget(_ParamRow("Tone", 0, 100, int(fx.tone * 100),
                                   lambda v: f"{v}%", set_tone))
        layout.addWidget(_ParamRow("Mix", 0, 100, int(fx.mix * 100),
                                   lambda v: f"{v}%", set_mix))
        return group

    def _build_bitcrusher(self) -> QGroupBox:
        fx = self.effects.bit_crusher

        def on_toggle(checked: bool) -> None:
            fx.enabled = bool(checked)
            self.parameters_changed.emit()

        group = self._make_group("Bit crusher", fx.enabled, on_toggle)
        layout = QVBoxLayout(group)

        def set_bits(v: int) -> None:
            fx.bit_depth = int(v)
            self.parameters_changed.emit()

        def set_rate(v: int) -> None:
            fx.rate_reduction = int(v)
            self.parameters_changed.emit()

        def set_mix(v: int) -> None:
            fx.mix = v / 100.0
            self.parameters_changed.emit()

        layout.addWidget(_ParamRow("Bit depth", 2, 16, int(fx.bit_depth),
                                   lambda v: f"{v}-bit", set_bits))
        layout.addWidget(_ParamRow("Rate reduction", 1, 32, int(fx.rate_reduction),
                                   lambda v: f"1/{v}", set_rate))
        layout.addWidget(_ParamRow("Mix", 0, 100, int(fx.mix * 100),
                                   lambda v: f"{v}%", set_mix))
        return group

    def _build_delay(self) -> QGroupBox:
        fx = self.effects.delay

        def on_toggle(checked: bool) -> None:
            fx.enabled = bool(checked)
            self.parameters_changed.emit()

        group = self._make_group("Delay", fx.enabled, on_toggle)
        layout = QVBoxLayout(group)

        def set_time(v: int) -> None:
            fx.time_ms = float(v)
            self.parameters_changed.emit()

        def set_fb(v: int) -> None:
            fx.feedback = v / 100.0
            self.parameters_changed.emit()

        def set_mix(v: int) -> None:
            fx.mix = v / 100.0
            self.parameters_changed.emit()

        layout.addWidget(_ParamRow("Time", 10, 1500, int(fx.time_ms),
                                   lambda v: f"{v} ms", set_time))
        layout.addWidget(_ParamRow("Feedback", 0, 95, int(fx.feedback * 100),
                                   lambda v: f"{v}%", set_fb))
        layout.addWidget(_ParamRow("Mix", 0, 100, int(fx.mix * 100),
                                   lambda v: f"{v}%", set_mix))
        return group

    def _build_reverb(self) -> QGroupBox:
        fx = self.effects.reverb

        def on_toggle(checked: bool) -> None:
            fx.enabled = bool(checked)
            self.parameters_changed.emit()

        group = self._make_group("Reverb", fx.enabled, on_toggle)
        layout = QVBoxLayout(group)

        def set_size(v: int) -> None:
            fx.room_size = v / 100.0
            self.parameters_changed.emit()

        def set_damp(v: int) -> None:
            fx.damping = v / 100.0
            self.parameters_changed.emit()

        def set_mix(v: int) -> None:
            fx.mix = v / 100.0
            self.parameters_changed.emit()

        layout.addWidget(_ParamRow("Room size", 0, 100, int(fx.room_size * 100),
                                   lambda v: f"{v}%", set_size))
        layout.addWidget(_ParamRow("Damping", 0, 100, int(fx.damping * 100),
                                   lambda v: f"{v}%", set_damp))
        layout.addWidget(_ParamRow("Mix", 0, 100, int(fx.mix * 100),
                                   lambda v: f"{v}%", set_mix))
        return group
