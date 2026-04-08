"""Synthesis mode controls panel.

Provides a tab widget with one tab per synthesis mode (Raw, Drone, Crunch).
Each tab exposes the parameters for that mode as labelled sliders. The panel
emits a single ``parameters_changed`` signal whenever any control changes;
the main window debounces this and re-renders.
"""

from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..engine.synthesiser import (
    CrunchParams,
    DroneParams,
    RawNoiseParams,
    SynthSettings,
)


def _make_slider(
    minimum: int,
    maximum: int,
    initial: int,
    on_change: Callable[[int], None],
) -> QSlider:
    """Create a horizontal slider wired to ``on_change``."""
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(initial)
    slider.valueChanged.connect(on_change)
    return slider


class _LabelledSlider(QWidget):
    """A slider with a name label and a live value label."""

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

        self.name_label = QLabel(name)
        self.value_label = QLabel(formatter(initial))
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setMinimumWidth(80)

        self.slider = _make_slider(minimum, maximum, initial, self._handle_change)

        layout.addWidget(self.name_label, 0, 0)
        layout.addWidget(self.value_label, 0, 1)
        layout.addWidget(self.slider, 1, 0, 1, 2)

    def _handle_change(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))
        self._on_change(value)


class SynthPanel(QWidget):
    """Synthesis controls grouped by mode."""

    parameters_changed = pyqtSignal()
    mode_changed = pyqtSignal(str)

    def __init__(self, settings: SynthSettings) -> None:
        super().__init__()
        self.settings = settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Synthesis")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_raw_tab(), "Raw Noise")
        self.tabs.addTab(self._build_drone_tab(), "Drone")
        self.tabs.addTab(self._build_crunch_tab(), "Crunch")
        self.tabs.currentChanged.connect(self._handle_tab_change)
        layout.addWidget(self.tabs)

        # Common: buffer duration
        duration_box = QGroupBox("Buffer")
        duration_layout = QVBoxLayout(duration_box)
        self.duration_slider = _LabelledSlider(
            "Buffer length",
            minimum=2,
            maximum=120,
            initial=int(settings.duration_s),
            formatter=lambda v: f"{v} s",
            on_change=self._set_duration,
        )
        duration_layout.addWidget(self.duration_slider)
        layout.addWidget(duration_box)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_raw_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        params: RawNoiseParams = self.settings.raw

        sr = _LabelledSlider(
            "Sample rate",
            minimum=8000,
            maximum=96000,
            initial=int(params.sample_rate),
            formatter=lambda v: f"{v} Hz",
            on_change=self._set_raw_sample_rate,
        )
        layout.addWidget(sr)

        bit_box = QGroupBox("Bit interpretation")
        bit_layout = QHBoxLayout(bit_box)
        self.raw_bit_combo = QComboBox()
        self.raw_bit_combo.addItems(["8-bit (one byte / sample)", "16-bit (paired bytes)"])
        self.raw_bit_combo.setCurrentIndex(0 if params.bit_mode == 8 else 1)
        self.raw_bit_combo.currentIndexChanged.connect(self._set_raw_bit_mode)
        bit_layout.addWidget(self.raw_bit_combo)
        layout.addWidget(bit_box)

        layout.addStretch()
        return widget

    def _build_drone_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        params: DroneParams = self.settings.drone

        layout.addWidget(_LabelledSlider(
            "Base frequency",
            minimum=20,
            maximum=400,
            initial=int(params.base_freq),
            formatter=lambda v: f"{v} Hz",
            on_change=self._set_drone_base_freq,
        ))

        wave_box = QGroupBox("Waveform")
        wave_layout = QHBoxLayout(wave_box)
        self.drone_wave_combo = QComboBox()
        self.drone_wave_combo.addItems(["sine", "saw", "triangle"])
        self.drone_wave_combo.setCurrentText(params.waveform)
        self.drone_wave_combo.currentTextChanged.connect(self._set_drone_waveform)
        wave_layout.addWidget(self.drone_wave_combo)
        layout.addWidget(wave_box)

        layout.addWidget(_LabelledSlider(
            "LFO speed",
            minimum=1,
            maximum=200,  # 0.01..2.0 Hz
            initial=int(params.lfo_speed * 100),
            formatter=lambda v: f"{v / 100.0:.2f} Hz",
            on_change=self._set_drone_lfo_speed,
        ))

        layout.addWidget(_LabelledSlider(
            "Density (voices)",
            minimum=1,
            maximum=12,
            initial=int(params.density),
            formatter=lambda v: f"{v}",
            on_change=self._set_drone_density,
        ))

        layout.addWidget(_LabelledSlider(
            "Evolution speed",
            minimum=1,
            maximum=50,  # 0.1..5.0
            initial=int(params.evolution_speed * 10),
            formatter=lambda v: f"{v / 10.0:.1f}x",
            on_change=self._set_drone_evolution,
        ))

        layout.addStretch()
        return widget

    def _build_crunch_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        params: CrunchParams = self.settings.crunch

        layout.addWidget(_LabelledSlider(
            "Output bit depth",
            minimum=2,
            maximum=16,
            initial=int(params.bit_depth),
            formatter=lambda v: f"{v}-bit",
            on_change=self._set_crunch_bit_depth,
        ))

        layout.addWidget(_LabelledSlider(
            "Sample rate reduction",
            minimum=1,
            maximum=32,
            initial=int(params.rate_reduction),
            formatter=lambda v: f"1/{v}",
            on_change=self._set_crunch_rate_reduction,
        ))

        layout.addWidget(_LabelledSlider(
            "Quantisation noise",
            minimum=0,
            maximum=100,
            initial=int(params.quantisation_noise * 100),
            formatter=lambda v: f"{v}%",
            on_change=self._set_crunch_qn,
        ))

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Slot callbacks
    # ------------------------------------------------------------------

    def _handle_tab_change(self, index: int) -> None:
        modes = ["raw", "drone", "crunch"]
        mode = modes[index] if 0 <= index < len(modes) else "raw"
        self.settings.mode = mode
        self.mode_changed.emit(mode)
        self.parameters_changed.emit()

    def _set_duration(self, value: int) -> None:
        self.settings.duration_s = float(value)
        self.parameters_changed.emit()

    # Raw
    def _set_raw_sample_rate(self, value: int) -> None:
        self.settings.raw.sample_rate = int(value)
        self.parameters_changed.emit()

    def _set_raw_bit_mode(self, index: int) -> None:
        self.settings.raw.bit_mode = 8 if index == 0 else 16
        self.parameters_changed.emit()

    # Drone
    def _set_drone_base_freq(self, value: int) -> None:
        self.settings.drone.base_freq = float(value)
        self.parameters_changed.emit()

    def _set_drone_waveform(self, text: str) -> None:
        self.settings.drone.waveform = text
        self.parameters_changed.emit()

    def _set_drone_lfo_speed(self, value: int) -> None:
        self.settings.drone.lfo_speed = value / 100.0
        self.parameters_changed.emit()

    def _set_drone_density(self, value: int) -> None:
        self.settings.drone.density = int(value)
        self.parameters_changed.emit()

    def _set_drone_evolution(self, value: int) -> None:
        self.settings.drone.evolution_speed = value / 10.0
        self.parameters_changed.emit()

    # Crunch
    def _set_crunch_bit_depth(self, value: int) -> None:
        self.settings.crunch.bit_depth = int(value)
        self.parameters_changed.emit()

    def _set_crunch_rate_reduction(self, value: int) -> None:
        self.settings.crunch.rate_reduction = int(value)
        self.parameters_changed.emit()

    def _set_crunch_qn(self, value: int) -> None:
        self.settings.crunch.quantisation_noise = value / 100.0
        self.parameters_changed.emit()
