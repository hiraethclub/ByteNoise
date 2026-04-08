"""Synthesis mode controls panel.

Provides a tab widget with one tab per synthesis mode (Raw, Drone, Crunch,
Space Drone). Each tab exposes the parameters for that mode as labelled
sliders. The panel emits a single ``parameters_changed`` signal whenever any
control changes; the main window debounces this and re-renders.

The panel also supports ``refresh()`` which re-reads every control's value
from the underlying ``SynthSettings`` and updates the widgets without firing
change signals. This is what makes undo/redo and randomise possible: they
mutate the model, then call ``refresh()``.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

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

from ..engine.synthesiser import SynthSettings


class _LabelledSlider(QWidget):
    """A slider with a name label and a live value label.

    Exposes ``set_value()`` which updates both the slider and the value label
    *without* firing the change callback. Used by the panel's ``refresh()``.
    """

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

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(initial)
        self.slider.valueChanged.connect(self._handle_change)

        layout.addWidget(self.name_label, 0, 0)
        layout.addWidget(self.value_label, 0, 1)
        layout.addWidget(self.slider, 1, 0, 1, 2)

    def _handle_change(self, value: int) -> None:
        self.value_label.setText(self._formatter(value))
        self._on_change(value)

    def set_value(self, value: int) -> None:
        """Update the slider's value silently (no change callback fired)."""
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.slider.blockSignals(False)
        self.value_label.setText(self._formatter(int(value)))


class SynthPanel(QWidget):
    """Synthesis controls grouped by mode."""

    parameters_changed = pyqtSignal()
    mode_changed = pyqtSignal(str)

    # Index <-> mode key map.
    _MODES = ["raw", "drone", "crunch", "space"]

    def __init__(self, settings: SynthSettings) -> None:
        super().__init__()
        self.settings = settings

        # Refreshable widget registry: (widget, getter callable returning int).
        # Combo boxes track (combo, getter returning option index).
        self._sliders: List[Tuple[_LabelledSlider, Callable[[], int]]] = []
        self._combos: List[Tuple[QComboBox, Callable[[], int]]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Synthesis")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_raw_tab(), "Raw Noise")
        self.tabs.addTab(self._build_drone_tab(), "Drone")
        self.tabs.addTab(self._build_crunch_tab(), "Crunch")
        self.tabs.addTab(self._build_space_tab(), "Space Drone")
        self.tabs.currentChanged.connect(self._handle_tab_change)
        layout.addWidget(self.tabs)

        # Common: buffer duration
        duration_box = QGroupBox("Buffer")
        duration_layout = QVBoxLayout(duration_box)
        self._add_slider(
            duration_layout,
            "Buffer length",
            2, 600,
            getter=lambda: int(self.settings.duration_s),
            setter=lambda v: setattr(self.settings, "duration_s", float(v)),
            formatter=lambda v: f"{v} s",
        )
        layout.addWidget(duration_box)

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
    ) -> _LabelledSlider:
        """Create a labelled slider, wire its setter, and register for refresh."""
        def on_change(value: int) -> None:
            setter(value)
            self.parameters_changed.emit()

        slider = _LabelledSlider(name, minimum, maximum, getter(), formatter, on_change)
        layout.addWidget(slider)
        self._sliders.append((slider, getter))
        return slider

    def _add_combo(
        self,
        layout,
        items: List[str],
        getter: Callable[[], int],
        setter: Callable[[int], None],
    ) -> QComboBox:
        """Create a combo box wired to setter and register for refresh."""
        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentIndex(getter())

        def on_change(index: int) -> None:
            setter(index)
            self.parameters_changed.emit()

        combo.currentIndexChanged.connect(on_change)
        layout.addWidget(combo)
        self._combos.append((combo, getter))
        return combo

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_raw_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self._add_slider(
            layout, "Sample rate", 8000, 96000,
            getter=lambda: int(self.settings.raw.sample_rate),
            setter=lambda v: setattr(self.settings.raw, "sample_rate", int(v)),
            formatter=lambda v: f"{v} Hz",
        )

        bit_box = QGroupBox("Bit interpretation")
        bit_layout = QHBoxLayout(bit_box)
        self._add_combo(
            bit_layout,
            ["8-bit (one byte / sample)", "16-bit (paired bytes)"],
            getter=lambda: 0 if self.settings.raw.bit_mode == 8 else 1,
            setter=lambda i: setattr(self.settings.raw, "bit_mode", 8 if i == 0 else 16),
        )
        layout.addWidget(bit_box)

        layout.addStretch()
        return widget

    def _build_drone_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self._add_slider(
            layout, "Base frequency", 20, 400,
            getter=lambda: int(self.settings.drone.base_freq),
            setter=lambda v: setattr(self.settings.drone, "base_freq", float(v)),
            formatter=lambda v: f"{v} Hz",
        )

        wave_box = QGroupBox("Waveform")
        wave_layout = QHBoxLayout(wave_box)
        wave_options = ["sine", "saw", "triangle"]
        self._add_combo(
            wave_layout,
            wave_options,
            getter=lambda: wave_options.index(self.settings.drone.waveform)
                if self.settings.drone.waveform in wave_options else 0,
            setter=lambda i: setattr(self.settings.drone, "waveform", wave_options[i]),
        )
        layout.addWidget(wave_box)

        self._add_slider(
            layout, "LFO speed", 1, 200,
            getter=lambda: int(self.settings.drone.lfo_speed * 100),
            setter=lambda v: setattr(self.settings.drone, "lfo_speed", v / 100.0),
            formatter=lambda v: f"{v / 100.0:.2f} Hz",
        )

        self._add_slider(
            layout, "Density (voices)", 1, 12,
            getter=lambda: int(self.settings.drone.density),
            setter=lambda v: setattr(self.settings.drone, "density", int(v)),
            formatter=lambda v: f"{v}",
        )

        self._add_slider(
            layout, "Evolution speed", 1, 50,
            getter=lambda: int(self.settings.drone.evolution_speed * 10),
            setter=lambda v: setattr(self.settings.drone, "evolution_speed", v / 10.0),
            formatter=lambda v: f"{v / 10.0:.1f}x",
        )

        layout.addStretch()
        return widget

    def _build_crunch_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self._add_slider(
            layout, "Output bit depth", 2, 16,
            getter=lambda: int(self.settings.crunch.bit_depth),
            setter=lambda v: setattr(self.settings.crunch, "bit_depth", int(v)),
            formatter=lambda v: f"{v}-bit",
        )

        self._add_slider(
            layout, "Sample rate reduction", 1, 32,
            getter=lambda: int(self.settings.crunch.rate_reduction),
            setter=lambda v: setattr(self.settings.crunch, "rate_reduction", int(v)),
            formatter=lambda v: f"1/{v}",
        )

        self._add_slider(
            layout, "Quantisation noise", 0, 100,
            getter=lambda: int(self.settings.crunch.quantisation_noise * 100),
            setter=lambda v: setattr(self.settings.crunch, "quantisation_noise", v / 100.0),
            formatter=lambda v: f"{v}%",
        )

        layout.addStretch()
        return widget

    def _build_space_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self._add_slider(
            layout, "Static amount", 0, 100,
            getter=lambda: int(self.settings.space.static_amount * 100),
            setter=lambda v: setattr(self.settings.space, "static_amount", v / 100.0),
            formatter=lambda v: f"{v}%",
        )

        self._add_slider(
            layout, "Fade speed", 1, 50,
            getter=lambda: int(self.settings.space.fade_speed * 10),
            setter=lambda v: setattr(self.settings.space, "fade_speed", v / 10.0),
            formatter=lambda v: f"{v / 10.0:.1f}x",
        )

        self._add_slider(
            layout, "Signal clarity", 0, 100,
            getter=lambda: int(self.settings.space.signal_clarity * 100),
            setter=lambda v: setattr(self.settings.space, "signal_clarity", v / 100.0),
            formatter=lambda v: f"{v}%",
        )

        self._add_slider(
            layout, "Drift rate", 1, 200,
            getter=lambda: int(self.settings.space.drift_rate * 100),
            setter=lambda v: setattr(self.settings.space, "drift_rate", v / 100.0),
            formatter=lambda v: f"{v / 100.0:.2f} Hz",
        )

        self._add_slider(
            layout, "Burst threshold", 0, 95,
            getter=lambda: int(self.settings.space.burst_threshold * 100),
            setter=lambda v: setattr(self.settings.space, "burst_threshold", v / 100.0),
            formatter=lambda v: f"{v}%",
        )

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Public refresh API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read all controls from ``self.settings`` without firing changes."""
        for slider, getter in self._sliders:
            try:
                slider.set_value(getter())
            except Exception:
                pass
        for combo, getter in self._combos:
            try:
                combo.blockSignals(True)
                combo.setCurrentIndex(int(getter()))
                combo.blockSignals(False)
            except Exception:
                combo.blockSignals(False)

        # Sync the active tab with self.settings.mode
        try:
            idx = self._MODES.index(self.settings.mode)
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(idx)
            self.tabs.blockSignals(False)
        except ValueError:
            self.tabs.blockSignals(False)

    # ------------------------------------------------------------------
    # Slot callbacks
    # ------------------------------------------------------------------

    def _handle_tab_change(self, index: int) -> None:
        mode = self._MODES[index] if 0 <= index < len(self._MODES) else "raw"
        self.settings.mode = mode
        self.mode_changed.emit(mode)
        self.parameters_changed.emit()
