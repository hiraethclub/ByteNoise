"""Modal dialog for the MIDI export parameters.

Lives between the user clicking ``File -> Export MIDI...`` and the actual
file being written. Lets them pick a tempo, scale, octave range, and how
densely to sample the input bytes. Returns a ``MidiExportParams`` instance
when accepted.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
    QDoubleSpinBox,
)

from ..engine.midi_export import MidiExportParams, SCALES


class MidiExportDialog(QDialog):
    """Asks the user for MIDI export parameters."""

    def __init__(self, parent=None, defaults: Optional[MidiExportParams] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export MIDI")
        self.setModal(True)
        self._params = defaults or MidiExportParams()

        outer = QVBoxLayout(self)
        form = QFormLayout()
        outer.addLayout(form)

        self.tempo = QSpinBox()
        self.tempo.setRange(30, 280)
        self.tempo.setValue(self._params.tempo_bpm)
        self.tempo.setSuffix(" BPM")
        form.addRow("Tempo", self.tempo)

        self.note_length = QDoubleSpinBox()
        self.note_length.setRange(0.05, 4.0)
        self.note_length.setSingleStep(0.05)
        self.note_length.setDecimals(2)
        self.note_length.setValue(self._params.note_length_beats)
        self.note_length.setSuffix(" beats")
        form.addRow("Note length", self.note_length)

        self.scale = QComboBox()
        for name in SCALES:
            self.scale.addItem(name.replace("_", " "), userData=name)
        # Restore the current scale if it's a known one.
        for i in range(self.scale.count()):
            if self.scale.itemData(i) == self._params.scale:
                self.scale.setCurrentIndex(i)
                break
        form.addRow("Scale", self.scale)

        self.base_octave = QSpinBox()
        self.base_octave.setRange(0, 8)
        self.base_octave.setValue(self._params.base_octave)
        form.addRow("Base octave", self.base_octave)

        self.octaves = QSpinBox()
        self.octaves.setRange(1, 6)
        self.octaves.setValue(self._params.octaves)
        form.addRow("Octave range", self.octaves)

        self.bytes_per_note = QSpinBox()
        self.bytes_per_note.setRange(1, 4096)
        self.bytes_per_note.setValue(self._params.bytes_per_note)
        form.addRow("Bytes per note", self.bytes_per_note)

        self.vmin = QSpinBox()
        self.vmin.setRange(1, 127)
        self.vmin.setValue(self._params.velocity_min)
        form.addRow("Velocity min", self.vmin)

        self.vmax = QSpinBox()
        self.vmax.setRange(1, 127)
        self.vmax.setValue(self._params.velocity_max)
        form.addRow("Velocity max", self.vmax)

        self.max_notes = QSpinBox()
        self.max_notes.setRange(16, 65535)
        self.max_notes.setValue(self._params.max_notes)
        form.addRow("Max notes", self.max_notes)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def params(self) -> MidiExportParams:
        """Return a fresh ``MidiExportParams`` reflecting the dialog state."""
        return MidiExportParams(
            tempo_bpm=self.tempo.value(),
            note_length_beats=float(self.note_length.value()),
            base_octave=self.base_octave.value(),
            octaves=self.octaves.value(),
            scale=self.scale.currentData() or "pentatonic_minor",
            bytes_per_note=self.bytes_per_note.value(),
            velocity_min=self.vmin.value(),
            velocity_max=max(self.vmin.value(), self.vmax.value()),
            max_notes=self.max_notes.value(),
        )
