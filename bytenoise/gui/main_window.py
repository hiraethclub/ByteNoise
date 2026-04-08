"""Main application window for ByteNoise.

Layout:
    +---------------------------------------------------------------+
    | [Load File] filename | [Randomise] [Undo] [Redo]              |
    +-----------------+-----------------------+---------------------+
    |                 |    Waveform           |                     |
    |  Synth panel    +-----------------------+   Effects panel     |
    |  (left)         |    Spectrum           |   (right)           |
    |                 |                       |                     |
    +-----------------+-----------------------+---------------------+
    | Play  Pause  Stop   Volume [====]            [Export...]      |
    +---------------------------------------------------------------+

The main window owns the synthesis settings, the effects chain, the audio
engine, and the loaded file. It debounces parameter changes via a single
QTimer so dragging sliders doesn't render dozens of buffers per second.
After each render the new state is pushed onto the undo history.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - soundfile may be missing in some environments.
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

from ..engine import synthesiser
from ..engine.audio_engine import AudioEngine
from ..engine.effects import EffectsChain
from ..engine.file_reader import LoadedFile, load_file
from ..engine.state import HistoryManager, randomise
from ..engine.synthesiser import SynthSettings
from .effects_panel import EffectsPanel
from .spectrum import SpectrumView
from .synth_panel import SynthPanel
from .transport import TransportBar
from .waveform import WaveformView


# Debounce time for parameter-driven re-renders.
RENDER_DEBOUNCE_MS = 120


class MainWindow(QMainWindow):
    """Top level window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ByteNoise")
        self.resize(1280, 760)

        # Core state
        self.synth_settings = SynthSettings()
        self.effects_chain = EffectsChain()
        self.engine = AudioEngine()
        self.history = HistoryManager(max_entries=50)
        self.loaded_file: Optional[LoadedFile] = None
        self.current_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        # When True, the next render won't push a new history entry. Used by
        # undo/redo so that applying a snapshot doesn't itself create one.
        self._suppress_history_push: bool = False

        self._build_ui()
        self._wire_signals()

        # Debounce timer for re-rendering on parameter changes.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_now)

        # Seed the history with the initial state.
        self.history.push(self.synth_settings, self.effects_chain)
        self._update_history_buttons()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Top bar ---
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 6, 8, 6)
        self.load_btn = QPushButton("Load file...")
        self.load_btn.setMinimumWidth(120)
        top_layout.addWidget(self.load_btn)
        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("color: #cdd; padding-left: 10px;")
        top_layout.addWidget(self.file_info_label)
        top_layout.addStretch()

        self.randomise_btn = QPushButton("Randomise")
        self.randomise_btn.setToolTip("Randomise all synth and effect parameters")
        top_layout.addWidget(self.randomise_btn)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.setEnabled(False)
        top_layout.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setToolTip("Redo (Ctrl+Shift+Z / Ctrl+Y)")
        self.redo_btn.setEnabled(False)
        top_layout.addWidget(self.redo_btn)

        outer.addWidget(top_bar)

        # --- Main splitter ---
        splitter = QSplitter(Qt.Horizontal)

        self.synth_panel = SynthPanel(self.synth_settings)
        splitter.addWidget(self.synth_panel)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.addWidget(QLabel("Waveform"))
        self.waveform = WaveformView()
        center_layout.addWidget(self.waveform, 2)
        center_layout.addWidget(QLabel("Spectrum"))
        self.spectrum = SpectrumView()
        center_layout.addWidget(self.spectrum, 1)
        splitter.addWidget(center)

        self.effects_panel = EffectsPanel(self.effects_chain)
        splitter.addWidget(self.effects_panel)

        splitter.setSizes([280, 640, 360])
        outer.addWidget(splitter, 1)

        # --- Bottom transport bar ---
        self.transport = TransportBar()
        outer.addWidget(self.transport)

        self.statusBar().showMessage("Load a file to begin.")

    def _wire_signals(self) -> None:
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.synth_panel.parameters_changed.connect(self._schedule_render)
        self.synth_panel.mode_changed.connect(self._on_mode_changed)
        self.effects_panel.parameters_changed.connect(self._schedule_render)

        self.randomise_btn.clicked.connect(self._on_randomise)
        self.undo_btn.clicked.connect(self._on_undo)
        self.redo_btn.clicked.connect(self._on_redo)

        self.transport.play_clicked.connect(self._on_play)
        self.transport.pause_clicked.connect(self._on_pause)
        self.transport.stop_clicked.connect(self._on_stop)
        self.transport.export_clicked.connect(self._on_export)
        self.transport.volume_changed.connect(self.engine.set_volume)
        self.engine.set_volume(0.8)

        # Keyboard shortcuts.
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._on_undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._on_redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._on_redo)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._on_randomise)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open any file",
            "",
            "All files (*)",
        )
        if not path:
            return
        try:
            loaded = load_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not load file:\n{exc}")
            return
        if loaded.size == 0:
            QMessageBox.warning(self, "Empty file", "That file has no bytes to play.")
            return

        self.loaded_file = loaded
        self.file_info_label.setText(loaded.info_string())
        self.statusBar().showMessage(f"Loaded {loaded.name}")
        self._render_now()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _on_mode_changed(self, _mode: str) -> None:
        # Mode change implies sample rate may differ - render fully now.
        self._schedule_render()

    def _schedule_render(self) -> None:
        self._render_timer.start(RENDER_DEBOUNCE_MS)

    def _render_now(self) -> None:
        if self.loaded_file is None or self.loaded_file.size == 0:
            return
        try:
            raw = synthesiser.render(self.loaded_file.data, self.synth_settings)
            sample_rate = synthesiser.get_sample_rate(self.synth_settings)
            processed = self.effects_chain.process(raw, sample_rate)
            # Soft normalise so loud effects don't clip the speakers.
            peak = float(np.max(np.abs(processed))) if processed.size else 0.0
            if peak > 1.0:
                processed = (processed / peak).astype(np.float32)
            self.current_buffer = processed
            self.engine.set_buffer(processed, sample_rate)
            self.waveform.set_buffer(processed)
            self.spectrum.set_buffer(processed, sample_rate)
        except Exception as exc:  # pragma: no cover - render errors should not crash GUI
            self.statusBar().showMessage(f"Render error: {exc}")
            return

        # Push the new state onto history (skipped during undo/redo apply).
        if not self._suppress_history_push:
            self.history.push(self.synth_settings, self.effects_chain)
            self._update_history_buttons()

    # ------------------------------------------------------------------
    # Randomise / Undo / Redo
    # ------------------------------------------------------------------

    def _on_randomise(self) -> None:
        randomise(self.synth_settings, self.effects_chain)
        self.synth_panel.refresh()
        self.effects_panel.refresh()
        self.statusBar().showMessage("Randomised parameters")
        # Render immediately (so the history snapshot reflects what the user
        # actually hears) rather than waiting for the debounce.
        self._render_now()

    def _on_undo(self) -> None:
        if not self.history.undo(self.synth_settings, self.effects_chain):
            return
        self._apply_history_state("Undo")

    def _on_redo(self) -> None:
        if not self.history.redo(self.synth_settings, self.effects_chain):
            return
        self._apply_history_state("Redo")

    def _apply_history_state(self, label: str) -> None:
        """Refresh GUI from the model and re-render without pushing history."""
        self.synth_panel.refresh()
        self.effects_panel.refresh()
        self._suppress_history_push = True
        try:
            self._render_now()
        finally:
            self._suppress_history_push = False
        self._update_history_buttons()
        self.statusBar().showMessage(label)

    def _update_history_buttons(self) -> None:
        self.undo_btn.setEnabled(self.history.can_undo())
        self.redo_btn.setEnabled(self.history.can_redo())

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _on_play(self) -> None:
        if self.current_buffer.size == 0:
            self.statusBar().showMessage("Nothing to play - load a file first.")
            return
        self.engine.play()
        self.statusBar().showMessage("Playing")

    def _on_pause(self) -> None:
        self.engine.pause()
        self.statusBar().showMessage("Paused")

    def _on_stop(self) -> None:
        self.engine.stop()
        self.statusBar().showMessage("Stopped")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        if self.current_buffer.size == 0:
            QMessageBox.information(self, "Nothing to export",
                                    "Load a file and render some audio first.")
            return
        if sf is None:
            QMessageBox.critical(self, "Missing dependency",
                                 "soundfile is not installed - cannot export.")
            return

        default_name = "bytenoise.wav"
        if self.loaded_file is not None:
            stem = self.loaded_file.name.rsplit(".", 1)[0] or "bytenoise"
            default_name = f"{stem}_bytenoise.wav"

        # Offer both WAV and FLAC. The selected filter determines the format
        # we pass to soundfile, with the file extension as a fallback.
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export audio",
            default_name,
            "WAV files (*.wav);;FLAC files (*.flac)",
        )
        if not path:
            return

        # Decide format from selected filter, falling back to extension.
        ext = os.path.splitext(path)[1].lower()
        if "flac" in selected_filter.lower() or ext == ".flac":
            fmt = "FLAC"
            subtype = "PCM_16"
            if ext != ".flac":
                path = path + ".flac"
        else:
            fmt = "WAV"
            subtype = "PCM_16"
            if ext != ".wav":
                path = path + ".wav"

        progress = QProgressDialog(f"Exporting {fmt}...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        try:
            sample_rate = synthesiser.get_sample_rate(self.synth_settings)
            buf = np.clip(self.current_buffer, -1.0, 1.0)
            sf.write(path, buf, sample_rate, format=fmt, subtype=subtype)
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Export failed",
                                 f"Could not write {fmt}:\n{exc}")
            return
        progress.close()
        self.statusBar().showMessage(f"Exported to {path}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        try:
            self.engine.shutdown()
        finally:
            super().closeEvent(event)
