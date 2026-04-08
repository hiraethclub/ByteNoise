"""Main application window for ByteNoise.

Layout:
    +---------------------------------------------------------------+
    | [Load File]  filename.ext  |  size  |  byte stats             |
    +-----------------+-----------------------+---------------------+
    |                 |                       |                     |
    |  Synth panel    |    Waveform display   |   Effects panel     |
    |  (left)         |    (centre)           |   (right)           |
    |                 |                       |                     |
    +-----------------+-----------------------+---------------------+
    | Play  Pause  Stop   Volume [====]            [Export WAV]     |
    +---------------------------------------------------------------+

The main window owns the synthesis settings, the effects chain, the audio
engine, and the loaded file. It debounces parameter changes via a single
QTimer so dragging sliders doesn't render dozens of buffers per second.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
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
from ..engine.synthesiser import SynthSettings
from .effects_panel import EffectsPanel
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
        self.loaded_file: Optional[LoadedFile] = None
        self.current_buffer: np.ndarray = np.zeros(0, dtype=np.float32)

        self._build_ui()
        self._wire_signals()

        # Debounce timer for re-rendering on parameter changes.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_now)

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
        center_layout.addWidget(self.waveform, 1)
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

        self.transport.play_clicked.connect(self._on_play)
        self.transport.pause_clicked.connect(self._on_pause)
        self.transport.stop_clicked.connect(self._on_stop)
        self.transport.export_clicked.connect(self._on_export)
        self.transport.volume_changed.connect(self.engine.set_volume)
        self.engine.set_volume(0.8)

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
        except Exception as exc:  # pragma: no cover - render errors should not crash GUI
            self.statusBar().showMessage(f"Render error: {exc}")

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
                                 "soundfile is not installed - cannot export WAV.")
            return

        default_name = "bytenoise.wav"
        if self.loaded_file is not None:
            stem = self.loaded_file.name.rsplit(".", 1)[0] or "bytenoise"
            default_name = f"{stem}_bytenoise.wav"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export WAV",
            default_name,
            "WAV files (*.wav)",
        )
        if not path:
            return

        progress = QProgressDialog("Exporting WAV...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        try:
            sample_rate = synthesiser.get_sample_rate(self.synth_settings)
            buf = np.clip(self.current_buffer, -1.0, 1.0)
            sf.write(path, buf, sample_rate, subtype="PCM_16")
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "Export failed", f"Could not write WAV:\n{exc}")
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
