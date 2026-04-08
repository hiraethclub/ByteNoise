"""Main application window for ByteNoise.

Layout:
    +---------------------------------------------------------------+
    | Menu bar (File / Edit / Help)                                 |
    +---------------------------------------------------------------+
    | [Load file] [Add layer] [x] Watch    [Randomise] [Undo] [Redo]|
    +-----------------+-----------------------+---------------------+
    |  Layers panel   |    Waveform           |                     |
    |                 +-----------------------+                     |
    |  Synth panel    |    Spectrum           |   Effects panel     |
    |                 |                       |                     |
    +-----------------+-----------------------+---------------------+
    | Play  Pause  Stop   Volume [====]            [Export...]      |
    +---------------------------------------------------------------+

The main window owns the synthesis settings, the effects chain, the audio
engine, and a list of loaded files (layers). Multiple layers are summed
before the effects chain runs, so the user can mix the textures of several
files into one patch.

It also debounces parameter changes via a single QTimer so dragging sliders
doesn't render dozens of buffers per second. After each render the new state
is pushed onto the undo history.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QFileSystemWatcher, QRectF, QTimer
from PyQt5.QtGui import (
    QColor,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
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
from ..engine.midi_export import MidiExportParams, export_midi
from ..engine.presets import load_preset, save_preset
from ..engine.state import HistoryManager, randomise
from ..engine.synthesiser import SynthSettings
from .effects_panel import EffectsPanel
from .layers_panel import LayersPanel
from .midi_export_dialog import MidiExportDialog
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
        self.resize(1320, 800)

        # Core state
        self.synth_settings = SynthSettings()
        self.effects_chain = EffectsChain()
        self.engine = AudioEngine()
        self.history = HistoryManager(max_entries=50)
        self.loaded_files: List[LoadedFile] = []
        self.current_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        # When True, the next render won't push a new history entry. Used by
        # undo/redo so that applying a snapshot doesn't itself create one.
        self._suppress_history_push: bool = False

        # File system watcher used by the auto re-render feature. Watches the
        # paths of every layer; whenever any of them changes on disk we reload
        # that layer and trigger a re-render.
        self._fs_watcher = QFileSystemWatcher(self)
        self._fs_watcher.fileChanged.connect(self._on_file_changed)
        self._watch_enabled: bool = False

        # Last-used MIDI export params, so reopening the dialog remembers them.
        self._midi_params = MidiExportParams()

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
        self._build_menu_bar()

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
        self.load_btn.setMinimumWidth(110)
        self.load_btn.setToolTip("Load a single file (replaces all layers)")
        top_layout.addWidget(self.load_btn)

        self.add_layer_btn = QPushButton("Add layer...")
        self.add_layer_btn.setToolTip("Add another file as a new layer")
        top_layout.addWidget(self.add_layer_btn)

        self.watch_check = QCheckBox("Watch")
        self.watch_check.setToolTip(
            "Re-render automatically when any loaded file changes on disk"
        )
        top_layout.addWidget(self.watch_check)

        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setStyleSheet("color: #cdd; padding-left: 10px;")
        top_layout.addWidget(self.file_info_label, 1)

        self.randomise_btn = QPushButton("Randomise")
        self.randomise_btn.setToolTip("Randomise all synth and effect parameters (Ctrl+R)")
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

        # Left column: layers panel + synth panel.
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.layers_panel = LayersPanel()
        left_layout.addWidget(self.layers_panel)
        self.synth_panel = SynthPanel(self.synth_settings)
        left_layout.addWidget(self.synth_panel, 1)
        splitter.addWidget(left_col)

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

        splitter.setSizes([300, 620, 360])
        outer.addWidget(splitter, 1)

        # --- Bottom transport bar ---
        self.transport = TransportBar()
        outer.addWidget(self.transport)

        self.statusBar().showMessage("Load a file to begin.")

    def _build_menu_bar(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")

        load_action = QAction("&Load file...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._on_load_clicked)
        file_menu.addAction(load_action)

        add_layer_action = QAction("&Add layer...", self)
        add_layer_action.setShortcut("Ctrl+L")
        add_layer_action.triggered.connect(self._on_add_layer_clicked)
        file_menu.addAction(add_layer_action)

        file_menu.addSeparator()

        save_preset_action = QAction("&Save preset...", self)
        save_preset_action.setShortcut("Ctrl+S")
        save_preset_action.triggered.connect(self._on_save_preset)
        file_menu.addAction(save_preset_action)

        load_preset_action = QAction("Load &preset...", self)
        load_preset_action.setShortcut("Ctrl+P")
        load_preset_action.triggered.connect(self._on_load_preset)
        file_menu.addAction(load_preset_action)

        file_menu.addSeparator()

        export_audio_action = QAction("&Export audio...", self)
        export_audio_action.setShortcut("Ctrl+E")
        export_audio_action.triggered.connect(self._on_export)
        file_menu.addAction(export_audio_action)

        export_midi_action = QAction("Export &MIDI...", self)
        export_midi_action.setShortcut("Ctrl+M")
        export_midi_action.triggered.connect(self._on_export_midi)
        file_menu.addAction(export_midi_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = bar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        randomise_action = QAction("Ra&ndomise", self)
        randomise_action.setShortcut("Ctrl+R")
        randomise_action.triggered.connect(self._on_randomise)
        edit_menu.addAction(randomise_action)

        help_menu = bar.addMenu("&Help")
        about_action = QAction("&About ByteNoise", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _wire_signals(self) -> None:
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.add_layer_btn.clicked.connect(self._on_add_layer_clicked)
        self.watch_check.toggled.connect(self._on_watch_toggled)
        self.synth_panel.parameters_changed.connect(self._schedule_render)
        self.synth_panel.mode_changed.connect(self._on_mode_changed)
        self.effects_panel.parameters_changed.connect(self._schedule_render)
        self.layers_panel.layer_removed.connect(self._on_layer_removed)
        self.layers_panel.gain_changed.connect(self._schedule_render)

        self.randomise_btn.clicked.connect(self._on_randomise)
        self.undo_btn.clicked.connect(self._on_undo)
        self.redo_btn.clicked.connect(self._on_redo)

        self.transport.play_clicked.connect(self._on_play)
        self.transport.pause_clicked.connect(self._on_pause)
        self.transport.stop_clicked.connect(self._on_stop)
        self.transport.export_clicked.connect(self._on_export)
        self.transport.volume_changed.connect(self.engine.set_volume)
        self.engine.set_volume(0.8)

        # Keyboard shortcuts not already covered by the menu bar.
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._on_redo)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _pick_file(self, title: str) -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(self, title, "", "All files (*)")
        return path or None

    def _try_load(self, path: str) -> Optional[LoadedFile]:
        try:
            loaded = load_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not load file:\n{exc}")
            return None
        if loaded.size == 0:
            QMessageBox.warning(self, "Empty file", "That file has no bytes to play.")
            return None
        return loaded

    def _on_load_clicked(self) -> None:
        path = self._pick_file("Open any file")
        if not path:
            return
        loaded = self._try_load(path)
        if loaded is None:
            return
        # Replace all existing layers with this one file.
        self._set_layers([loaded])
        self.statusBar().showMessage(f"Loaded {loaded.name}")
        self._render_now()

    def _on_add_layer_clicked(self) -> None:
        path = self._pick_file("Add file as a new layer")
        if not path:
            return
        loaded = self._try_load(path)
        if loaded is None:
            return
        self._set_layers(self.loaded_files + [loaded])
        self.statusBar().showMessage(f"Added layer {loaded.name}")
        self._render_now()

    def _on_layer_removed(self, index: int) -> None:
        if not (0 <= index < len(self.loaded_files)):
            return
        new_layers = list(self.loaded_files)
        removed = new_layers.pop(index)
        self._set_layers(new_layers)
        self.statusBar().showMessage(f"Removed layer {removed.name}")
        self._render_now()

    def _set_layers(self, layers: List[LoadedFile]) -> None:
        """Update the layer list, the panel, the file watcher, and the label."""
        self.loaded_files = list(layers)
        self.layers_panel.set_layers(self.loaded_files)
        self._refresh_file_info_label()
        self._refresh_watcher_paths()

    def _refresh_file_info_label(self) -> None:
        if not self.loaded_files:
            self.file_info_label.setText("No file loaded")
        elif len(self.loaded_files) == 1:
            self.file_info_label.setText(self.loaded_files[0].info_string())
        else:
            total = sum(f.size for f in self.loaded_files)
            self.file_info_label.setText(
                f"{len(self.loaded_files)} layers  |  {_format_total_size(total)}"
            )

    # ------------------------------------------------------------------
    # File watcher
    # ------------------------------------------------------------------

    def _on_watch_toggled(self, checked: bool) -> None:
        self._watch_enabled = bool(checked)
        self._refresh_watcher_paths()
        if self._watch_enabled:
            self.statusBar().showMessage("Auto re-render on file change: ON")
        else:
            self.statusBar().showMessage("Auto re-render on file change: OFF")

    def _refresh_watcher_paths(self) -> None:
        """Sync the QFileSystemWatcher's watched-paths list with the layers."""
        currently_watched = list(self._fs_watcher.files())
        if currently_watched:
            self._fs_watcher.removePaths(currently_watched)
        if not self._watch_enabled:
            return
        paths = [f.path for f in self.loaded_files if f.path and os.path.isfile(f.path)]
        if paths:
            self._fs_watcher.addPaths(paths)

    def _on_file_changed(self, path: str) -> None:
        # Some editors replace the file atomically (delete + create); QFSW
        # stops watching it when that happens, so re-add it after a short
        # delay before doing the reload.
        QTimer.singleShot(80, lambda: self._reload_layer(path))

    def _reload_layer(self, path: str) -> None:
        # Re-add the path to the watcher in case it was unhooked.
        if self._watch_enabled and os.path.isfile(path) and path not in self._fs_watcher.files():
            self._fs_watcher.addPath(path)

        for i, layer in enumerate(self.loaded_files):
            if layer.path != path:
                continue
            try:
                fresh = load_file(path)
            except Exception as exc:
                self.statusBar().showMessage(f"Watch reload failed: {exc}")
                return
            if fresh.size == 0:
                self.statusBar().showMessage(f"Watched file is empty: {os.path.basename(path)}")
                return
            # Preserve the user's gain setting on the layer.
            fresh.gain = layer.gain
            self.loaded_files[i] = fresh

        self.layers_panel.set_layers(self.loaded_files)
        self._refresh_file_info_label()
        self.statusBar().showMessage(f"Reloaded {os.path.basename(path)}")
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
        if not self.loaded_files:
            return
        try:
            sample_rate = synthesiser.get_sample_rate(self.synth_settings)
            # Render every layer through the synth and sum them, weighted by
            # each layer's per-file gain. Effects then run on the combined mix
            # so layers share the same texture.
            buffers: List[np.ndarray] = []
            total_gain = 0.0
            for layer in self.loaded_files:
                if layer.size == 0:
                    continue
                raw = synthesiser.render(layer.data, self.synth_settings)
                buffers.append(raw * float(layer.gain))
                total_gain += float(layer.gain)
            if not buffers:
                return
            mixed = buffers[0]
            for b in buffers[1:]:
                mixed = mixed + b
            # Average so adding layers doesn't cause runaway clipping; the
            # per-layer gain still controls relative loudness.
            denom = max(1.0, float(len(buffers)))
            mixed = (mixed / denom).astype(np.float32)

            processed = self.effects_chain.process(mixed, sample_rate)

            # Soft normalise so loud effects don't clip the speakers.
            peak = float(np.max(np.abs(processed))) if processed.size else 0.0
            if peak > 1.0:
                processed = (processed / peak).astype(np.float32)
            self.current_buffer = processed
            self.engine.set_buffer(processed, sample_rate)
            self.waveform.set_buffer(processed)
            self.spectrum.set_buffer(processed, sample_rate)
            self.effects_panel.update_meters()
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
    # Presets
    # ------------------------------------------------------------------

    def _on_save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save preset", "bytenoise_preset.json",
            "ByteNoise preset (*.json);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_preset(path, self.synth_settings, self.effects_chain)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save preset:\n{exc}")
            return
        self.statusBar().showMessage(f"Saved preset to {os.path.basename(path)}")

    def _on_load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load preset", "",
            "ByteNoise preset (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            load_preset(path, self.synth_settings, self.effects_chain)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Could not load preset:\n{exc}")
            return
        self.synth_panel.refresh()
        self.effects_panel.refresh()
        self.statusBar().showMessage(f"Loaded preset {os.path.basename(path)}")
        self._render_now()

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
    # Export (audio)
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

        default_name = self._default_export_stem() + ".wav"

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
    # Export (MIDI)
    # ------------------------------------------------------------------

    def _on_export_midi(self) -> None:
        if not self.loaded_files:
            QMessageBox.information(self, "Nothing to export",
                                    "Load a file first.")
            return

        dialog = MidiExportDialog(self, defaults=self._midi_params)
        if dialog.exec_() != dialog.Accepted:
            return
        params = dialog.params()
        self._midi_params = params

        default_name = self._default_export_stem() + ".mid"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDI", default_name,
            "MIDI files (*.mid);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".mid"):
            path += ".mid"

        # MIDI is built from the first layer's bytes - layering is an audio
        # concept and doesn't translate cleanly to a single MIDI track.
        primary = self.loaded_files[0]
        try:
            n_notes = export_midi(path, primary.data, params)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed",
                                 f"Could not write MIDI:\n{exc}")
            return
        self.statusBar().showMessage(
            f"Exported MIDI to {os.path.basename(path)} ({n_notes} notes)"
        )

    def _default_export_stem(self) -> str:
        if not self.loaded_files:
            return "bytenoise"
        primary = self.loaded_files[0]
        stem = primary.name.rsplit(".", 1)[0] or "bytenoise"
        return f"{stem}_bytenoise"

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _on_about(self) -> None:
        from .. import __version__
        msg = QMessageBox(self)
        msg.setWindowTitle("About ByteNoise")
        msg.setIconPixmap(self._make_welsh_flag(160, 100))
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            f"<b>ByteNoise {__version__}</b><br><br>"
            "Convert any file's raw bytes into sound.<br><br>"
            "Four synthesis modes, thirteen effects, "
            "multi-file layering, presets, MIDI export, "
            "and a file watcher for live reloads.<br><br>"
            "Created by Aisling de Gr&aacute;s<br>"
            "<a href=\"mailto:aisling@hiraeth.club\">aisling@hiraeth.club</a>"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        # Make the mailto link clickable.
        inner_label = msg.findChild(QLabel, "qt_msgbox_label")
        if inner_label is not None:
            inner_label.setOpenExternalLinks(True)
            inner_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg.exec_()

    @staticmethod
    def _make_welsh_flag(width: int, height: int) -> QPixmap:
        """Draw a small Welsh flag (Y Ddraig Goch) into a QPixmap.

        Top half white, bottom half green, with a stylised red dragon
        silhouette in the middle. We don't ship a binary asset, so the
        dragon is composed from a few painter primitives (head, body,
        tail, legs, wing) - intentionally cartoony but recognisable
        as a creature on the Welsh colours.
        """
        pix = QPixmap(width, height)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)

        half = height // 2

        # Background stripes.
        p.fillRect(0, 0, width, half, QColor(255, 255, 255))
        p.fillRect(0, half, width, height - half, QColor(0, 135, 73))

        # Welsh red.
        red = QColor(206, 17, 38)
        p.setPen(Qt.NoPen)
        p.setBrush(red)

        # Drawn in a 160x100 reference frame, scaled to the requested size.
        ref_w, ref_h = 160.0, 100.0
        sx = width / ref_w
        sy = height / ref_h

        def rx(v: float) -> float:
            return v * sx

        def ry(v: float) -> float:
            return v * sy

        # ----- Body: a long rounded oval running across the middle. -----
        body = QRectF(rx(40), ry(38), rx(80), ry(28))
        p.drawRoundedRect(body, rx(12), ry(12))

        # ----- Head: ellipse on the left with a triangular snout. -----
        head = QRectF(rx(8), ry(28), rx(40), ry(26))
        p.drawEllipse(head)
        snout = QPainterPath()
        snout.moveTo(rx(0), ry(40))
        snout.lineTo(rx(20), ry(34))
        snout.lineTo(rx(20), ry(46))
        snout.closeSubpath()
        p.drawPath(snout)
        # Tongue / fire poking out of the snout.
        tongue = QPainterPath()
        tongue.moveTo(rx(0), ry(40))
        tongue.lineTo(rx(-8), ry(38))
        tongue.lineTo(rx(-4), ry(42))
        tongue.lineTo(rx(-8), ry(44))
        tongue.lineTo(rx(0), ry(44))
        tongue.closeSubpath()
        p.drawPath(tongue)

        # ----- Wing: a couple of triangular peaks above the body. -----
        wing = QPainterPath()
        wing.moveTo(rx(50), ry(38))
        wing.lineTo(rx(60), ry(14))
        wing.lineTo(rx(72), ry(22))
        wing.lineTo(rx(82), ry(12))
        wing.lineTo(rx(94), ry(22))
        wing.lineTo(rx(98), ry(38))
        wing.closeSubpath()
        p.drawPath(wing)

        # ----- Tail: curling out from the rear. -----
        tail = QPainterPath()
        tail.moveTo(rx(118), ry(40))
        tail.cubicTo(rx(140), ry(30), rx(156), ry(48), rx(158), ry(72))
        tail.lineTo(rx(150), ry(74))
        tail.cubicTo(rx(146), ry(56), rx(132), ry(50), rx(118), ry(54))
        tail.closeSubpath()
        p.drawPath(tail)

        # ----- Four legs hanging below the body. -----
        leg_w = rx(6)
        leg_h = ry(18)
        for lx in (50, 66, 92, 108):
            p.fillRect(QRectF(rx(lx), ry(60), leg_w, leg_h), red)
            # Foot tip.
            p.fillRect(QRectF(rx(lx) - rx(1), ry(60) + leg_h - ry(2), leg_w + rx(2), ry(3)), red)

        # ----- Eye (white dot on the head). -----
        eye_size = max(2, int(rx(3)))
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(int(rx(20)), int(ry(36)), eye_size, eye_size)

        # ----- Border last so it stays crisp on top. -----
        p.setPen(QPen(QColor(40, 40, 40), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, width - 1, height - 1)

        p.end()
        return pix

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        try:
            self.engine.shutdown()
        finally:
            super().closeEvent(event)


def _format_total_size(size: int) -> str:
    val = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0:
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PB"
