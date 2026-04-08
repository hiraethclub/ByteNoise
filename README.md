# ByteNoise

**Turn any file into sound.**

ByteNoise is a desktop instrument that reads the raw bytes of a file — an
image, an executable, a novel, your tax return — and renders them as audio.
Four synthesis modes interpret those bytes in very different ways, and a
chain of thirteen effects shapes the result into anything from glitchy noise
to slow, drifting drones.

Built with PyQt5, NumPy and SciPy. No external audio DSP libraries.

## Features

### Synthesis modes

- **Raw Noise** — The bytes *are* the sample stream. Choose 8-bit or paired
  16-bit interpretation and a sample rate from 8 kHz to 96 kHz.
- **Drone** — The bytes drive a bank of detuned oscillators (sine, saw, or
  triangle) with an LFO, voice density, and evolution speed.
- **Crunch** — Bit-depth reduction, sample-rate decimation, and quantisation
  noise for aggressive digital grit.
- **Space Drone** — Long-form ambient: static bed, slow fades, signal clarity,
  drift, and byte-triggered bursts.

### Effects chain (13, in a fixed order)

Parametric EQ · Low-pass filter · High-pass filter · Ring modulator ·
Distortion · Chorus · Flanger · Phaser · Granular · Paulstretch · Bit crusher ·
Delay · Reverb.

Every effect has an enable toggle, a wet/dry mix where it makes sense, and a
live peak meter so you can see what's active.

### Workflow

- **Multi-file layering** — Load several files at once. Each gets its own gain
  slider; the synth renders all of them in parallel and the effects chain runs
  on the summed mix.
- **Preset save/load** — Full patch state (synth mode, parameters, all effects)
  serialised to JSON. Unknown fields are ignored, so presets survive version
  upgrades in both directions.
- **File watcher** — Tick the *Watch* box and ByteNoise re-renders
  automatically whenever one of the loaded files changes on disk. Handles
  atomic editor saves (delete-and-replace).
- **Undo / redo / randomise** — Every parameter change is snapshotted.
  Randomise rolls new values for the current mode and the full effects chain.
- **MIDI export** — Convert the byte stream into a Standard MIDI File. Pick a
  tempo, scale (9 scales from chromatic to phrygian to whole-tone), octave
  range, bytes-per-note, and velocity range. Pure-Python writer, no external
  MIDI dependency.
- **Audio export** — WAV output at the engine's sample rate.

### GUI

Dark theme. Live waveform, spectrum, and per-effect peak meters. Transport
bar with play/stop, loop toggle and master level meter.

## Install

```bash
git clone https://github.com/hiraethclub/bytenoise.git
cd bytenoise
pip install -r requirements.txt
```

Requires Python 3.9+ and a working audio output device. On Linux you'll also
want PortAudio (`sudo apt install libportaudio2`).

## Run

```bash
python main.py
```

or

```bash
python -m bytenoise.main
```

## Usage

1. **Load a file** from *File → Load file*, or click *Load file* on the top
   bar. Add more layers with *File → Add layer...*.
2. **Pick a synth mode** in the Synthesis tabs on the left and move the
   sliders.
3. **Enable effects** in the chain on the right. Each effect has a checkbox,
   its own parameters, and a wet/dry mix.
4. **Hit play** on the transport bar at the bottom.
5. **Save the patch** with *File → Save preset...* — it's just JSON.
6. **Export** audio (WAV) or MIDI from the *File* menu.

### Keyboard shortcuts

| Action       | Shortcut        |
|--------------|-----------------|
| Undo         | `Ctrl+Z`        |
| Redo         | `Ctrl+Shift+Z`  |
| Randomise    | `Ctrl+R`        |
| Save preset  | `Ctrl+S`        |
| Load preset  | `Ctrl+O`        |
| Quit         | `Ctrl+Q`        |

## Project layout

```
bytenoise/
  engine/
    file_reader.py    # loading files into byte buffers
    synthesiser.py    # the four synth modes
    effects.py        # all 13 DSP effects + the chain
    audio_engine.py   # sounddevice playback wrapper
    state.py          # undo/redo snapshot manager
    presets.py        # JSON save/load
    midi_export.py    # Standard MIDI File writer
  gui/
    main_window.py    # top-level Qt window, menus, wiring
    synth_panel.py    # per-mode parameter tabs
    effects_panel.py  # effects chain UI
    layers_panel.py   # loaded-file list with per-layer gain
    waveform.py       # live waveform view
    spectrum.py       # live spectrum view
    level_meter.py    # master output meter
    transport.py      # play/stop/loop bar
    midi_export_dialog.py  # MIDI export parameter dialog
  main.py             # QApplication entry point
```

## Credits

Created by Aisling de Grás — <aisling@hiraeth.club>

Released under the GNU General Public License v3.0. See [LICENSE](LICENSE).
