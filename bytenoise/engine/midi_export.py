"""MIDI file export.

Turns the loaded file's bytes into a Standard MIDI File: each "byte group"
becomes a note whose pitch comes from the byte value mapped onto a musical
scale. This is the same spirit as the audio synth modes - the file's bytes
deterministically drive everything - but the output is symbolic so the user
can drop it into a DAW and assign their own instruments.

Standard MIDI File format reference:
    Header chunk:  "MThd" + length(4) + format(2) + tracks(2) + division(2)
    Track chunk:   "MTrk" + length(4) + events
    Event:         delta-time(varint) + status + data

We write SMF format 0 (single track) with a tempo meta event followed by
note on/off pairs.

No external dependencies - just bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------


# Each scale is a list of semitone offsets within an octave.
SCALES = {
    "chromatic":          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major":              [0, 2, 4, 5, 7, 9, 11],
    "minor":              [0, 2, 3, 5, 7, 8, 10],
    "pentatonic_minor":   [0, 3, 5, 7, 10],
    "pentatonic_major":   [0, 2, 4, 7, 9],
    "blues":              [0, 3, 5, 6, 7, 10],
    "dorian":             [0, 2, 3, 5, 7, 9, 10],
    "phrygian":           [0, 1, 3, 5, 7, 8, 10],
    "whole_tone":         [0, 2, 4, 6, 8, 10],
}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class MidiExportParams:
    """User-tunable parameters for the MIDI export."""

    tempo_bpm: int = 110
    note_length_beats: float = 0.25  # 1/16 by default
    base_octave: int = 3              # C3 = MIDI 48
    octaves: int = 3                  # range = octaves * scale_size notes
    scale: str = "pentatonic_minor"
    bytes_per_note: int = 16          # group bytes (averaged) before mapping
    velocity_min: int = 50
    velocity_max: int = 115
    max_notes: int = 4096             # cap so huge files don't make huge MIDIs


# ---------------------------------------------------------------------------
# Variable-length quantity encoding (used by SMF for delta times)
# ---------------------------------------------------------------------------


def _vlq(value: int) -> bytes:
    """Encode an integer as a SMF variable-length quantity (big-endian, 7-bit)."""
    value = max(0, int(value))
    if value == 0:
        return b"\x00"
    parts: List[int] = []
    parts.append(value & 0x7F)
    value >>= 7
    while value > 0:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(parts))


# ---------------------------------------------------------------------------
# Note generation
# ---------------------------------------------------------------------------


def _bytes_to_notes(
    byte_data: np.ndarray,
    params: MidiExportParams,
) -> List[Tuple[int, int]]:
    """Map a byte array to a list of (midi_note, velocity) tuples.

    Bytes are averaged in groups of ``bytes_per_note`` so the resulting note
    sequence is reasonable even for large files. Pitch comes from the average
    byte value mapped through the chosen scale; velocity comes from the byte
    deviation within the group (loud = lots of variance).
    """
    if byte_data.size == 0:
        return []

    group = max(1, int(params.bytes_per_note))
    n_groups = byte_data.size // group
    if n_groups == 0:
        # Whole file is shorter than one group - use the whole thing as one note.
        n_groups = 1
        group = byte_data.size

    n_groups = min(n_groups, max(1, int(params.max_notes)))

    used = byte_data[: n_groups * group].astype(np.float32)
    grid = used.reshape(n_groups, group)
    means = grid.mean(axis=1) / 255.0           # 0..1
    stds = grid.std(axis=1) / 128.0             # 0..~1
    stds = np.clip(stds, 0.0, 1.0)

    # Build the available pitch list: octaves * scale_size unique semitones.
    scale_offsets = SCALES.get(params.scale, SCALES["pentatonic_minor"])
    octaves = max(1, int(params.octaves))
    pitches: List[int] = []
    base_midi = max(0, min(108, 12 * (int(params.base_octave) + 1)))  # C of octave
    for oct_i in range(octaves):
        for off in scale_offsets:
            note = base_midi + 12 * oct_i + off
            if 0 <= note <= 127:
                pitches.append(note)
    if not pitches:
        pitches = [60]  # C4 fallback

    # Map mean -> index in the pitch list.
    idxs = np.clip((means * len(pitches)).astype(np.int32), 0, len(pitches) - 1)
    notes = [pitches[i] for i in idxs]

    # Map std -> velocity.
    vmin = int(np.clip(params.velocity_min, 1, 127))
    vmax = int(np.clip(params.velocity_max, vmin, 127))
    vels = (vmin + (vmax - vmin) * stds).astype(np.int32)
    vels = np.clip(vels, 1, 127)

    return list(zip(notes, [int(v) for v in vels]))


# ---------------------------------------------------------------------------
# Track writer
# ---------------------------------------------------------------------------


def _build_track(notes: List[Tuple[int, int]], params: MidiExportParams) -> bytes:
    """Build the MTrk payload (events + end-of-track meta event)."""
    ppq = 480
    note_ticks = max(1, int(round(ppq * float(params.note_length_beats))))

    events = bytearray()

    # Tempo meta event: FF 51 03 tt tt tt (microseconds per quarter note).
    bpm = max(20, min(300, int(params.tempo_bpm)))
    us_per_qn = int(round(60_000_000 / bpm))
    events += _vlq(0)
    events += b"\xFF\x51\x03"
    events += us_per_qn.to_bytes(3, "big")

    # Program change (acoustic piano = 0) just so DAWs that ignore patches
    # still get something audible by default.
    events += _vlq(0)
    events += b"\xC0\x00"

    # Note events. We chain them back-to-back: each note's "off" lands at
    # the same tick as the next note's "on" so the result is a continuous
    # rhythmic line at the chosen note length.
    channel = 0
    note_on = 0x90 | channel
    note_off = 0x80 | channel
    for i, (pitch, vel) in enumerate(notes):
        # Delta from the previous event.
        events += _vlq(0)
        events += bytes([note_on, pitch & 0x7F, vel & 0x7F])
        events += _vlq(note_ticks)
        events += bytes([note_off, pitch & 0x7F, 0x40])

    # End of track.
    events += _vlq(0)
    events += b"\xFF\x2F\x00"

    return bytes(events)


def _build_smf(track: bytes) -> bytes:
    """Wrap a single track in MThd + MTrk chunks (format 0)."""
    ppq = 480
    header = b"MThd" + (6).to_bytes(4, "big")
    header += (0).to_bytes(2, "big")          # format 0
    header += (1).to_bytes(2, "big")          # 1 track
    header += ppq.to_bytes(2, "big")          # division (ticks per quarter)

    mtrk = b"MTrk" + len(track).to_bytes(4, "big") + track
    return header + mtrk


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def export_midi(
    path: str,
    byte_data: np.ndarray,
    params: MidiExportParams,
) -> int:
    """Write a MIDI file built from ``byte_data`` to ``path``.

    Returns the number of notes written.
    """
    notes = _bytes_to_notes(byte_data, params)
    track = _build_track(notes, params)
    smf = _build_smf(track)
    with open(path, "wb") as fh:
        fh.write(smf)
    return len(notes)
