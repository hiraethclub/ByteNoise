"""Synthesis modes for ByteNoise.

Three modes are provided in v1:
    * Raw Noise - direct interpretation of bytes as audio samples.
    * Drone - bytes drive a small bank of oscillators that crossfade over
      time, producing slowly evolving tones.
    * Crunch - lo-fi raw byte interpretation with bit depth reduction and
      sample rate reduction.

All modes return a mono ``np.float32`` buffer in the range ``[-1.0, 1.0]``.

The synthesisers are intentionally written so that the actual file bytes
strongly influence the output - identical files always sound the same, and
different files always sound different. There is no random seeding hidden in
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RawNoiseParams:
    """Parameters for the Raw Noise synthesis mode."""

    sample_rate: int = 44100
    bit_mode: int = 8  # 8 or 16


@dataclass
class DroneParams:
    """Parameters for the Drone synthesis mode."""

    sample_rate: int = 44100
    base_freq: float = 80.0          # lowest oscillator frequency in Hz
    waveform: str = "sine"           # one of: sine, saw, triangle
    lfo_speed: float = 0.5           # Hz, slow modulation
    density: int = 4                 # number of simultaneous voices
    evolution_speed: float = 1.0     # 1.0 = move once through file per buffer


@dataclass
class CrunchParams:
    """Parameters for the Crunch synthesis mode."""

    sample_rate: int = 44100
    bit_depth: int = 6               # 2..16 bits
    rate_reduction: int = 4          # sample-and-hold factor (>=1)
    quantisation_noise: float = 0.1  # 0..1 amount of added noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_length(data: np.ndarray, n: int) -> np.ndarray:
    """Tile or truncate ``data`` so it has exactly ``n`` elements.

    Used so that short files still produce a full-length buffer (we loop the
    bytes), while long files only consume what is needed.
    """
    if data.size == 0:
        return np.zeros(n, dtype=data.dtype)
    if data.size >= n:
        return data[:n]
    repeats = int(np.ceil(n / data.size))
    return np.tile(data, repeats)[:n]


def _bytes_to_float8(byte_data: np.ndarray) -> np.ndarray:
    """Map uint8 bytes (0..255) to float32 in -1.0..1.0."""
    if byte_data.size == 0:
        return np.zeros(0, dtype=np.float32)
    return (byte_data.astype(np.float32) - 127.5) / 127.5


def _bytes_to_float16(byte_data: np.ndarray) -> np.ndarray:
    """Pair consecutive bytes (little-endian) into signed 16-bit floats."""
    if byte_data.size < 2:
        return np.zeros(0, dtype=np.float32)
    # Drop a trailing odd byte if necessary.
    if byte_data.size % 2 != 0:
        byte_data = byte_data[:-1]
    # numpy view -> little-endian int16
    int16 = byte_data.view(np.int16)
    return int16.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Raw Noise
# ---------------------------------------------------------------------------


def synth_raw_noise(
    byte_data: np.ndarray,
    duration_s: float,
    params: RawNoiseParams,
) -> np.ndarray:
    """Render Raw Noise mode.

    Each byte (or pair of bytes) becomes one audio sample. The output buffer
    has ``int(duration_s * sample_rate)`` samples; if the file is shorter we
    loop the data.
    """
    n = max(1, int(duration_s * params.sample_rate))

    if params.bit_mode == 16:
        floats = _bytes_to_float16(byte_data)
    else:
        floats = _bytes_to_float8(byte_data)

    if floats.size == 0:
        return np.zeros(n, dtype=np.float32)

    out = _ensure_length(floats, n).astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------


def _waveform(phase: np.ndarray, kind: str) -> np.ndarray:
    """Generate a periodic waveform from a phase array (0..1 wrapping).

    Phase here is wrapped into [0, 1) before evaluation.
    """
    p = phase - np.floor(phase)
    if kind == "saw":
        return (2.0 * p - 1.0).astype(np.float32)
    if kind == "triangle":
        return (2.0 * np.abs(2.0 * p - 1.0) - 1.0).astype(np.float32)
    # default: sine
    return np.sin(2.0 * np.pi * p).astype(np.float32)


def synth_drone(
    byte_data: np.ndarray,
    duration_s: float,
    params: DroneParams,
) -> np.ndarray:
    """Render Drone mode.

    The file's bytes are split into ``density`` interleaved streams. For each
    stream we look at successive chunks - each chunk's mean byte value selects
    a frequency in a musical range, and we crossfade between consecutive chunk
    frequencies as we progress through the buffer. The result is a slowly
    evolving multi-voice drone whose pitch contour is determined by the file.
    """
    sr = params.sample_rate
    n = max(1, int(duration_s * sr))
    density = max(1, int(params.density))

    if byte_data.size == 0:
        return np.zeros(n, dtype=np.float32)

    # Frequency range: base_freq up to base_freq * 6 (about 2.5 octaves).
    base = float(max(20.0, params.base_freq))
    top = base * 6.0

    # Number of "frames" per voice - how many distinct pitches we step through
    # over the buffer length. Higher evolution_speed = more frames, faster
    # movement. Keep at least 4 frames so there's something to crossfade.
    frames_per_voice = max(4, int(8 * float(params.evolution_speed)))

    # We chunk the file so each voice gets its own slice of bytes for variety.
    # Voice v reads bytes starting at offset v*step.
    voice_offset_step = max(1, byte_data.size // density)

    out = np.zeros(n, dtype=np.float32)

    # Slow LFO applied to the overall amplitude for breathing motion.
    lfo_phase = (np.arange(n, dtype=np.float32) / sr) * float(params.lfo_speed)
    lfo = 0.5 + 0.5 * _waveform(lfo_phase, "sine")

    sample_indices = np.arange(n, dtype=np.float32)
    inv_sr = 1.0 / sr

    for v in range(density):
        # Build a per-voice byte sequence by rolling the data so each voice
        # starts at a different file offset.
        offset = (v * voice_offset_step) % byte_data.size
        rolled = np.roll(byte_data, -offset)

        # Pick `frames_per_voice` evenly spaced sample positions through the
        # rolled byte stream and read those bytes as control values.
        idxs = np.linspace(0, rolled.size - 1, frames_per_voice).astype(np.int64)
        control_bytes = rolled[idxs].astype(np.float32) / 255.0  # 0..1
        # Map control to a frequency in [base, top].
        freqs = base + control_bytes * (top - base)

        # Crossfade between successive frame frequencies. We compute the
        # current frequency at each sample by linearly interpolating between
        # the per-frame freqs along time. Then we integrate to get phase so
        # that frequency changes don't cause clicks.
        frame_positions = np.linspace(0.0, frames_per_voice - 1, n).astype(np.float32)
        f_lo = np.floor(frame_positions).astype(np.int64)
        f_hi = np.minimum(f_lo + 1, frames_per_voice - 1)
        frac = frame_positions - f_lo.astype(np.float32)
        inst_freq = freqs[f_lo] * (1.0 - frac) + freqs[f_hi] * frac

        # Phase = cumulative integral of frequency. Use cumsum * (1/sr).
        phase = np.cumsum(inst_freq) * inv_sr
        # Add per-voice phase offset for richness.
        phase = phase + (v * 0.137)

        voice = _waveform(phase, params.waveform)

        # Slight per-voice gain variation derived from the file (more byte
        # data influence). Voice gain is between 0.4 and 1.0.
        avg_byte = float(rolled.mean()) / 255.0
        gain = 0.4 + 0.6 * avg_byte

        out += voice * gain

    # Normalise voice sum and apply LFO breathing envelope.
    out /= float(density)
    out *= lfo

    # Soft amplitude envelope at the very start/end to prevent clicks.
    fade_len = min(2048, n // 8)
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = fade_in[::-1]
        out[:fade_len] *= fade_in
        out[-fade_len:] *= fade_out

    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Crunch
# ---------------------------------------------------------------------------


def synth_crunch(
    byte_data: np.ndarray,
    duration_s: float,
    params: CrunchParams,
) -> np.ndarray:
    """Render Crunch mode.

    Starts from raw byte interpretation, then applies bit depth reduction and
    sample rate reduction (sample-and-hold) to make it lo-fi and gritty.
    """
    sr = params.sample_rate
    n = max(1, int(duration_s * sr))

    floats = _bytes_to_float8(byte_data)
    if floats.size == 0:
        return np.zeros(n, dtype=np.float32)

    base = _ensure_length(floats, n).astype(np.float32, copy=False).copy()

    # --- Bit depth reduction ---
    bits = max(2, min(16, int(params.bit_depth)))
    levels = float(2 ** bits) - 1.0
    # Map -1..1 -> 0..1 -> quantise -> back to -1..1
    quantised = np.round((base + 1.0) * 0.5 * levels) / levels
    base = quantised * 2.0 - 1.0

    # --- Sample rate reduction (sample-and-hold) ---
    factor = max(1, int(params.rate_reduction))
    if factor > 1:
        # Replace each block of `factor` samples with the first sample of that
        # block. We do this with reshape + broadcast for speed.
        truncated_len = (n // factor) * factor
        if truncated_len > 0:
            held = base[:truncated_len].reshape(-1, factor)
            held = np.repeat(held[:, 0:1], factor, axis=1).reshape(-1)
            base[:truncated_len] = held

    # --- Quantisation noise ---
    qn = float(max(0.0, min(1.0, params.quantisation_noise)))
    if qn > 0.0:
        # Deterministic, file-driven noise: cycle through the byte data again
        # but at a different stride so it sounds different from the carrier.
        stride = max(1, byte_data.size // 7 + 1)
        noise_bytes = byte_data[(np.arange(n) * stride) % byte_data.size]
        noise = (noise_bytes.astype(np.float32) - 127.5) / 127.5
        base = np.clip(base + noise * qn * 0.5, -1.0, 1.0)

    return base.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Top level dispatcher
# ---------------------------------------------------------------------------


@dataclass
class SynthSettings:
    """Bundle of all synthesis parameters used by the engine."""

    mode: str = "raw"  # raw | drone | crunch
    raw: RawNoiseParams = field(default_factory=RawNoiseParams)
    drone: DroneParams = field(default_factory=DroneParams)
    crunch: CrunchParams = field(default_factory=CrunchParams)
    duration_s: float = 30.0


def render(byte_data: np.ndarray, settings: SynthSettings) -> np.ndarray:
    """Render a synthesis buffer using the chosen mode."""
    if settings.mode == "drone":
        return synth_drone(byte_data, settings.duration_s, settings.drone)
    if settings.mode == "crunch":
        return synth_crunch(byte_data, settings.duration_s, settings.crunch)
    return synth_raw_noise(byte_data, settings.duration_s, settings.raw)


def get_sample_rate(settings: SynthSettings) -> int:
    """Return the sample rate associated with the active mode."""
    if settings.mode == "drone":
        return settings.drone.sample_rate
    if settings.mode == "crunch":
        return settings.crunch.sample_rate
    return settings.raw.sample_rate
