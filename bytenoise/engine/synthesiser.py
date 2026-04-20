"""Synthesis modes for ByteNoise.

Modes available:
    * Raw Noise - direct interpretation of bytes as audio samples.
    * Drone - bytes drive a small bank of oscillators that crossfade over
      time, producing slowly evolving tones.
    * Crunch - lo-fi raw byte interpretation with bit depth reduction and
      sample rate reduction.
    * Space Drone (v2) - band-pass filtered noise bursts simulating shortwave
      radio: signal fading, ionospheric drift, static textures, distant
      sine fragments.

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
from scipy import signal


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


@dataclass
class SpaceDroneParams:
    """Parameters for the Space Drone synthesis mode (v2)."""

    sample_rate: int = 44100
    static_amount: float = 0.3      # 0..1 background noise floor
    fade_speed: float = 1.0         # 0.1..5.0 how fast bands cut in/out
    signal_clarity: float = 0.5     # 0..1 how strong the sine fragments are
    drift_rate: float = 0.15        # Hz, slow ionospheric drift LFO
    burst_threshold: float = 0.55   # 0..1 envelope threshold (higher = sparser)
    slowness: float = 0.5           # 0..1, higher = more glacial evolution
    warmth: float = 0.5            # 0..1, higher = consonant harmonics over noise


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
# Space Drone (v2)
# ---------------------------------------------------------------------------


# Fixed set of frequency bands the synth uses. Roughly logarithmic, covering
# the AM/SW radio range plus a couple of high bands for hiss content.
_SPACE_BANDS = (
    (80.0, 200.0),
    (200.0, 500.0),
    (500.0, 1100.0),
    (1100.0, 2400.0),
    (2400.0, 5000.0),
    (5000.0, 10000.0),
)


_CONSONANT_RATIOS = [1.0, 5.0/4, 4.0/3, 3.0/2, 5.0/3, 2.0, 5.0/2, 3.0]


def synth_space_drone(
    byte_data: np.ndarray,
    duration_s: float,
    params: SpaceDroneParams,
) -> np.ndarray:
    """Render Space Drone mode.

    Recipe:
        1. Build a deterministic noise stream from the file bytes.
        2. Filter that noise into a fixed set of overlapping frequency bands.
        3. Build a per-band amplitude envelope from a different slice of the
           file - threshold it so most of the time the band is silent and
           only "bursts" through occasionally.
        4. Apply a slow drift LFO to the band gains to simulate ionospheric
           propagation fades.
        5. Add a faint static noise floor.
        6. Sprinkle in a few sine fragments at byte-derived frequencies for
           the "distant signal" feel.

    The whole thing is driven by the file bytes - same file, same output;
    different file, audibly different shortwave chatter.
    """
    sr = params.sample_rate
    n = max(1, int(duration_s * sr))
    nyq = 0.5 * sr

    if byte_data.size == 0:
        return np.zeros(n, dtype=np.float32)

    slowness = float(np.clip(params.slowness, 0.0, 1.0))
    warmth = float(np.clip(params.warmth, 0.0, 1.0))
    slow_factor = max(0.05, 1.0 - slowness * 0.95)

    # ----- Step 1: deterministic noise base from file bytes -----
    base_noise = _bytes_to_float8(byte_data)
    base_noise = _ensure_length(base_noise, n).astype(np.float32, copy=False)

    # ----- Step 4 (precomputed): drift LFO at params.drift_rate Hz -----
    t = np.arange(n, dtype=np.float32) / sr
    effective_drift = float(params.drift_rate) * slow_factor
    drift = 0.5 + 0.5 * np.sin(2.0 * np.pi * effective_drift * t).astype(
        np.float32
    )

    # ----- Steps 2 + 3: per-band filtered noise * envelope -----
    out = np.zeros(n, dtype=np.float32)
    band_count = len(_SPACE_BANDS)
    fade_speed = max(0.05, float(params.fade_speed) * slow_factor)
    threshold = float(np.clip(params.burst_threshold, 0.0, 0.95))

    env_points = max(4, int(duration_s * 3.0 * fade_speed))

    for i, (lo, hi) in enumerate(_SPACE_BANDS):
        hi_clipped = min(hi, nyq * 0.99)
        if hi_clipped <= lo:
            continue
        sos = signal.iirfilter(
            N=4,
            Wn=[lo / nyq, hi_clipped / nyq],
            btype="band",
            ftype="butter",
            output="sos",
        )
        band = signal.sosfilt(sos, base_noise).astype(np.float32)

        # Build the envelope from a band-specific slice of bytes so each band
        # has its own pattern of bursts. Stride keeps adjacent bands distinct.
        offset = (i * (byte_data.size // band_count)) % byte_data.size
        stride = max(1, byte_data.size // (env_points * 3) + i + 1)
        ctrl_idx = (np.arange(env_points) * stride + offset) % byte_data.size
        ctrl = byte_data[ctrl_idx].astype(np.float32) / 255.0

        # Per-band normalisation: rescale this band's control values to span
        # 0..1 regardless of the file's overall byte distribution. Without
        # this, files made mostly of mid-range bytes (e.g. ASCII text) would
        # never cross the burst threshold and the band would stay silent.
        cmin = float(ctrl.min())
        cmax = float(ctrl.max())
        if cmax > cmin + 1e-6:
            ctrl = (ctrl - cmin) / (cmax - cmin)

        # Stretch the control points to the buffer length with linear interp,
        # then threshold so quiet bytes mute the band entirely (this is what
        # gives the "burst" character).
        env_x = np.linspace(0.0, env_points - 1.0, n).astype(np.float32)
        env = np.interp(env_x, np.arange(env_points, dtype=np.float32), ctrl)
        env = np.maximum(0.0, env - threshold)
        env *= 1.0 / max(1e-3, 1.0 - threshold)
        smooth_win = max(8, int(sr * 0.02 / fade_speed * (1.0 + slowness * 4.0)))
        if smooth_win > 1 and smooth_win < n:
            kernel = np.ones(smooth_win, dtype=np.float32) / smooth_win
            env = np.convolve(env, kernel, mode="same")

        # Drift modulation: each band fades in/out of the drift LFO at a
        # slightly different phase so it sounds like multiple distant stations
        # rolling against each other.
        phase_shift = (i * 0.27) % 1.0
        drift_band = 0.4 + 0.6 * np.roll(drift, int(phase_shift * n))

        # Per-band gain compensation: band-pass filtering loses energy, so
        # boost each band before summing. Final peak normalisation at the end
        # keeps the buffer in range.
        out += band * env * drift_band * 4.0

    # ----- Step 5: static noise floor -----
    static_amount = float(np.clip(params.static_amount, 0.0, 1.0)) * max(0.1, 1.0 - warmth * 0.7)
    if static_amount > 0.0:
        # High-pass the noise so the floor sits above the band content
        # rather than rumbling underneath it.
        sos = signal.iirfilter(
            N=2, Wn=2000.0 / nyq, btype="high", ftype="butter", output="sos"
        )
        static = signal.sosfilt(sos, base_noise).astype(np.float32)
        out += static * static_amount * 0.4

    # ----- Step 6: sine fragments (harmonic when warmth is high) -----
    clarity = float(np.clip(params.signal_clarity, 0.0, 1.0))
    if clarity > 0.0:
        n_fragments = 3 + int(warmth * 5)
        root_byte = float(byte_data[0]) / 255.0
        root_freq = 100.0 + root_byte * 300.0

        for k in range(n_fragments):
            byte_idx = (k * (byte_data.size // n_fragments + 1)) % byte_data.size
            freq_byte = float(byte_data[byte_idx]) / 255.0
            random_freq = 200.0 + freq_byte * 1800.0
            harmonic_freq = root_freq * _CONSONANT_RATIOS[k % len(_CONSONANT_RATIOS)]
            while harmonic_freq > 2000.0:
                harmonic_freq *= 0.5
            while harmonic_freq < 100.0:
                harmonic_freq *= 2.0
            freq = random_freq * (1.0 - warmth) + harmonic_freq * warmth
            sine = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

            offset = ((k + 1) * 1009) % byte_data.size
            stride = max(1, byte_data.size // (env_points * 4) + 1)
            ctrl_idx = (np.arange(env_points) * stride + offset) % byte_data.size
            ctrl = byte_data[ctrl_idx].astype(np.float32) / 255.0
            cmin = float(ctrl.min())
            cmax = float(ctrl.max())
            if cmax > cmin + 1e-6:
                ctrl = (ctrl - cmin) / (cmax - cmin)
            env_x = np.linspace(0.0, env_points - 1.0, n).astype(np.float32)
            env = np.interp(env_x, np.arange(env_points, dtype=np.float32), ctrl)
            frag_thresh = max(0.3, 0.7 - warmth * 0.4)
            env = np.maximum(0.0, env - frag_thresh) * (1.0 / (1.0 - frag_thresh))
            frag_smooth = max(8, int(sr * 0.05 * (1.0 + slowness * 3.0)))
            if frag_smooth > 1 and frag_smooth < n:
                kernel = np.ones(frag_smooth, dtype=np.float32) / frag_smooth
                env = np.convolve(env, kernel, mode="same")

            out += sine * env * clarity * (0.25 + warmth * 0.35)

    # Click guard at edges.
    fade_len = min(2048, n // 8)
    if fade_len > 0:
        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        out[:fade_len] *= ramp
        out[-fade_len:] *= ramp[::-1]

    # Soft normalise so the output sits comfortably in [-1, 1].
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Top level dispatcher
# ---------------------------------------------------------------------------


@dataclass
class SynthSettings:
    """Bundle of all synthesis parameters used by the engine."""

    mode: str = "raw"  # raw | drone | crunch | space
    raw: RawNoiseParams = field(default_factory=RawNoiseParams)
    drone: DroneParams = field(default_factory=DroneParams)
    crunch: CrunchParams = field(default_factory=CrunchParams)
    space: SpaceDroneParams = field(default_factory=SpaceDroneParams)
    duration_s: float = 30.0


def render(byte_data: np.ndarray, settings: SynthSettings) -> np.ndarray:
    """Render a synthesis buffer using the chosen mode."""
    if settings.mode == "drone":
        return synth_drone(byte_data, settings.duration_s, settings.drone)
    if settings.mode == "crunch":
        return synth_crunch(byte_data, settings.duration_s, settings.crunch)
    if settings.mode == "space":
        return synth_space_drone(byte_data, settings.duration_s, settings.space)
    return synth_raw_noise(byte_data, settings.duration_s, settings.raw)


def get_sample_rate(settings: SynthSettings) -> int:
    """Return the sample rate associated with the active mode."""
    if settings.mode == "drone":
        return settings.drone.sample_rate
    if settings.mode == "crunch":
        return settings.crunch.sample_rate
    if settings.mode == "space":
        return settings.space.sample_rate
    return settings.raw.sample_rate
