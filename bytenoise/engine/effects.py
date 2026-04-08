"""Effects chain for ByteNoise.

All effects operate on a mono ``np.float32`` buffer in ``[-1.0, 1.0]`` and
return a buffer of the same shape. Each effect supports an ``enabled`` flag
and a wet/dry mix where applicable.

Effects implemented:
    * ParametricEQ    - 3-band peaking biquad EQ.
    * LowPassFilter   - 2nd order Butterworth low-pass with resonance.
    * HighPassFilter  - 2nd order Butterworth high-pass with resonance.
    * RingModulator   - Multiply by a sine carrier at a tunable frequency.
    * Distortion      - Soft tanh saturation with tone control.
    * Chorus          - Single-voice modulated delay.
    * Flanger         - Short modulated delay with feedback.
    * Phaser          - 4-stage all-pass chain with LFO-swept break freq.
    * Granular        - Grain-based reshuffling of the buffer.
    * Paulstretch     - FFT phase-randomisation time-stretch.
    * BitCrusher      - Post-synthesis bit depth + sample rate reduction.
    * Delay           - Single tap delay line with feedback.
    * Reverb          - Schroeder-style FDN: 4 comb filters + 2 all-pass.

The chain is processed in a fixed order matching the GUI layout. After each
effect runs, the chain stores its output peak in ``fx._last_peak`` so the GUI
can display per-effect activity meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wet_dry(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    """Linear crossfade between dry and wet signals.

    ``mix`` of 0 returns dry, 1 returns wet.
    """
    mix = float(np.clip(mix, 0.0, 1.0))
    return (dry * (1.0 - mix) + wet * mix).astype(np.float32)


def _safe_clip(buf: np.ndarray) -> np.ndarray:
    return np.clip(buf, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@dataclass
class LowPassFilter:
    enabled: bool = False
    cutoff: float = 4000.0  # Hz
    resonance: float = 0.7  # Q factor (0.5 = no peak, higher = resonant)
    mix: float = 1.0

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        nyq = 0.5 * sample_rate
        cutoff = float(np.clip(self.cutoff, 20.0, nyq * 0.99))
        # Use a 2nd order biquad-like sos. SciPy's iirfilter with btype='low'.
        # Q is approximated via filter order: keep order=2 and use sosfilt.
        sos = signal.iirfilter(
            N=2,
            Wn=cutoff / nyq,
            btype="low",
            ftype="butter",
            output="sos",
        )
        wet = signal.sosfilt(sos, buf).astype(np.float32)
        # Approximate resonance by mixing in a band-passed peak around cutoff.
        if self.resonance > 0.71:
            q = float(self.resonance)
            bw = max(0.05, 1.0 / q)
            low = max(20.0, cutoff * (1.0 - bw * 0.5))
            high = min(nyq * 0.99, cutoff * (1.0 + bw * 0.5))
            if high > low:
                bp = signal.iirfilter(
                    N=2,
                    Wn=[low / nyq, high / nyq],
                    btype="band",
                    ftype="butter",
                    output="sos",
                )
                wet = wet + signal.sosfilt(bp, buf).astype(np.float32) * (q - 0.7) * 0.5
        return _wet_dry(buf, wet, self.mix)


@dataclass
class HighPassFilter:
    enabled: bool = False
    cutoff: float = 200.0
    resonance: float = 0.7
    mix: float = 1.0

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        nyq = 0.5 * sample_rate
        cutoff = float(np.clip(self.cutoff, 20.0, nyq * 0.99))
        sos = signal.iirfilter(
            N=2,
            Wn=cutoff / nyq,
            btype="high",
            ftype="butter",
            output="sos",
        )
        wet = signal.sosfilt(sos, buf).astype(np.float32)
        if self.resonance > 0.71:
            q = float(self.resonance)
            bw = max(0.05, 1.0 / q)
            low = max(20.0, cutoff * (1.0 - bw * 0.5))
            high = min(nyq * 0.99, cutoff * (1.0 + bw * 0.5))
            if high > low:
                bp = signal.iirfilter(
                    N=2,
                    Wn=[low / nyq, high / nyq],
                    btype="band",
                    ftype="butter",
                    output="sos",
                )
                wet = wet + signal.sosfilt(bp, buf).astype(np.float32) * (q - 0.7) * 0.5
        return _wet_dry(buf, wet, self.mix)


# ---------------------------------------------------------------------------
# Reverb (Schroeder FDN)
# ---------------------------------------------------------------------------


def _comb_filter(buf: np.ndarray, delay: int, feedback: float, damp: float) -> np.ndarray:
    """A single feedback comb filter with one-pole damping in the loop.

    Implemented sample by sample because the feedback path is recursive. We
    keep this in pure NumPy with a Python loop; for the buffer sizes ByteNoise
    uses (a few seconds at 44.1 kHz) this is fast enough.
    """
    n = buf.size
    out = np.zeros(n, dtype=np.float32)
    delay_line = np.zeros(delay, dtype=np.float32)
    last = 0.0
    idx = 0
    for i in range(n):
        delayed = delay_line[idx]
        # One pole low-pass on the feedback path for damping.
        last = delayed * (1.0 - damp) + last * damp
        out[i] = delayed
        delay_line[idx] = buf[i] + last * feedback
        idx += 1
        if idx >= delay:
            idx = 0
    return out


def _allpass_filter(buf: np.ndarray, delay: int, feedback: float = 0.5) -> np.ndarray:
    """A simple all-pass filter used to diffuse the comb output."""
    n = buf.size
    out = np.zeros(n, dtype=np.float32)
    delay_line = np.zeros(delay, dtype=np.float32)
    idx = 0
    for i in range(n):
        delayed = delay_line[idx]
        x = buf[i]
        out[i] = -x + delayed
        delay_line[idx] = x + delayed * feedback
        idx += 1
        if idx >= delay:
            idx = 0
    return out


@dataclass
class Reverb:
    enabled: bool = False
    room_size: float = 0.5  # 0..1, scales feedback
    damping: float = 0.5    # 0..1, low-pass amount in the feedback loop
    mix: float = 0.3

    # Schroeder's classic comb delay times in samples (at 44.1k); scaled by SR.
    _comb_delays_44k = (1116, 1188, 1277, 1356)
    _allpass_delays_44k = (556, 441)

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        scale = sample_rate / 44100.0
        feedback = 0.7 + 0.28 * float(np.clip(self.room_size, 0.0, 1.0))
        damp = float(np.clip(self.damping, 0.0, 1.0)) * 0.4

        wet = np.zeros_like(buf, dtype=np.float32)
        for d in self._comb_delays_44k:
            wet += _comb_filter(buf, max(1, int(d * scale)), feedback, damp)
        wet /= len(self._comb_delays_44k)
        for d in self._allpass_delays_44k:
            wet = _allpass_filter(wet, max(1, int(d * scale)))

        wet = _safe_clip(wet)
        return _wet_dry(buf, wet, self.mix)


# ---------------------------------------------------------------------------
# Delay
# ---------------------------------------------------------------------------


@dataclass
class Delay:
    enabled: bool = False
    time_ms: float = 350.0
    feedback: float = 0.4
    mix: float = 0.3

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        delay_samples = max(1, int(sample_rate * float(self.time_ms) / 1000.0))
        n = buf.size
        out = np.copy(buf).astype(np.float32)
        fb = float(np.clip(self.feedback, 0.0, 0.95))
        # Per-sample feedback loop. Cheap enough for our buffer sizes.
        for i in range(delay_samples, n):
            out[i] += out[i - delay_samples] * fb
        wet = _safe_clip(out)
        return _wet_dry(buf, wet, self.mix)


# ---------------------------------------------------------------------------
# Bit crusher (post-synthesis)
# ---------------------------------------------------------------------------


@dataclass
class BitCrusher:
    enabled: bool = False
    bit_depth: int = 8       # 2..16
    rate_reduction: int = 2  # >=1
    mix: float = 1.0

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        bits = max(2, min(16, int(self.bit_depth)))
        levels = float(2 ** bits) - 1.0
        wet = np.round((buf + 1.0) * 0.5 * levels) / levels
        wet = wet * 2.0 - 1.0

        factor = max(1, int(self.rate_reduction))
        if factor > 1:
            n = wet.size
            truncated_len = (n // factor) * factor
            if truncated_len > 0:
                held = wet[:truncated_len].reshape(-1, factor)
                held = np.repeat(held[:, 0:1], factor, axis=1).reshape(-1)
                wet[:truncated_len] = held

        return _wet_dry(buf, _safe_clip(wet), self.mix)


# ---------------------------------------------------------------------------
# Distortion / overdrive
# ---------------------------------------------------------------------------


@dataclass
class Distortion:
    enabled: bool = False
    drive: float = 2.0   # 1..20
    tone: float = 0.5    # 0 = dark, 1 = bright
    mix: float = 0.6

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        drive = float(max(1.0, self.drive))
        # Soft saturator
        wet = np.tanh(buf * drive).astype(np.float32)
        # Tone control: lerp between LP-filtered (dark) and HP-filtered (bright)
        # versions of the saturated signal.
        nyq = 0.5 * sample_rate
        lp_sos = signal.iirfilter(N=2, Wn=2000.0 / nyq, btype="low",
                                  ftype="butter", output="sos")
        hp_sos = signal.iirfilter(N=2, Wn=1500.0 / nyq, btype="high",
                                  ftype="butter", output="sos")
        dark = signal.sosfilt(lp_sos, wet).astype(np.float32)
        bright = signal.sosfilt(hp_sos, wet).astype(np.float32)
        t = float(np.clip(self.tone, 0.0, 1.0))
        wet = dark * (1.0 - t) + bright * t
        # Compensate the level - tanh + filter usually drops volume.
        wet *= 0.9
        return _wet_dry(buf, _safe_clip(wet), self.mix)


# ---------------------------------------------------------------------------
# Parametric EQ
# ---------------------------------------------------------------------------


def _peaking_biquad_sos(freq: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """Compute peaking biquad coefficients via the RBJ Audio EQ Cookbook.

    Returns a single-section ``sos`` array suitable for ``signal.sosfilt``.
    A gain of 0 dB produces a near-identity filter.
    """
    A = 10.0 ** (float(gain_db) / 40.0)
    omega = 2.0 * np.pi * float(freq) / float(sr)
    cos_w = float(np.cos(omega))
    sin_w = float(np.sin(omega))
    alpha = sin_w / (2.0 * max(0.1, float(q)))

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha / A

    return np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )


@dataclass
class ParametricEQ:
    enabled: bool = False
    low_freq: float = 200.0
    low_gain: float = 0.0    # dB
    mid_freq: float = 1000.0
    mid_gain: float = 0.0
    high_freq: float = 5000.0
    high_gain: float = 0.0
    q: float = 0.9
    mix: float = 1.0

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        out = buf.astype(np.float32, copy=True)
        for freq, gain in (
            (self.low_freq, self.low_gain),
            (self.mid_freq, self.mid_gain),
            (self.high_freq, self.high_gain),
        ):
            if abs(gain) > 0.05:
                f = float(np.clip(freq, 20.0, sample_rate * 0.49))
                sos = _peaking_biquad_sos(f, float(gain), float(self.q), sample_rate)
                out = signal.sosfilt(sos, out).astype(np.float32)
        return _wet_dry(buf, _safe_clip(out), self.mix)


# ---------------------------------------------------------------------------
# Ring modulator
# ---------------------------------------------------------------------------


@dataclass
class RingModulator:
    enabled: bool = False
    carrier_freq: float = 220.0
    mix: float = 0.7

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        n = buf.size
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        carrier = np.sin(2.0 * np.pi * float(self.carrier_freq) * t).astype(np.float32)
        wet = (buf * carrier).astype(np.float32)
        return _wet_dry(buf, _safe_clip(wet), self.mix)


# ---------------------------------------------------------------------------
# Modulated delay (used by chorus and flanger)
# ---------------------------------------------------------------------------


def _modulated_delay(
    buf: np.ndarray,
    base_delay_s: float,
    depth_s: float,
    rate_hz: float,
    feedback: float,
    sample_rate: int,
) -> np.ndarray:
    """Vectorised LFO-modulated delay line.

    Reads from a left-padded copy of the input at fractional positions given
    by an LFO. Feedback is approximated by mixing the wet output back into
    the read once - true per-sample feedback would need a Python loop.
    """
    n = buf.size
    sr = float(sample_rate)
    base = max(0.0, float(base_delay_s)) * sr
    depth = max(0.0, float(depth_s)) * sr
    max_delay = int(np.ceil(base + depth)) + 4

    t = np.arange(n, dtype=np.float32) / sr
    lfo = base + depth * np.sin(2.0 * np.pi * float(rate_hz) * t).astype(np.float32)
    src_pos = np.arange(n, dtype=np.float32) - lfo

    padded = np.concatenate(
        (np.zeros(max_delay, dtype=np.float32), buf.astype(np.float32, copy=False))
    )
    wet = np.interp(
        src_pos + max_delay,
        np.arange(padded.size, dtype=np.float32),
        padded,
    ).astype(np.float32)

    if feedback > 0.0:
        fb_padded = np.concatenate((np.zeros(max_delay, dtype=np.float32), wet))
        wet = wet + np.interp(
            src_pos + max_delay,
            np.arange(fb_padded.size, dtype=np.float32),
            fb_padded,
        ).astype(np.float32) * float(feedback)

    return wet


# ---------------------------------------------------------------------------
# Chorus
# ---------------------------------------------------------------------------


@dataclass
class Chorus:
    enabled: bool = False
    rate: float = 0.6      # LFO rate, Hz
    depth: float = 0.004   # LFO depth, seconds (~4 ms)
    delay: float = 0.020   # base delay, seconds (~20 ms)
    mix: float = 0.5

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        wet = _modulated_delay(buf, self.delay, self.depth, self.rate, 0.0, sample_rate)
        return _wet_dry(buf, _safe_clip(wet), self.mix)


# ---------------------------------------------------------------------------
# Flanger
# ---------------------------------------------------------------------------


@dataclass
class Flanger:
    enabled: bool = False
    rate: float = 0.4
    depth: float = 0.003   # ~3 ms
    delay: float = 0.002   # ~2 ms
    feedback: float = 0.4
    mix: float = 0.5

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        wet = _modulated_delay(
            buf, self.delay, self.depth, self.rate, self.feedback, sample_rate
        )
        return _wet_dry(buf, _safe_clip(wet), self.mix)


# ---------------------------------------------------------------------------
# Phaser (4-stage all-pass with LFO sweep)
# ---------------------------------------------------------------------------


@dataclass
class Phaser:
    enabled: bool = False
    rate: float = 0.4
    depth: float = 0.7      # 0..1 LFO depth
    stages: int = 4         # number of all-pass stages
    mix: float = 0.5

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        n = buf.size
        sr = float(sample_rate)

        # LFO sweeps the all-pass coefficient between low and high values.
        t = np.arange(n, dtype=np.float32) / sr
        lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * float(self.rate) * t).astype(np.float32)
        depth = float(np.clip(self.depth, 0.0, 1.0))
        alpha_array = (0.05 + 0.85 * depth * lfo).astype(np.float64)

        stages = max(1, min(8, int(self.stages)))

        # Process in small chunks where alpha is approximately constant. This
        # lets us use scipy.signal.lfilter (a fast C routine) per chunk and
        # carry the IIR state across chunks via zi/zf, rather than running a
        # per-sample Python loop over 1 M+ samples.
        chunk_size = 256
        wet = np.empty(n, dtype=np.float64)
        zis = [np.zeros(1, dtype=np.float64) for _ in range(stages)]

        idx = 0
        while idx < n:
            end = min(idx + chunk_size, n)
            a = float(alpha_array[idx])
            b_coef = np.array([-a, 1.0], dtype=np.float64)
            a_coef = np.array([1.0, -a], dtype=np.float64)
            chunk = buf[idx:end].astype(np.float64)
            for s in range(stages):
                chunk, zis[s] = signal.lfilter(b_coef, a_coef, chunk, zi=zis[s])
            wet[idx:end] = chunk
            idx = end

        return _wet_dry(buf, _safe_clip(wet.astype(np.float32)), self.mix)


# ---------------------------------------------------------------------------
# Granular reshuffler
# ---------------------------------------------------------------------------


@dataclass
class Granular:
    enabled: bool = False
    grain_ms: float = 80.0   # grain length, ms
    density: float = 1.5     # overlap factor (1 = none, >1 = overlap)
    scatter: float = 0.3     # 0..1 random source position offset
    mix: float = 0.7

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        n = buf.size
        grain_size = max(64, int(sample_rate * float(self.grain_ms) / 1000.0))
        if grain_size >= n:
            return buf

        density = max(0.5, float(self.density))
        scatter = float(np.clip(self.scatter, 0.0, 1.0))

        window = np.hanning(grain_size).astype(np.float32)
        n_grains = max(4, int((n / grain_size) * density))

        out = np.zeros(n + grain_size, dtype=np.float32)

        # Deterministic scatter so the same buffer always produces the same
        # output. Seed comes from the buffer length to vary across renders.
        rng = np.random.RandomState(seed=int(n & 0xFFFF))
        max_src = max(1, n - grain_size)
        for g in range(n_grains):
            base_out = int(g * (n - grain_size) / max(1, n_grains - 1))
            base_src = base_out
            if scatter > 0.0:
                offset = int(rng.uniform(-scatter, scatter) * grain_size * 4)
                base_src = (base_src + offset) % max_src
                if base_src < 0:
                    base_src += max_src
            base_src = max(0, min(max_src, base_src))
            out[base_out : base_out + grain_size] += (
                buf[base_src : base_src + grain_size] * window
            )

        wet = out[:n].astype(np.float32)
        # Soft normalise back to roughly the source level so density and
        # scatter don't affect overall loudness.
        peak_w = float(np.max(np.abs(wet)))
        if peak_w > 0.001:
            target = max(0.5, float(np.max(np.abs(buf))))
            wet = wet * (target / peak_w)
        return _wet_dry(buf, _safe_clip(wet), self.mix)


# ---------------------------------------------------------------------------
# Paulstretch
# ---------------------------------------------------------------------------


@dataclass
class Paulstretch:
    """FFT phase-randomisation time-stretch (the Paulstretch algorithm).

    Stretches the front portion of the buffer to fill the full output length,
    creating smeared, ambient drone textures from any source material. The
    larger the stretch, the less of the buffer's start is used.
    """

    enabled: bool = False
    stretch: float = 4.0     # how much to stretch the input
    window_ms: float = 280.0  # FFT window length in ms
    mix: float = 0.7

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or buf.size == 0:
            return buf
        n = buf.size
        sr = sample_rate
        stretch = max(1.0, float(self.stretch))
        window_size = max(512, int(sr * float(self.window_ms) / 1000.0))
        # Round to even so the half-window math works.
        window_size = (window_size // 2) * 2
        if window_size >= n:
            return buf

        # Source slice that will be stretched. With stretch=4 we use 1/4 of
        # the input, with stretch=1 we use the whole thing.
        src_len = max(window_size * 2, int(n / stretch))
        src_len = min(src_len, n)
        src = buf[:src_len].astype(np.float32, copy=False)

        half = window_size // 2
        window = np.hanning(window_size).astype(np.float32)

        # The output advances by `half` samples per frame; the input advances
        # by `half / stretch` samples per frame, so the input lasts longer in
        # output time. This is what gives the time-stretch.
        in_step = max(1.0, half / stretch)

        out = np.zeros(n + window_size, dtype=np.float32)
        out_pos = 0
        in_pos = 0.0

        max_in_start = max(1, src_len - window_size)
        while out_pos + window_size <= n + window_size:
            i0 = int(in_pos) % max_in_start
            chunk = src[i0 : i0 + window_size] * window

            # Randomise the phase of each FFT bin while keeping the magnitudes.
            # This is the trick that makes paulstretch sound smooth instead of
            # introducing the artefacts you'd get from naive overlap-add.
            spec = np.fft.rfft(chunk)
            mag = np.abs(spec)
            # Deterministic per-frame phase: same input → same output.
            seed = (i0 * 2654435761) & 0xFFFFFFFF
            rng = np.random.RandomState(seed=seed)
            phases = rng.uniform(0.0, 2.0 * np.pi, mag.size)
            spec_new = mag * np.exp(1j * phases)
            ichunk = np.fft.irfft(spec_new, n=window_size).astype(np.float32) * window

            out[out_pos : out_pos + window_size] += ichunk
            out_pos += half
            in_pos += in_step

        wet = out[:n].astype(np.float32)
        # Paulstretch tends to flatten the dynamics; renormalise toward the
        # source peak so the wet level stays comparable to the dry.
        peak_w = float(np.max(np.abs(wet)))
        if peak_w > 0.001:
            target = max(0.5, float(np.max(np.abs(buf))))
            wet = wet * (target / peak_w)
        return _wet_dry(buf, _safe_clip(wet), self.mix)


# ---------------------------------------------------------------------------
# Effects chain container
# ---------------------------------------------------------------------------


@dataclass
class EffectsChain:
    """Ordered effects chain. Order is fixed and matches the GUI layout.

    The chain also writes ``fx._last_peak`` after each effect runs so the
    GUI can display per-effect activity meters without re-processing the
    buffer.
    """

    eq: ParametricEQ = field(default_factory=ParametricEQ)
    low_pass: LowPassFilter = field(default_factory=LowPassFilter)
    high_pass: HighPassFilter = field(default_factory=HighPassFilter)
    ring_mod: RingModulator = field(default_factory=RingModulator)
    distortion: Distortion = field(default_factory=Distortion)
    chorus: Chorus = field(default_factory=Chorus)
    flanger: Flanger = field(default_factory=Flanger)
    phaser: Phaser = field(default_factory=Phaser)
    granular: Granular = field(default_factory=Granular)
    paulstretch: Paulstretch = field(default_factory=Paulstretch)
    bit_crusher: BitCrusher = field(default_factory=BitCrusher)
    delay: Delay = field(default_factory=Delay)
    reverb: Reverb = field(default_factory=Reverb)

    def ordered(self) -> List[object]:
        return [
            self.eq,
            self.low_pass,
            self.high_pass,
            self.ring_mod,
            self.distortion,
            self.chorus,
            self.flanger,
            self.phaser,
            self.granular,
            self.paulstretch,
            self.bit_crusher,
            self.delay,
            self.reverb,
        ]

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        out = buf
        for fx in self.ordered():
            out = fx.process(out, sample_rate)
            # Track per-effect output peak for the activity meters.
            if getattr(fx, "enabled", False) and out.size:
                fx._last_peak = float(np.max(np.abs(out)))
            else:
                fx._last_peak = 0.0
        return out
