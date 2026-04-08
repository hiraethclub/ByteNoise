"""Effects chain for ByteNoise.

All effects operate on a mono ``np.float32`` buffer in ``[-1.0, 1.0]`` and
return a buffer of the same shape. Each effect supports an ``enabled`` flag
and a wet/dry mix where applicable.

Effects implemented:
    * LowPassFilter   - 2nd order Butterworth low-pass with resonance.
    * HighPassFilter  - 2nd order Butterworth high-pass with resonance.
    * Reverb          - Schroeder-style FDN: 4 comb filters + 2 all-pass.
    * Delay           - Single tap delay line with feedback.
    * BitCrusher      - Post-synthesis bit depth + sample rate reduction.
    * Distortion      - Soft tanh saturation with tone control.

The chain is processed in a fixed order matching the GUI layout.
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
# Effects chain container
# ---------------------------------------------------------------------------


@dataclass
class EffectsChain:
    """Ordered effects chain. Order is fixed and matches the GUI layout."""

    low_pass: LowPassFilter = field(default_factory=LowPassFilter)
    high_pass: HighPassFilter = field(default_factory=HighPassFilter)
    distortion: Distortion = field(default_factory=Distortion)
    bit_crusher: BitCrusher = field(default_factory=BitCrusher)
    delay: Delay = field(default_factory=Delay)
    reverb: Reverb = field(default_factory=Reverb)

    def ordered(self) -> List[object]:
        return [
            self.low_pass,
            self.high_pass,
            self.distortion,
            self.bit_crusher,
            self.delay,
            self.reverb,
        ]

    def process(self, buf: np.ndarray, sample_rate: int) -> np.ndarray:
        out = buf
        for fx in self.ordered():
            out = fx.process(out, sample_rate)
        return out
