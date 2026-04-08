"""State helpers: randomise + undo/redo history.

The model state for ByteNoise is the pair (SynthSettings, EffectsChain).
This module provides:

    * ``randomise(settings, chain)`` - mutate both in place with sensible
      random values, leaving the active mode and the buffer duration alone.
    * ``HistoryManager`` - a simple deepcopy-based undo/redo stack that
      stores snapshots of (settings, chain) and can apply them back to the
      live model objects when the user undoes or redoes.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import List, Optional, Tuple

from .effects import EffectsChain
from .synthesiser import SynthSettings


# ---------------------------------------------------------------------------
# Randomise
# ---------------------------------------------------------------------------


def randomise(settings: SynthSettings, chain: EffectsChain) -> None:
    """Mutate ``settings`` and ``chain`` in place with random sensible values.

    The current ``mode`` and ``duration_s`` are preserved so the user doesn't
    accidentally jump to a different synth or change their buffer length.
    """
    rng = random.Random()

    # --- Synth params ---
    raw = settings.raw
    raw.sample_rate = rng.choice([8000, 11025, 22050, 44100, 48000])
    raw.bit_mode = rng.choice([8, 16])

    drone = settings.drone
    drone.base_freq = rng.uniform(30.0, 250.0)
    drone.waveform = rng.choice(["sine", "saw", "triangle"])
    drone.lfo_speed = rng.uniform(0.05, 1.5)
    drone.density = rng.randint(2, 8)
    drone.evolution_speed = rng.uniform(0.3, 3.0)

    crunch = settings.crunch
    crunch.bit_depth = rng.randint(2, 12)
    crunch.rate_reduction = rng.randint(1, 16)
    crunch.quantisation_noise = rng.uniform(0.0, 0.6)

    space = settings.space
    space.static_amount = rng.uniform(0.0, 0.6)
    space.fade_speed = rng.uniform(0.4, 3.0)
    space.signal_clarity = rng.uniform(0.0, 0.8)
    space.drift_rate = rng.uniform(0.05, 0.8)
    space.burst_threshold = rng.uniform(0.3, 0.75)

    # --- Effects: each has a 50% chance of being enabled with random params ---
    lp = chain.low_pass
    lp.enabled = rng.random() < 0.5
    lp.cutoff = rng.uniform(300.0, 12000.0)
    lp.resonance = rng.uniform(0.5, 3.5)
    lp.mix = rng.uniform(0.5, 1.0)

    hp = chain.high_pass
    hp.enabled = rng.random() < 0.4
    hp.cutoff = rng.uniform(40.0, 1500.0)
    hp.resonance = rng.uniform(0.5, 2.5)
    hp.mix = rng.uniform(0.5, 1.0)

    dist = chain.distortion
    dist.enabled = rng.random() < 0.5
    dist.drive = rng.uniform(1.5, 12.0)
    dist.tone = rng.uniform(0.0, 1.0)
    dist.mix = rng.uniform(0.3, 0.9)

    bc = chain.bit_crusher
    bc.enabled = rng.random() < 0.4
    bc.bit_depth = rng.randint(3, 12)
    bc.rate_reduction = rng.randint(1, 8)
    bc.mix = rng.uniform(0.5, 1.0)

    delay = chain.delay
    delay.enabled = rng.random() < 0.5
    delay.time_ms = rng.uniform(60.0, 800.0)
    delay.feedback = rng.uniform(0.1, 0.7)
    delay.mix = rng.uniform(0.2, 0.6)

    rev = chain.reverb
    rev.enabled = rng.random() < 0.6
    rev.room_size = rng.uniform(0.2, 0.95)
    rev.damping = rng.uniform(0.0, 0.9)
    rev.mix = rng.uniform(0.15, 0.55)


# ---------------------------------------------------------------------------
# History (undo/redo)
# ---------------------------------------------------------------------------


Snapshot = Tuple[SynthSettings, EffectsChain]


class HistoryManager:
    """Bounded deepcopy-based undo/redo stack for (settings, chain) pairs.

    Usage:
        history = HistoryManager()
        history.push(settings, chain)         # initial state
        ... user changes parameters, render runs, then ...
        history.push(settings, chain)         # snapshot the new state
        ...
        if history.can_undo():
            history.undo(settings, chain)     # mutates the args in place

    Snapshots are deep copies, so mutating the live model after pushing does
    not affect history entries.
    """

    def __init__(self, max_entries: int = 50) -> None:
        self._stack: List[Snapshot] = []
        self._cursor: int = -1
        self._max = max_entries

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def can_undo(self) -> bool:
        return self._cursor > 0

    def can_redo(self) -> bool:
        return -1 < self._cursor < len(self._stack) - 1

    def __len__(self) -> int:
        return len(self._stack)

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push(self, settings: SynthSettings, chain: EffectsChain) -> None:
        """Append a snapshot of the current state.

        If the cursor is not at the top of the stack (i.e. the user has
        undone some changes and is now making new ones) the redo tail is
        discarded, matching standard editor semantics.
        """
        snap = (copy.deepcopy(settings), copy.deepcopy(chain))

        # Skip if identical to the current top - prevents the history from
        # filling up with no-op pushes when nothing actually changed.
        if (
            self._cursor >= 0
            and self._cursor < len(self._stack)
            and _snapshot_equal(self._stack[self._cursor], snap)
        ):
            return

        # Truncate redo tail.
        if self._cursor < len(self._stack) - 1:
            del self._stack[self._cursor + 1 :]

        self._stack.append(snap)
        # Trim from the front if we exceeded the cap.
        if len(self._stack) > self._max:
            drop = len(self._stack) - self._max
            del self._stack[:drop]
        self._cursor = len(self._stack) - 1

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def undo(self, settings: SynthSettings, chain: EffectsChain) -> bool:
        if not self.can_undo():
            return False
        self._cursor -= 1
        self._apply(self._stack[self._cursor], settings, chain)
        return True

    def redo(self, settings: SynthSettings, chain: EffectsChain) -> bool:
        if not self.can_redo():
            return False
        self._cursor += 1
        self._apply(self._stack[self._cursor], settings, chain)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _apply(snap: Snapshot, settings: SynthSettings, chain: EffectsChain) -> None:
        """Copy the snapshot's field values into the live ``settings`` /
        ``chain`` objects in place.

        We mutate the existing objects (rather than reassigning) so that any
        external code holding references - notably the GUI panels - sees the
        new values without having to be re-wired.
        """
        snap_settings, snap_chain = snap
        snap_settings = copy.deepcopy(snap_settings)
        snap_chain = copy.deepcopy(snap_chain)

        for f in dataclasses.fields(settings):
            setattr(settings, f.name, getattr(snap_settings, f.name))
        for f in dataclasses.fields(chain):
            setattr(chain, f.name, getattr(snap_chain, f.name))


def _snapshot_equal(a: Snapshot, b: Snapshot) -> bool:
    """Compare two snapshots field by field. Used to skip duplicate pushes."""
    return (
        dataclasses.asdict(a[0]) == dataclasses.asdict(b[0])
        and dataclasses.asdict(a[1]) == dataclasses.asdict(b[1])
    )
