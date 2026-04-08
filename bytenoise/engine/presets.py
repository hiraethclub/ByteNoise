"""Preset save/load.

A preset is a JSON file containing the full state of a ByteNoise patch:
the synth settings (mode + per-mode parameters + duration) and the effects
chain (every effect's parameters and enabled flag).

We serialise via ``dataclasses.asdict`` for the writing side and walk the
fields recursively for the reading side. Reading mutates the live ``settings``
and ``chain`` objects in place so any GUI panels referencing them stay valid
- exactly the same trick the undo/redo manager uses when applying snapshots.

The format is intentionally permissive: unknown fields are ignored, missing
fields keep their existing value. This means presets saved with older
ByteNoise versions still load on newer ones (and vice versa) - any new
parameters just stay at the user's current value.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict

from .effects import EffectsChain
from .synthesiser import SynthSettings


PRESET_VERSION = 1


def serialise(settings: SynthSettings, chain: EffectsChain) -> Dict[str, Any]:
    """Return a JSON-friendly dict describing the given state."""
    return {
        "format": "bytenoise-preset",
        "version": PRESET_VERSION,
        "settings": dataclasses.asdict(settings),
        "chain": dataclasses.asdict(chain),
    }


def save_preset(path: str, settings: SynthSettings, chain: EffectsChain) -> None:
    """Write a preset JSON file to ``path``."""
    payload = serialise(settings, chain)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_preset(path: str, settings: SynthSettings, chain: EffectsChain) -> None:
    """Read a preset JSON file from ``path`` and apply it to the live objects.

    Unknown fields are ignored, missing fields keep their current values.
    Raises ``ValueError`` if the file isn't a valid preset.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict) or payload.get("format") != "bytenoise-preset":
        raise ValueError("Not a ByteNoise preset file")

    settings_dict = payload.get("settings") or {}
    chain_dict = payload.get("chain") or {}

    _apply_dict_to_dataclass(settings, settings_dict)
    _apply_dict_to_dataclass(chain, chain_dict)


def _apply_dict_to_dataclass(target: Any, data: Dict[str, Any]) -> None:
    """Walk a dict and copy matching values into a dataclass instance.

    Nested dataclasses are recursed into so each leaf field gets the value
    from the dict if present. Type coercion is left to whatever the field's
    constructor accepted - we just call ``setattr``.
    """
    if not dataclasses.is_dataclass(target) or not isinstance(data, dict):
        return
    for f in dataclasses.fields(target):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(target, f.name)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply_dict_to_dataclass(current, value)
        else:
            try:
                setattr(target, f.name, value)
            except Exception:
                # Skip unsettable fields silently rather than failing the load.
                pass
