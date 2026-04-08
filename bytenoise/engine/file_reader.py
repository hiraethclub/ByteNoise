"""File reading utilities.

Loads any file as raw bytes and exposes basic statistics about its content.
The byte data feeds the synthesis modes downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LoadedFile:
    """Container for a file loaded as raw bytes."""

    path: str
    name: str
    size: int
    data: np.ndarray  # uint8 array of raw bytes
    histogram: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))

    @property
    def mean_byte(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.data.mean())

    @property
    def std_byte(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.data.std())

    def info_string(self) -> str:
        """Return a short, human readable description for the GUI top bar."""
        if self.size == 0:
            return f"{self.name} (empty)"
        size_str = _format_size(self.size)
        return (
            f"{self.name}  |  {size_str}  |  "
            f"mean={self.mean_byte:.1f}  std={self.std_byte:.1f}"
        )


def load_file(path: str, max_bytes: Optional[int] = None) -> LoadedFile:
    """Load a file from disk as raw bytes.

    Parameters
    ----------
    path:
        Absolute or relative path to a file.
    max_bytes:
        Optional cap on bytes read. Useful for very large files where the user
        only wants to scrub through the start. ``None`` reads the whole file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not a file: {path}")

    with open(path, "rb") as fh:
        if max_bytes is None:
            raw = fh.read()
        else:
            raw = fh.read(max_bytes)

    data = np.frombuffer(raw, dtype=np.uint8).copy()
    histogram = np.bincount(data, minlength=256).astype(np.int64) if data.size else np.zeros(256, dtype=np.int64)

    return LoadedFile(
        path=path,
        name=os.path.basename(path),
        size=len(raw),
        data=data,
        histogram=histogram,
    )


def _format_size(size: int) -> str:
    """Pretty print a byte count."""
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(size)
    for unit in units:
        if val < 1024.0 or unit == units[-1]:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{size} B"
