"""Convenience launcher: ``python main.py``.

Delegates to :mod:`bytenoise.main` so the package layout stays clean.
"""

from bytenoise.main import main

if __name__ == "__main__":
    raise SystemExit(main())
