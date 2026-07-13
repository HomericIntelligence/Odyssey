"""Odyssey — pip-installable wrapper around the Mojo library.

This wheel bundles the compiled `odyssey.mojopkg` so consumers can
``pip install odyssey`` and then use the package from Mojo code:

    >>> import odyssey
    >>> odyssey.mojopkg_path()
    '/path/to/site-packages/odyssey/_data/odyssey.mojopkg'
    >>> odyssey.import_dir()
    '/path/to/site-packages/odyssey/_data'

Then in your Mojo project:

    pixi run mojo run -I "$(python -c 'import odyssey; print(odyssey.import_dir())')" main.mojo

The wheel is pure-Python — no compiled bindings. Mojo→Python bindings via
`@export def PyInit_*` + `mojo build --emit shared-lib` are out of scope
here; see Mojo's Python-interop docs if you need them.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _get_version
from pathlib import Path

__all__ = ["mojopkg_path", "import_dir", "__version__"]

# Single source of truth: python/pyproject.toml [project].version, read at
# import time via importlib.metadata. The fallback is hit only in source/editable
# installs that lack dist-info (e.g. running from a fresh checkout).
try:
    __version__: str = _get_version("odyssey")
except PackageNotFoundError:
    __version__ = "0.0.0"

# The .mojopkg ships as package data under odyssey/_data/.
# Resolve relative to this module's file — works identically whether the
# package was installed by pip, dropped on PYTHONPATH, or imported from
# a source checkout. No reliance on importlib.resources (which Semgrep
# flags) and no backport dependency.
_DATA_DIR = Path(__file__).resolve().parent / "_data"


def mojopkg_path() -> str:
    """Absolute filesystem path to the bundled `odyssey.mojopkg`."""
    return str(_DATA_DIR / "odyssey.mojopkg")


def import_dir() -> str:
    """Directory to pass to ``mojo run -I`` so the package can be imported.

    Returns the parent directory of `odyssey.mojopkg`.
    """
    return str(_DATA_DIR)
