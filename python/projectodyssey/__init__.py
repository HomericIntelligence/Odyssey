"""ProjectOdyssey — pip-installable wrapper around the Mojo library.

This wheel bundles the compiled `projectodyssey.mojopkg` so consumers can
``pip install projectodyssey`` and then use the package from Mojo code:

    >>> import projectodyssey
    >>> projectodyssey.mojopkg_path()
    '/path/to/site-packages/projectodyssey/_data/projectodyssey.mojopkg'
    >>> projectodyssey.import_dir()
    '/path/to/site-packages/projectodyssey/_data'

Then in your Mojo project:

    pixi run mojo run -I "$(python -c 'import projectodyssey; print(projectodyssey.import_dir())')" main.mojo

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
    __version__: str = _get_version("projectodyssey")
except PackageNotFoundError:
    __version__ = "0.0.0"

# The .mojopkg ships as package data under projectodyssey/_data/.
# Resolve relative to this module's file — works identically whether the
# package was installed by pip, dropped on PYTHONPATH, or imported from
# a source checkout. No reliance on importlib.resources (which Semgrep
# flags) and no backport dependency.
_DATA_DIR = Path(__file__).resolve().parent / "_data"


def mojopkg_path() -> str:
    """Absolute filesystem path to the bundled `projectodyssey.mojopkg`."""
    return str(_DATA_DIR / "projectodyssey.mojopkg")


def import_dir() -> str:
    """Directory to pass to ``mojo run -I`` so the package can be imported.

    Returns the parent directory of `projectodyssey.mojopkg`.
    """
    return str(_DATA_DIR)
