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

import os

# nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib
# Safe to use here: this wheel pins requires-python = ">=3.10" (see pyproject.toml).
from importlib.resources import files

__all__ = ["mojopkg_path", "import_dir", "__version__"]

# Kept in sync manually with mojo.toml and python/pyproject.toml.
# When this drifts, the wheel build is the source of truth (it embeds this).
__version__ = "0.1.0"


def mojopkg_path() -> str:
    """Absolute filesystem path to the bundled `projectodyssey.mojopkg`."""
    return os.fspath(files(__package__) / "_data" / "projectodyssey.mojopkg")


def import_dir() -> str:
    """Directory to pass to ``mojo run -I`` so the package can be imported.

    Returns the parent directory of `projectodyssey.mojopkg`.
    """
    return os.fspath(files(__package__) / "_data")
