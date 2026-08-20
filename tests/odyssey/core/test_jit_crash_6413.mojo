"""Permanent JIT canary for modular/modular#6413.

The historical failure occurred while initializing the Python interop layer:
Mojo's JIT selected AVX-512 instructions after the host or hypervisor masked
those features. Keep this test intentionally small and run it without any
CPU-target override. On a vulnerable toolchain/runner, the process exits with
SIGILL; on a fixed toolchain it completes normally.
"""

from std.python import Python


def test_python_import_jit_path() raises:
    """Exercise the Python initialization and string handling JIT path."""
    # Repeated imports keep this canary useful if the JIT regression is
    # intermittent while remaining cheap enough for every comprehensive run.
    for _ in range(200):
        var os_module = Python.import_module("os")
        _ = os_module


def main() raises:
    print("Running modular/modular#6413 JIT canary...")
    test_python_import_jit_path()
    print("PASS: modular/modular#6413 JIT canary")
