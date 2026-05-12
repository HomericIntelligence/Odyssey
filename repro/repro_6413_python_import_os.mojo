"""Standalone repro candidate for modular/modular#6413 (Backtrace B).

Captured backtrace (CI run 25649843238, Mojo 1.0.0b2.dev2026050805):

    Thread 1 "mojo" received signal SIGILL, Illegal instruction.
    #0  _strip ()         at oss/modular/mojo/stdlib/std/collections/string/string_slice.mojo:1035
    #1  rstrip ()         at oss/modular/mojo/stdlib/std/collections/string/string_slice.mojo:1104
    #5  __init__ ()       at oss/modular/mojo/stdlib/std/python/python.mojo:45
    #7  KGEN_CompilerRT_GetOrCreateGlobalIndexed () from libKGENCompilerRTShared.so
    #11 import_module ()  at oss/modular/mojo/stdlib/std/python/python.mojo:238
    #12 test_substitute_simple_env_var () at tests/configs/test_env_vars.mojo:21

This file exercises the same code path with NO ProjectOdyssey state — pure stdlib —
so that if it crashes in CI, Modular has a one-file repro Modular asked for in
the issue thread.

Run under the gdb wrapper to capture an ELF core if it crashes:

    bash scripts/mojo-under-gdb.sh /tmp/cores repro/repro_6413_python_import_os.mojo

Repeat 60+ times — the production rate was ~30% per CI job; expect single-digit
percentage at this scale, so loops both inside main() AND in the calling shell
to amplify.
"""

from std.python import Python


def main() raises:
    # Inner loop attacks the KGEN_CompilerRT_GetOrCreateGlobalIndexed path
    # (Python global init) and the string_slice._strip path used by Python.__init__.
    # 200 iterations keeps wall time short while still hitting the global
    # initializer many times (Python re-imports are cached, but the rstrip path
    # in Python.__init__ runs per call).
    for _ in range(200):
        var os_mod = Python.import_module("os")
        _ = os_mod
