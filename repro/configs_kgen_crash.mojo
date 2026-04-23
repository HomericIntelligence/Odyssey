"""Minimal reproducer for Configs group 100% KGEN crash.

Every test in tests/configs/ crashes at libKGENCompilerRTShared.so+0x6d4ab
before any output. This file isolates the minimum import that triggers it.

The crash is a KGEN JIT buffer overflow during compilation of shared.utils.config,
which imports std.python (CPython interop) and defines Dict[String, ConfigValue]
with ConfigValue containing List[String]. The combination triggers the same
__fortify_fail_abort seen across the project.

Run:
    pixi run mojo repro/configs_kgen_crash.mojo

Expected: Either runs successfully (KGEN doesn't overflow) or crashes with
    libKGENCompilerRTShared.so execution crashed
before any output, confirming the compilation-phase KGEN overflow.
"""

from shared.utils.config import Config


def main() raises:
    print("If you see this, the KGEN crash did NOT occur on this run.")
    var c = Config()
    c.set("key", "value")
    print("Config created successfully:", c.get_string("key"))
