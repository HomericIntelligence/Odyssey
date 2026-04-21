# Reproducer: JIT crash via package-level import volume (shared.core).
#
# ADR-015: This file was moved from tests/shared/core/test_jit_crash_heavy_import.mojo
# to repro/ because its purpose IS to reproduce the crash, not to test production logic.
# It MUST NOT be converted to targeted imports — keeping the package-level import is
# the entire point.
#
# Background: `from shared.core import` forces the Mojo JIT to compile all 37K+ lines
# across 60+ modules via __init__.mojo. Run 30+ times to observe intermittent
# 'execution crashed' (libKGENCompilerRTShared.so+0x6d4ab/+0x6a686/+0x6e157).
#
# See docs/dev/mojo-jit-crash-workaround.md for root-cause analysis.
# See repro/issues/jit-compilation-volume-crash.md for upstream issue template.
#
# To run: mojo repro/repro_jit_heavy_import_test.mojo
# Expected (bug): error: execution crashed (before any output, intermittent)
# Expected (fixed): prints "PASS"
"""JIT crash reproduction: heavy import via shared.core (package-level)."""
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core import relu, sigmoid, matmul, softmax
from std.testing import assert_true


def test_basic_tensor_creation() raises:
    var t = zeros([2, 3], DType.float32)
    assert_true(t.shape()[0] == 2, "shape[0] should be 2")
    assert_true(t.shape()[1] == 3, "shape[1] should be 3")


def main() raises:
    print("Running JIT crash heavy import test...")
    test_basic_tensor_creation()
    print("PASS")
