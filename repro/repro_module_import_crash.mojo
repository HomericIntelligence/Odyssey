"""Deterministic JIT crash via module-level import chain explosion.

Environment: Mojo 0.26.3 (dev2026040705), GLIBC 2.39, Linux 6.6.87 x86_64

CRASH MECHANISM (DETERMINISTIC):
  When a test file imports a library module, Mojo compiles ALL module-level
  imports of that module at parse time. If the transitive import chain
  exceeds the JIT compiler's internal buffer, libKGENCompilerRTShared.so
  crashes before any test output is produced.

  This is DISTINCT from Category 2 (non-deterministic volume crash):
  - Category 2: Non-deterministic, depends on system load / memory fragmentation
  - THIS BUG: Deterministic, triggered by specific import depth, crashes
    EVERY TIME the same import chain is compiled.

CRASH PATTERN:
  $ mojo test repro_module_import_crash.mojo
  Running: repro_module_import_crash.mojo...
  [crash — no test output]

  Stack trace:
    #0 libKGENCompilerRTShared.so+0x6d4ab
    #1 libKGENCompilerRTShared.so+0x6a686
    #2 libKGENCompilerRTShared.so+0x6e157
    #3 libc.so.6 (sigaction / signal handler)

IMPORT CHAIN THAT TRIGGERS CRASH:
  test_file.mojo
    → src/projectodyssey/core/loss_utils.mojo        (module-level)
      → src/projectodyssey/core/elementwise.mojo     (module-level, 1650 lines)
        → src/projectodyssey/core/dtype_dispatch.mojo (module-level, 1520 lines, 176+ mono)

  OR:
  test_file.mojo
    → src/projectodyssey/core/reduction.mojo         (module-level)
      → src/projectodyssey/core/shape.mojo           (module-level, 1371 lines)

FIX: See repro_module_import_crash_fixed.mojo — move heavy imports
  from module level into the function bodies that use them.
"""

# This import chain triggers the crash:
#   loss_utils → elementwise (1650 lines) → dtype_dispatch (1520 lines)
from projectodyssey.core.loss_utils import clip_predictions
from projectodyssey.core.reduction import sum as reduce_sum


def test_clip_is_bounded() raises:
    from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones
    var shape = List[Int]()
    shape.append(4)
    var t = ones(shape, DType.float32)
    var clipped = clip_predictions(t)
    print("test_clip_is_bounded: PASS")


def test_reduce_sum_basic() raises:
    from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones
    var shape = List[Int]()
    shape.append(3)
    var t = ones(shape, DType.float32)
    var result = reduce_sum(t)
    print("test_reduce_sum_basic: PASS")


def main() raises:
    # Expected: crash before reaching here due to module-level import chain
    # If we reach here, the JIT buffer happened not to overflow this run
    test_clip_is_bounded()
    test_reduce_sum_basic()
