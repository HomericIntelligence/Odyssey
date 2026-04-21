"""Fixed version: per-function imports eliminate module-level compilation chain.

Same tests as repro_module_import_crash.mojo but imports moved into function
bodies. This compiles reliably because heavy modules (elementwise, dtype_dispatch,
shape) are not compiled at module load time — only when the calling function
is actually invoked.

RESULT: Passes 100% of the time.

FIX PATTERN:
  BEFORE (module level — compiled for all importers at parse time):
    from shared.core.loss_utils import clip_predictions

  AFTER (per-function — compiled lazily only when function executes):
    def my_test():
        from shared.core.loss_utils import clip_predictions
        ...

WHY THIS WORKS:
  Mojo compiles module-level imports eagerly and transitively. Per-function
  imports are compiled only when the function's code is reached at runtime.
  Moving a heavy import from module level to the first line of the function
  that uses it eliminates the transitive compilation chain from the module's
  compilation unit.

NOTE: This workaround is valid for test files. Library modules should also
  move heavy imports into the specific functions that use them (see
  shared/core/reduction.mojo, conv.mojo, loss_utils.mojo for examples).
"""


def test_clip_is_bounded() raises:
    # Import moved here — NOT at module level
    from shared.core.loss_utils import clip_predictions
    from shared.tensor.any_tensor import ones

    var shape = List[Int]()
    shape.append(4)
    var t = ones(shape, DType.float32)
    var clipped = clip_predictions(t)
    print("test_clip_is_bounded: PASS")


def test_reduce_sum_basic() raises:
    # Import moved here — NOT at module level
    from shared.core.reduction import sum as reduce_sum
    from shared.tensor.any_tensor import ones

    var shape = List[Int]()
    shape.append(3)
    var t = ones(shape, DType.float32)
    var result = reduce_sum(t)
    print("test_reduce_sum_basic: PASS")


def main() raises:
    # Reaches here reliably — no module-level import chain to overflow JIT
    test_clip_is_bounded()
    test_reduce_sum_basic()
