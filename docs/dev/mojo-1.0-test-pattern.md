# Mojo 1.0.0b2 Test Pattern

Reference for converting test files from `mojo test` (0.26.x) to the
`TestSuite` framework (1.0.0b2).

---

## The new pattern

Every test file becomes an executable runnable via `mojo run`. The
`TestSuite.discover_tests` macro auto-discovers all `def test_*()`
functions in the module.

```mojo
"""Unit tests for shared.core.dtype_dispatch.

Run with: mojo run tests/shared/core/test_dtype_dispatch.mojo
Or via:   pixi run python scripts/run_mojo_tests.py tests/shared/core/test_dtype_dispatch.mojo
"""

from testing import TestSuite, assert_equal, assert_raises

# Imports under test
from shared.core.dtype_dispatch import elementwise_unary, dispatch_unary
from shared.tensor.any_tensor import AnyTensor


# ----------------------------------------------------------------------------
# Tests — every function starting with `test_` is auto-discovered
# ----------------------------------------------------------------------------


def test_elementwise_unary_float32() raises:
    """Elementwise unary op applied with FP32 specialization."""
    var t = AnyTensor([3], DType.float32)
    # ... assertions ...
    assert_equal(t._numel, 3)


def test_elementwise_unary_unsupported_dtype_raises() raises:
    """Unsupported dtype must raise a descriptive Error."""
    with assert_raises():
        # body that should raise
        var t = AnyTensor([3], DType.bool)
        _ = dispatch_unary[some_op](t)


# ----------------------------------------------------------------------------
# Entry point — required for `mojo run` to discover & execute tests
# ----------------------------------------------------------------------------


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

---

## Conversion checklist (per test file)

When migrating a `tests/**/test_*.mojo` file from 0.26 to 1.0:

1. **Imports**:
   - `from testing import ...` → `from testing import TestSuite, assert_equal, assert_raises, ...`
   - Drop any `from std.testing import ...` (now consolidated under `testing`).

2. **Test functions**:
   - Ensure every test function:
     - Starts with `test_`
     - Is declared `def` (not `fn`)
     - Has `raises` if it uses any `assert_*` (they all raise)
   - Top-level helper functions that don't start with `test_` are not run.

3. **Append a `main()`**:
   - Add the `def main() raises:` block at the bottom (see template above).
   - Inside, exactly: `TestSuite.discover_tests[__functions_in_module()]().run()`

4. **Remove**:
   - Any `if __name__ == "__main__":` blocks (Mojo doesn't use this idiom).
   - Any old `mojo test` runner code.

5. **Verify**:
   - `pixi run mojo run tests/path/test_foo.mojo` — should print test results.

---

## Repo-specific runner

Once `scripts/run_mojo_tests.py` is in place (Phase B), prefer that over
direct `mojo run`:

```bash
# Run one file:
pixi run python scripts/run_mojo_tests.py tests/shared/core/test_shape.mojo

# Run a directory:
pixi run python scripts/run_mojo_tests.py tests/shared/core/

# Filter by glob (replicates `just test-group PATH PATTERN`):
pixi run python scripts/run_mojo_tests.py tests/shared/core/ "test_dtype*.mojo"
```

The runner walks for `test_*.mojo`, runs each via `mojo run`, captures
stdout/stderr/exit, and prints a per-file summary.

---

## See also

- [Mojo 1.0 migration recipe](mojo-1.0-migration-recipe.md) — error class fixes
- Modular's reference test:
  `~/.agent-brain/modular/mojo/examples/testing/test/my_math/test_inc.mojo`
- Modular's stdlib tests (real-world examples):
  `~/.agent-brain/modular/mojo/stdlib/test/builtin/`
- Official testing docs: <https://mojolang.org/docs/tools/testing>
