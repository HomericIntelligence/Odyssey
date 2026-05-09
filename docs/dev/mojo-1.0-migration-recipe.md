# Mojo 1.0.0b2 Migration Recipe

Per-error-class fix recipes discovered while migrating ProjectOdyssey from
Mojo `0.26.3.0.dev2026040705` to `1.0.0b2.dev2026050805`.

This is the swarm-agent reference: when an agent hits an error in Phase D
(source migration), it looks up the error class here and applies the recipe.

---

## How to use this document

1. Run `pixi run mojo build <file> -I .` (or `mojo package` for libraries).
2. Match the first error message against a recipe section below.
3. Apply the fix, re-run build, repeat.
4. If the error is not listed here, escalate: file an upstream issue at
   `modular/modular`, document a workaround in this file, then continue.

---

## Recipe 1: Parametric `def` value passed as parameter (verified)

### Symptom

```text
error: invalid call to '__call__': '__call__' parameter '_Self' has
'def[T: DType](Scalar[T]) -> Scalar[T]' type, but value has type
'AnyTrait[def[T: DType](Scalar[T]) -> Scalar[T]]'
```

Or:

```text
error: 'apply' parameter 'op' has 'def[T: DType](Scalar[T]) -> Scalar[T]'
type, but value has type 'def my_op[T: DType](x: Scalar[T]) -> Scalar[T]'
```

### Root cause

In Mojo 1.0, every `def` has a unique nominal type bound to its declaration. A
parametric function value is no longer assignable to a structural function type
unless that type is declared as a "thin" function type.

Reference: `modular/modular` v1.0.0b1 release notes —
*"Function literal types unique per definition; two `def(Int) -> Int`
functions no longer interchangeable; use `def(Int) thin -> Int`"*

### Fix

Add `thin` to the function-value parameter type, immediately before `->`:

```mojo
# Before (0.26.x):
def apply[
    dtype: DType, op: def[T: DType](Scalar[T]) -> Scalar[T]
](v: Scalar[dtype]) -> Scalar[dtype]:
    return op[dtype](v)

# After (1.0.0b2):
def apply[
    dtype: DType, op: def[T: DType](Scalar[T]) thin -> Scalar[T]
](v: Scalar[dtype]) -> Scalar[dtype]:
    return op[dtype](v)  # call site unchanged
```

**Call sites do NOT need to change.** Both `op(arg)` (implicit parameter) and
`op[T](arg)` (explicit parameter) work with `thin` parameters.

### Verified in

- `shared/core/dtype_dispatch.mojo:87,130,207,254,342,384,462,517,574`
  (9 `def[T: DType]` parameters; all fixed by adding `thin`)
- `/tmp/probe_c.mojo`, `/tmp/probe_d.mojo` (minimal repros)

---

## Recipe 2: `unified` keyword removed (verified compile error)

### Symptom

```text
error: unknown function effect 'unified', expected 'raises', 'capturing',
'thin', or 'register_passable'
    def normalize_kernel[width: Int](idx: Int) unified {mut}:
                                               ^
error: expected ':' in function definition
    def normalize_kernel[width: Int](idx: Int) unified {mut}:
                                                       ^
```

### Root cause

Mojo 1.0 removed the `unified` function effect entirely. Closure capture is now
expressed exclusively via `{...}` capture lists.

Reference: v1.0.0b1 release notes —
*"unified-closure semantics using explicit capture lists in braces"*.

### Fix (verified — TWO-STEP)

The fix is **two steps**: remove `unified` from the function effect position,
AND replace `{mut}` with an explicit per-closure capture list `{var <name1>, var <name2>, ...}`.

```mojo
# Before (0.26):
@parameter
def vectorized_add[width: Int](idx: Int) unified {mut}:
    var a_vec = a_ptr.load[width=width](idx)
    var b_vec = b_ptr.load[width=width](idx)
    result_ptr.store[width=width](idx, a_vec + b_vec)

# After (1.0):
@parameter
def vectorized_add[width: Int](idx: Int) {var a_ptr, var b_ptr, var result_ptr}:
    var a_vec = a_ptr.load[width=width](idx)
    var b_vec = b_ptr.load[width=width](idx)
    result_ptr.store[width=width](idx, a_vec + b_vec)
```

The `{mut}` shorthand from 0.26 (which captured *everything* mutably) is gone.
You must list each captured variable by name with the `var` keyword.

**This is NOT mechanical**: each closure's capture list depends on which
enclosing-scope variables it reads or writes. Don't bulk-rewrite — fix each
closure individually after a quick scan.

### Verified in

- Modular's own stdlib: `mojo/stdlib/test/algorithm/test_vectorize.mojo`
  uses `def add_two[width: Int](idx: Int) {var vector}:` for closures that
  capture the `vector` Span.
- Modular's own stdlib: `mojo/stdlib/test/algorithm/test_vectorize.mojo` —
  `def double_buf[simd_width: Int](idx: Int) {var buf}:`
- (NB) The 1.0 vectorize docstring example uses `{mut}` — that example is
  stale. The actual `vectorize.mojo` parameter signature is
  `func: def[width: Int](idx: Int) -> None` (no `thin`), and closures match
  it via explicit `{var ...}` capture lists, NOT via `{mut}`.

### Partial fix in this repo

A first-pass commit on the bump branch dropped the entire `unified {mut}`
suffix from 47 closures across 9 files (see git log entry titled
"chore(mojo-1.0): partial unified-keyword removal"). That removed the parser
errors but left `vectorize` call-site type mismatches because the closures
no longer carry capture lists. Phase D's mechanical wave needs to reapply the
correct `{var <captured_vars>}` capture list per closure. Files affected:

- `shared/core/activation_simd.mojo`
- `shared/core/matmul.mojo`
- `shared/core/normalization_simd.mojo`
- `shared/tensor/typed/activation_simd.mojo`
- `shared/tensor/typed/arithmetic_contiguous.mojo`
- `shared/tensor/typed/arithmetic_simd.mojo`
- `shared/tensor/typed/numerical_safety.mojo`
- `shared/training/gradient_clipping.mojo`
- `shared/training/mixed_precision.mojo`

---

## Recipe 3: `UnsafePointer` non-null (currently warning, will be error)

### Symptom

```text
warning: UnsafePointer is non-null by design, so Bool(ptr) is no longer
meaningful. To model a null pointer, use `Optional[UnsafePointer[...]]` and
check with `Bool(opt_ptr)` / `!= None`.
        if self._refcount:
           ~~~~^~~~~~~~~~
```

### Root cause

`UnsafePointer` no longer conforms to `Boolable` or `Defaultable`. It is now
non-null by design with zero-overhead layout where null is the `None` niche of
`Optional`.

### Fix

Two cases:

**Case A** — the field can legitimately be null (e.g. uninitialized refcount):

```mojo
# Before:
struct AnyTensor:
    var _refcount: UnsafePointer[Int]  # may be null

    def is_initialized(self) -> Bool:
        if self._refcount:  # warning
            return True
        return False

# After:
struct AnyTensor:
    var _refcount: Optional[UnsafePointer[Int]]  # explicitly nullable

    def is_initialized(self) -> Bool:
        if self._refcount is not None:
            return True
        return False
```

**Case B** — the field is always non-null (e.g. always set in `__init__`):

```mojo
# Before:
if self._refcount:
    self._refcount.destroy_pointee()

# After (just remove the check):
self._refcount.destroy_pointee()
```

Audit each call site to decide which case applies. Default to Case A when
unsure — a wrapped `Optional` is zero-cost in 1.0.

### Verified in

- `shared/tensor/any_tensor.mojo:452,475,508` (warnings; not yet fixed)

---

## Recipe 4: `mojo test` subcommand removed (planned)

### Symptom

```text
mojo: error: no such command 'test'
```

### Fix

Replace `mojo test <path>` with the new `TestSuite` framework + `mojo run`:

**Per-test-file boilerplate** (add to each `tests/**/test_*.mojo`):

```mojo
from testing import TestSuite, assert_equal, assert_raises

def test_something() raises:
    assert_equal(2 + 2, 4)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

Run with `mojo run tests/path/test_foo.mojo`.

The repo's test runner is being replaced wholesale in Phase B. Once
`scripts/run_mojo_tests.py` exists, agents should NOT invoke `mojo run`
directly — call `pixi run python scripts/run_mojo_tests.py <path>` instead.

### Reference

- Modular's reference: `modular/modular:mojo/examples/testing/test/my_math/test_inc.mojo`
- Modular stdlib tests: `modular/modular:mojo/stdlib/test/builtin/`

---

## Recipe 5+: TBD as Phase D agents discover them

When a swarm agent encounters an error pattern not listed above:

1. Add a new "Recipe N" section here using the same structure (symptom, root
   cause, fix, verified in).
2. Cite the source: official changelog, release notes, or empirical probe.
3. Cross-link from the agent's PR description.

This file is the source of truth for all 1.0.0b2 migration patterns. Keep it
current.

---

## See also

- [`mojo-1.0-test-pattern.md`](mojo-1.0-test-pattern.md) — TestSuite cheatsheet
- [`mojo-1.0-migration-status.md`](mojo-1.0-migration-status.md) — per-file
  status (output of Phase C survey)
- v1.0.0b1 release notes:
  <https://raw.githubusercontent.com/modular/modular/main/mojo/docs/releases/v1.0.0b1.md>
- Nightly changelog:
  <https://raw.githubusercontent.com/modular/modular/main/mojo/docs/nightly-changelog.md>
