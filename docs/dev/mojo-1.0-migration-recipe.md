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

## Recipe 5: TRAIT_CALL — Bare callable type as runtime argument (verified)

### Symptom

```text
error: invalid call to '__call__': value passed to '' cannot be converted
from type value '_Self' to an instance of '_Self'; did you mean to instantiate '_Self'?
```

Typically triggered at the call site of a function parameter declared as a bare
callable type:

```mojo
def benchmark_function(func: def() raises -> None, ...) raises -> BenchmarkStatistics:
    func()  # ← error fires here
```

### Root cause

In Mojo 1.0, bare `def(...) -> ...` in a **runtime argument position** is no
longer a structural function type that can accept any compatible callable.
Each named `def` declaration has a unique nominal type; passing a named function
where a bare structural type is expected fails unless the function is made a
**compile-time parameter** (`[FuncType: def(...) -> ...]`).

Contrast with Mojo 0.26.3, where bare `def() raises -> None` in an argument
list acted as an implicit structural type, and any matching callable could be
passed at runtime.

### Fix

Convert the callable argument from a **runtime argument** to a **compile-time
parameter** using Mojo 1.0's parametric function syntax:

**Before (0.26.3):**

```mojo
def benchmark_function(
    func: def() raises -> None,
    warmup_iters: Int = 10,
) raises -> BenchmarkStatistics:
    func()
```

**After (1.0):**

```mojo
def benchmark_function[
    FuncType: def() raises -> None
](
    func: FuncType,
    warmup_iters: Int = 10,
) raises -> BenchmarkStatistics:
    func()
```

The same pattern applies to struct methods (use `[...]` before `(self, ...)`)
and to callables with arguments (`def(AnyTensor) raises -> AnyTensor`):

```mojo
# Before:
def check_gradients(
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    backward_fn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
    ...
) raises -> Bool:
    var output = forward_fn(input)
    var grad = backward_fn(grad_output, input)

# After:
def check_gradients[
    FwdFn: def(AnyTensor) raises -> AnyTensor,
    BwdFn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
](
    forward_fn: FwdFn,
    backward_fn: BwdFn,
    ...
) raises -> Bool:
    var output = forward_fn(input)
    var grad = backward_fn(grad_output, input)
```

When the parametric function calls another parametric function in its body,
the compiler infers the concrete `FuncType` from the argument, so no explicit
`[FuncType]` annotation is needed at internal call sites.

### Verified in

- `shared/benchmarking/runner.mojo:223,230,349` (free functions + struct method)
- `shared/utils/profiling.mojo:436,470` (two overloads)
- `shared/testing/gradient_checker.mojo:232,236,471,742,948,952,1170,1179,1618,1630,1763`
- `shared/testing/property_testing.mojo:223`
- `shared/training/loops/training_loop.mojo:66,69,79,252`
- `shared/training/loops/validation_loop.mojo:48,149,272,333`
- `shared/training/script_runner.mojo:137`
- `shared/training/trainer.mojo:229`

---

## Recipe 6: STD_OS_ATOMIC — atomic module relocation + null UnsafePointer

**Symptom (verbatim):**

```text
error: unable to locate module 'atomic'
error: no matching function in initialization
        self.head = UnsafePointer[_FreeListNode, origin=MutAnyOrigin]()
                                                                     ^
```

**Root cause:** Mojo 1.0 moved `std.os.atomic` to `std.atomic` and made
`UnsafePointer` non-nullable by design. The old no-arg `UnsafePointer[T,
MutAnyOrigin]()` constructor that returned a null pointer was removed; the
1.0 stance is "model nullable pointers with `Optional[UnsafePointer[...]]`."

A keyword-only escape hatch is still available for code that genuinely needs
to construct a literal null pointer: `UnsafePointer[T, O](_unsafe_null=())`.

**Fix:**

```mojo
# Before (0.26):
from std.os.atomic import Atomic
self.head = UnsafePointer[_FreeListNode, origin=MutAnyOrigin]()

# After (1.0):
from std.atomic import Atomic
self.head = UnsafePointer[_FreeListNode, MutAnyOrigin](_unsafe_null=())
```

For new code, prefer `Optional[UnsafePointer[...]]` instead of the
`_unsafe_null=()` keyword — the keyword form is an escape hatch, not the
recommended pattern. The Atomic API also renamed `Consistency` -> `Ordering`
and `MONOTONIC` -> `RELAXED`, and reordered `compare_exchange` to take
`success_ordering` before `failure_ordering`. Our codebase only uses the
defaults, so we did not need to update those call sites.

**Verified in:** `shared/base/memory_pool.mojo` (commit f6d2fa47c).

---

## Recipe 7: DYNAMIC_TRAIT — `def(...)` struct fields are no longer dynamic

**Symptom (verbatim):**

```text
error: dynamic traits not supported yet, please use a compile time generic
       instead of 'def(Float32) -> Float32'
```

**Root cause:** In Mojo 0.26 a struct could store an arbitrary `def(...)`
function value as a field, with dynamic dispatch at call time. Mojo 1.0
forbids this — every callable used as a value must be known at compile
time. The two ways forward:

1. **Promote the function to a struct parameter** (compile-time generic):

   ```mojo
   struct Lam[F: SomeCallableTrait](...):
       var f: Self.F
   ```

2. **Wrap the callable in a struct that conforms to a small callable
   trait**, and parameterize on that trait. This is currently the only
   pattern that works for fields, because raw `def`-typed parameters fail
   to coerce at the call site (named functions have unique types, not the
   declared `def(...)` type).

**Fix (trait-callable pattern, verified to compile):**

```mojo
trait Float32Fn(Copyable, Movable, ImplicitlyDestructible):
    def __call__(self, x: Float32) -> Float32: ...

struct DoubleFn(Copyable, Movable, Float32Fn):
    def __init__(out self): pass
    def __call__(self, x: Float32) -> Float32:
        return x * 2.0

struct Lam[F: Float32Fn](Copyable, Movable):
    var f: Self.F
    def __init__(out self, var f: Self.F):
        self.f = f^
```

**Caveats encountered during D5:**

- `struct S[F: comptime_alias_of_def_type]` followed by `var f: Self.F`
  triggered a compiler crash in `1.0.0b2.dev2026050805`. File a Modular
  issue if you hit this; until then, prefer the trait-callable pattern
  above.
- Promoting a struct to be parametric cascades through every call site.
  Type-erasure wrappers (e.g. `AnyTransform`) lose the ability to wrap the
  parametric struct without also being parametric.
- If the cascade is too large to land in one wave, **stub the field with
  an `Int` placeholder, accept the function arg in `__init__` and discard
  it, raise `Error` from `__call__`, and gate every site with
  `# TODO(mojo-1.0)`**. This unblocks the package compile and clearly
  delegates the real refactor (e.g. to Phase E along with the test files).

**Verified in:** `shared/data/generic_transforms.mojo` (commit 279e726c8;
stubbed pending Phase E refactor).

---

## Recipe 8: ABSOLUTE_IMPORT_DOUBLING — relative imports for tightly-coupled package cycles

**Symptom (verbatim):**

```text
error: cannot implicitly convert 'AnyTensor' value to 'AnyTensor'
error: invalid call to 'save_tensor': value passed to 'tensor' cannot
       be converted from 'AnyTensor' to 'AnyTensor'
```

**Root cause:** When two modules inside the same package both import a
shared type by absolute path AND one of them is also imported back into
the type's own module (a cycle), Mojo's package compiler may compile the
shared module twice — producing two distinct type identities for the same
struct. Values of "type A" cannot then be passed to functions expecting
"type A" because they are technically different types under the hood.

The historical workaround (already documented at the top of
`shared/tensor/tensor_io.mojo`) is to use **relative imports** for sibling
modules inside the cycle. Phase D1's relative->absolute conversion wave
flipped these imports back to absolute and silently re-introduced the
doubling.

**Fix:** in every file inside the cycle, use relative imports for sibling
modules:

```mojo
# Before (post-D1):
from shared.tensor.any_tensor import AnyTensor
from shared.tensor.tensor_io import save_tensor

# After (D5 fix):
# NOTE: relative imports REQUIRED — see tensor_io.mojo top-of-file
# docstring for the type-doubling rationale.
from .any_tensor import AnyTensor
from .tensor_io import save_tensor
```

For `shared/tensor/`, the cycle is `any_tensor` <-> `tensor_io` <->
`tensor_creation`. Other `shared/tensor/` files (`factories`, `typed/*`,
etc.) can keep absolute imports because they are not part of the cycle.

**Future-proofing:** when a future migration wave touches imports inside
a package, leave the explanatory `# NOTE` comments in place. The doubling
is invisible until you call across the cycle.

**Verified in:** `shared/tensor/any_tensor.mojo`,
`shared/tensor/tensor_io.mojo`, `shared/tensor/tensor_creation.mojo`
(commit 4a4f20338).

---

## Recipe 9+: TBD as Phase D agents discover them

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
