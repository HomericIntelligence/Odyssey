# Mojo 1.0.0b2 Migration Recipe

Per-error-class fix recipes discovered while migrating Odyssey from
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

- `src/odyssey/core/dtype_dispatch.mojo:87,130,207,254,342,384,462,517,574`
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
@always_inline
def vectorized_add[width: Int](idx: Int) {var a_ptr, var b_ptr, var result_ptr}:
    var a_vec = a_ptr.load[width=width](idx)
    var b_vec = b_ptr.load[width=width](idx)
    result_ptr.store[width=width](idx, a_vec + b_vec)
```

The `{mut}` shorthand from 0.26 (which captured *everything* mutably) is gone.
You must list each captured variable by name with the `var` keyword.

**IMPORTANT — decorator must change too**: `@parameter` does NOT combine with
`{var ...}` capture lists in 1.0 (parser rejects `def ... (...) {var ...}:`
when the function carries `@parameter`). Use `@always_inline` instead.
`@parameter` was the 0.26 way to inline-monomorphize a closure into the
parent function's body; in 1.0 that's expressed via `@always_inline`.

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

- `src/odyssey/core/activation_simd.mojo`
- `src/odyssey/core/matmul.mojo`
- `src/odyssey/core/normalization_simd.mojo`
- `src/odyssey/tensor/typed/activation_simd.mojo`
- `src/odyssey/tensor/typed/arithmetic_contiguous.mojo`
- `src/odyssey/tensor/typed/arithmetic_simd.mojo`
- `src/odyssey/tensor/typed/numerical_safety.mojo`
- `src/odyssey/training/gradient_clipping.mojo`
- `src/odyssey/training/mixed_precision.mojo`

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

- `src/odyssey/tensor/any_tensor.mojo:452,475,508` (warnings; not yet fixed)

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

- `src/odyssey/benchmarking/runner.mojo:223,230,349` (free functions + struct method)
- `src/odyssey/utils/profiling.mojo:436,470` (two overloads)
- `src/odyssey/testing/gradient_checker.mojo:232,236,471,742,948,952,1170,1179,1618,1630,1763`
- `src/odyssey/testing/property_testing.mojo:223`
- `src/odyssey/training/loops/training_loop.mojo:66,69,79,252`
- `src/odyssey/training/loops/validation_loop.mojo:48,149,272,333`
- `src/odyssey/training/script_runner.mojo:137`
- `src/odyssey/training/trainer.mojo:229`

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

**Verified in:** `src/odyssey/base/memory_pool.mojo` (commit f6d2fa47c).

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

**Fix A (trait-callable pattern, verified to compile):**

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

**Fix B (thin function parameter — verified in E3, simpler for non-capturing defs):**

If the function is **non-capturing** (does not close over runtime variables),
use `thin` in the parameter type. Thin function parameters can be used as
compile-time struct parameters and called via `Self.F(...)` without triggering
the `capturing` restriction on `Transform.__call__`:

```mojo
# Module-level helper to dispatch through a thin F
def _apply_f32[F: def(Float32) thin -> Float32](data: AnyTensor) raises -> AnyTensor:
    var result = List[Float32](capacity=data.num_elements())
    for i in range(data.num_elements()):
        result.append(F(Float32(data[i])))
    return AnyTensor(result^)

struct LambdaTransform[F: def(Float32) thin -> Float32](Copyable, Movable, Transform):
    var _dummy: Int  # needed so synthesised copy constructor works
    def __init__(out self): self._dummy = 0
    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        return _apply_f32[Self.F](data)
```

Call sites change from `LambdaTransform(f)` to `LambdaTransform[f]()`.
Local non-capturing `def` functions are automatically `thin` in Mojo 1.0.

**Type-erasure wrappers when using thin parameters:**

`AnyTransform` cannot store `Optional[LambdaTransform[F]]` for an unknown `F`.
Instead, store a `Optional[def(AnyTensor) raises thin -> AnyTensor]` field and
provide a static factory method:

```mojo
struct AnyTransform(Copyable, Movable, Transform):
    var _lambda_fn: Optional[def(AnyTensor) raises thin -> AnyTensor]
    # ... other optional fields for stateful transforms (ClampTransform, etc.)

    @staticmethod
    def from_lambda[F: def(Float32) thin -> Float32]() -> AnyTransform:
        # _apply_f32[F] is statically specialised here; the result is thin
        # and can be stored in the Optional field.
        return AnyTransform(_apply_f32[F])
```

Call sites change from `AnyTransform(LambdaTransform(f))` to
`AnyTransform.from_lambda[f]()`.

Key constraint: `_apply_f32[F]` must be resolved inside a **static method**
(or module-level function) where `F` is a compile-time parameter — NOT inside
a struct instance method. If called from within a parametric struct's instance
method, the result is `capturing` and cannot be stored as a thin field.

**Caveats encountered during D5:**

- `struct S[F: comptime_alias_of_def_type]` followed by `var f: Self.F`
  triggered a compiler crash in `1.0.0b2.dev2026050805`. File a Modular
  issue if you hit this; until then, prefer Fix A or Fix B above.
- Promoting a struct to be parametric cascades through every call site.
  Type-erasure wrappers (e.g. `AnyTransform`) lose the ability to wrap the
  parametric struct without also being parametric — use the static factory
  approach shown in Fix B instead.
- If the cascade is too large to land in one wave, **stub the field with
  an `Int` placeholder, accept the function arg in `__init__` and discard
  it, raise `Error` from `__call__`, and gate every site with
  `# TODO(mojo-1.0)`**. This unblocks the package compile and clearly
  delegates the real refactor (e.g. to Phase E along with the test files).

**Verified in:** `src/odyssey/data/generic_transforms.mojo` (D5 commit: stubbed;
E3 commit: fully implemented with thin-parameter pattern).

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
`src/odyssey/tensor/tensor_io.mojo`) is to use **relative imports** for sibling
modules inside the cycle. Phase D1's relative->absolute conversion wave
flipped these imports back to absolute and silently re-introduced the
doubling.

**Fix:** in every file inside the cycle, use relative imports for sibling
modules:

```mojo
# Before (post-D1):
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_io import save_tensor

# After (D5 fix):
# NOTE: relative imports REQUIRED — see tensor_io.mojo top-of-file
# docstring for the type-doubling rationale.
from .any_tensor import AnyTensor
from .tensor_io import save_tensor
```

For `src/odyssey/tensor/`, the cycle is `any_tensor` <-> `tensor_io` <->
`tensor_creation`. Other `src/odyssey/tensor/` files (`factories`, `typed/*`,
etc.) can keep absolute imports because they are not part of the cycle.

**Future-proofing:** when a future migration wave touches imports inside
a package, leave the explanatory `# NOTE` comments in place. The doubling
is invisible until you call across the cycle.

**Verified in:** `src/odyssey/tensor/any_tensor.mojo`,
`src/odyssey/tensor/tensor_io.mojo`, `src/odyssey/tensor/tensor_creation.mojo`
(commit 4a4f20338).

---

## Recipe 9: STRING_SUBSTR — `String.substr()` and `String[:n]` removed

**Symptom (verbatim):**

```text
error: 'String' value has no attribute 'substr'
error: no matching method in call to '__getitem__'
    var s = my_str[:6]
            ~^~~~~~~~~
note: candidate not viable: missing 1 required keyword-only argument: 'byte'
```

**Root cause:** In Mojo 1.0, `String.substr(i, j)` was removed entirely.
The `String.__getitem__(Slice)` overload now requires an explicit `byte=True`
keyword at the call site, but the `s[i:j]` subscript syntax does not yet
support keyword-only arguments inside brackets — so `s[:n]` also fails.

**Fix:** Use byte-level slicing via `as_bytes()` and reconstruct a String with
`String(unsafe_from_utf8=...)`:

```mojo
# Before (0.26):
var short = s.substr(0, 6)
var short2 = s[:6]

# After (1.0):
def str_trunc(s: String, n: Int) -> String:
    """Truncate string to n bytes (ASCII-safe)."""
    var b = s.as_bytes()
    if len(b) <= n:
        return s
    return String(unsafe_from_utf8=b[:n])

var short = str_trunc(s, 6)
```

Add `str_trunc` (or equivalent) as a module-level helper. Do NOT inline the
pattern at every call site — it's verbose and error-prone.

**Note:** `unsafe_from_utf8` is named because it skips UTF-8 validation. For
ASCII-only strings (e.g. float representations) this is safe. For user-supplied
or non-ASCII strings, use `String(from_utf8=b[:n])` instead (may raise).

**Verified in:** `benchmarks/bench_matmul.mojo` (12 call sites),
`benchmarks/reporter.mojo` (1 call site).

---

## Recipe 10: STRING_LJUST — `String.ljust()` removed

**Symptom (verbatim):**

```text
error: 'String' value has no attribute 'ljust'
error: 'StringLiteral["Operation"]' value has no attribute 'ljust'
```

**Root cause:** `String.ljust(width)` (and `.rjust()`, `.center()`) were
removed in Mojo 1.0. There is no direct replacement in the stdlib.

**Fix:** Manual padding with string concatenation and a conditional:

```mojo
# Before (0.26):
print(name.ljust(10), value.ljust(8))

# After (1.0):
var name_pad = name + " " * (10 - len(name)) if len(name) < 10 else name
var val_pad  = value + " " * (8 - len(value)) if len(value) < 8 else value
print(name_pad, val_pad)
```

For repeated use, extract a helper:

```mojo
def ljust(s: String, width: Int) -> String:
    if len(s) >= width:
        return s
    return s + " " * (width - len(s))
```

**Verified in:** `benchmarks/bench_simd.mojo` (6 call sites across function
body and header print).

---

## Recipe 11: TEMPLATE_PLACEHOLDER — Mojo 1.0 rejects `{{...}}` mustache syntax in structural positions

**Symptom (verbatim):**

```text
error: expected struct name
    struct {{name}}(Module):
           ^
error: expected module name
    from {{module_path}} import *
         ^
error: expected construct name to import
    from odyssey.nn import Module, {{imports}}
                                  ^
```

**Root cause:** Mojo 1.0's parser (stricter than 0.26) rejects `{{...}}`
mustache/Jinja tokens in syntactically-sensitive positions: struct names,
import module paths, and import symbol lists. In 0.26 the parser may have
silently ignored these; in 1.0 they are hard parse errors.

**Fix:** Replace every `{{X}}` that appears in a structural Mojo position with
an identifier-safe token ending in `_PLACEHOLDER` (e.g. `Layer_PLACEHOLDER`,
`Model_PLACEHOLDER`, `Dataset_PLACEHOLDER`). Add a `RENDERING CONVENTION`
comment block at the top of the template documenting the token map so the
scaffold renderer knows what to substitute.

Placeholders that appear only inside string literals, docstrings, or comments
can remain as `{{X}}` — they don't cause parse errors.

```mojo
# Before (causes parse error in 1.0):
struct {{name}}(Module):
    ...

# After:
# RENDERING CONVENTION: Layer_PLACEHOLDER -> {{name}}
struct Layer_PLACEHOLDER(Module):
    ...
```

Also update stale import paths simultaneously:

- `from odyssey.nn import Module` → `from odyssey.core.module import Module`
- `from odyssey.datasets import Dataset` → `from odyssey.data._datasets_core import Dataset`

**Verified in:** `.templates/layer_template.mojo`, `.templates/model_template.mojo`,
`.templates/dataset_template.mojo`, `.templates/tests_template.mojo`,
`.templates/training_template.mojo`.

---

## Recipe 12: LIST_ITER_DEREF — `for x in list: x[]` dereference removed

**Symptom (verbatim):**

```text
error: '_ListIter[Int, origin_of(indices)].Element' is not subscriptable,
it does not implement the `__getitem__`/`__setitem__` methods
    var sample = self[idx[]]
                          ^
```

**Root cause:** In Mojo 0.26, iterating a `List[T]` yielded a reference
wrapper, and `x[]` was needed to dereference it. In Mojo 1.0, `for x in list`
yields the value directly — `x[]` is no longer valid.

**Fix:** Remove the `[]` dereference:

```mojo
# Before (0.26):
for idx in indices:
    process(idx[])

# After (1.0):
for idx in indices:
    process(idx)
```

**Verified in:** `.templates/dataset_template.mojo` (1 call site).

---

## Recipe 9+: TBD as Phase E agents discover them

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
