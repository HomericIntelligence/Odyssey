# Mojo 1.0.0b2 Migration Status — Phase C Compile Survey

**Date**: 2026-05-08
**Mojo version surveyed**: `1.0.0b2.dev2026050805`
**Survey scope**: `src/odyssey/` and `tools/` source files
(excluding `tests/`, `.pixi/`, `worktrees/`, `repro/`, `examples/`)

## Summary

- **Total source files surveyed**: 189
- **Compile-clean files**: 110 (library files with only deprecation warnings or no issues)
- **Compile-broken files**: 79 (at least one hard `error:` line above "no main" message)
- **Timeouts**: 0

Per-class file counts (sorted descending):

| Error Class | Files |
| --- | --- |
| RELATIVE_IMPORT | 34 |
| THIN | 22 |
| UNIFIED | 16 |
| TRAIT_CALL | 12 |
| OTHER | 8 |
| STD_OS_ATOMIC | 2 |
| DYNAMIC_TRAIT | 1 |

Note: Some files appear in multiple buckets (e.g. a file can have both RELATIVE_IMPORT and UNIFIED
errors). Total unique broken files = 79.

### Warning-only (not counted as broken)

- **UNSAFEPOINTER_BOOL**: 2 files (`src/odyssey/tensor/any_tensor.mojo`,
  `src/odyssey/tensor/tensor.mojo`) — 6 occurrences of `if self._refcount:` that now trigger
  `UnsafePointer is non-null by design` warnings. Currently warnings only; not counted in
  broken total.

---

## Per-class file lists

### RELATIVE_IMPORT (34 files)

**Symptom**: `error: cannot import relative to a top-level package`

**Root cause**: In Mojo 1.0, `from .foo import bar` (dot-relative imports) are no longer allowed
when compiling a file that is part of a top-level package. The pattern `from .submodule import X`
must be changed to `from package.submodule import X`.

- `src/odyssey/core/activation.mojo` — `from .gradient_types import GradientPair`
- `src/odyssey/core/activation_simd.mojo` — `from .activation_ops import ...`
- `src/odyssey/core/arithmetic.mojo` — `from .tensor_types import ...`
- `src/odyssey/core/arithmetic_simd.mojo` — relative import in conditional block
- `src/odyssey/core/attention.mojo` — `from .attention_types import ...`
- `src/odyssey/core/conv.mojo` — `from .conv_types import ...`
- `src/odyssey/core/dropout.mojo` — `from .dropout_ops import ...`
- `src/odyssey/core/elementwise.mojo` — `from .elementwise_types import ...`
- `src/odyssey/core/layers/batchnorm.mojo` — `from .layer_types import ...`
- `src/odyssey/core/layers/conv2d.mojo` — `from .conv2d_types import ...`
- `src/odyssey/core/layers/linear.mojo` — `from .linear_types import ...`
- `src/odyssey/core/layers/relu.mojo` — `from .relu_types import ...`
- `src/odyssey/core/lazy_eval.mojo` — relative import
- `src/odyssey/core/linear.mojo` — `from .linear_ops import ...`
- `src/odyssey/core/loss.mojo` — `from .loss_types import ...`
- `src/odyssey/core/loss_utils.mojo` — `from .loss_ops import ...`
- `src/odyssey/core/matmul.mojo` — relative import
- `src/odyssey/core/matrix.mojo` — relative import
- `src/odyssey/core/normalization.mojo` — relative import
- `src/odyssey/core/normalization_simd.mojo` — relative import
- `src/odyssey/core/pooling.mojo` — relative import
- `src/odyssey/core/reduction.mojo` — relative import
- `src/odyssey/core/sequential.mojo` — relative import
- `src/odyssey/core/strassen.mojo` — relative import
- `src/odyssey/core/types/__init__.mojo` — relative import
- `src/odyssey/core/types/mxfp4.mojo` — relative import
- `src/odyssey/core/types/nvfp4.mojo` — relative import
- `src/odyssey/tensor/any_tensor.mojo` — `from .tensor_types import ...`
- `src/odyssey/tensor/factories.mojo` — `from .tensor_types import ...`
- `src/odyssey/tensor/tensor.mojo` — relative import
- `src/odyssey/tensor/tensor_creation.mojo` — relative import
- `src/odyssey/tensor/tensor_io.mojo` — relative import
- `src/odyssey/tensor/tensor_utils.mojo` — relative import
- `src/odyssey/utils/__init__.mojo` — relative import

### THIN (22 files)

**Symptom**: `invalid call to 'X': parameter 'op' has 'def[T: DType](...) -> ...' type,`
`but value has type 'def my_op[T: DType](...) -> ...'`

**Root cause**: In Mojo 1.0, parametric `def` function types used as value parameters require the
`thin` keyword before `->` in the parameter type annotation. Fix recipe is in
`docs/dev/mojo-1.0-migration-recipe.md` (already verified on `src/odyssey/core/dtype_dispatch.mojo`).

**Primary broken files** (files with their own THIN errors, not just cascade):

- `src/odyssey/core/arithmetic.mojo` — `_dispatch_broadcast_binary` parameter `op` type mismatch
- `src/odyssey/core/elementwise.mojo` — `_dispatch_float_unary_typed` parameter `op` type mismatch
- `src/odyssey/tensor/any_tensor.mojo` — `_anytensor_binary_op` parameter `op` type mismatch
- `src/odyssey/tensor/typed/arithmetic.mojo` — `__call__` parameter `_Self` type mismatch
- `src/odyssey/tensor/typed/elementwise.mojo` — `__call__` parameter `_Self` type mismatch

**Cascade-broken files** (fail because they import broken THIN files):

- `src/odyssey/__init__.mojo` — imports arithmetic+elementwise
- `src/odyssey/autograd/__init__.mojo` — imports arithmetic
- `src/odyssey/autograd/backward_ops.mojo` — imports arithmetic
- `src/odyssey/autograd/functional.mojo` — imports arithmetic+elementwise
- `src/odyssey/autograd/optimizers.mojo` — imports elementwise
- `src/odyssey/autograd/tape.mojo` — imports arithmetic
- `src/odyssey/autograd/variable.mojo` — imports arithmetic
- `src/odyssey/core/__init__.mojo` — imports arithmetic+elementwise
- `src/odyssey/core/layers/__init__.mojo` — imports arithmetic via any_tensor
- `src/odyssey/testing/layer_testers.mojo` — imports any_tensor (THIN) + gradient_checker (TRAIT_CALL)
- `src/odyssey/training/optimizers/__init__.mojo` — imports elementwise
- `src/odyssey/training/optimizers/adam.mojo` — imports arithmetic
- `src/odyssey/training/optimizers/adamw.mojo` — imports arithmetic
- `src/odyssey/training/optimizers/lars.mojo` — imports arithmetic
- `src/odyssey/training/optimizers/optimizer_utils.mojo` — imports elementwise
- `src/odyssey/training/optimizers/rmsprop.mojo` — imports arithmetic
- `src/odyssey/training/optimizers/sgd.mojo` — imports arithmetic
- `src/odyssey/training/trainer.mojo` — imports any_tensor (THIN) + training_loop (TRAIT_CALL)

### UNIFIED (16 files)

**Symptom**: `error: unknown function effect 'unified', expected 'raises', 'capturing', 'thin', or 'register_passable'`

**Root cause**: The `unified` function effect was removed in Mojo 1.0. Drop the `unified` keyword;
keep the `{...}` capture list if the function captures variables.

- `src/odyssey/core/activation_simd.mojo` — multiple `unified` uses on SIMD kernel functions
- `src/odyssey/core/arithmetic_contiguous.mojo` — `unified` on arithmetic kernel
- `src/odyssey/core/matmul.mojo` — `unified` on matmul kernel
- `src/odyssey/core/normalization_simd.mojo` — `unified` on normalization kernel
- `src/odyssey/core/numerical_safety.mojo` — (via typed/numerical_safety)
- `src/odyssey/tensor/typed/activation_simd.mojo` — multiple `unified` uses
- `src/odyssey/tensor/typed/arithmetic_contiguous.mojo` — multiple `unified` uses
- `src/odyssey/tensor/typed/arithmetic_simd.mojo` — multiple `unified` uses
- `src/odyssey/tensor/typed/numerical_safety.mojo` — multiple `unified` uses
- `src/odyssey/training/__init__.mojo` — imports broken training submodule
- `src/odyssey/training/gradient_clipping.mojo` — `unified` on clipping function
- `src/odyssey/training/mixed_precision.mojo` — `unified` on precision-cast functions
- `src/odyssey/training/optimizers/lars.mojo` — cascade from arithmetic THIN
- `src/odyssey/training/optimizers/optimizer_utils.mojo` — cascade from arithmetic THIN
- `src/odyssey/training/optimizers/sgd.mojo` — cascade from arithmetic THIN
- `src/odyssey/training/precision_config.mojo` — `unified` on config function

### TRAIT_CALL (12 files)

**Symptom**: `error: invalid call to '__call__': value passed to '' cannot be converted`
`from type value '_Self' to an instance of '_Self'; did you mean to instantiate '_Self'?`

**Root cause**: In Mojo 1.0, calling a trait-typed callable value now requires explicit
instantiation syntax changes. Functions stored as trait parameters need updated call syntax.
This affects benchmarking harnesses, testing infrastructure, and training loops that store
user-provided callbacks as trait parameters.

- `src/odyssey/benchmarking/__init__.mojo` — runner uses trait callback
- `src/odyssey/benchmarking/runner.mojo` — 3 call sites: lines 223, 230, 349
- `src/odyssey/testing/__init__.mojo` — re-exports testing with trait callbacks
- `src/odyssey/testing/gradient_checker.mojo` — trait-typed forward function
- `src/odyssey/testing/layer_testers.mojo` — also has THIN cascade
- `src/odyssey/testing/property_testing.mojo` — `property_fn` trait call
- `src/odyssey/training/loops/__init__.mojo` — re-exports broken loop
- `src/odyssey/training/loops/training_loop.mojo` — epoch callback trait call
- `src/odyssey/training/loops/validation_loop.mojo` — validation callback trait call (line 272)
- `src/odyssey/training/script_runner.mojo` — runner callback trait call
- `src/odyssey/training/trainer.mojo` — also has THIN cascade
- `src/odyssey/utils/profiling.mojo` — profiler function trait call (lines 436, 470)

### STD_OS_ATOMIC (2 files)

**Symptom**: `error: unable to locate module 'atomic'`

**Root cause**: The `atomic` module path changed in Mojo 1.0. `from memory import Atomic` or the
old `std.os.atomic` import path no longer works.

- `src/odyssey/base/__init__.mojo` — cascade from memory_pool
- `src/odyssey/base/memory_pool.mojo` — `from memory import Atomic` at line 41; also has secondary
  errors: `statement indentation must match the rest of the block` (line 144) and multiple
  `no matching function in initialization` errors (lines 399, 409, 418, 438)

### DYNAMIC_TRAIT (1 file)

**Symptom**: `error: dynamic traits not supported yet, please use a compile time generic`
`instead of 'def(Float32) -> Float32'`

**Root cause**: In Mojo 1.0, storing a `def` type directly in a struct field (as a runtime
dynamic callable) is not supported. The pattern must be changed to use compile-time generics
or a different abstraction.

- `src/odyssey/data/generic_transforms.mojo` — struct field of type `def(Float32) -> Float32` (line 99)
  and `def(AnyTensor) raises -> Bool` (line 158)

### OTHER (8 files)

Files with errors that don't fit existing categories:

- `src/odyssey/base/__init__.mojo` — cascade from memory_pool
  (STD_OS_ATOMIC + `no matching function in initialization` on Atomic initialization)
- `src/odyssey/base/memory_pool.mojo` — `unable to locate module 'atomic'` (STD_OS_ATOMIC);
  secondary: `statement indentation must match the rest of the block` (line 144);
  `no matching function in initialization` (lines 399, 409, 418, 438, 460) — Atomic API changed
- `src/odyssey/core/conv.mojo` — `use of unknown declaration 'out_h'` (line 428),
  `'out_w'` (line 429) — variable scoping change after relative import failure
- `src/odyssey/core/layers/batchnorm.mojo` — `use of unknown declaration 'new_running_mean'`
  (line 149), `'new_running_var'` (150), `'output'` (152) — variable scoping changed
- `src/odyssey/core/normalization_simd.mojo` — `use of unknown declaration 'scalar_result'`
  (line 119) — variable scoping changed in 1.0
- `src/odyssey/core/pooling.mojo` — `use of unknown declaration 'out_h'` (lines 78, 354),
  `'out_w'` (line 79) — variable scoping change
- `src/odyssey/data/formats/cifar_loader.mojo` — `use of unknown declaration 'CIFAR10_CHANNELS'`
  (line 83), `'CIFAR10_BYTES_PER_IMAGE'` (86), `'CIFAR100_BYTES_PER_IMAGE'` (88)
  — constants no longer visible, likely relative import failure or scoping change
- `src/odyssey/testing/fuzz_core.mojo` — `use of unknown declaration 'dtype_to_string'`
  (line 784) — utility function moved or renamed

---

## OTHER bucket details

### src/odyssey/base/memory_pool.mojo

```text
src/odyssey/base/memory_pool.mojo:41:9: error: unable to locate module 'atomic'
src/odyssey/base/memory_pool.mojo:144:17: error: statement indentation must match the rest of the block; adjust to align
src/odyssey/base/memory_pool.mojo:399:70: error: no matching function in initialization
```

The `atomic` module has moved. The subsequent `no matching function in initialization` errors (at
lines 399, 409, 418, 438, 460) are all on `Atomic` construction — the Atomic API changed or
the type is now initialized differently.

### src/odyssey/core/conv.mojo (secondary from RELATIVE_IMPORT)

```text
src/odyssey/core/conv.mojo:16:7: error: cannot import relative to a top-level package
src/odyssey/core/conv.mojo:428:22: error: use of unknown declaration 'out_h'
src/odyssey/core/conv.mojo:429:21: error: use of unknown declaration 'out_w'
```

Secondary `use of unknown declaration` errors likely caused by a scoping change in 1.0 where
variables declared in `if`/`for` blocks no longer escape their scope.

### src/odyssey/core/layers/batchnorm.mojo (secondary from RELATIVE_IMPORT)

```text
src/odyssey/core/layers/batchnorm.mojo:21:7: error: cannot import relative to a top-level package
src/odyssey/core/layers/batchnorm.mojo:149:33: error: use of unknown declaration 'new_running_mean'
src/odyssey/core/layers/batchnorm.mojo:150:32: error: use of unknown declaration 'new_running_var'
```

Same scoping pattern as conv.mojo — variables assigned inside a conditional block referenced
outside it.

### src/odyssey/core/normalization_simd.mojo (secondary from RELATIVE_IMPORT + UNIFIED)

```text
src/odyssey/core/normalization_simd.mojo:71:7: error: cannot import relative to a top-level package
src/odyssey/core/normalization_simd.mojo:119:16: error: use of unknown declaration 'scalar_result'
```

### src/odyssey/core/pooling.mojo (secondary from RELATIVE_IMPORT)

```text
src/odyssey/core/pooling.mojo:11:7: error: cannot import relative to a top-level package
src/odyssey/core/pooling.mojo:78:22: error: use of unknown declaration 'out_h'
src/odyssey/core/pooling.mojo:79:21: error: use of unknown declaration 'out_w'
```

### src/odyssey/data/formats/cifar_loader.mojo

```text
src/odyssey/data/formats/cifar_loader.mojo:83:25: error: use of unknown declaration 'CIFAR10_CHANNELS'
src/odyssey/data/formats/cifar_loader.mojo:86:36: error: use of unknown declaration 'CIFAR10_BYTES_PER_IMAGE'
src/odyssey/data/formats/cifar_loader.mojo:88:36: error: use of unknown declaration 'CIFAR100_BYTES_PER_IMAGE'
```

Constants defined in `src/odyssey/data/constants.mojo` are no longer visible. Either the import
was done via relative import (which now fails) or the constant names changed.

### src/odyssey/testing/fuzz_core.mojo

```text
src/odyssey/testing/fuzz_core.mojo:784:11: error: use of unknown declaration 'dtype_to_string'
```

The function `dtype_to_string` was presumably renamed or moved in the stdlib or in the
codebase during refactoring.

---

## Compile-clean files (sample)

These 10 files compiled with no hard errors (only warnings allowed):

1. `src/odyssey/autograd/grad_utils.mojo`
2. `src/odyssey/autograd/optimizer_base.mojo`
3. `src/odyssey/autograd/schedulers.mojo`
4. `src/odyssey/autograd/tape_types.mojo`
5. `src/odyssey/base/broadcasting.mojo`
6. `src/odyssey/base/defaults.mojo`
7. `src/odyssey/base/dtype_ordinal.mojo`
8. `src/odyssey/base/math_constants.mojo`
9. `src/odyssey/core/activation_constants.mojo`
10. `src/odyssey/core/activation_ops.mojo`

(110 total clean files — these are primarily leaf modules with no internal relative imports and
no THIN/UNIFIED/TRAIT_CALL patterns.)

---

## Recommended migration wave structure

Wave ordering is driven by dependency: fix the leaves first, then the dependents compile.

### D1 — Haiku-mechanical: RELATIVE_IMPORT (34 files, ~1–2 lines each)

**Assignment**: Haiku (pure find-replace, no logic)

**What**: Replace `from .submodule import X` with `from odyssey.submodule import X` (or the
correct absolute package path). This is purely mechanical — grep and replace.

**Groups by module** (fix together so tests can verify incrementally):

- Group 1: `src/odyssey/tensor/` — any_tensor, tensor, factories, tensor_creation, tensor_io,
  tensor_utils (6 files)
- Group 2: `src/odyssey/core/types/` — `__init__`, mxfp4, nvfp4 (3 files)
- Group 3: `src/odyssey/core/` top-level — activation, arithmetic, arithmetic_simd, attention,
  conv, dropout, elementwise, lazy_eval, linear, loss, loss_utils, matmul, matrix,
  normalization, pooling, reduction, sequential, strassen (18 files)
- Group 4: `src/odyssey/core/activation_simd`, `normalization_simd` (2 files; also have UNIFIED)
- Group 5: `src/odyssey/core/layers/` — batchnorm, conv2d, linear, relu (4 files; batchnorm also has OTHER)
- Group 6: `src/odyssey/utils/__init__` (1 file)

**Estimated effort**: 1 agent pass, ~30 min

### D2 — Haiku-mechanical: UNIFIED (16 files, ~1 keyword removal each)

**Assignment**: Haiku (pure keyword deletion)

**What**: Remove `unified` from function effect position. Keep `{...}` capture lists.

**Pure UNIFIED files** (no other issues after D1 fixes RELATIVE_IMPORT):

- `src/odyssey/core/arithmetic_contiguous.mojo`
- `src/odyssey/core/numerical_safety.mojo` (via tensor/typed/numerical_safety)
- `src/odyssey/tensor/typed/activation_simd.mojo`
- `src/odyssey/tensor/typed/arithmetic_contiguous.mojo`
- `src/odyssey/tensor/typed/arithmetic_simd.mojo`
- `src/odyssey/tensor/typed/numerical_safety.mojo`
- `src/odyssey/training/gradient_clipping.mojo`
- `src/odyssey/training/mixed_precision.mojo`
- `src/odyssey/training/precision_config.mojo`

**Mixed RELATIVE+UNIFIED** (fix RELATIVE_IMPORT first in D1, then these in D2):

- `src/odyssey/core/activation_simd.mojo`
- `src/odyssey/core/matmul.mojo`
- `src/odyssey/core/normalization_simd.mojo`
- `src/odyssey/training/__init__.mojo` (cascade, may auto-fix after dependencies fixed)
- `src/odyssey/training/optimizers/lars.mojo` (cascade THIN+UNIFIED — after D3)
- `src/odyssey/training/optimizers/optimizer_utils.mojo` (cascade THIN+UNIFIED — after D3)
- `src/odyssey/training/optimizers/sgd.mojo` (cascade THIN+UNIFIED — after D3)

**Estimated effort**: 1 agent pass, ~20 min

### D3 — Sonnet: THIN (5 source files, cascades 17 dependents)

**Assignment**: Sonnet (requires understanding parametric def type semantics)

**What**: Add `thin` keyword to parametric def parameter type annotations. Fix recipe is already
documented and verified in `docs/dev/mojo-1.0-migration-recipe.md`.

**Primary files to fix** (fixing these will unblock all 17 cascade-broken files):

- `src/odyssey/core/arithmetic.mojo` — `_dispatch_broadcast_binary` op parameter
- `src/odyssey/core/elementwise.mojo` — `_dispatch_float_unary_typed` op parameter
- `src/odyssey/tensor/any_tensor.mojo` — `_anytensor_binary_op` op parameter (also needs D1 relative import fix)
- `src/odyssey/tensor/typed/arithmetic.mojo` — `__call__` _Self parameter
- `src/odyssey/tensor/typed/elementwise.mojo` — `__call__` _Self parameter

**After these 5 are fixed, cascade fixes unblock**:
All 17 autograd, core, and optimizer files listed under THIN.

**Estimated effort**: 1–2 Sonnet passes, ~45 min

### D4 — Sonnet: TRAIT_CALL (12 files)

**Assignment**: Sonnet (requires understanding trait callable call-site syntax)

**What**: Update call sites where a trait-typed callable (`_Self`) is invoked. The new syntax
may require `.value()` unwrap or explicit type annotation at the call site.

**Files**:

- `src/odyssey/benchmarking/runner.mojo` — 3 call sites
- `src/odyssey/testing/gradient_checker.mojo` — 3+ call sites
- `src/odyssey/testing/property_testing.mojo` — 2 call sites
- `src/odyssey/training/loops/training_loop.mojo` — epoch callback
- `src/odyssey/training/loops/validation_loop.mojo` — validation callback (line 272)
- `src/odyssey/training/script_runner.mojo` — runner callback
- `src/odyssey/utils/profiling.mojo` — profiler callback (lines 436, 470)

After fixing: benchmarking/\_\_init\_\_, testing/\_\_init\_\_, training/loops/\_\_init\_\_,
training/trainer.mojo will auto-unblock.

**Estimated effort**: 1–2 Sonnet passes, ~60 min

### D5 — Opus: OTHER + STD_OS_ATOMIC + DYNAMIC_TRAIT (scattered)

**Assignment**: Opus (architectural investigation needed)

**Files and issues**:

- `src/odyssey/base/memory_pool.mojo` (STD_OS_ATOMIC) — Atomic module path changed; investigate
  `from memory import Atomic` vs new location; fix `no matching function in initialization`
  on Atomic construction (likely constructor API changed).
- `src/odyssey/data/generic_transforms.mojo` (DYNAMIC_TRAIT) — Storing `def(Float32) -> Float32`
  in struct field; must redesign to use compile-time generic parameter (parametric struct)
  or a trait-based approach.
- `src/odyssey/data/formats/cifar_loader.mojo` (OTHER) — Constants `CIFAR10_CHANNELS`,
  `CIFAR10_BYTES_PER_IMAGE`, `CIFAR100_BYTES_PER_IMAGE` not visible; check if import
  path changed or if they need to be imported explicitly.
- `src/odyssey/testing/fuzz_core.mojo` (OTHER) — `dtype_to_string` not found at line 784;
  trace what module it came from and fix import.
- Secondary scoping errors in `src/odyssey/core/conv.mojo`, `src/odyssey/core/pooling.mojo`,
  `src/odyssey/core/layers/batchnorm.mojo`, `src/odyssey/core/normalization_simd.mojo` —
  variables declared inside `if`/`for` blocks that are referenced outside them; in Mojo 1.0
  block-scoped variables may no longer escape. These need careful reading to restructure
  the scope (hoist variable declarations before the block).

**Warning-only cleanup** (UNSAFEPOINTER_BOOL — not blocking, do last):

- `src/odyssey/tensor/any_tensor.mojo` — 3 occurrences of `if self._refcount:`
- `src/odyssey/tensor/tensor.mojo` — 3 occurrences of `if self._refcount:`
- Fix: change to `if self._refcount[] > 0:` or `if self._refcount != UnsafePointer[Int]():`

**Estimated effort**: 1 Opus pass for investigation, then 1 Sonnet pass for implementation,
~2–3 hours total
