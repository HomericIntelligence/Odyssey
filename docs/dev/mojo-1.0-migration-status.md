# Mojo 1.0.0b2 Migration Status — Phase C Compile Survey

**Date**: 2026-05-08
**Mojo version surveyed**: `1.0.0b2.dev2026050805`
**Survey scope**: `shared/` and `tools/` source files (excluding `tests/`, `.pixi/`, `worktrees/`, `repro/`, `examples/`)

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

- **UNSAFEPOINTER_BOOL**: 2 files (`shared/tensor/any_tensor.mojo`,
  `shared/tensor/tensor.mojo`) — 6 occurrences of `if self._refcount:` that now trigger
  `UnsafePointer is non-null by design` warnings. Currently warnings only; not counted in
  broken total.

---

## Per-class file lists

### RELATIVE_IMPORT (34 files)

**Symptom**: `error: cannot import relative to a top-level package`

**Root cause**: In Mojo 1.0, `from .foo import bar` (dot-relative imports) are no longer allowed
when compiling a file that is part of a top-level package. The pattern `from .submodule import X`
must be changed to `from package.submodule import X`.

- `shared/core/activation.mojo` — `from .gradient_types import GradientPair`
- `shared/core/activation_simd.mojo` — `from .activation_ops import ...`
- `shared/core/arithmetic.mojo` — `from .tensor_types import ...`
- `shared/core/arithmetic_simd.mojo` — relative import in conditional block
- `shared/core/attention.mojo` — `from .attention_types import ...`
- `shared/core/conv.mojo` — `from .conv_types import ...`
- `shared/core/dropout.mojo` — `from .dropout_ops import ...`
- `shared/core/elementwise.mojo` — `from .elementwise_types import ...`
- `shared/core/layers/batchnorm.mojo` — `from .layer_types import ...`
- `shared/core/layers/conv2d.mojo` — `from .conv2d_types import ...`
- `shared/core/layers/linear.mojo` — `from .linear_types import ...`
- `shared/core/layers/relu.mojo` — `from .relu_types import ...`
- `shared/core/lazy_eval.mojo` — relative import
- `shared/core/linear.mojo` — `from .linear_ops import ...`
- `shared/core/loss.mojo` — `from .loss_types import ...`
- `shared/core/loss_utils.mojo` — `from .loss_ops import ...`
- `shared/core/matmul.mojo` — relative import
- `shared/core/matrix.mojo` — relative import
- `shared/core/normalization.mojo` — relative import
- `shared/core/normalization_simd.mojo` — relative import
- `shared/core/pooling.mojo` — relative import
- `shared/core/reduction.mojo` — relative import
- `shared/core/sequential.mojo` — relative import
- `shared/core/strassen.mojo` — relative import
- `shared/core/types/__init__.mojo` — relative import
- `shared/core/types/mxfp4.mojo` — relative import
- `shared/core/types/nvfp4.mojo` — relative import
- `shared/tensor/any_tensor.mojo` — `from .tensor_types import ...`
- `shared/tensor/factories.mojo` — `from .tensor_types import ...`
- `shared/tensor/tensor.mojo` — relative import
- `shared/tensor/tensor_creation.mojo` — relative import
- `shared/tensor/tensor_io.mojo` — relative import
- `shared/tensor/tensor_utils.mojo` — relative import
- `shared/utils/__init__.mojo` — relative import

### THIN (22 files)

**Symptom**: `invalid call to 'X': parameter 'op' has 'def[T: DType](...) -> ...' type,`
`but value has type 'def my_op[T: DType](...) -> ...'`

**Root cause**: In Mojo 1.0, parametric `def` function types used as value parameters require the
`thin` keyword before `->` in the parameter type annotation. Fix recipe is in
`docs/dev/mojo-1.0-migration-recipe.md` (already verified on `shared/core/dtype_dispatch.mojo`).

**Primary broken files** (files with their own THIN errors, not just cascade):

- `shared/core/arithmetic.mojo` — `_dispatch_broadcast_binary` parameter `op` type mismatch
- `shared/core/elementwise.mojo` — `_dispatch_float_unary_typed` parameter `op` type mismatch
- `shared/tensor/any_tensor.mojo` — `_anytensor_binary_op` parameter `op` type mismatch
- `shared/tensor/typed/arithmetic.mojo` — `__call__` parameter `_Self` type mismatch
- `shared/tensor/typed/elementwise.mojo` — `__call__` parameter `_Self` type mismatch

**Cascade-broken files** (fail because they import broken THIN files):

- `shared/__init__.mojo` — imports arithmetic+elementwise
- `shared/autograd/__init__.mojo` — imports arithmetic
- `shared/autograd/backward_ops.mojo` — imports arithmetic
- `shared/autograd/functional.mojo` — imports arithmetic+elementwise
- `shared/autograd/optimizers.mojo` — imports elementwise
- `shared/autograd/tape.mojo` — imports arithmetic
- `shared/autograd/variable.mojo` — imports arithmetic
- `shared/core/__init__.mojo` — imports arithmetic+elementwise
- `shared/core/layers/__init__.mojo` — imports arithmetic via any_tensor
- `shared/testing/layer_testers.mojo` — imports any_tensor (THIN) + gradient_checker (TRAIT_CALL)
- `shared/training/optimizers/__init__.mojo` — imports elementwise
- `shared/training/optimizers/adam.mojo` — imports arithmetic
- `shared/training/optimizers/adamw.mojo` — imports arithmetic
- `shared/training/optimizers/lars.mojo` — imports arithmetic
- `shared/training/optimizers/optimizer_utils.mojo` — imports elementwise
- `shared/training/optimizers/rmsprop.mojo` — imports arithmetic
- `shared/training/optimizers/sgd.mojo` — imports arithmetic
- `shared/training/trainer.mojo` — imports any_tensor (THIN) + training_loop (TRAIT_CALL)

### UNIFIED (16 files)

**Symptom**: `error: unknown function effect 'unified', expected 'raises', 'capturing', 'thin', or 'register_passable'`

**Root cause**: The `unified` function effect was removed in Mojo 1.0. Drop the `unified` keyword;
keep the `{...}` capture list if the function captures variables.

- `shared/core/activation_simd.mojo` — multiple `unified` uses on SIMD kernel functions
- `shared/core/arithmetic_contiguous.mojo` — `unified` on arithmetic kernel
- `shared/core/matmul.mojo` — `unified` on matmul kernel
- `shared/core/normalization_simd.mojo` — `unified` on normalization kernel
- `shared/core/numerical_safety.mojo` — (via typed/numerical_safety)
- `shared/tensor/typed/activation_simd.mojo` — multiple `unified` uses
- `shared/tensor/typed/arithmetic_contiguous.mojo` — multiple `unified` uses
- `shared/tensor/typed/arithmetic_simd.mojo` — multiple `unified` uses
- `shared/tensor/typed/numerical_safety.mojo` — multiple `unified` uses
- `shared/training/__init__.mojo` — imports broken training submodule
- `shared/training/gradient_clipping.mojo` — `unified` on clipping function
- `shared/training/mixed_precision.mojo` — `unified` on precision-cast functions
- `shared/training/optimizers/lars.mojo` — cascade from arithmetic THIN
- `shared/training/optimizers/optimizer_utils.mojo` — cascade from arithmetic THIN
- `shared/training/optimizers/sgd.mojo` — cascade from arithmetic THIN
- `shared/training/precision_config.mojo` — `unified` on config function

### TRAIT_CALL (12 files)

**Symptom**: `error: invalid call to '__call__': value passed to '' cannot be converted`
`from type value '_Self' to an instance of '_Self'; did you mean to instantiate '_Self'?`

**Root cause**: In Mojo 1.0, calling a trait-typed callable value now requires explicit
instantiation syntax changes. Functions stored as trait parameters need updated call syntax.
This affects benchmarking harnesses, testing infrastructure, and training loops that store
user-provided callbacks as trait parameters.

- `shared/benchmarking/__init__.mojo` — runner uses trait callback
- `shared/benchmarking/runner.mojo` — 3 call sites: lines 223, 230, 349
- `shared/testing/__init__.mojo` — re-exports testing with trait callbacks
- `shared/testing/gradient_checker.mojo` — trait-typed forward function
- `shared/testing/layer_testers.mojo` — also has THIN cascade
- `shared/testing/property_testing.mojo` — `property_fn` trait call
- `shared/training/loops/__init__.mojo` — re-exports broken loop
- `shared/training/loops/training_loop.mojo` — epoch callback trait call
- `shared/training/loops/validation_loop.mojo` — validation callback trait call (line 272)
- `shared/training/script_runner.mojo` — runner callback trait call
- `shared/training/trainer.mojo` — also has THIN cascade
- `shared/utils/profiling.mojo` — profiler function trait call (lines 436, 470)

### STD_OS_ATOMIC (2 files)

**Symptom**: `error: unable to locate module 'atomic'`

**Root cause**: The `atomic` module path changed in Mojo 1.0. `from memory import Atomic` or the
old `std.os.atomic` import path no longer works.

- `shared/base/__init__.mojo` — cascade from memory_pool
- `shared/base/memory_pool.mojo` — `from memory import Atomic` at line 41; also has secondary
  errors: `statement indentation must match the rest of the block` (line 144) and multiple
  `no matching function in initialization` errors (lines 399, 409, 418, 438)

### DYNAMIC_TRAIT (1 file)

**Symptom**: `error: dynamic traits not supported yet, please use a compile time generic`
`instead of 'def(Float32) -> Float32'`

**Root cause**: In Mojo 1.0, storing a `def` type directly in a struct field (as a runtime
dynamic callable) is not supported. The pattern must be changed to use compile-time generics
or a different abstraction.

- `shared/data/generic_transforms.mojo` — struct field of type `def(Float32) -> Float32` (line 99)
  and `def(AnyTensor) raises -> Bool` (line 158)

### OTHER (8 files)

Files with errors that don't fit existing categories:

- `shared/base/__init__.mojo` — cascade from memory_pool
  (STD_OS_ATOMIC + `no matching function in initialization` on Atomic initialization)
- `shared/base/memory_pool.mojo` — `unable to locate module 'atomic'` (STD_OS_ATOMIC);
  secondary: `statement indentation must match the rest of the block` (line 144);
  `no matching function in initialization` (lines 399, 409, 418, 438, 460) — Atomic API changed
- `shared/core/conv.mojo` — `use of unknown declaration 'out_h'` (line 428),
  `'out_w'` (line 429) — variable scoping change after relative import failure
- `shared/core/layers/batchnorm.mojo` — `use of unknown declaration 'new_running_mean'`
  (line 149), `'new_running_var'` (150), `'output'` (152) — variable scoping changed
- `shared/core/normalization_simd.mojo` — `use of unknown declaration 'scalar_result'`
  (line 119) — variable scoping changed in 1.0
- `shared/core/pooling.mojo` — `use of unknown declaration 'out_h'` (lines 78, 354),
  `'out_w'` (line 79) — variable scoping change
- `shared/data/formats/cifar_loader.mojo` — `use of unknown declaration 'CIFAR10_CHANNELS'`
  (line 83), `'CIFAR10_BYTES_PER_IMAGE'` (86), `'CIFAR100_BYTES_PER_IMAGE'` (88)
  — constants no longer visible, likely relative import failure or scoping change
- `shared/testing/fuzz_core.mojo` — `use of unknown declaration 'dtype_to_string'`
  (line 784) — utility function moved or renamed

---

## OTHER bucket details

### shared/base/memory_pool.mojo

```text
shared/base/memory_pool.mojo:41:9: error: unable to locate module 'atomic'
shared/base/memory_pool.mojo:144:17: error: statement indentation must match the rest of the block; adjust to align
shared/base/memory_pool.mojo:399:70: error: no matching function in initialization
```

The `atomic` module has moved. The subsequent `no matching function in initialization` errors (at
lines 399, 409, 418, 438, 460) are all on `Atomic` construction — the Atomic API changed or
the type is now initialized differently.

### shared/core/conv.mojo (secondary from RELATIVE_IMPORT)

```text
shared/core/conv.mojo:16:7: error: cannot import relative to a top-level package
shared/core/conv.mojo:428:22: error: use of unknown declaration 'out_h'
shared/core/conv.mojo:429:21: error: use of unknown declaration 'out_w'
```

Secondary `use of unknown declaration` errors likely caused by a scoping change in 1.0 where
variables declared in `if`/`for` blocks no longer escape their scope.

### shared/core/layers/batchnorm.mojo (secondary from RELATIVE_IMPORT)

```text
shared/core/layers/batchnorm.mojo:21:7: error: cannot import relative to a top-level package
shared/core/layers/batchnorm.mojo:149:33: error: use of unknown declaration 'new_running_mean'
shared/core/layers/batchnorm.mojo:150:32: error: use of unknown declaration 'new_running_var'
```

Same scoping pattern as conv.mojo — variables assigned inside a conditional block referenced
outside it.

### shared/core/normalization_simd.mojo (secondary from RELATIVE_IMPORT + UNIFIED)

```text
shared/core/normalization_simd.mojo:71:7: error: cannot import relative to a top-level package
shared/core/normalization_simd.mojo:119:16: error: use of unknown declaration 'scalar_result'
```

### shared/core/pooling.mojo (secondary from RELATIVE_IMPORT)

```text
shared/core/pooling.mojo:11:7: error: cannot import relative to a top-level package
shared/core/pooling.mojo:78:22: error: use of unknown declaration 'out_h'
shared/core/pooling.mojo:79:21: error: use of unknown declaration 'out_w'
```

### shared/data/formats/cifar_loader.mojo

```text
shared/data/formats/cifar_loader.mojo:83:25: error: use of unknown declaration 'CIFAR10_CHANNELS'
shared/data/formats/cifar_loader.mojo:86:36: error: use of unknown declaration 'CIFAR10_BYTES_PER_IMAGE'
shared/data/formats/cifar_loader.mojo:88:36: error: use of unknown declaration 'CIFAR100_BYTES_PER_IMAGE'
```

Constants defined in `shared/data/constants.mojo` are no longer visible. Either the import
was done via relative import (which now fails) or the constant names changed.

### shared/testing/fuzz_core.mojo

```text
shared/testing/fuzz_core.mojo:784:11: error: use of unknown declaration 'dtype_to_string'
```

The function `dtype_to_string` was presumably renamed or moved in the stdlib or in the
codebase during refactoring.

---

## Compile-clean files (sample)

These 10 files compiled with no hard errors (only warnings allowed):

1. `shared/autograd/grad_utils.mojo`
2. `shared/autograd/optimizer_base.mojo`
3. `shared/autograd/schedulers.mojo`
4. `shared/autograd/tape_types.mojo`
5. `shared/base/broadcasting.mojo`
6. `shared/base/defaults.mojo`
7. `shared/base/dtype_ordinal.mojo`
8. `shared/base/math_constants.mojo`
9. `shared/core/activation_constants.mojo`
10. `shared/core/activation_ops.mojo`

(110 total clean files — these are primarily leaf modules with no internal relative imports and
no THIN/UNIFIED/TRAIT_CALL patterns.)

---

## Recommended migration wave structure

Wave ordering is driven by dependency: fix the leaves first, then the dependents compile.

### D1 — Haiku-mechanical: RELATIVE_IMPORT (34 files, ~1–2 lines each)

**Assignment**: Haiku (pure find-replace, no logic)

**What**: Replace `from .submodule import X` with `from shared.submodule import X` (or the
correct absolute package path). This is purely mechanical — grep and replace.

**Groups by module** (fix together so tests can verify incrementally):

- Group 1: `shared/tensor/` — any_tensor, tensor, factories, tensor_creation, tensor_io,
  tensor_utils (6 files)
- Group 2: `shared/core/types/` — `__init__`, mxfp4, nvfp4 (3 files)
- Group 3: `shared/core/` top-level — activation, arithmetic, arithmetic_simd, attention,
  conv, dropout, elementwise, lazy_eval, linear, loss, loss_utils, matmul, matrix,
  normalization, pooling, reduction, sequential, strassen (18 files)
- Group 4: `shared/core/activation_simd`, `normalization_simd` (2 files; also have UNIFIED)
- Group 5: `shared/core/layers/` — batchnorm, conv2d, linear, relu (4 files; batchnorm also has OTHER)
- Group 6: `shared/utils/__init__` (1 file)

**Estimated effort**: 1 agent pass, ~30 min

### D2 — Haiku-mechanical: UNIFIED (16 files, ~1 keyword removal each)

**Assignment**: Haiku (pure keyword deletion)

**What**: Remove `unified` from function effect position. Keep `{...}` capture lists.

**Pure UNIFIED files** (no other issues after D1 fixes RELATIVE_IMPORT):

- `shared/core/arithmetic_contiguous.mojo`
- `shared/core/numerical_safety.mojo` (via tensor/typed/numerical_safety)
- `shared/tensor/typed/activation_simd.mojo`
- `shared/tensor/typed/arithmetic_contiguous.mojo`
- `shared/tensor/typed/arithmetic_simd.mojo`
- `shared/tensor/typed/numerical_safety.mojo`
- `shared/training/gradient_clipping.mojo`
- `shared/training/mixed_precision.mojo`
- `shared/training/precision_config.mojo`

**Mixed RELATIVE+UNIFIED** (fix RELATIVE_IMPORT first in D1, then these in D2):

- `shared/core/activation_simd.mojo`
- `shared/core/matmul.mojo`
- `shared/core/normalization_simd.mojo`
- `shared/training/__init__.mojo` (cascade, may auto-fix after dependencies fixed)
- `shared/training/optimizers/lars.mojo` (cascade THIN+UNIFIED — after D3)
- `shared/training/optimizers/optimizer_utils.mojo` (cascade THIN+UNIFIED — after D3)
- `shared/training/optimizers/sgd.mojo` (cascade THIN+UNIFIED — after D3)

**Estimated effort**: 1 agent pass, ~20 min

### D3 — Sonnet: THIN (5 source files, cascades 17 dependents)

**Assignment**: Sonnet (requires understanding parametric def type semantics)

**What**: Add `thin` keyword to parametric def parameter type annotations. Fix recipe is already
documented and verified in `docs/dev/mojo-1.0-migration-recipe.md`.

**Primary files to fix** (fixing these will unblock all 17 cascade-broken files):

- `shared/core/arithmetic.mojo` — `_dispatch_broadcast_binary` op parameter
- `shared/core/elementwise.mojo` — `_dispatch_float_unary_typed` op parameter
- `shared/tensor/any_tensor.mojo` — `_anytensor_binary_op` op parameter (also needs D1 relative import fix)
- `shared/tensor/typed/arithmetic.mojo` — `__call__` _Self parameter
- `shared/tensor/typed/elementwise.mojo` — `__call__` _Self parameter

**After these 5 are fixed, cascade fixes unblock**:
All 17 autograd, core, and optimizer files listed under THIN.

**Estimated effort**: 1–2 Sonnet passes, ~45 min

### D4 — Sonnet: TRAIT_CALL (12 files)

**Assignment**: Sonnet (requires understanding trait callable call-site syntax)

**What**: Update call sites where a trait-typed callable (`_Self`) is invoked. The new syntax
may require `.value()` unwrap or explicit type annotation at the call site.

**Files**:

- `shared/benchmarking/runner.mojo` — 3 call sites
- `shared/testing/gradient_checker.mojo` — 3+ call sites
- `shared/testing/property_testing.mojo` — 2 call sites
- `shared/training/loops/training_loop.mojo` — epoch callback
- `shared/training/loops/validation_loop.mojo` — validation callback (line 272)
- `shared/training/script_runner.mojo` — runner callback
- `shared/utils/profiling.mojo` — profiler callback (lines 436, 470)

After fixing: benchmarking/\_\_init\_\_, testing/\_\_init\_\_, training/loops/\_\_init\_\_,
training/trainer.mojo will auto-unblock.

**Estimated effort**: 1–2 Sonnet passes, ~60 min

### D5 — Opus: OTHER + STD_OS_ATOMIC + DYNAMIC_TRAIT (scattered)

**Assignment**: Opus (architectural investigation needed)

**Files and issues**:

- `shared/base/memory_pool.mojo` (STD_OS_ATOMIC) — Atomic module path changed; investigate
  `from memory import Atomic` vs new location; fix `no matching function in initialization`
  on Atomic construction (likely constructor API changed).
- `shared/data/generic_transforms.mojo` (DYNAMIC_TRAIT) — Storing `def(Float32) -> Float32`
  in struct field; must redesign to use compile-time generic parameter (parametric struct)
  or a trait-based approach.
- `shared/data/formats/cifar_loader.mojo` (OTHER) — Constants `CIFAR10_CHANNELS`,
  `CIFAR10_BYTES_PER_IMAGE`, `CIFAR100_BYTES_PER_IMAGE` not visible; check if import
  path changed or if they need to be imported explicitly.
- `shared/testing/fuzz_core.mojo` (OTHER) — `dtype_to_string` not found at line 784;
  trace what module it came from and fix import.
- Secondary scoping errors in `shared/core/conv.mojo`, `shared/core/pooling.mojo`,
  `shared/core/layers/batchnorm.mojo`, `shared/core/normalization_simd.mojo` —
  variables declared inside `if`/`for` blocks that are referenced outside them; in Mojo 1.0
  block-scoped variables may no longer escape. These need careful reading to restructure
  the scope (hoist variable declarations before the block).

**Warning-only cleanup** (UNSAFEPOINTER_BOOL — not blocking, do last):

- `shared/tensor/any_tensor.mojo` — 3 occurrences of `if self._refcount:`
- `shared/tensor/tensor.mojo` — 3 occurrences of `if self._refcount:`
- Fix: change to `if self._refcount[] > 0:` or `if self._refcount != UnsafePointer[Int]():`

**Estimated effort**: 1 Opus pass for investigation, then 1 Sonnet pass for implementation,
~2–3 hours total
