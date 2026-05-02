<!-- markdownlint-disable -->

# AnyTensor → Tensor[dtype] + AnyTensor Migration Plan

**Epic**: [#4998](https://github.com/HomericIntelligence/ProjectOdyssey/issues/4998)
**Goal**: Split AnyTensor into two types following Mojo conventions:
- **`Tensor[dtype: DType]`** — compile-time typed, SIMD-like. `tensor[i] = value` just works.
- **`AnyTensor`** — runtime-typed, type-erased. For collections, I/O, trait interfaces. Uses `.set(i, val)` for element access.

Both share a `TensorLike` trait interface. Zero-copy conversion between them via `AnyTensor.as_tensor[dtype]()` and `Tensor.as_any()`.

**Review**: Completeness review integrated below (4 blockers, 8 high, 6 medium, 4 low).

### Review Findings

#### BLOCKERS (4)

**B1: Auto-parameterization does NOT work for return types.** Verified by compilation:
```mojo
# FAILS — "failed to infer parameter 'dtype'":
fn relu(t: Tensor) -> Tensor: ...

# WORKS — call sites still infer dt from arguments:
fn relu[dt: DType](t: Tensor[dt]) -> Tensor[dt]: ...
```
All 480 functions returning AnyTensor need explicit `[dt: DType]` parameters. Line count estimates revised upward 30-50%.

**B2: `lazy_expression.mojo` / `lazy_eval.mojo` missing from all phases.** These hold `var _tensors: List[AnyTensor]` (line 111) and interact with core ops and collections. Added to Phase 5.

**B3: `comptime Tensor = AnyTensor` naming collision.** `shared/__init__.mojo:82` defines `comptime Tensor = AnyTensor`. Creating a new `struct Tensor[dtype: DType]` while this alias exists causes a naming collision. Additionally, 8 test files define local `comptime Tensor = AnyTensor` aliases. **Fix**: Phase 1b must remove this alias BEFORE the new struct is importable. Replace with proper re-export from `shared/tensor/`. Update test files with local aliases too.

**B4: `as_tensor[dtype]()`/`as_any()` refcount protocol unspecified.** The zero-copy conversion code in section 11.8 (lines 896-916) uses `...` to elide the critical refcount detail. AnyTensor's refcount protocol (`any_tensor.mojo:435-489`) increments in `__copyinit__`, decrements in `__del__`, frees at 0. If `as_tensor()` creates a `Tensor[dtype]` WITHOUT incrementing the shared refcount, then when the source AnyTensor is destroyed (Mojo's ASAP destruction), the refcount drops to 0 and frees memory — leaving the Tensor[dtype] with a dangling pointer.

**Fix**: Both `as_tensor()` and `as_any()` MUST:
1. Share the SAME `_refcount` pointer (not allocate new)
2. Increment refcount during construction
3. Both `Tensor[dtype].__del__` and `AnyTensor.__del__` decrement the shared refcount

Required internal constructor for `Tensor[dtype]`:
```mojo
# Internal constructor for zero-copy conversion (NOT public)
fn __init__(out self, data: UnsafePointer[Scalar[dtype]], shape: List[Int],
            strides: List[Int], refcount: UnsafePointer[Int], numel: Int,
            is_view: Bool, allocated_size: Int, original_numel_quantized: Int):
    self._data = data
    self._shape = shape
    self._strides = strides
    self._refcount = refcount
    self._refcount[] += 1  # CRITICAL: shared ownership
    self._numel = numel
    self._is_view = is_view
    self._allocated_size = allocated_size
    self._original_numel_quantized = original_numel_quantized
```

Required test: create AnyTensor, convert to Tensor[dtype], let AnyTensor go out of scope, verify Tensor[dtype] data is still valid. With shared refcount, ASAP destruction of AnyTensor only decrements (2→1), keeping memory alive.

#### HIGH (8)

**H1: Slice pointer arithmetic double-offset.** Current: `result._data = self._data + offset_bytes` where `offset_bytes = start * strides * dtype_size`. For typed `UnsafePointer[Scalar[dtype]]`, pointer arithmetic auto-scales by element size — the `* dtype_size` must be REMOVED. Silent wrong data at runtime if missed. (`any_tensor.mojo:721-733`)

**H2: I/O boundary underspecified.** `save_tensor()` takes `UnsafePointer[UInt8]`. Solution: `Tensor[dtype].save()` → `as_any().save()`. Loading always returns `AnyTensor` → `as_tensor[dtype]()`. (`tensor_io.mojo:34-65, 181-201`)

**H3: Phase 3→5 circular dependency.** `concatenate()`, `stack()`, `split()` in `shape.mojo` take `List[AnyTensor]` — core ops but use collections. Moved to Phase 5. (`shape.mojo:432, 585, 654, 721`)

**H4: `__str__`/`__repr__` use `_get_float64` internally.** Need typed access for `Tensor[dtype]`. Also hardcodes `"AnyTensor(["` string. Added to Phase 1. (`any_tensor.mojo:3247-3317`)

**H5: Scope underestimated 30-50%.** 480 functions need explicit `[dt: DType]` on signatures (not just type name change).

**H6: Eager instantiation binary bloat.** Mojo uses eager instantiation (confirmed by blog post investigation). 317 functions x 11 dtypes = 3,487 worst-case. Mitigation: restrict to 3 float types (float16/32/64) initially.

**H7: Module/Sequential trait boundary forces AnyTensor round-trips.** `shared/core/module.mojo:86` defines `fn forward(mut self, input: AnyTensor) raises -> AnyTensor`. Mojo 0.26.1 doesn't support parametric trait methods. This means: (a) layers like `Linear[dtype]` must still accept AnyTensor in their `forward()` signature; (b) `Sequential2[T0, T1]` chains layers through AnyTensor, losing type safety between layers; (c) every inter-layer boundary does AnyTensor→Tensor[dtype]→compute→as_any() round-trip. **Impact**: Type safety benefits of Tensor[dtype] apply only INSIDE individual layer implementations, not across the composition boundary. **Key detail**: Only `Linear` and `ReLULayer` implement Module. `BatchNorm2dLayer`, `Conv2dLayer`, `DropoutLayer` do NOT — they can be freely parameterized without trait constraints.

**H8: Total scope underestimated ~2x.** Phase 1 estimate of ~200 lines changed is wrong — renaming the struct in the 4,703-line any_tensor.mojo + updating `shared/__init__.mojo` + adding `as_tensor()` is ~1,500 lines. Phase 7 estimate of ~3,000 lines across 386 test files (8 lines/file avg) is wrong — 3,975 creation calls + 412 import lines + set() calls = ~5,600-8,000 lines. Revised total: ~15,700 lines (up from 9,700).

#### MEDIUM (6)

- **M1**: In-place operators (`__iadd__` etc.) use `_get_float64`/`_set_float64` round-trip — pre-existing precision bug (`any_tensor.mojo:3013-3079`)
- **M2**: `__hash__` uses `_get_float64` — precision loss for int types (`any_tensor.mojo:3319-3369`)
- **M3**: `Hashable` conformance not specified in `TensorLike` trait design
- **M4**: SIMD helpers (`_relu_simd_float32`/`_float64`) should merge to single parametric version
- **M5**: Circular import in `tensor_io.mojo` (lines 3-14) must be preserved during rename
- **M6**: Phase 4→5 trait dependency. `Linear` and `ReLULayer` implement the `Module` trait (`module.mojo:69`). They can't be parameterized until Module's `forward` signature changes from `AnyTensor` to `AnyTensor`. Phase ordering must be: traits first (Phase 5a), then Module-implementing layers (Phase 4b).

#### LOW (4)

- `__str__` hardcodes `"AnyTensor(["` — string-assertion tests break
- Reflected operators (`__radd__` etc.) not listed but delegate to non-reflected (auto-migrate)
- `__neg__` has 12 dtype branches — not explicitly listed in "remove bitcasts" steps
- `_extensor_binary_arith` (`any_tensor.mojo:3728-3796`) transformation not explicitly mapped. This is the central dispatch pattern used by `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__pow__`. New typed version: `fn _tensor_binary_arith[dtype: DType, op: fn(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype]](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]` — zero dispatch cascade, broadcasting logic unchanged. AnyTensor version preserved until Phase 7e.

#### VERIFIED CLAIMS

| Claim | Status |
|-------|--------|
| `comptime AnyTensor = AnyTensor` works as alias | VERIFIED |
| Operator overloads work with `Self` on parametric struct | VERIFIED |
| Cross-dtype `Tensor[f32] + Tensor[f64]` is compile-time error | VERIFIED |
| Zero-copy bitcast view safe (no ASAP UAF) | VERIFIED (requires shared refcount — see B4) |
| `Batch[data_dtype, label_dtype]` multi-param struct works | VERIFIED |
| Memory pool dtype-agnostic (no impact) | VERIFIED |
| All binary ops reject dtype mismatch at runtime | VERIFIED |

#### BUGS FOUND (fixed in PR #5001)

- `conv2d`: no dtype guard on kernel/bias vs input (`conv.mojo:342-476`)
- `batch_norm2d`: no dtype guard on gamma/beta/running_stats vs input (`normalization.mojo:29-341`)
- `attention`: no dtype guard on mask vs scores (`attention.mojo:127-128`)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [How SIMD Works (The Target Behavior)](#2-how-simd-works)
3. [Why AnyTensor Can't Do This Today](#3-why-extensor-cant-do-this-today)
4. [The Parametric Solution](#4-the-parametric-solution)
5. [Critical Design Decisions](#5-critical-design-decisions)
6. [Impact Assessment](#6-impact-assessment)
7. [Migration Strategy (17 Sub-Phases)](#7-migration-strategy-17-sub-phases)
8. [Phase Details](#8-phase-details)
9. [Risk Analysis](#9-risk-analysis)
10. [Verification Plan](#10-verification-plan)
11. [Corner Cases & Blockers](#11-corner-cases--blockers-deep-analysis)
12. [References](#12-references)

---

## 1. Problem Statement

AnyTensor stores `_dtype` as a **runtime** field (`var _dtype: DType`). This means:

- `__getitem__` returns a fixed type (`Float32`) regardless of the tensor's actual dtype
- `tensor[i] = Float64(x)` fails: "cannot implicitly convert 'Float64' to 'Float32'"
- `tensor[i] = Float16(x)` fails: same error
- `tensor[i] = 3.14159` on a float64 tensor truncates to Float32 precision
- Mojo does NOT dispatch `obj[i] = val` to `__setitem__` — it uses `__getitem__` as an lvalue

The current workaround (`set()` method) introduces precision-losing Float64 round-trips and requires callers to use a non-standard API.

**Source**: `shared/core/any_tensor.mojo:798` — `__getitem__` returns `Float32`
**Source**: `shared/core/any_tensor.mojo:116` — `var _dtype: DType` runtime field

---

## 2. How SIMD Works

SIMD is **parametric on DType at compile time**: `SIMD[DType.float32, 4]`

```mojo
var s = SIMD[DType.float32, 4](0.0)
s[0] = 3.14159       # FloatLiteral → Scalar[float32] ✓
s[1] = Float32(2.0)  # Same type ✓
s[2] = 42            # IntLiteral → Scalar[float32] ✓
s[3] = Float64(1.0)  # ERROR: different type ✗ (strict, by design)
```

Key behaviors:
- `__getitem__` returns `Scalar[Self.dtype]` — the correct type at compile time
- Assignment works with same-type values and literals only
- No implicit widening or narrowing between float types
- The compiler selects a single concrete code path per instantiation

**Source**: Mojo SIMD docs — https://docs.modular.com/mojo/std/builtin/simd/SIMD/
**Verified**: Tested locally with `SIMD[DType.float32, 4]` and `SIMD[DType.float64, 4]`

---

## 3. Why AnyTensor Can't Do This Today

### 3.1 Runtime DType Prevents Parametric Return Types

```mojo
# Current — runtime dtype
struct AnyTensor:
    var _dtype: DType  # runtime value
    fn __getitem__(self, i: Int) raises -> Float32:  # FIXED return type
        return self._get_float32(i)
```

`__getitem__` can't return `Scalar[self._dtype]` because `self._dtype` is a runtime value and Mojo requires compile-time parameters for generic types.

### 3.2 Massive Runtime Branching

The codebase contains:
- **177 dtype branch checks** inside `any_tensor.mojo` alone
- **174 dtype branch checks** across consumer files
- **708 `_data.bitcast[T]()` calls** across the `shared/` directory
- **158 bitcast calls** inside `any_tensor.mojo`

Each of these represents code that should be monomorphized by the compiler instead of branching at runtime.

### 3.3 The `__setitem__` Dead Code Problem

Mojo's `obj[i] = val` uses `__getitem__` as an lvalue, not `__setitem__`. All `__setitem__` overloads are dead code for subscript assignment syntax. The 12 `set()` overloads are a workaround, not a solution.

**Source**: `shared/core/any_tensor.mojo:922-992` — 12 `set()` overloads
**Source**: Verified by test — `__setitem__` is never called via `[i]=` syntax

---

## 4. The Parametric Solution

### 4.1 New Struct Definition

```mojo
struct Tensor[dtype: DType]:
    var _data: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var _shape: List[Int]
    var _strides: List[Int]
    var _numel: Int
    var _is_view: Bool
    var _refcount: UnsafePointer[Int, origin=MutAnyOrigin]
    var _allocated_size: Int
    var _original_numel_quantized: Int
    # _dtype field REMOVED — now Self.dtype (compile-time parameter)
```

### 4.2 Simplified __getitem__ / Assignment

```mojo
fn __getitem__(self, index: Int) raises -> Scalar[Self.dtype]:
    var idx = self._resolve_index(index)
    return self._data[idx]

# No __setitem__ needed! Mojo handles [i]= via lvalue from __getitem__:
# tensor[i] = 3.14159       → FloatLiteral converts to Scalar[dtype] ✓
# tensor[i] = Float32(x)    → same type for float32 tensor ✓
# tensor[i] = Float64(x)    → same type for float64 tensor ✓
```

### 4.3 Simplified Internal Methods

```mojo
# BEFORE: 6 branching internal methods (177 branch checks)
fn _get_float64(self, index: Int) -> Float64:
    if self._dtype == DType.float16: ...
    elif self._dtype == DType.float32: ...
    elif self._dtype == DType.float64: ...
    elif self._dtype == DType.bfloat16: ...
    else: ...

# AFTER: Direct typed access (0 branches)
fn _get(self, index: Int) -> Scalar[Self.dtype]:
    return self._data[index]

fn _set(mut self, index: Int, value: Scalar[Self.dtype]):
    self._data[index] = value
```

### 4.4 Factory Functions

```mojo
# BEFORE: runtime dtype parameter
fn zeros(shape: List[Int], dtype: DType) raises -> AnyTensor: ...

# AFTER: compile-time dtype parameter
fn zeros[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]: ...

# Usage:
var t = zeros[DType.float32]([3, 4])
t[0] = 3.14  # Just works — Scalar[float32] lvalue
```

---

## 5. Critical Design Decisions

### 5.1 Runtime-to-Compile-Time Promotion

**Problem**: Mojo does NOT allow runtime values as compile-time parameters.

```mojo
var runtime_dtype: DType = DType.float32  # runtime value
var t = AnyTensor[runtime_dtype]([3])       # COMPILE ERROR
```

**Solution**: Where dtype must be determined at runtime (e.g., loading from file, user config), use the existing `dtype_dispatch` pattern to fan out:

```mojo
fn load_tensor(path: String) raises -> ???:
    var dtype = read_dtype_from_header(path)
    if dtype == DType.float32:
        return load_typed[DType.float32](path)
    elif dtype == DType.float64:
        return load_typed[DType.float64](path)
    # ...
```

This is the same pattern already used in `_extensor_binary_arith` at `any_tensor.mojo:3728`.

### 5.2 Heterogeneous Collections

**Problem**: `List[AnyTensor]` works today because all AnyTensors are the same type. With parametric AnyTensor, `List[Tensor[DType.float32]]` can't hold `Tensor[DType.float64]`.

**Solution**: Define a trait `Tensor` that `Tensor[dtype]` conforms to. Use `List[AnyTensor]` (trait object) for heterogeneous collections:

```mojo
trait Tensor:
    fn shape(self) -> List[Int]
    fn numel(self) -> Int
    fn dtype(self) -> DType
    # ...

struct Tensor[dtype: DType](Tensor):
    # ...
```

**Source**: `shared/core/traits.mojo` — already defines trait patterns for the codebase.

### 5.3 Cross-DType Operations

**Problem**: `shared/training/mixed_precision.mojo` intentionally converts between dtypes:
```mojo
fn convert_to_fp32_master(params: AnyTensor) -> AnyTensor  # float16 → float32
```

**Solution**: With parametric types, this becomes explicit and type-safe:
```mojo
fn convert_to_fp32_master(params: Tensor[DType.float16]) -> Tensor[DType.float32]
```

Add a `cast` method for explicit conversions:
```mojo
fn cast[target: DType](self) raises -> Tensor[target]:
    var result = Tensor[target](self._shape)
    for i in range(self._numel):
        result._data[i] = self._data[i].cast[target]()
    return result^
```

**Source**: `shared/training/mixed_precision.mojo:8,5` — 8 AnyTensor params, 5 returns, 6 dtype branches

### 5.4 Memory Pool Compatibility

**Current**: `_data = pooled_alloc(total_bytes)` returns `UnsafePointer[UInt8]`
**After**: `_data = pooled_alloc(total_bytes).bitcast[Scalar[Self.dtype]]()` — one bitcast at allocation time, zero bitcasts thereafter.

**Source**: `shared/base/memory_pool.mojo` — `pooled_alloc` / `pooled_free` byte-level API (moved from `shared/core/` in Phase 0)

### 5.5 Module/Sequential Stays on AnyTensor

**Problem**: Mojo 0.26.1 doesn't support parametric trait methods. The `Module` trait (`shared/core/module.mojo:86`) defines `fn forward(mut self, input: AnyTensor) raises -> AnyTensor`. This signature cannot become `fn forward[dt: DType](mut self, input: Tensor[dt]) raises -> Tensor[dt]`.

**Consequence**: Type safety benefits of `Tensor[dtype]` apply only INSIDE individual layer implementations, not across the composition boundary:
- `Linear[DType.float32].forward()` internally works with `Tensor[DType.float32]`
- But the Module interface forces `forward(AnyTensor) -> AnyTensor`
- `Sequential2[T0, T1]` chains layers through AnyTensor, doing boundary conversion at each step

**Key detail**: Only `Linear` and `ReLULayer` implement `Module`. `BatchNorm2dLayer`, `Conv2dLayer`, `DropoutLayer` do NOT implement Module — they are free to parameterize without trait constraints.

**Future**: When Mojo adds parametric trait methods, introduce `TypedModule[dtype: DType]` with `fn forward(mut self, input: Tensor[dtype]) -> Tensor[dtype]` and `TypedSequential2[dtype, T0, T1]`.

---

## 6. Impact Assessment

### 6.1 Quantitative Summary

| Metric | Count |
|--------|-------|
| Functions taking AnyTensor as parameter | 368 |
| Functions returning AnyTensor | 480 |
| Runtime dtype branch checks (any_tensor.mojo) | 177 |
| Runtime dtype branch checks (consumers) | 174 |
| `_data.bitcast[T]()` calls (total) | 708 |
| Factory functions to update | 14 |
| `set()` overloads to remove | 12 |
| Internal getter/setter methods to simplify | 6 |
| Type conversion methods | 16 |
| Test files using `DType.float32` | 386 |
| Source files importing AnyTensor | 73 |

### 6.2 Top 10 Most Impacted Files

| Rank | File | Total Signals | Why Hard |
|------|------|--------------|----------|
| 1 | `shared/core/any_tensor.mojo` | 263 | The struct itself — every method changes |
| 2 | `shared/core/elementwise.mojo` | 99 | 27 function signatures change |
| 3 | `shared/core/activation.mojo` | 79 | 18 dtype branches, 29 return sites |
| 4 | `shared/core/arithmetic_contiguous.mojo` | 67 | 40 dtype comparisons in 4 functions |
| 5 | `shared/core/matrix.mojo` | 64 | 48 bitcast calls, partially migrated |
| 6 | `shared/core/dtype_dispatch.mojo` | 59 | The dispatch infra itself — may be replaced |
| 7 | `shared/core/arithmetic.mojo` | 58 | 15 function signatures |
| 8 | `shared/core/numerical_safety.mojo` | 51 | 21 dtype branches, 21 bitcasts |
| 9 | `shared/core/strassen.mojo` | 40 | 34 bitcasts in 3 long functions |
| 10 | `shared/core/activation_simd.mojo` | 43 | 24 bitcasts in parametric kernels |

**Source**: Consumer audit agent — full per-file table available

### 6.3 Files Already Using Parametric Patterns

These files already use `[dtype: DType]` function parameters and will be easiest to migrate:
- `shared/core/conv.mojo` — `_conv2d_kernel[dtype: DType]()` pattern
- `shared/core/elementwise.mojo` — `dispatch_unary`/`dispatch_binary`
- `shared/core/dtype_dispatch.mojo` — the dispatch infrastructure itself
- `shared/core/matrix.mojo` — `_matmul_2d_1d_impl[dtype]()` (partially migrated)

**Source**: Consumer audit — "Pattern 2: Parametric `@parameter if`" section

---

## 7. Migration Strategy (17 Sub-Phases)

### Overview

```
Phase 0:  Package split (shared/base + shared/tensor + shared/core)
    ↓
Phase 1a: Create Tensor[dtype] + TensorLike trait in shared/tensor/ (additive only)
Phase 1b: Rename struct AnyTensor → AnyTensor, add alias, fix naming collision (B3)
    ↓
Phase 2:  Factory functions return Tensor[dtype]
    ↓
Phase 3a: Single-tensor shape ops (reshape, as_contiguous, flatten, etc.)
Phase 3b: Arithmetic + SIMD ops
Phase 3c: Elementwise + activation + comparison ops
Phase 3d: Matrix + reduction + conv ops
    ↓
Phase 4a: Parameterize non-Module layers (BatchNorm, Conv2d, Dropout)
    ↓
Phase 5a: Update trait signatures (Module, Differentiable, etc.) → AnyTensor
    ↓ (traits must update BEFORE Module-implementing layers can be parameterized — see M6)
Phase 4b: Parameterize Module layers (Linear, ReLU) + Sequential
    ↓
Phase 5b: Collection ops (concatenate, stack, split, lazy_expr) → List[AnyTensor]
Phase 5c: Optimizer + Variable + gradient types → AnyTensor
    ↓
Phase 6:  Training, autograd tape, data pipelines
    ↓
Phase 7a: Tests — shared/core/ (~180 files)
Phase 7b: Tests — models/ (~45 files)
Phase 7c: Tests — training/ + autograd/ + data/ (~120 files)
Phase 7d: Tests — integration/ + remaining (~40 files)
Phase 7e: Final cleanup — remove alias, rename any_tensor.mojo → any_tensor.mojo
```

**Key ordering change**: Phase 5a (traits) moves BEFORE Phase 4b (Module layers), because `Linear` and `ReLULayer` implement `Module` and can't be parameterized until Module accepts AnyTensor (see M6).

Each sub-phase is its own PR. Each follows the 12-step workflow (Plan → Test → Implement → Review → Fix → Cleanup).

### Revised Scope Estimates

| Phase | Lines | Notes |
|-------|-------|-------|
| Phase 0 | ~500 | Move files + update import paths |
| Phase 1a | ~800 new | New Tensor[dtype], TensorLike |
| Phase 1b | ~1,500 | Rename struct, alias, collision fix |
| Phase 2 | ~200 new | Typed factory functions |
| Phase 3a-d | ~2,500 | Core op overloads across ~20 files |
| Phase 4a | ~500 | Non-Module layer parameterization |
| Phase 5a | ~300 | Trait signature updates |
| Phase 4b | ~400 | Module layer + Sequential |
| Phase 5b-c | ~1,400 | Collection ops + optimizer/variable |
| Phase 6 | ~1,500 | Training, autograd, data |
| Phase 7a-d | ~5,600 | Test updates (4 sub-PRs by directory) |
| Phase 7e | ~500 | Final cleanup |
| **Total** | **~15,700** | Up from original 9,700 estimate (see H8) |

---

## 8. Phase Details

### Phase 1: Tensor[dtype] + AnyTensor + TensorLike Trait

**Goal**: Create `Tensor[dtype]` alongside the existing AnyTensor (renamed to `AnyTensor`), with a shared `TensorLike` trait.

**Files**: `shared/tensor/tensor.mojo` (NEW), `shared/tensor/tensor_traits.mojo` (NEW), `shared/tensor/any_tensor.mojo` (MOVED from `shared/core/`), `shared/base/` (NEW — extracted from `shared/core/`)

**3-Layer Package Architecture** (prerequisite: Phase 0 package split):
```
shared/base/             # LAYER 1: Zero tensor dependencies
    __init__.mojo
    memory_pool.mojo     # pooled_alloc/pooled_free (moved from core)
    broadcasting.mojo    # broadcast_shapes (moved from core, pure List[Int] functions)
    dtype_ordinal.mojo   # dtype-to-ordinal mapping (moved from core)
    defaults.mojo        # constants (moved from core)
    math_constants.mojo  # constants (moved from core)
    numerical_constants.mojo
    activation_constants.mojo
    optimizer_constants.mojo
    error_utils.mojo
    types/               # dtype_aliases, fp_constants, mxfp4, nvfp4 (moved from core)

shared/tensor/           # LAYER 2: Imports base only
    __init__.mojo
    tensor.mojo          # NEW struct Tensor[dtype: DType]
    tensor_traits.mojo   # NEW trait TensorLike
    any_tensor.mojo        # struct AnyTensor (MOVED from core, filename kept until Phase 7e)
    gradient_types.mojo  # GradientPair/Triple/Quad (moved from core)
    validation.mojo      # shape/dtype validation (moved from core)
    tensor_io.mojo       # save/load (moved from core)

shared/core/             # LAYER 3: Imports base + tensor
    __init__.mojo
    arithmetic.mojo      # (and all 40+ operation files)
    traits.mojo          # Differentiable, Parameterized, Model, Loss, Optimizer
    module.mojo          # Module trait
    sequential.mojo      # Sequential2-5
    layers/              # Linear, Conv2D, BatchNorm, etc.
    lazy_expression.mojo
    lazy_eval.mojo
    dtype_dispatch.mojo
    ...
```

**Dependency chain verified** (no circular deps):
- `any_tensor.mojo` imports: `memory_pool` (→base), `broadcasting` (→base), `dtype_ordinal` (→base) ✓
- `broadcasting.mojo` imports: nothing from shared (pure stdlib) ✓
- `validation.mojo` imports: `extensor` (→tensor, same package) ✓
- `traits.mojo` imports: `extensor` (→tensor, cross-package but acyclic) ✓

**File rename strategy**: Do NOT rename `any_tensor.mojo` → `any_tensor.mojo` until Phase 7e. Keep filename throughout migration to avoid breaking 569 import paths early. Add `comptime AnyTensor = AnyTensor` alias inside the file.

**Estimated scope**: Phase 0 (~500 lines import path updates), Phase 1a (~800 lines new), Phase 1b (~1,500 lines changed)

#### Phase 0 — Package Split (NEW — prerequisite)

Move files from `shared/core/` to create the 3-layer architecture described above. Each moved file gets updated import paths (`from shared.base.X import Y`). `shared/core/__init__.mojo` and `shared/__init__.mojo` re-export from new locations for backward compat. ~500 lines of import path updates. This is a mechanical move with zero behavior changes.

#### Phase 1a — Create Tensor[dtype] + TensorLike Trait (additive only)
- Create `shared/tensor/tensor.mojo`: `struct Tensor[dtype: DType = DType.float32]`
- Create `shared/tensor/tensor_traits.mojo`: `trait TensorLike(Copyable, Movable)`
- Design `TensorLike` trait interface (`numel`, `shape`, `dtype`, `ndim`; decide on `Hashable`)
- Design `Tensor[dtype]` struct: fields, `__getitem__` returning `Scalar[Self.dtype]`, `__init__`, `as_any()`, `cast[target]()`
- Design `__str__`/`__repr__` using typed `self._data[i]` access (not `_get_float64`), output `"Tensor(["`
- Design `__hash__` using typed access (fix int precision loss, see M2)
- Implement shared refcount protocol for `as_any()` (see B4)
- Pointer arithmetic does NOT multiply by `dtype_size` (typed pointer auto-scales)
- Compilation time guard: run `time just package` after creating Tensor[dtype], record baseline
- Note: all functions returning `Tensor` must use `fn foo[dt: DType](t: Tensor[dt]) -> Tensor[dt]` (auto-param fails for return types, see B1)
- Zero existing code changes in this phase

#### Phase 1b — Rename Struct + Fix Collision
- Rename `struct AnyTensor` → `struct AnyTensor` inside `any_tensor.mojo` (keep filename)
- Add `comptime AnyTensor = AnyTensor` alias for backward compat
- Add `as_tensor[dtype: DType]() -> Tensor[dtype]` method with shared refcount (see B4)
- Conform AnyTensor to `TensorLike` trait
- Remove `comptime Tensor = AnyTensor` from `shared/__init__.mojo:82` (see B3)
- Add proper Tensor[dtype] re-export from shared/tensor/
- Update `shared/core/__init__.mojo` and `shared/__init__.mojo` exports

#### Phase 1.2 — Commit Plan
- Commit plan documentation

#### Phase 1.3 — Test (TDD)
- Write tests for `Tensor[dtype]` element access: `t[i] = 3.14159`, `t[i] = Float32(x)`, `t[i] = 42`
- Write tests for `AnyTensor.as_tensor[dtype]()` — valid and dtype-mismatch cases
- Write tests for `Tensor.as_any()` round-trip
- Write tests for `TensorLike` trait conformance

#### Phase 1.4 — Review Tests
- Review test design for coverage and edge cases

#### Phase 1.5 — Commit Tests
- Commit test files

#### Phase 1.6 — Implement
- Create `shared/tensor/` module with `struct Tensor[dtype: DType = DType.float32](TensorLike)`
- Implement `__getitem__` returning `Scalar[Self.dtype]`, `__str__`, `__repr__`, `__hash__` with typed access
- Implement `as_any() -> AnyTensor` (zero-copy, bitcast to UInt8)
- Implement `cast[target: DType]() -> Tensor[target]`
- Rename AnyTensor → AnyTensor in `shared/core/any_tensor.mojo`, add `as_tensor[dtype]()`, conform to `TensorLike`
- Update `shared/core/__init__.mojo` and `shared/__init__.mojo` exports
- Add `comptime AnyTensor = AnyTensor` alias for backward compat
- Ensure pointer arithmetic does NOT multiply by dtype_size (typed pointer auto-scales)

#### Phase 1.7 — Review Implementation
- Review against plan, check trait conformance, verify zero-copy conversions

#### Phase 1.8 — Commit Implementation
- Commit implementation

#### Phase 1.9 — Final Review
- End-to-end review: plan + tests + implementation coherence
- Verify `tensor[i] = value` just works for all supported types/literals

#### Phase 1.10 — Fix & Validate
- Address review findings
- Run `just package` and `just test-mojo` via Podman
- Verify no regressions in existing tests

#### Phase 1.11 — Commit Fixes
- Commit review fixes

#### Phase 1.12 — Cleanup
- Remove any dead code from Phase 1
- Update docs/ADRs if needed
- Verify CI green

### Phase 2: Factory Functions

**Goal**: Add `Tensor[dtype]`-returning factory functions alongside existing ones.

**Files**: `shared/core/tensor.mojo`, `shared/core/initializers.mojo`

**Estimated scope**: ~200 lines new

#### Phase 2.1 — Plan
- Design typed factory signatures: `fn zeros[dtype: DType](shape) -> Tensor[dtype]`
- Decide: new functions or overloads? (New functions recommended — keep existing for AnyTensor)
- Cover all 14: `zeros`, `ones`, `full`, `empty`, `arange`, `eye`, `linspace`, `*_like`, `nan_tensor`, `inf_tensor`, `neg_inf_tensor`, `randn`

#### Phase 2.2 — Commit Plan

#### Phase 2.3 — Test (TDD)
- `var t = zeros[DType.float32]([3, 4])` — verify type inference, element access
- `var t = ones[DType.float64]([2])` — verify literal assignment works at float64 precision
- `var t2 = zeros_like(t)` — verify dtype propagation

#### Phase 2.4 — Review Tests
#### Phase 2.5 — Commit Tests
#### Phase 2.6 — Implement
- Add typed factory functions that create `Tensor[dtype]` using `pooled_alloc` + bitcast
- `*_like` functions use auto-parameterization: `fn zeros_like(t: Tensor) -> Tensor`

#### Phase 2.7 — Review Implementation
#### Phase 2.8 — Commit Implementation
#### Phase 2.9 — Final Review
#### Phase 2.10 — Fix & Validate
#### Phase 2.11 — Commit Fixes
#### Phase 2.12 — Cleanup

### Phase 3: Core Operations Accept Tensor[dtype] (4 sub-PRs)

**Goal**: Migrate core ops to accept `Tensor[dtype]`. Split into 4 sub-PRs for manageable PR sizes.

**Estimated scope**: ~2,500 lines changed across ~20 files

**Key transformation** — `_extensor_binary_arith` (`any_tensor.mojo:3728-3796`) becomes:
```mojo
fn _tensor_binary_arith[
    dtype: DType, op: fn(Scalar[dtype], Scalar[dtype]) -> Scalar[dtype]
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    # Zero dtype dispatch — dtype is compile-time parameter
    # Broadcasting logic unchanged from _extensor_binary_arith
    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = Tensor[dtype](result_shape)
    for i in range(result.numel()):
        result._data[idx] = op(a._data[a_idx], b._data[b_idx])
    return result^
```
AnyTensor version (`_extensor_binary_arith`) preserved until Phase 7e. Same pattern for `_extensor_compare_op`.

#### Phase 3a — Single-tensor shape ops
**Files**: `shape.mojo` (single-tensor ops ONLY: `reshape`, `as_contiguous`, `flatten`, `flatten_to_2d`, `squeeze`, `unsqueeze`, `expand_dims`, `permute`, `broadcast_to`)
- NOTE: `concatenate`/`stack`/`split`/`tile`/`repeat` deferred to Phase 5b (they take `List`)

#### Phase 3b — Arithmetic + SIMD ops
**Files**: `arithmetic.mojo`, `arithmetic_contiguous.mojo`, `arithmetic_simd.mojo`
- Add `Tensor[dtype]` overloads alongside existing AnyTensor functions
- Use explicit parametric signatures: `fn add[dt: DType](a: Tensor[dt], b: Tensor[dt]) -> Tensor[dt]`
- Eliminate runtime dtype branches, remove bitcasts

#### Phase 3c — Elementwise + activation + comparison ops
**Files**: `elementwise.mojo`, `activation.mojo`, `activation_simd.mojo`, `comparison.mojo`
- Merge SIMD helpers: `_relu_simd_float32`/`_float64` → single `_relu_simd[dt: DType]` (see M4)

#### Phase 3d — Matrix + reduction + conv ops
**Files**: `matrix.mojo`, `matmul.mojo`, `reduction.mojo`, `reduction_ops.mojo`, `conv.mojo`, `strassen.mojo`

Each sub-phase follows the 12-step workflow (Plan → Test → Implement → Review → Fix → Cleanup).

### Phase 4a: Parameterize Non-Module Layers

**Goal**: Parameterize layer structs that do NOT implement the Module trait.

**Files**: `layers/batchnorm.mojo`, `layers/conv2d.mojo`, `layers/dropout.mojo`, `normalization.mojo`, `normalization_simd.mojo`, `pooling.mojo`, `attention.mojo`, `loss.mojo`, `loss_utils.mojo`, `dtype_cast.mojo`, `numerical_safety.mojo`

**Estimated scope**: ~500 lines changed

- Parameterize: `BatchNorm2dLayer[dtype: DType = DType.float32]`, `Conv2dLayer[dtype: DType = DType.float32]`
- `DropoutLayer` has no AnyTensor fields — may not need parameterization
- Design `cast[target]()` for `dtype_cast.mojo`
- I/O: `Tensor[dtype].save()` → `self.as_any().save()`. Loading returns `AnyTensor`
- Preserve circular import workaround in `tensor_io.mojo` (lines 3-14)

### Phase 5a: Update Trait Signatures → AnyTensor

**Goal**: Change trait signatures from `AnyTensor` to `AnyTensor`. Must happen BEFORE Phase 4b (see M6).

**Files**: `traits.mojo`, `module.mojo`

**Estimated scope**: ~300 lines changed

- `Module.forward(mut self, input: AnyTensor) raises -> AnyTensor`
- `Module.parameters(self) raises -> List[AnyTensor]`
- `Differentiable.forward/backward` → AnyTensor
- `Parameterized.parameters/gradients` → `List[AnyTensor]`
- `Optimizer.step` → `List[AnyTensor]`
- Note: These traits stay on AnyTensor permanently (not Tensor[dtype]) because Mojo 0.26.1 doesn't support parametric trait methods and collections require type erasure (see section 5.5)

### Phase 4b: Parameterize Module Layers + Sequential

**Goal**: Parameterize layers that implement Module. Only possible after Phase 5a updates Module's signature.

**Files**: `layers/linear.mojo`, `layers/relu.mojo`, `sequential.mojo`

**Estimated scope**: ~400 lines changed

- `Linear[dtype: DType = DType.float32]` implements `Module` with AnyTensor boundary:
  - `forward(mut self, input: AnyTensor) -> AnyTensor` converts at boundary
  - Internal computation uses `Tensor[dtype]` for type safety
- `ReLULayer` — implements Module (no dtype fields, may not need parameterization)
- `Sequential2-5` — keep Module-conforming, work with AnyTensor at boundaries

### Phase 5b: Collection Operations → List[AnyTensor]

**Goal**: Migrate operations that take `List[AnyTensor]` to `List[AnyTensor]`.

**Files**: `shape.mojo` (collection ops), `lazy_expression.mojo`, `lazy_eval.mojo`

**Estimated scope**: ~600 lines changed

- `concatenate`, `stack`, `split`, `split_with_indices`, `tile`, `repeat` → accept `List[AnyTensor]` (H3)
- `TensorExpr._tensors: List[AnyTensor]` (B2)
- `lazy_eval.mojo` dispatch updated

### Phase 5c: Optimizer + Variable + Gradient Types → AnyTensor

**Goal**: Migrate autograd infrastructure to AnyTensor.

**Files**: `autograd/variable.mojo`, `autograd/optimizers.mojo`, `autograd/tape_types.mojo`, `gradient_types.mojo`, `testing/models.mojo`

**Estimated scope**: ~800 lines changed

- `Variable.data: AnyTensor` (keeps runtime dtype for autograd tape)
- All gradient containers: AnyTensor fields (`GradientPair`, `GradientTriple`, `GradientQuad`, etc.)
- `SavedTensors.tensors: List[AnyTensor]`, `VariableRegistry.grads: List[AnyTensor]`
- All 3 optimizer systems updated: functional (training/optimizers/), autograd (autograd/optimizers.mojo), trait-based (traits.mojo)
- Optimizer velocity/moment buffers: `List[AnyTensor]`
- `state_dict() -> Dict[String, AnyTensor]`
- Boundary conversions: `param.as_tensor[dtype]()` before compute

### Phase 6: Training, Autograd, Data Pipelines

**Goal**: Migrate training infrastructure, autograd, data loading, metrics.

**Files**: `training/mixed_precision.mojo`, `training/gradient_ops.mojo`, `training/metrics/*.mojo`, `autograd/tape_types.mojo`, `autograd/functional.mojo`, `data/transforms.mojo`, `data/_datasets_core.mojo`, `data/cache.mojo`, `testing/assertions.mojo`, `testing/gradient_checker.mojo`, `utils/file_io.mojo`

**Estimated scope**: ~1000 lines changed across 15+ files

#### Phase 6.1 — Plan
- Design `mixed_precision.mojo` with explicit cross-dtype signatures: `fn convert(Tensor[f16]) -> Tensor[f32]`
- Design `Batch` struct: `Batch[data_dtype: DType, label_dtype: DType]` or `data: Tensor[dtype]` + `labels: AnyTensor`
- Design file I/O boundary: `load_tensor() -> AnyTensor` (runtime dtype from file)
- Plan `PrecisionConfig` integration with `AnyTensor` (runtime dtype selection)

#### Phase 6.2 — Commit Plan
#### Phase 6.3 — Test (TDD)
- Mixed-precision: `convert_fp16_to_fp32(Tensor[f16]) -> Tensor[f32]`
- Data loading: `load_tensor() -> AnyTensor`, then `as_tensor[f32]()`
- Gradient checker with `Tensor[dtype]`
- Metrics with typed tensors

#### Phase 6.4 — Review Tests
#### Phase 6.5 — Commit Tests
#### Phase 6.6 — Implement
- Update mixed_precision with explicit typed signatures
- Update data pipeline: `Batch` uses typed data + `AnyTensor` labels
- File I/O returns `AnyTensor` with dispatch at boundary
- Update gradient checker, assertions, metrics

#### Phase 6.7 — Review Implementation
#### Phase 6.8 — Commit Implementation
#### Phase 6.9 — Final Review
#### Phase 6.10 — Fix & Validate
#### Phase 6.11 — Commit Fixes
#### Phase 6.12 — Cleanup

### Phase 7: Tests + Final Cleanup (5 sub-PRs)

**Goal**: Migrate all 395+ test files, remove AnyTensor alias, rename file, remove all workarounds.

**Estimated scope**: ~6,100 lines changed across 395+ files

**Mechanical replacement patterns** (for parallel sub-agents):
- `zeros([3, 4], DType.float32)` → `zeros[DType.float32]([3, 4])`
- `var t: AnyTensor = ...` → `var t = ...` (type inference)
- `tensor.set(i, val)` → `tensor[i] = val`
- `tensor._set_float64(i, val)` → `tensor[i] = Scalar[dtype](val)`
- Import path updates from `shared.core.any_tensor` → `shared.tensor.extensor`

#### Phase 7a — Tests: shared/core/ (~180 files, ~2,500 lines)
- Core tensor operation tests
- Activation, arithmetic, matrix, shape, comparison, reduction tests
- Add regression test: `tensor[i] = 3.14159` on float64 tensor preserves full precision

#### Phase 7b — Tests: models/ (~45 files, ~700 lines)
- Model layer tests, forward/backward pass tests

#### Phase 7c — Tests: training/ + autograd/ + data/ (~120 files, ~1,800 lines)
- Training loop tests, optimizer tests, gradient tests, data loading tests

#### Phase 7d — Tests: integration/ + remaining (~40 files, ~600 lines)
- Integration tests, packaging tests, benchmark tests

#### Phase 7e — Final Cleanup (~500 lines)
- Remove `comptime AnyTensor = AnyTensor` alias
- Rename file `any_tensor.mojo` → `any_tensor.mojo`, update ALL import paths
- Remove deprecated `set()` overloads from AnyTensor (if no longer needed)
- Remove dead code: old internal setters/getters (`_get_float64`, `_set_float64`, etc.)
- Remove `dtype_dispatch.mojo` if fully superseded
- Full CI green check + performance regression check
- Update CLAUDE.md, ADRs, documentation
- Close epic #4998

---

## 9. Risk Analysis

### 9.1 High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| `List[AnyTensor]` breaks — heterogeneous collections | Batches, model params | Use `AnyTensor` for heterogeneous collections |
| Runtime dtype determination (file loading, config) | Data pipeline | Keep dispatch fan-out at boundary |
| Compilation time explosion (13 dtype instantiations) | CI slowdown | Profile; consider supporting fewer dtypes |
| Mojo compiler bugs with complex parametric types | Blocked | Test incrementally; file upstream issues |

### 9.2 Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| `mixed_precision.mojo` cross-dtype semantics | Training | Make explicit: `AnyTensor[float16] → AnyTensor[float32]` |
| `__init__` can't take runtime dtype | All callers | Move dtype to compile-time parameter position |
| Quantization methods (FP8, MXFP4, NVFP4) | Inference | These produce `AnyTensor[uint8]` — explicit output type |

### 9.3 Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory pool compatibility | Allocation | Single bitcast at alloc time |
| Import patterns | All files | No change to imports |
| Test migration volume (386 files) | Time | Mechanical; sub-agents can parallelize |

---

## 10. Verification Plan

### Per-Phase Verification

Each phase must pass before proceeding:
1. `just package` — compiles with `--Werror`
2. `just test-mojo` — all tests pass
3. No precision regressions — existing test tolerances unchanged

### Key Test Cases

```bash
# Phase 1: Basic element access
just test-group "tests/shared/core" "test_extensor_setitem.mojo"
just test-group "tests/shared/core" "test_creation_part1.mojo"

# Phase 4: Core operations
just test-group "tests/shared/core" "test_backward_*.mojo"
just test-group "tests/shared/core" "test_arithmetic*.mojo"

# Phase 5: Higher-level
just test-group "tests/shared/core" "test_activation*.mojo"
just test-group "tests/shared/core" "test_normalization*.mojo"

# Phase 7: Full suite
just test-mojo
```

### Precision Verification

The parametric refactor should IMPROVE precision:
- `tensor[i] = 3.14159` on float64 tensor → stores full Float64 precision (currently truncates to Float32)
- `tensor[i] = Float32(x)` on float32 tensor → stores exact Float32 bits (currently round-trips through Float64)

---

## 11. Corner Cases & Blockers (Deep Analysis)

This section documents findings from 4 parallel research agents that audited every AnyTensor usage pattern in the codebase and tested Mojo 0.26.1 parametric capabilities.

### 11.1 BLOCKER: No Trait Objects / Existential Types in Mojo 0.26.1

`List[TensorLike]` where `TensorLike` is a trait does NOT compile. Mojo has no vtable/fat-pointer dispatch for trait objects. Each `Tensor[DType.float32]` and `Tensor[DType.float64]` are completely distinct types that cannot be unified in a single collection.

**Impact on codebase** — these patterns ALL break:

| Pattern | Location | Why It Breaks |
|---------|----------|---------------|
| `fn parameters(self) -> List[AnyTensor]` | `traits.mojo:136,151` | Can't hold mixed-dtype params |
| `fn step(mut self, params: List[AnyTensor])` | `traits.mojo:626` | Optimizer receives mixed list |
| `var velocities: List[AnyTensor]` | `autograd/optimizers.mojo:100` | Created from `param.dtype()` at runtime |
| `fn state_dict(self) -> Dict[String, AnyTensor]` | `testing/models.mojo:934` | Mixed-dtype state |
| `var G_buffers: Dict[Int, AnyTensor]` | `autograd/optimizers.mojo:895` | Per-param dtype |
| `Dict[Int, Tuple[AnyTensor, AnyTensor]]` | `data/cache.mojo:61` | (float data, int label) tuples |
| `var data: List[AnyTensor]` / `var targets: List[AnyTensor]` | `.templates/dataset_template.mojo:22-23` | float images + int labels |
| `var train_images: AnyTensor` + `var train_labels: AnyTensor` | `training/dataset_loaders.mojo:40-43` | Different dtypes in same struct |

**Workaround**: `Variant[Tensor[DType.float32], Tensor[DType.float64], ...]` — tagged union. Verified to compile. But requires `isa[T]()` + `unsafe_get[T]()` dispatch at every access site.

### 11.2 BLOCKER: Runtime DType Determination

These patterns cannot use compile-time dtype:

| Pattern | Location | Source of Runtime DType |
|---------|----------|------------------------|
| `var dtype = _parse_tensor_dtype(dtype_str)` | `utils/file_io.mojo:357` | Loaded from file header |
| `var compute_dtype: DType` | `training/precision_config.mojo:144` | Config struct field |
| `if dtype_str == "float16": dtype = DType.float16` | `papers/_template/examples/train.mojo:138-146` | CLI argument |
| `AnyTensor(grad.shape(), grad.dtype())` | `autograd/optimizers.mojo:163` (19 sites) | Derived from another tensor |
| `zeros([channels], x.dtype())` | `normalization.mojo:120-121` (hundreds of sites) | Derived from input tensor |

The `zeros(shape, tensor.dtype())` pattern appears **hundreds of times**. With parametric AnyTensor, these become `zeros[dtype](shape)` where `dtype` is the compile-time parameter of the input tensor — this works. But file I/O and CLI paths require a dispatch fan-out at the boundary.

### 11.3 BLOCKER: AnyTensor as Struct Fields (Cascading Parameterization)

Every struct storing `AnyTensor` must either become parametric or use type erasure:

| Struct | Fields | File |
|--------|--------|------|
| `BatchNorm` | `gamma, beta, running_mean, running_var` | `layers/batchnorm.mojo:34-40` |
| `Conv2D` | `weight, bias` | `layers/conv2d.mojo:34-36` |
| `Linear` | `weight, bias` | `layers/linear.mojo:30-31` |
| `MultiHeadAttention` | `wq, wk, wv, wo` | `attention.mojo:398-404` |
| `Variable` | `data` | `autograd/variable.mojo:87` |
| `LossResult` | `loss, grad` | `autograd/functional.mojo:216-217` |
| `GradientPair/Triple/ConvGradients` | `grad_input, grad_weight, ...` | `gradient_types.mojo` |
| `Batch` | `data, labels` (DIFFERENT dtypes) | `data/_datasets_core.mojo:65-66` |
| All model structs | 10-30 weight fields each | `examples/*/model.mojo` |

**Critical**: `Batch` holds `data` (float) and `labels` (int) — cannot share a single dtype parameter.

**Critical**: `Variable` becoming `Variable[dtype]` makes the entire autograd tape parametric.

### 11.4 BLOCKER: Trait Signatures Reference AnyTensor

All 5 core traits reference `AnyTensor`:

```
Differentiable:  fn forward(input: AnyTensor) -> AnyTensor
Parameterized:   fn parameters() -> List[AnyTensor]
Model:           fn forward(input: AnyTensor) -> AnyTensor
Loss:            fn compute(pred: AnyTensor, target: AnyTensor) -> AnyTensor
Optimizer:       fn step(params: List[AnyTensor])
```

Trait methods can't be parametric on additional type parameters in Mojo 0.26.1.

### 11.5 Missing DType Guards (Bugs Found — Fix Regardless)

3 functions silently corrupt data on dtype mismatch:

| Function | File | Missing Guard |
|----------|------|---------------|
| `conv2d(x, kernel, bias)` | `conv.mojo:342-476` | kernel/bias dtype not checked vs x |
| `batch_norm2d(x, gamma, beta, ...)` | `normalization.mojo:29-341` | gamma/beta/running_* dtype not checked |
| `scaled_dot_product_attention_masked` | `attention.mojo:127-128` | Mask dtype not checked |

### 11.6 Cross-DType Policy: Hard Reject Everywhere

The audit found that **every** binary AnyTensor operation rejects mismatched dtypes with a hard error — there is zero dtype promotion in the codebase. 20+ guard sites across arithmetic, comparison, matmul, loss, optimizers, autograd. This is GOOD for the parametric migration — the compiler will enforce what runtime checks currently do.

### 11.7 Mojo 0.26.1 Parametric Capabilities (Verified by Compilation)

| Capability | Works? |
|------------|--------|
| `struct Tensor[dtype: DType]` | **Yes** |
| Conforming to traits (`Copyable`, `Movable`, custom) | **Yes** |
| `__copyinit__` / `__moveinit__` on parametric struct | **Yes** |
| `@parameter if dtype == DType.X:` in methods | **Yes** |
| Default parameter: `[dtype: DType = DType.float32]` | **Yes** |
| Type inference: `var t = zeros[DType.float32]([3])` | **Yes** |
| `sizeof[Scalar[Self.dtype]]()` | **Yes** |
| `List[TensorLike]` (trait objects) | **No** |
| `Variant[AnyTensor[float32], AnyTensor[float64]]` | **Yes** |
| Runtime dtype → compile-time parameter | **No** |
| Parametric trait methods | **No** |
| `rebind` for cross-type casting | **No** (same-type assertion only) |

### 11.8 Revised Architecture: Dual-Type Design (Tensor + AnyTensor)

Given the blockers, the migration uses **two coexisting types** following Mojo naming conventions (`AnyType`, `AnyOrigin` → `AnyTensor`):

```mojo
# Shared interface trait
trait TensorLike(Copyable, Movable):
    fn numel(self) -> Int
    fn shape(self) -> List[Int]
    fn dtype(self) -> DType
    fn ndim(self) -> Int

# Compile-time typed tensor — SIMD-like, for computation
struct Tensor[dtype: DType = DType.float32](TensorLike):
    var _data: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var _shape: List[Int]
    var _strides: List[Int]
    # ...

    fn __getitem__(self, i: Int) raises -> Scalar[Self.dtype]:
        return self._data[self._resolve_index(i)]
    # tensor[i] = value just works — same as SIMD

    fn as_any(self) -> AnyTensor:
        """Zero-copy conversion to type-erased tensor."""
        return AnyTensor(self._data.bitcast[UInt8](), Self.dtype, self._shape, ...)

    fn cast[target: DType](self) raises -> Tensor[target]:
        """Explicit dtype conversion."""
        ...

# Runtime-typed tensor — for collections, I/O, trait interfaces
struct AnyTensor(TensorLike):
    var _data: UnsafePointer[UInt8, origin=MutAnyOrigin]
    var _dtype: DType  # runtime
    var _shape: List[Int]
    var _strides: List[Int]
    # ... (current AnyTensor internals)

    fn as_tensor[dtype: DType](self) raises -> Tensor[dtype]:
        """Zero-copy conversion to typed tensor. Raises on dtype mismatch."""
        if self._dtype != dtype:
            raise Error("dtype mismatch: expected " + String(dtype) + ", got " + String(self._dtype))
        return Tensor[dtype](self._data.bitcast[Scalar[dtype]](), self._shape, ...)

    fn set(mut self, index: Int, value: Float64) raises:
        """Runtime-typed element assignment (no SIMD-like [i]= possible)."""
        ...
```

**Usage patterns**:
```mojo
# Computation — full type safety, SIMD-like assignment
var x = zeros[DType.float32]([3, 4])   # returns Tensor[DType.float32]
x[0] = 3.14159                          # just works
var y = relu(x)                          # auto-parameterized

# Collections — type-erased
var params = List[AnyTensor]()
params.append(weight.as_any())
params.append(bias.as_any())

# At boundary — one runtime check, then typed
var w = params[0].as_tensor[DType.float32]()  # raises if mismatch
w[0] = 3.14159  # now typed, just works

# I/O — runtime dtype at load, dispatch at boundary
var loaded = load_from_file("model.bin")  # returns AnyTensor
if loaded.dtype() == DType.float32:
    var t = loaded.as_tensor[DType.float32]()
    process(t)  # typed path
```

**Naming convention**:
| Type | Role | Follows |
|------|------|---------|
| `Tensor[dtype]` | Compile-time typed | `SIMD[dtype, size]` |
| `AnyTensor` | Runtime typed / type-erased | `AnyType`, `AnyOrigin` |
| `TensorLike` | Trait interface | `Writable`, `Copyable`, `Movable` |

### 11.9 Mojo 0.26.1 Features That Help the Migration

Several Mojo features significantly reduce migration friction:

**Automatic Parameterization** — Functions can accept unbound parametric types and the compiler auto-parameterizes:

```mojo
# Instead of writing:
fn relu[dtype: DType](tensor: Tensor[dtype]) -> Tensor[dtype]: ...

# You can write (compiler infers the parameter):
fn relu(tensor: Tensor) -> Tensor: ...
# Equivalent to: fn relu[dtype: DType, //](tensor: Tensor[dtype]) -> Tensor[dtype]
```

This means many function signatures won't need explicit `[dtype: DType]` — just change the type name from `AnyTensor` to `Tensor` and the compiler handles the rest.

Source: [Parameters docs](https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/parameters/index.mdx) — "Automatic parameterization" section

**Parameter Inference** — DType can be inferred from arguments:

```mojo
fn zeros[dtype: DType](shape: List[Int]) -> Tensor[dtype]: ...
var t = zeros[DType.float32]([3, 4])  # Compiler infers Tensor[DType.float32]
```

Source: [Parameters docs](https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/parameters/index.mdx) — "Parameter inference" section

**Default Parameter Values** — `Tensor[dtype: DType = DType.float32]` allows `Tensor(shape)` without explicit dtype:

```mojo
struct Tensor[dtype: DType = DType.float32]:
    ...
var t = Tensor([3, 4])  # defaults to float32
```

Source: [Parameters docs](https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/parameters/index.mdx) — "Optional parameters" section

**Conditional Conformance** — Methods can require stricter constraints on the type parameter:

```mojo
struct Tensor[dtype: DType]:
    # Only available when dtype is float:
    def to_string[FloatType: Writable, //](self: Tensor[FloatType]) -> String: ...
```

Source: [Traits docs](https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/traits.mdx) — "Conditional conformance" section

**Associated Types via comptime** — Traits can define associated types:

```mojo
trait Stacklike:
    comptime EltType: Copyable
    fn push(mut self, var item: Self.EltType): ...
```

Source: [Traits docs](https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/traits.mdx) — "Associated types" section

**Key Limitations from docs**:
- "All parameters must resolve at compile time" — no runtime-to-parameter promotion
- "Conditional conformance is limited" — compiler can't always recognize specialized conformance
- "Can't add traits to existing types" — no retroactive conformance
- "`DType.float64` isn't a type, it's a value" — can't declare `var x: DType.float64`
- Destructors can't raise errors
- `@parameter if` branches with different types need `rebind()` at the boundaries

### 11.10 Mojo 0.26.1 Documentation References

| Topic | URL |
|-------|-----|
| Parameters (compile-time, inference, auto-parameterization) | https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/parameters/index.mdx |
| Structs (parametric structs, fields) | https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/structs/index.mdx |
| Traits (conformance, associated types, conditional conformance) | https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/traits.mdx |
| Lifecycle — constructors | https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/lifecycle/life.mdx |
| Lifecycle — destructors, ASAP policy | https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/lifecycle/death.mdx |
| UnsafePointer (alloc, bitcast, origin) | https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/pointers/unsafe-pointers.mdx |
| Types (SIMD, Scalar, DType system) | https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/types.mdx |
| Metaprogramming (materialization, comptime) | https://github.com/modular/modular/blob/modular/v26.1/mojo/docs/manual/metaprogramming/materialization.mdx |

---

## 12. References

### Codebase Sources

| File | Relevance |
|------|-----------|
| `shared/core/any_tensor.mojo` | The struct — 4704 lines, 177 dtype branches, 158 bitcasts |
| `shared/core/dtype_dispatch.mojo` | Existing parametric dispatch infrastructure |
| `shared/core/dtype_ordinal.mojo` | Ordinal mapping for dispatch fan-out |
| `shared/core/conv.mojo` | Best example of fully-parametric kernel pattern |
| `shared/core/arithmetic_contiguous.mojo` | Worst dtype-branching case (40 branches) |
| `shared/core/normalization.mojo` | Struct-field propagation example |
| `shared/training/mixed_precision.mojo` | Cross-dtype edge case |
| `shared/core/traits.mojo` | Existing trait patterns |
| `shared/core/memory_pool.mojo` | Pool allocation API |
| `.claude/shared/mojo-anti-patterns.md` | UAF via bitcast documentation |

### External Sources

| Source | URL |
|--------|-----|
| Mojo SIMD documentation | https://docs.modular.com/mojo/std/builtin/simd/SIMD/ |
| Mojo parameters documentation | https://docs.modular.com/mojo/manual/parameters/ |
| Python Array API Standard | https://data-apis.org/array-api/latest/ |
| Modular upstream issue (bitcast UAF) | https://github.com/modular/modular/issues/6187 |

### Previous Work

| Item | Reference |
|------|-----------|
| PR #4997 | Short-term fix: `set()` method, `_resolve_index`, tolerance adjustments |
| Issue #4998 | This epic — parametric AnyTensor migration |
| Skill: mojo-setitem-lvalue-semantics | ProjectMnemosyne — documents the `__getitem__` lvalue discovery |
| Skill: extensor-parametric-dtype-migration | ProjectMnemosyne — architecture research |

---

## Prompt for Implementation

Use the following prompt to start each phase:

```
Implement Phase N of the AnyTensor parametric DType migration (issue #4998).

Read ~/AnyTensorRefactor.md for the full plan. Focus on Phase N only.

Key constraints:
- Tensor[dtype: DType] must behave like SIMD — tensor[i] = value just works
- No precision-losing type round-trips
- Each phase must compile and pass tests before moving to the next
- Use Podman for all builds and tests (just podman-up, just shell)
- Never use NATIVE=1
- Create a PR for each phase, linked to issue #4998

Start by reading the current state of the files listed in Phase N,
then implement the changes incrementally, verifying compilation after
each major change.
```
