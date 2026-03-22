# ADR-012: Parametric DType Tensor Architecture

**Status**: Accepted

**Date**: 2026-03-21

**Issue Reference**: [Issue #4998](https://github.com/HomericIntelligence/ProjectOdyssey/issues/4998)

**Decision Owner**: ML Odyssey Team

## Executive Summary

Split ExTensor into two types: `Tensor[dtype: DType]` (compile-time typed, SIMD-like
element access) and `AnyTensor` (runtime-typed, type-erased for collections/I/O/trait
interfaces). Both conform to a shared `TensorLike` trait and support zero-copy conversion
via `as_tensor[dtype]()` and `as_any()` with shared reference counting. The new types live
in a `shared/tensor/` package while existing files remain in `shared/core/`.

## Context

### Problem Statement

ExTensor stores `_dtype` as a **runtime** field (`var _dtype: DType`) and uses
type-erased `UnsafePointer[UInt8]` storage. This causes three categories of problems:

1. **Wrong return types**: `__getitem__` always returns `Float32` regardless of actual
   tensor dtype. On a float64 tensor, `tensor[i] = 3.14159` truncates to Float32
   precision.

2. **Dead `__setitem__`**: Mojo's `obj[i] = val` uses `__getitem__` as an lvalue, not
   `__setitem__`. All 5 `__setitem__` overloads are dead code for subscript assignment.
   The 12 `set()` overloads are a workaround introducing precision-losing Float64
   round-trips.

3. **Massive runtime branching**: 177 dtype branch checks in `extensor.mojo`, 174 in
   consumers, and 708 `_data.bitcast[T]()` calls. All of this should be monomorphized
   by the compiler instead of branching at runtime.

**Source**: `shared/core/extensor.mojo:798` (`__getitem__` returns `Float32`),
`shared/core/extensor.mojo:116` (`var _dtype: DType` runtime field),
`shared/core/extensor.mojo:922-992` (12 `set()` overloads)

### Constraints

- Mojo v0.26.1 does not support parametric trait methods, so `Module.forward()` cannot
  accept `Tensor[dtype]` in its signature
- Mojo v0.26.1 does not support variadic generic parameters (`[*Ts: Module]`)
- Mojo v0.26.1 has re-export chain limitations (#3754) that prevent transparent imports
  through intermediate `__init__.mojo` files
- `shared/__init__.mojo:82` defines `comptime Tensor = ExTensor`, creating a naming
  collision with the new `struct Tensor[dtype: DType]` (B3)
- 522 files import ExTensor across the codebase (~15,700 lines to migrate)
- Mojo uses eager monomorphization, risking binary bloat with many dtype instantiations

### Requirements

- `tensor[i] = value` must work with correct precision for all float dtypes
- `__getitem__` must return `Scalar[Self.dtype]` (compile-time typed)
- Zero-copy conversion between `Tensor[dtype]` and `AnyTensor`
- All existing tests must continue to pass during migration (backward compatibility)
- `Module` trait must remain functional (stays on AnyTensor at boundaries)
- Reference counting must be correct across type conversions (no dangling pointers)

## Decision

### Solution Overview

Introduce a dual-type system:

- **`Tensor[dtype: DType]`** -- compile-time typed tensor with
  `UnsafePointer[Scalar[Self.dtype]]` storage. Element access returns the correct type.
  Used inside layer implementations and for type-safe arithmetic.

- **`AnyTensor`** -- the existing ExTensor renamed. Runtime-typed with
  `UnsafePointer[UInt8]` storage. Used at trait boundaries (Module.forward),
  heterogeneous collections (`List[AnyTensor]`), serialization/I/O, and autograd tape.

- **`TensorLike` trait** -- shared interface (`numel`, `shape`, `dtype`, `ndim`) that
  both types conform to.

The new files live in `shared/tensor/` (tensor.mojo, tensor_traits.mojo). Existing files
remain in `shared/core/` -- no package reorganization. `comptime ExTensor = AnyTensor`
alias provides backward compatibility during the migration.

### Technical Details

#### Tensor[dtype] struct

```mojo
struct Tensor[dtype: DType = DType.float32](TensorLike):
    var _data: UnsafePointer[Scalar[Self.dtype], origin=MutAnyOrigin]
    var _shape: List[Int]
    var _strides: List[Int]
    var _numel: Int
    var _is_view: Bool
    var _refcount: UnsafePointer[Int, origin=MutAnyOrigin]
    var _allocated_size: Int
    var _original_numel_quantized: Int
    # NOTE: No _dtype field -- it's Self.dtype (compile-time parameter)

    fn __getitem__(self, index: Int) raises -> Scalar[Self.dtype]:
        """Returns the correct type at compile time -- no branching."""
        return self._data[index]  # Typed pointer, direct access

    fn as_any(self) -> AnyTensor:
        """Zero-copy conversion to runtime-typed tensor (shared refcount)."""
        ...

    fn cast[target: DType](self) raises -> Tensor[target]:
        """Element-by-element conversion to a different dtype."""
        ...
```

#### Zero-copy conversion with shared refcount (B4 fix)

```mojo
# Internal constructor for zero-copy conversion (NOT public API)
fn __init__(out self, data: UnsafePointer[Scalar[dtype]],
            shape: List[Int], strides: List[Int],
            refcount: UnsafePointer[Int], numel: Int,
            is_view: Bool, allocated_size: Int,
            original_numel_quantized: Int):
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

Both `Tensor[dtype].__del__` and `AnyTensor.__del__` decrement the shared `_refcount`
and free memory only when it reaches 0. This prevents dangling pointers when either
side is destroyed by Mojo's ASAP destruction.

#### Typed pointer arithmetic (H1)

```mojo
# WRONG -- typed pointers auto-scale, this double-counts:
result._data = self._data + start * strides[axis] * dtype_size

# CORRECT -- typed UnsafePointer[Scalar[dtype]] auto-scales by element size:
result._data = self._data + start * strides[axis]
```

#### Module boundary pattern (H7)

```mojo
struct Linear[dtype: DType = DType.float32]:
    var weight: Tensor[dtype]
    var bias: Tensor[dtype]

    # Module trait requires AnyTensor signature
    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        # Convert at boundary: AnyTensor -> Tensor[dtype]
        var typed_input = input.as_tensor[dtype]()
        # Type-safe computation inside
        var result = matmul[dtype](typed_input, self.weight)
        result = add[dtype](result, self.bias)
        # Convert back at boundary: Tensor[dtype] -> AnyTensor
        return result.as_any()
```

#### Factory functions (B1 fix)

```mojo
# Auto-parameterization does NOT work for return types:
fn relu(t: Tensor) -> Tensor: ...  # FAILS: "failed to infer parameter 'dtype'"

# All functions need explicit [dt: DType] parameter:
fn relu[dt: DType](t: Tensor[dt]) -> Tensor[dt]: ...  # WORKS
fn zeros[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]: ...
```

#### Package structure (no physical reorganization)

```text
shared/
    tensor/              # NEW -- only 3 new files
        __init__.mojo
        tensor.mojo      # struct Tensor[dtype: DType]
        tensor_traits.mojo  # trait TensorLike
    core/                # UNCHANGED -- all existing files stay here
        extensor.mojo    # struct AnyTensor (renamed from ExTensor)
        ...              # all existing operation files unchanged
```

## Rationale

### Key Factors

1. **Type safety**: `Tensor[dtype]` catches type mismatches at compile time. Cross-dtype
   `Tensor[f32] + Tensor[f64]` is a compile error, not a silent precision loss.

2. **Precision correctness**: `tensor[i] = 3.14159` on a float64 tensor stores full
   Float64 precision instead of truncating to Float32.

3. **SIMD-like ergonomics**: `Tensor[dtype]` follows Mojo's own SIMD design pattern
   (`SIMD[DType.float32, 4]`). `__getitem__` returns the correct scalar type, and
   `tensor[i] = value` just works via lvalue semantics.

4. **Zero-cost abstraction**: Monomorphization eliminates all 177 runtime dtype branches
   inside Tensor[dtype] methods. The compiler generates optimized code for each
   instantiated dtype.

5. **Mojo naming conventions**: `AnyTensor` follows Mojo stdlib patterns (`AnyType`,
   `AnyOrigin`, `AnyPointer`) for type-erased wrappers.

### Trade-offs Accepted

1. **~15,700 lines to migrate** across 522 files for full precision correctness and
   type safety. Mitigated by `comptime ExTensor = AnyTensor` alias providing backward
   compatibility during the 11-PR migration.

2. **Module stays on AnyTensor**: Type safety benefits apply only INSIDE individual
   layer implementations, not across the Module composition boundary. This is a Mojo
   v0.26.1 limitation -- when parametric trait methods are added, `TypedModule[dtype]`
   can be introduced.

3. **Binary bloat risk**: Each Tensor[dtype] instantiation generates separate code.
   Mitigated by defaulting to `DType.float32` and monitoring compile times. Most ML
   workloads use 1-3 float types.

4. **No package reorganization**: Keeping files in `shared/core/` avoids 500+ import
   path changes but means the 3-layer architecture exists logically, not physically.
   This is the pragmatic choice given Mojo's re-export chain limitation (#3754).

## Consequences

### Positive

- `tensor[i] = value` works with correct precision for all dtypes
- 177 runtime dtype branches eliminated inside Tensor[dtype]
- 708 `_data.bitcast[T]()` calls reduced to 1 per allocation
- 12 `set()` overloads can be removed (no longer needed)
- Compile-time type checking prevents silent precision loss
- Foundation for future `TypedModule[dtype]` when Mojo adds parametric trait methods

### Negative

- ~15,700 lines of migration work across 11 PRs
- Module/Sequential stays on AnyTensor, requiring boundary conversions
- Potential compilation time increase from monomorphization
- Two tensor types to maintain (Tensor[dtype] and AnyTensor)

### Neutral

- AnyTensor preserves ALL current functionality unchanged
- Memory pool is dtype-agnostic (no changes needed)
- Existing tests pass unchanged via `comptime ExTensor = AnyTensor` alias

## Alternatives Considered

### Alternative 1: Keep ExTensor runtime-typed with improved set()

**Description**: Keep the current architecture but add better `set()` overloads that
avoid precision loss.

**Pros**:

- Zero migration cost
- No new types to maintain

**Cons**:

- Does not fix `__getitem__` return type (still Float32)
- `tensor[i] = value` still broken (Mojo lvalue semantics)
- 177 runtime dtype branches remain
- Non-standard `set()` API forever

**Why Rejected**: Does not solve the fundamental problem -- `__getitem__` always returns
Float32 regardless of actual dtype, and Mojo's lvalue semantics prevent `__setitem__`
from working.

### Alternative 2: Single parametric type with Variant for collections

**Description**: Replace ExTensor with only `Tensor[dtype]`, using `Variant` for
heterogeneous collections.

**Pros**:

- Single type, simpler mental model
- Full type safety everywhere

**Cons**:

- `Variant[Tensor[f16], Tensor[f32], Tensor[f64], ...]` is unwieldy
- Module trait cannot use parametric types (Mojo 0.26.1 limitation)
- Loses ergonomics for collections, autograd tape, I/O

**Why Rejected**: Too complex. The Variant approach loses the ergonomics of a single
collection type and doesn't solve the Module trait boundary problem.

### Alternative 3: Python-style duck typing with runtime checks

**Description**: Keep runtime typing but add runtime assertions for type mismatches.

**Pros**:

- Minimal code changes
- Catches mismatches at runtime

**Cons**:

- Does not leverage Mojo's compile-time type system
- No performance benefit (runtime branches remain)
- Errors caught at runtime, not compile time

**Why Rejected**: Doesn't leverage Mojo's strengths. The whole point of Mojo is
compile-time type safety and zero-cost abstractions.

## Implementation Plan

The migration is structured as 11 PRs with clear dependencies. See
`docs/dev/extensor-refactor-plan.md` for the complete 17 sub-phase plan with per-file
details.

### Phase 1: Foundation (PRs 1-2)

- [ ] ADR-012 + design documentation
- [ ] Tensor[dtype] struct + TensorLike trait + AnyTensor rename + B3/B4 fixes

### Phase 2: Core Operations (PRs 3-5)

- [ ] Typed factory functions (zeros[dtype], ones[dtype], etc.)
- [ ] Shape + arithmetic ops with Tensor[dtype] overloads
- [ ] Elementwise/activation + matrix/reduction/conv ops

### Phase 3: Layer System (PRs 6-7)

- [ ] Parameterize non-Module layers (BatchNorm, Conv2d, Dropout)
- [ ] Update trait signatures to AnyTensor + parameterize Module layers

### Phase 4: Infrastructure (PRs 8-9)

- [ ] Collection ops + optimizer/variable migration to AnyTensor
- [ ] Training/autograd/data pipeline migration

### Phase 5: Testing and Cleanup (PRs 10-11)

- [ ] Test migration (~576 files, 4 parallel sub-PRs)
- [ ] Final cleanup: remove aliases, rename files, remove dead code

### Success Criteria

- [ ] `tensor[i] = 3.14159` on float64 tensor preserves full Float64 precision
- [ ] All existing tests pass (no regressions)
- [ ] `just package` compiles without errors
- [ ] `just test-mojo` passes all tests
- [ ] Zero-copy as_tensor/as_any conversion is safe (B4 refcount test passes)
- [ ] No precision-losing Float64 round-trips in Tensor[dtype] operations
- [ ] Module/Sequential functional with AnyTensor boundary conversions

## References

### Related ADRs

- [ADR-009](ADR-009-heap-corruption-workaround.md): Heap corruption workaround
  (test file splitting) -- relevant for test migration phases

### Related Issues

- [Issue #4998](https://github.com/HomericIntelligence/ProjectOdyssey/issues/4998):
  Epic tracking the full migration
- [PR #4997](https://github.com/HomericIntelligence/ProjectOdyssey/pull/4997):
  Short-term precision fix (set() method improvements)
- [PR #5001](https://github.com/HomericIntelligence/ProjectOdyssey/pull/5001):
  DType guards for conv2d, batch_norm2d, attention

### External Documentation

- [Mojo SIMD Documentation](https://docs.modular.com/mojo/std/builtin/simd/SIMD/):
  Target behavior for Tensor[dtype]
- [ExTensor Refactor Plan](../dev/extensor-refactor-plan.md): Complete 17 sub-phase
  migration plan with review findings

## Revision History

| Version | Date       | Author         | Changes     |
| ------- | ---------- | -------------- | ----------- |
| 1.0     | 2026-03-21 | ML Odyssey Team | Initial ADR |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-012-parametric-dtype-tensor-architecture.md`
- **Status**: Accepted
- **Review Frequency**: As-needed
- **Next Review**: After Phase 5 completion
- **Supersedes**: None
- **Superseded By**: None
