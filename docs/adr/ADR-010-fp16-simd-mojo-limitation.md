# ADR-010: FP16 SIMD Limitation in Mojo v0.26.1

**Status**: Accepted

**Date**: 2026-03-07

**Issue Reference**: [Issue #3291](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3291)

**Decision Owner**: Development Team

## Executive Summary

Mojo v0.26.1 does not support FP16 as a SIMD element type. The `SIMD[DType.float16, N]`
vectorized load/store fails to compile. Mixed-precision training functions that convert
between FP16 and FP32 use scalar loops as a workaround until Mojo adds FP16 SIMD support.

## Context

### Problem Statement

Mixed-precision training requires efficient FP16↔FP32 conversion in two functions:

- `convert_to_fp32_master()`: converts FP16 model params → FP32 master weights before optimizer step
- `update_model_from_master()`: copies FP32 master weights → FP16 model params after optimizer step

The natural implementation uses SIMD vectorization for ~4x throughput (matching the FP32→FP32 path).
However, `SIMD[DType.float16, N]` is not a valid type in Mojo v0.26.1: the compiler rejects it with
a type error, preventing SIMD load/store on FP16 buffers.

### Key Findings

1. **Compiler limitation, not a design choice** — `SIMD[DType.float16, N]` fails to compile
2. **FP32→FP32 SIMD path works correctly** — The vectorized path is proven and used elsewhere
3. **Scalar loop is correct and safe** — One-element-at-a-time conversion produces correct results
4. **Performance penalty is bounded** — FP16 conversion is ~10-15x slower than FP32→FP32 SIMD
5. **Tracked in issue #3015** — No upstream Mojo issue filed yet

### Constraints

- Cannot use SIMD vectorization for FP16 without Mojo compiler support
- Mixed-precision training must remain functional for correctness
- Scalar workaround is acceptable until Mojo adds FP16 SIMD support

### Requirements

- `convert_to_fp32_master()` must correctly convert FP16→FP32 for all valid inputs
- `update_model_from_master()` must correctly convert FP32→FP16 for all valid inputs
- Performance regression must be documented and bounded

## Decision

Use scalar element-by-element loops for FP16↔FP32 conversion paths in
`convert_to_fp32_master()` and `update_model_from_master()`.

### Solution Overview

Both functions branch on `dtype()`:

- **FP32→FP32**: SIMD vectorized path (`_convert_fp32_to_fp32_simd`, `_update_fp32_from_fp32_simd`)
- **FP16↔FP32**: Scalar loop (`for i in range(size): dst_ptr[i] = cast(src_ptr[i])`)
- **Other dtypes**: Generic scalar path via `_get_float64` / `_set_float64`

The scalar FP16 paths are in `shared/training/mixed_precision.mojo`.

### Technical Details

Scalar conversion pattern used:

```mojo
# FP16 SIMD blocked by Mojo v0.26.1 limitation; using scalar loop.
# See docs/adr/ADR-010-fp16-simd-mojo-limitation.md for rationale.
var src_ptr = params._data.bitcast[Float16]()
var dst_ptr = result._data.bitcast[Float32]()
for i in range(size):
    dst_ptr[i] = Float32(src_ptr[i])
```

When Mojo adds FP16 SIMD support, replace the scalar loop with:

```mojo
alias simd_width = simdwidthof[Float16]()
for i in range(0, size, simd_width):
    var v = src_ptr.load[width=simd_width](i).cast[DType.float32]()
    dst_ptr.store[width=simd_width](i, v)
```

## Rationale

### Key Factors

1. **Correctness over performance**: Scalar loop produces identical numeric results
2. **Minimal change**: Scalar loop requires no structural changes to the function
3. **Clear supersession path**: When Mojo adds FP16 SIMD, the fix is a targeted replacement
4. **Unblocks mixed-precision training**: FP16 training works correctly with scalar path

### Trade-offs Accepted

1. FP16↔FP32 conversion is ~10-15x slower than FP32→FP32 SIMD path
2. All other paths (FP32→FP32, generic) are unaffected

## Consequences

### Positive

- Mixed-precision training is fully functional
- Code is correct and well-tested
- Limitation is centrally documented (this ADR)

### Negative

- FP16↔FP32 conversion is ~10-15x slower than the vectorized path
- Performance gap grows with tensor size

### Neutral

- FP32-only training is unaffected (SIMD path still used)
- BF16 and other dtype paths use the generic scalar fallback regardless

## Alternatives Considered

### Alternative 1: Keep duplicate inline comments

**Description**: Document the limitation in each function's docstring and cross-reference
between functions (prior approach before this ADR).

**Pros**:

- No separate ADR file needed
- Context visible in editor while reading the function

**Cons**:

- Duplicate documentation must be kept in sync across two functions
- Harder to find when Mojo adds FP16 SIMD support (no single place to update)
- Verbose docstrings obscure the key implementation detail

**Why Rejected**: Centralizing in an ADR is the established project pattern and
reduces maintenance burden.

### Alternative 2: Python interop for FP16 conversion

**Description**: Call Python's NumPy for FP16↔FP32 conversion via interop.

**Pros**:

- NumPy has optimized FP16 paths

**Cons**:

- Adds Python/NumPy dependency to the hot training path
- Interop overhead likely exceeds scalar loop overhead for small tensors
- Contradicts Mojo-first language policy (ADR-001)

**Why Rejected**: Unnecessary complexity; scalar loop is simpler and sufficient.

### Alternative 3: Wait for Mojo FP16 SIMD support

**Description**: Do not implement FP16 mixed-precision training until Mojo supports FP16 SIMD.

**Pros**:

- No workaround needed

**Cons**:

- Blocks all FP16 mixed-precision training work
- Timeline for Mojo FP16 SIMD support is unknown

**Why Rejected**: Mixed-precision training is a required feature; correctness is sufficient
to proceed.

## Implementation Plan

### Phase 1: Consolidate documentation (COMPLETE)

- [x] Create this ADR as single source of truth
- [x] Replace verbose inline comments in `convert_to_fp32_master()` with ADR cross-reference
- [x] Replace verbose inline comment in `update_model_from_master()` with ADR cross-reference
- [x] Add ADR-010 to `docs/adr/README.md` index

### Phase 2: Supersession (FUTURE — when Mojo adds FP16 SIMD)

- [ ] Implement SIMD vectorized FP16↔FP32 paths
- [ ] Mark this ADR as Superseded
- [ ] Remove scalar workaround comments
- [ ] Verify ~4x speedup matches FP32→FP32 SIMD performance

### Success Criteria

- [x] `convert_to_fp32_master()` FP16 path produces correct results
- [x] `update_model_from_master()` FP16 path produces correct results
- [x] Limitation documented in single canonical location

## References

### Related Issues

- [Issue #3291](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3291):
  Consolidate FP16 SIMD limitation references (this ADR)
- [Issue #3072](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3072): Mixed-precision conversion implementation
- [Issue #3015](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3015): FP16 SIMD limitation tracking

### Affected Files

- `shared/training/mixed_precision.mojo` — `convert_to_fp32_master()` and `update_model_from_master()`

### Related ADRs

- [ADR-001](ADR-001-language-selection-tooling.md): Language selection — Mojo-first policy

## Revision History

| Version | Date       | Author      | Changes                          |
| ------- | ---------- | ----------- | -------------------------------- |
| 1.0     | 2026-03-07 | Claude Code | Initial ADR documenting FP16 SIMD limitation and scalar workaround |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-010-fp16-simd-mojo-limitation.md`
- **Status**: Accepted
- **Review Frequency**: As-needed (review on Mojo upgrade)
- **Next Review**: On Mojo 0.27+ upgrade (when FP16 SIMD may be available)
- **Supersedes**: None
- **Superseded By**: None (will be superseded when Mojo adds FP16 SIMD support)
