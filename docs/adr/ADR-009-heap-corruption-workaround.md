# ADR-009: Heap Corruption Workaround for Mojo Runtime Bug

**Status**: Resolved (2026-03-20)

**Resolution**: Bitcast UAF fixed by ADR-013 (Mojo 0.25.x runtime fix). No active workarounds
remain in tree. Last `# ADR-009` annotation removed; all test files use standard isolation
grouping for parallelism/timeout management only.

**Date**: 2025-12-30

**Issue Reference**: [Issue #3120](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3120)

**Decision Owner**: Development Team

## Executive Summary

Mojo 0.26.1's JIT compiler had a bug where ASAP (As Soon As Possible) destruction would
destroy a tensor's backing memory before a `bitcast` write completed, causing heap corruption
after approximately 15 test functions in a single file. The workaround was to split test files
into smaller units (≤10 `test_` functions per file). This ADR is **RESOLVED** — the root cause
was identified in March 2026 and fixed without file splitting.

## Context

### Problem Statement

Running test files with more than ~15 `test_` functions caused intermittent crashes with
the signature:

```text
#0 libKGENCompilerRTShared.so+0x3cb78b
#1 libKGENCompilerRTShared.so+0x3c93c6
```

These crashes were non-deterministic and varied by test order. ASAN analysis showed
heap-use-after-free in `bitcast` operations on `AnyTensor`.

### Root Cause

The Mojo compiler's ASAP destruction optimization would free a tensor's backing memory
(via `pooled_free`) before the `bitcast` write completed. This meant:

```mojo
var raw = UnsafePointer[UInt8](bitcast=tensor._data)  # reads _data
# ASAP: tensor destroyed here, _data freed
raw[0] = value  # heap-use-after-free
```

This was later formally documented as the "bitcast UAF" and fixed in ADR-013 by replacing
`bitcast` with `set()` methods throughout `AnyTensor`.

## Decision

Applied two-layer workaround:

1. **File splitting**: Limit each test file to ≤10 `test_` functions to reduce JIT memory
   pressure and make crashes non-deterministic (didn't always trigger in small files)
2. **`continue-on-error: true`**: Added to CI jobs to allow partial test results through

## Resolution

**ADR-009 is fully resolved as of 2026-03-20.**

The root cause (bitcast UAF in `AnyTensor`) was fixed by:

- Replacing all `UnsafePointer[T](bitcast=tensor._data)` patterns with `tensor.set()` and
  `tensor.get()` methods (see [ADR-013](ADR-013-slice-view-destructor-fix.md))
- Removing `continue-on-error: true` from CI
- File splitting workaround is no longer required

The separate upstream JIT compiler non-determinism issue
(`libKGENCompilerRTShared.so` ASLR-dependent crash) was fixed in
[modular/modular#6413](https://github.com/modular/modular/issues/6413).

## Consequences

### Positive (Historical)

- Reduced crash frequency enough to make CI usable during the investigation period
- Made individual test files smaller and more focused

### Negative (Historical)

- Many test files were artificially split, creating unnecessary file proliferation
- `continue-on-error` masked real test failures

### Resolution Impact

- Test files can now be consolidated (no longer bound by ≤10 function rule)
- CI no longer masks failures via `continue-on-error`

## References

- [ADR-013](ADR-013-slice-view-destructor-fix.md): Actual root cause fix (bitcast UAF)
- [ADR-016](ADR-016-jit-crash-mitigations-consolidated-status.md): Consolidated JIT-crash mitigation status and audit closure
- [modular/modular#6413](https://github.com/modular/modular/issues/6413): upstream JIT compiler fix (AVX-512 mis-emission)

---

## Document Metadata

- **Location**: `/docs/adr/ADR-009-heap-corruption-workaround.md`
- **Status**: Resolved
- **Resolved**: 2026-03-20
- **Superseded By**: [ADR-013](ADR-013-slice-view-destructor-fix.md) (actual fix)
