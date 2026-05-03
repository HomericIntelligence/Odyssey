# ADR-013: Slice View Destructor Fix

**Status**: Accepted

**Date**: 2026-03-24

**Issue Reference**: [Issue #5056](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5056)

**Decision Owner**: ML Odyssey Team

## Executive Summary

`AnyTensor.slice()` created zero-copy views by offsetting `_data` into the parent tensor's
allocation, but `__del__` called `pooled_free` unconditionally on every tensor -- including
views. Freeing an offset pointer is undefined behavior: ASAN catches it as "bad-free", and
without ASAN it silently corrupts the heap allocator metadata. After ~15-17 test functions
with tensor allocations, the accumulated corruption triggered an abort in
`libKGENCompilerRTShared.so`. The fix is a one-line guard: check `_is_view` before freeing
`_data`.

## Context

### Problem Statement

`AnyTensor.slice()` (at `any_tensor.mojo:677`) creates a zero-copy view by computing an
offset pointer into the parent tensor's data allocation:

```mojo
var batch = data.slice(start, end)
# batch._data = data._data + (start * stride * element_size)
# batch._is_view = True
```

`AnyTensor.__del__` (at `any_tensor.mojo:491`) calls `pooled_free(self._data,
self._allocated_size)` unconditionally -- regardless of whether `_data` is an owned
allocation or an offset pointer into another tensor's allocation.

When `_is_view == True`, `_data` is **not** the pointer returned by `malloc`. It is
`parent._data + offset`. Passing this offset pointer to `free()` is undefined behavior
per the C standard. The allocator expects the exact pointer it handed out; an interior
pointer corrupts the allocator's bookkeeping metadata (free-list pointers, chunk headers,
etc.).

The corruption is silent. Each bad-free slightly damages the heap, but the allocator
continues to function. After ~15-17 test functions that allocate and destroy tensors
(each bad-free compounding the corruption), the metadata becomes inconsistent enough that
a subsequent `malloc` or `free` triggers an abort:

```text
SIGABRT in libKGENCompilerRTShared.so+0x3cb78b
```

This crash signature matches the heap-corruption bug (the bitcast UAF bug),
which led to months of misclassification as "Mojo JIT compiler flakiness."

### Minimum Reproducer

```mojo
var data = ones([8, 2, 4, 4], DType.float32)
var batch = data.slice(4, 8)  # _data = data._data + 512 bytes
# batch.__del__() -> pooled_free(offset_ptr) -> bad-free -> heap corruption
```

Under ASAN, this immediately reports:

```text
ERROR: AddressSanitizer: attempting free on address which was not malloc()-ed
```

Without ASAN, the corruption accumulates silently until the allocator aborts.

### Constraints

- `slice()` must remain zero-copy for performance (batch slicing in training loops)
- The fix must not break the existing reference-counting scheme for shared data
- Must be backward-compatible with all existing callsites

### Requirements

- Views must not free the parent's data allocation
- Owned tensors must continue to free their data on destruction
- The fix must be verifiable with ASAN to confirm no further bad-frees

## Decision

### Solution Overview

Guard the `pooled_free` call in `__del__` with an `_is_view` check. Views do not own
their `_data` pointer -- it points into the parent's allocation, which is managed by the
parent's reference count. Only non-view tensors (those that obtained `_data` from the
allocator) should free it.

### Technical Details

The fix in `AnyTensor.__del__`:

```mojo
fn __del__(owned self):
    # ... other cleanup ...
    if not self._is_view:
        pooled_free(self._data, self._allocated_size)
```

Previously, the destructor unconditionally called:

```mojo
fn __del__(owned self):
    # ... other cleanup ...
    pooled_free(self._data, self._allocated_size)  # BAD: _data may be offset pointer
```

The `_is_view` field already existed and was set correctly by `slice()`. It was simply
never consulted during destruction.

## Rationale

### Key Factors

1. **Correctness**: Freeing an offset pointer is undefined behavior. The fix eliminates
   UB from every `slice()` call in the codebase.

2. **Simplicity**: A one-line conditional is the minimal correct fix. No new fields,
   no architectural changes, no new allocation strategies required.

3. **Existing infrastructure**: The `_is_view` flag was already maintained by `slice()`.
   The only missing piece was checking it in `__del__`.

### Trade-offs Accepted

1. Views depend on the parent tensor outliving them (or the shared refcount keeping the
   data alive). This is the existing contract and is not changed by this fix.

## Consequences

### Positive

- Eliminates silent heap corruption from every `slice()` view destruction
- Fixes the "flaky" CI crashes that occurred after ~15-17 test functions
- 3-line ASAN reproducer makes regression testing trivial
- No performance cost (a single branch in the destructor)

### Negative

- None. The fix is a one-line conditional that enforces an invariant that should have
  existed from the start.

### Neutral

- The `_is_view` field was already present and correctly maintained; this fix only
  adds the destructor check

## Alternatives Considered

### Alternative 1: Make slice() return a copy instead of a view

**Description**: Allocate new memory in `slice()` and copy the data, so the returned
tensor owns its `_data` pointer.

**Pros**:

- Every tensor owns its data, simplifying lifetime management
- No view/non-view distinction needed

**Cons**:

- Defeats the purpose of zero-copy batch slicing
- Training loops slice batches every iteration; copying would add significant overhead
- Increases peak memory usage (two copies of every batch)

**Why Rejected**: Unacceptable performance regression for the primary use case (batch
slicing in training loops).

### Alternative 2: Add `_ = parent` keepalive at every slice() callsite

**Description**: Keep the parent tensor alive at each callsite by binding it to `_`,
preventing early destruction.

**Pros**:

- No changes to `AnyTensor` internals

**Cons**:

- Fragile: every callsite must remember the keepalive pattern
- Easy to forget, leading to the same bug resurfacing
- Does not fix the fundamental issue (views still call `free` on offset pointers)

**Why Rejected**: Does not address the root cause. The destructor still frees an offset
pointer; the keepalive only delays when that happens.

### Alternative 3: Store the original base pointer in view tensors

**Description**: Add a `_base_data` field that stores the original `malloc`-returned
pointer, and always free `_base_data` instead of `_data`.

**Pros**:

- Views could participate in ownership without a separate flag

**Cons**:

- Adds a field to every tensor (8 bytes per tensor), even non-views
- Complicates the ownership model -- who frees the base pointer?
- The `_is_view` flag already distinguishes views from owners; adding another field is
  redundant

**Why Rejected**: Unnecessary complexity. The `_is_view` flag is sufficient and already
exists.

## Relationship to the Heap-Corruption File-Splitting Workaround

The heap-corruption workaround (late 2025, now resolved) applied file splitting to work
around what appeared to be a distinct crash. The root cause was identified as the same
bitcast UAF fixed by this ADR.

This ADR-013 bug produces **identical crash symptoms**:

- Same crash offset: `libKGENCompilerRTShared.so+0x3cb78b`
- Same threshold: ~15-17 test functions before abort
- Same misleading signal: appears to be a Mojo runtime bug

But the root cause is entirely different:

- **Heap-corruption workaround**: Mojo compiler bug -- ASAP destruction destroys tensor before bitcast write
  completes
- **ADR-013**: Our code bug -- `__del__` frees an offset pointer that was never
  `malloc`-ed

The file-splitting workaround from the heap-corruption workaround also inadvertently masked this bug: fewer
tests per file meant fewer tensor destructions, which meant less accumulated heap
corruption before process exit.

## Implementation Plan

### Phase 1: Fix (COMPLETE)

- [x] Add `_is_view` guard in `AnyTensor.__del__`
- [x] Verify fix under ASAN (no bad-free reports)
- [x] Run full test suite without file-splitting workaround

### Success Criteria

- [x] ASAN reports zero bad-free errors on slice() view destruction
- [x] Full test suite passes without crashes
- [x] No regression in tensor memory management (owned tensors still freed correctly)

## References

### Related ADRs

- [ADR-003](ADR-003-memory-pool-architecture.md): Memory pool architecture (`pooled_free`
  is the allocator entry point affected by the bad-free)

### Related Issues

- [Issue #5056](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5056):
  Slice view destructor bug
- [Issue #2942](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2942):
  Original heap corruption report

### Affected Files

- `shared/core/any_tensor.mojo` -- `__del__` method (line ~491), `slice()` method
  (line ~677)

## Revision History

| Version | Date | Author | Changes |
| --- | --- | --- | --- |
| 1.0 | 2026-03-24 | Claude Code | Initial ADR |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-013-slice-view-destructor-fix.md`
- **Status**: Accepted
- **Review Frequency**: N/A (bug fix, no ongoing review needed)
- **Next Review**: N/A
- **Supersedes**: None
- **Superseded By**: None
