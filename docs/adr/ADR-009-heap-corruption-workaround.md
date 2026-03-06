# ADR-009: Heap Corruption Workaround for Mojo Runtime Bug

**Status**: Accepted

**Date**: 2025-12-30

**Issue Reference**: [Issue #2942](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2942)

**Decision Owner**: Development Team

## Executive Summary

A heap corruption bug in Mojo 0.26.1 runtime causes crashes after running approximately 15 cumulative
layer tests in a single file. The workaround splits test files to have fewer than 10 tests each,
ensuring we stay well below the crash threshold.

## Context

### Problem Statement

Memory allocation crashes occur with heap corruption after running exactly 15 layer tests
cumulatively. The crash happens during `alloc[UInt8]()` in `ExTensor.__init__` when allocating
a small 1600-byte tensor, despite all previous allocations/deallocations completing successfully.

The crash manifests in `libKGENCompilerRTShared.so`, indicating a Mojo runtime/compiler issue
rather than a bug in our code.

### Key Findings

1. **NOT a bug in layer implementations** - Same operations work in different files
2. **NOT a bug in ExTensor memory management** - All allocations/frees tracked correctly
3. **Cannot create minimal reproduction** - Tried 17 different isolated test cases
4. **Requires exact sequence of 15 specific tests** - Any subset <15 works fine
5. **Heap corruption in Mojo allocator** - Crash in `libKGENCompilerRTShared.so`
6. **`^` operator in `__moveinit__` is NOT a fix** - Switching `_shape` and `_strides`
   from `.copy()` to `^` corrupts list element values when the source List has been
   mutated in-place (e.g. by `slice()`). The `.copy()` workaround in `__moveinit__`
   is correct and necessary for Mojo 0.26.1.

### Constraints

- Mojo 0.26.1 runtime bug - Cannot fix without Mojo upgrade
- Tests must still provide comprehensive coverage
- CI pipeline must remain reliable

### Requirements

- All layer tests must pass
- CI must not experience random crashes
- Tests must remain maintainable

## Decision

Split test files to contain fewer than 10 tests each, well below the observed crash threshold
of ~15 tests.

### Solution Overview

The monolithic `test_lenet5_layers.mojo` file was split into 5 smaller files:

1. `test_lenet5_conv_layers.mojo` - 6 tests
2. `test_lenet5_activation_layers.mojo` - 3 tests
3. `test_lenet5_pooling_layers.mojo` - 4 tests
4. `test_lenet5_fc_layers.mojo` - 9 tests
5. `test_lenet5_reshape_layers.mojo` - 2 tests

### Technical Details

Each split file follows the pattern:

```mojo
"""LeNet-5 [Layer Type] Layer Tests

Tests for [layer type] layers only.

Note: Split from monolithic test file due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See Issue #2942.
"""
```

## Rationale

### Key Factors

1. **Reliability**: Split files eliminate crashes entirely
2. **Minimal code change**: Only file organization changed, not test logic
3. **CI stability**: Each test file runs independently without issues

### Trade-offs Accepted

1. Multiple files instead of single file - Increased navigation overhead
2. Threshold buffer (10 tests) instead of exact (15 tests) - Safety margin
3. No true fix - Waiting for Mojo upgrade to resolve root cause

## Consequences

### Positive

- All tests pass reliably
- CI pipeline stable
- No changes to test logic required
- Clear documentation of issue

### Negative

- More test files to maintain
- Must remember limit when adding new tests
- Root cause unfixed (Mojo runtime bug)

### Neutral

- Test coverage unchanged
- Test execution time unchanged

## Alternatives Considered

### Alternative 1: Wait for Mojo fix

**Description**: Don't implement workaround, wait for Mojo 0.27+ to fix the bug.

**Pros**:

- No workaround needed
- Cleaner codebase

**Cons**:

- CI remains broken
- Unknown timeline for fix
- Blocks development

**Why Rejected**: Unacceptable to have broken CI indefinitely.

### Alternative 2: Run tests in separate processes

**Description**: Use a test runner that spawns separate processes for each test.

**Pros**:

- No file splitting needed
- Process isolation prevents corruption

**Cons**:

- Requires custom test runner
- Increased complexity
- Slower execution

**Why Rejected**: More complex than simple file splitting.

## Implementation Plan

### Phase 1: Split Test Files (COMPLETE)

- [x] Split `test_lenet5_layers.mojo` into 5 files
- [x] Rename original to `.DEPRECATED`
- [x] Verify all split files pass
- [x] Update CI to discover split files

### Phase 2: Safeguards (OPTIONAL)

- [ ] Add validation script for test file sizes
- [ ] Add pre-commit hook to check test file sizes
- [ ] Document limit in CLAUDE.md

### Success Criteria

- [x] All 24 LeNet-5 tests pass
- [x] No heap corruption crashes
- [x] CI pipeline stable

## References

### Related Issues

- [Issue #2942](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2942): Heap corruption bug report
- [Issue #2705](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2705): Flatten tests (closed)
- [Issue #2702](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2702): FC backward tests (closed)

### Affected Files

- `tests/models/test_lenet5_conv_layers.mojo`
- `tests/models/test_lenet5_activation_layers.mojo`
- `tests/models/test_lenet5_pooling_layers.mojo`
- `tests/models/test_lenet5_fc_layers.mojo`
- `tests/models/test_lenet5_reshape_layers.mojo`
- `tests/models/test_lenet5_layers.mojo.DEPRECATED`

## Revision History

| Version | Date       | Author      | Changes                                                              |
| ------- | ---------- | ----------- | -------------------------------------------------------------------- |
| 1.0     | 2025-12-30 | Claude Code | Initial ADR documenting workaround                                   |
| 1.1     | 2026-03-06 | Claude Code | Investigation: `^` in `__moveinit__` corrupts slice shapes in Mojo  |
|         |            |             | 0.26.1 — workaround (file splitting + `.copy()`) remains correct.   |
|         |            |             | Also fixes `mojo format` line-length in `layer_testers.mojo`.       |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-009-heap-corruption-workaround.md`
- **Status**: Accepted
- **Review Frequency**: As-needed (review on Mojo upgrade)
- **Next Review**: On Mojo 0.27+ upgrade
- **Supersedes**: None
- **Superseded By**: None (will be superseded when Mojo fix is available)
