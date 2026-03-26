# ADR-014: JIT Crash Retry Mitigation

**Status**: Accepted

**Date**: 2026-03-25

**Issue Reference**: [Issue #5108](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5108)

**Decision Owner**: Development Team

## Executive Summary

Non-deterministic `libKGENCompilerRTShared.so` JIT crashes in Mojo 0.26.1 cause ~40-60% of CI
runs to fail despite no bugs in test code. Since this is an upstream compiler bug
([modular/modular#6187](https://github.com/modular/modular/issues/6187)), we add a per-file
retry mechanism that retries only on JIT crash signatures, never on real test failures.

## Context

### Problem Statement

The Mojo 0.26.1 JIT compiler intermittently crashes with the signature:

```text
#0 libKGENCompilerRTShared.so+0x3cb78b
#1 libKGENCompilerRTShared.so+0x3c93c6
#2 libKGENCompilerRTShared.so+0x3cc397
#3 libc.so.6+0x45330
#4 <JIT-compiled code at random address>
/workspace/.pixi/envs/default/bin/mojo: error: execution crashed
```

The crash is non-deterministic: different test files crash on different runs with no code changes.
The same test file may pass or crash depending on ASLR, memory layout, and JIT cache state.

### What Has Already Been Done

| Approach | Result |
|----------|--------|
| Targeted submodule imports (ADR-009 era) | Reduced frequency but didn't eliminate |
| ADR-009 file splitting | No longer needed (bitcast UAF resolved) |
| `@always_inline` on helpers | Made things worse (reverted) |
| `continue-on-error: true` | Removed -- was masking real failures |

### Constraints

- Mojo 0.26.1 upstream bug -- cannot fix without Mojo upgrade
- Must not mask real test failures (assertion errors, compile errors)
- CI must remain a reliable signal for code quality

## Decision

Add a per-file retry wrapper (`scripts/test-with-retry.sh`) that detects JIT crashes by
checking for `execution crashed` in test output and retries the file once. Real test failures
are never retried.

### Solution Overview

**Layer 1 -- Per-file retry in `scripts/test-with-retry.sh`**:

- Runs `pixi run mojo --Werror -I "$REPO_ROOT" -I . "$test_file"` capturing output
- If test fails AND output contains `execution crashed` -> retry once
- If test fails without `execution crashed` -> fail immediately (exit 1)
- If retry also crashes -> report as persistent JIT crash (exit 2)
- Exit 0 on pass (first attempt or retry)

**Layer 2 -- Justfile integration**:

- `_test-group-inner` and `_test-mojo-inner` call the retry wrapper instead of `pixi run mojo`
  directly
- JIT crash count tracked separately in test summary output
- Failed tests annotated with `(JIT crash)` when applicable

### Technical Details

**Detection marker**: `execution crashed` -- the exact string Mojo 0.26.1 prints on JIT fault.

**MAX_RETRIES=1**: One retry is sufficient. Two consecutive JIT crashes on the same file
indicates a harder problem (possibly a real compilation issue). Configurable via
`TEST_WITH_RETRY_MAX` environment variable.

**Exit codes**:

| Code | Meaning | Retried? |
|------|---------|----------|
| 0 | Test passed | First attempt or after retry |
| 1 | Real test failure | Never |
| 2 | JIT crash persisted | Yes, once |

## Rationale

### Key Factors

1. **Correctness**: Real test failures are never masked or retried
2. **Simplicity**: Single bash script, no dependencies beyond coreutils
3. **Transparency**: Retry attempts are logged clearly in CI output
4. **Minimal blast radius**: Only affects the test execution path, no workflow structure changes

### Why Not Workflow-Level Retry?

GitHub Actions `continue-on-error` or step-level retry would retry the entire test group
(dozens of files), wasting CI minutes. Per-file retry is surgical: only the crashed file
is re-executed, and only once.

### Why MAX_RETRIES=1?

JIT crashes are transient. If a file crashes twice consecutively, it's likely a deterministic
issue (e.g., a file that always overflows the JIT buffer). Retrying more than once wastes CI
time without improving pass rates.

## Consequences

### Positive

- Reduces CI failure rate from ~40-60% to <5% (expected)
- Real test failures still block the workflow immediately
- Clear audit trail: JIT crash retries are logged with distinct messaging
- No structural changes to CI workflow

### Negative

- Slightly longer CI time for files that crash (one retry adds ~30s per file)
- Retry mechanism adds a layer of indirection in test execution

### Neutral

- Test coverage unchanged
- No changes to test code or imports

## Files Modified

| File | Change |
|------|--------|
| `scripts/test-with-retry.sh` | New -- core retry logic |
| `justfile` | `_test-group-inner` and `_test-mojo-inner` use retry wrapper |
| `.github/workflows/comprehensive-tests.yml` | Comment documenting retry mechanism |
| `docs/dev/mojo-jit-crash-workaround.md` | Added retry mitigation section |
| `tests/smoke/test_retry_script.py` | Smoke tests for retry behavior |

## References

- [Issue #5108](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5108) -- JIT crash comprehensive tracking
- [modular/modular#6187](https://github.com/modular/modular/issues/6187) -- Upstream Mojo bug
- [ADR-009](ADR-009-heap-corruption-workaround.md) -- Heap corruption workaround (resolved)
- [docs/dev/mojo-jit-crash-workaround.md](../../docs/dev/mojo-jit-crash-workaround.md) -- Import-explosion variant

---

## Document Metadata

- **Location**: `/docs/adr/ADR-014-jit-crash-retry-mitigation.md`
- **Status**: Accepted
- **Review Frequency**: When Mojo is upgraded past 0.26.1
- **Next Review**: After Mojo upgrade (see #4913)
- **Supersedes**: None
- **Superseded By**: Mojo upgrade (when available)

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-25 | Claude Code | Initial ADR documenting retry mitigation |
