# Mojo JIT Crash: `libKGENCompilerRTShared.so` (Mojo v0.26.1)

**Tracking**: Issue #3330, follow-up from #3120

## Problem

An intermittent crash in the Mojo JIT compiler causes `mojo test` to output `execution crashed`
and exit non-zero. The crash originates in `libKGENCompilerRTShared.so`, the Mojo runtime/compiler
shared library — it is **a Mojo v0.26.1 compiler bug, not a bug in test code**.

Sample crash output:

```text
execution crashed
```

This single line is the entire output. No test names, no stack trace, no assertion failures.

## Diagnosis: Compiler Flake vs. Test Bug

The key diagnostic is **where the crash appears relative to test output**:

| Symptom | Cause |
|---------|-------|
| `execution crashed` appears **before any test output** | Compiler flake — retry |
| `execution crashed` or segfault appears **after test output** | Likely a real test bug |
| Specific assertion failure message | Real test bug — investigate |

### Sample: Compiler Flake (retry to fix)

```text
execution crashed
```

No test names printed. Nothing ran. This is the JIT crash.

### Sample: Real Test Failure (investigate code)

```text
test_forward ... PASS
test_backward ... FAIL: assertion failed at line 42
  left:  0.5
  right: 0.6
```

Tests started running, then a specific test failed with a meaningful message.

## Workaround: CI Retry Pattern

Because the crash is non-deterministic, a retry loop resolves it in practice. The crash rarely
occurs twice in a row on the same test group.

### Shell Retry Loop

```bash
for attempt in 1 2 3; do
    if just test-group "$PATH" "$PATTERN"; then
        break
    fi
    if [ $attempt -lt 3 ]; then
        echo "Attempt $attempt failed, retrying in 30s..."
        sleep 30
    fi
done
```

### GitHub Actions: `nick-fields/retry`

```yaml
- name: Run test group (with retry for JIT crash)
  uses: nick-fields/retry@v3
  with:
    timeout_minutes: 15
    max_attempts: 3
    retry_wait_seconds: 30
    command: just test-group "${{ matrix.test-group.path }}" "${{ matrix.test-group.pattern }}"
```

### Current CI Mitigation (as of Mojo v0.26.1)

The `comprehensive-tests.yml` workflow currently uses `continue-on-error: true` for the known
flaky test groups (Core Tensors, Integration Tests, Benchmarking) as a stopgap. Adding a retry
action is the recommended long-term fix while still on Mojo v0.26.1.

## Relationship to Heap Corruption Bug (ADR-009)

This crash is **distinct** from the deterministic heap corruption crash described in
[ADR-009](../adr/ADR-009-heap-corruption-workaround.md):

| | JIT Crash (this doc) | Heap Corruption (ADR-009) |
|-|---------------------|--------------------------|
| **Trigger** | Non-deterministic, intermittent | After exactly ~15 cumulative tests in one file |
| **Output** | `execution crashed` before any test runs | Crash mid-run after test output |
| **Workaround** | Retry the test run | Split files to ≤10 tests each |
| **CI fix** | `continue-on-error` or retry action | File splitting (already applied) |

Both originate in `libKGENCompilerRTShared.so` but are separate bugs with different behaviors and
different workarounds.

## Long-Term Resolution

This crash is expected to be fixed in a future Mojo release. When upgrading past v0.26.1:

1. Remove the `continue-on-error` workarounds from `comprehensive-tests.yml`
2. Remove any retry steps added for this crash
3. Verify by running `just test-group` on Core Tensors, Integration Tests, and Benchmarking
   groups 5+ times in a row — they should pass consistently without retries
4. If the crash no longer appears, the Mojo runtime fix is confirmed

## References

- [Issue #3330](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3330) — Document JIT crash workaround
- [Issue #3120](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3120) — Core Loss test crashes (follow-up context)
- [ADR-009](../adr/ADR-009-heap-corruption-workaround.md) — Heap corruption workaround (related but distinct)
- `.github/workflows/comprehensive-tests.yml` — Current `continue-on-error` mitigation
