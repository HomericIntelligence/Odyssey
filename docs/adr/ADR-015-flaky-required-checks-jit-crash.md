# ADR-015: Flaky Required Checks -- JIT Crash in Core Types & Fuzz and Integration Tests

**Status**: Accepted -- Corrective Actions Open

**Date**: 2026-04-12

**Issue Reference**: [Issue #5108](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5108)

**Decision Owner**: Development Team

## Executive Summary

Non-deterministic `libKGENCompilerRTShared.so` JIT crashes in Mojo 0.26.3 cause the two
required CI checks -- `Core Types & Fuzz` and `Integration Tests` -- to fail on the majority
of `main` runs, blocking all PR auto-merges including pure-docs changes. The root cause is
JIT compilation volume overflow, not test code bugs. We commit to removing the per-file retry
mitigation (which is SUPERSEDED in ADR-014 but still wired in tree) and executing a targeted
import audit on the two failing test groups as the correct long-term fix.

## Context

### Problem Statement

Evidence collected 2026-04-12 from three consecutive `main` runs:

| Run ID | SHA | Failed Groups |
| --- | --- | --- |
| 24307240107 | e3a6690a6 | Core Utilities A/B, Configs, Benchmarks, Core Types & Fuzz, Integration Tests |
| 24307200634 | 1ab0ebb3b | Gradient Checking Tests, Core Gradient, Core Utilities A/B/C, Models |
| 24307185935 | c1f76120d | Gradient Checking, Data Utilities, Core Utilities A/B/C, Data Transforms, Examples, Misc Tests, Core Types & Fuzz |

Every run fails in a **different random subset** of groups with no code changes between them.
Pure-docs PRs #5219 and #5224 exhibited identical crash patterns, proving the failures are
not introduced by any PR change.

The required status checks in branch protection are `Core Types & Fuzz` and `Integration
Tests`. Because these groups fail on most runs, no PR can reliably auto-merge.

> **Note on stale workflow comments**: `.github/workflows/comprehensive-tests.yml` lines
> 316-318 and 356-358 contain inline comments labeling these two matrix entries as
> "non-blocking pending investigation". That characterization is stale -- branch protection
> is authoritative. Corrective action #3 removes the stale comments to align workflow
> source with reality.

### Crash Signature

All observed failures produce `execution crashed` as the **only output** -- no test names,
no assertion results. Per the diagnostic table in
[`docs/dev/mojo-jit-crash-workaround.md`](../dev/mojo-jit-crash-workaround.md)
(`## Diagnosis: Compiler Flake vs. Test Bug`):

| Symptom | Cause |
| --- | --- |
| `execution crashed` **before any test output** | JIT compilation volume overflow |
| `execution crashed` or segfault **after test output** | Likely a real test bug |

The observed pattern -- crash with zero preceding test output -- is the unambiguous signature
of a JIT compilation volume overflow, not a test correctness issue.

### Impact

- Every PR is blocked from auto-merging until a clean CI run clears the required checks
- Required re-runs (often 2-4x) consume CI budget without producing new signal
- Pure-docs and non-code changes are blocked at the same rate as functional changes
- Developer trust in CI is eroded when `main` itself shows the same failure rate

### What Has Already Been Done

| Approach | Status | Reference |
| --- | --- | --- |
| Heap corruption workaround (file splitting) | Resolved 2026-03-20 | ADR-009 |
| `continue-on-error: true` in workflow | Removed -- was masking real failures | ADR-009 |
| Targeted submodule import audit (126 test files) | Applied -- reduced but did not eliminate | [mojo-jit-crash-workaround.md](../dev/mojo-jit-crash-workaround.md) |
| Per-file retry script (`scripts/test-with-retry.sh`) | ADR-014 status: SUPERSEDED; script still wired in justfile | ADR-014 |
| Minimal reproducer + upstream issue filing | In progress | `repro/issues/jit-compilation-volume-crash.md` |

**Retry script drift**: ADR-014 prose says "the retry workaround has been removed", but
`scripts/test-with-retry.sh` still exists and is still called from `justfile:664`
(`_test-group-inner`) and `justfile:829` (`_test-mojo-inner`). The script defaults to
`MAX_RETRIES=1` via `TEST_WITH_RETRY_MAX` (unset everywhere in CI). This ADR resolves
the drift by committing to actually removing the script.

## Root Cause Analysis

The JIT crash is triggered by **compilation footprint** per
[`docs/dev/mojo-jit-crash-workaround.md`](../dev/mojo-jit-crash-workaround.md):

- Package-level `from shared.core import` forces the Mojo JIT to compile all 37,401 lines
  across 60+ source files (because `shared/core/__init__.mojo` eagerly re-exports 200+
  symbols from 40+ modules)
- This compilation volume intermittently overflows a JIT-internal buffer, triggering
  `__fortify_fail_abort` in `libKGENCompilerRTShared.so`
- The crash is non-deterministic because ASLR, memory layout, and JIT caching vary per run

The same root cause is documented under Mojo 0.26.3 in
`repro/issues/jit-compilation-volume-crash.md` (environment: Ubuntu 24.04, GLIBC 2.39).

**Why the per-file retry was insufficient**:

The retry script catches a crash in a single test file and re-executes that one file.
However, a CI matrix job (`Core Types & Fuzz`, `Integration Tests`) compiles and runs many
test files in sequence. If two independent files both trigger the JIT overflow in the same
job run, the job fails even with `MAX_RETRIES=1`. At the observed crash rates (~40-60%
of runs have at least one crash per group), two-crash scenarios occur frequently enough
to keep the required checks in a near-permanent failed state.

More fundamentally, retries mask the problem: they reward package-level imports by hiding
their cost and defer the import audit that is the correct fix.

**Upstream issue references**: ADR-014 and `mojo-jit-crash-workaround.md` cite
[modular/modular#6187](https://github.com/modular/modular/issues/6187);
`scripts/test-with-retry.sh` header cites
[modular/modular#6413](https://github.com/modular/modular/issues/6413).
Corrective action #5 includes confirming the canonical upstream issue number.

## Decision

### Solution Overview

1. **Remove the retry mitigation** -- delete `scripts/test-with-retry.sh` and replace its
   two justfile call sites with a direct `pixi run mojo` invocation. This makes ADR-014's
   SUPERSEDED status true in the tree and forces the import audit to be the fix.

2. **Audit imports in the two required-check test groups** -- convert any remaining
   package-level `from shared.core import` statements in `tests/core/types/` and
   `tests/shared/integration/` to targeted submodule imports per the Symbol-to-Submodule
   Mapping in `docs/dev/mojo-jit-crash-workaround.md`. Expected crash-rate reduction: ~95%.

3. **Do NOT remove the checks from `required_status_checks`** -- keeping them required
   maintains CI signal integrity. Removal is reserved as a fallback if action #1 alone
   does not move crash rate below 5% after a two-week observation window (see action #4).

### Technical Details

**Justfile sites to replace** when removing the retry script:

```text
justfile:664  (_test-group-inner):  bash "$REPO_ROOT/scripts/test-with-retry.sh" "$REPO_ROOT" "$test_file"
justfile:829  (_test-mojo-inner):   bash "$REPO_ROOT/scripts/test-with-retry.sh" "$REPO_ROOT" "$test_file"
```

Both should become direct invocations:

```bash
pixi run mojo --Werror -I "$REPO_ROOT" -I . "$test_file"
```

**Workflow matrix entries with stale comments to clean up** in
`.github/workflows/comprehensive-tests.yml`:

```text
lines 316-318  (Integration Tests):    comment "Integration tests (segfault-prone on CI, kept non-blocking)"
lines 356-358  (Core Types & Fuzz):    comment "Mojo JIT crashes on CI runners -- non-blocking pending investigation."
```

Both comments should be removed. The checks are required; the "non-blocking" framing is
incorrect and misleading.

## Corrective Actions

1. **[HIGH] Import audit** -- Enumerate test files in `tests/core/types/test_*.mojo` and
   `tests/shared/integration/test_*.mojo`. Convert any `from shared.core import` (package-level)
   to targeted submodule imports using the Symbol-to-Submodule Mapping table in
   `docs/dev/mojo-jit-crash-workaround.md`. Expected: reduces per-file JIT crash
   probability by ~95% based on the controlled experiment from 2026-03-15.
   **Verification**: Run CI on a branch with only import changes; compare crash rate over
   5+ runs.

2. **[HIGH] Remove retry script** -- Delete `scripts/test-with-retry.sh`. Replace its two
   justfile call sites (`_test-group-inner`, `_test-mojo-inner`) with a direct `pixi run mojo`
   invocation. Also remove `tests/smoke/test_retry_script.py` (validates the now-deleted
   script). This reconciles ADR-014's SUPERSEDED prose with the actual state of the tree.
   **Verification**: `git ls-files scripts/test-with-retry.sh` returns empty; CI runs
   do not reference the script; justfile `just test-group` completes without calling the
   wrapper.

3. **[LOW] Remove stale workflow comments** -- Edit `comprehensive-tests.yml:316-318` and
   `:356-358` to remove the "non-blocking pending investigation" / "segfault-prone,
   kept non-blocking" comments. The branch protection rules are authoritative; having
   contradictory "non-blocking" comments in workflow source creates confusion.
   **Verification**: `grep -n "non-blocking" .github/workflows/comprehensive-tests.yml`
   returns empty.

4. **[LOW] Fallback: remove from required checks** -- If crash rate does not fall below 5%
   within two weeks of completing action #1, temporarily remove `Core Types & Fuzz` and
   `Integration Tests` from `required_status_checks`. Unblocks the PR pipeline immediately
   but risks masking genuine failures. Should be paired with a Slack/GitHub notification
   rule to surface group failures during the non-required window.
   **Trigger**: >5% crash rate over a two-week post-audit observation window.

5. **[FUTURE] Track upstream Mojo fix** -- Confirm the canonical upstream issue
   (`modular/modular#6187` vs `#6413`; references are inconsistent across ADR-014,
   `mojo-jit-crash-workaround.md`, and `scripts/test-with-retry.sh`). Update
   `repro/issues/jit-compilation-volume-crash.md` with the 0.26.3 environment details
   already documented there. Revisit this ADR when Mojo is upgraded past 0.26.3.

## Alternatives Considered

### Alternative 1: Raise MAX_RETRIES to 2

Set `TEST_WITH_RETRY_MAX=2` via env var for the two required-check matrix entries in
`comprehensive-tests.yml`. This would absorb two-consecutive-crash scenarios.

**Why rejected**: Masking is not fixing. A higher retry budget defers the import audit
indefinitely, consumes extra CI minutes on known-flaky groups, and keeps a SUPERSEDED
mitigation alive in the tree. The user has explicitly directed removal of the retry
mechanism. Import hygiene is the correct fix.

### Alternative 2: Workflow-level job retry

Use GitHub Actions' built-in job retry (`retry-on-failure` or `continue-on-error`) to
re-run the entire matrix job on failure.

**Why rejected**: Retries the entire group (dozens of files passing cleanly) for each
crash. Wastes CI budget. Same fundamental masking problem as Alternative 1.

### Alternative 3: Remove from required_status_checks immediately

Remove the two checks from branch protection now, before any import audit.

**Why rejected**: Masks genuine test failures in the short window before the audit lands.
Reserved as fallback in action #4 only if the import audit does not move the needle.

### Alternative 4: Pin Mojo below 0.26.3

Downgrade to an earlier Mojo version where this crash did not manifest.

**Why rejected**: Other landed fixes and features depend on Mojo 0.26.3 and its specific
stdlib APIs. A downgrade would require reverting production code changes.

## Consequences

### Positive

- Removing the retry script makes ADR-014's SUPERSEDED status accurate in the tree
- Import audit eliminates the root cause rather than masking it
- CI becomes a reliable signal once crash rate drops to <5%
- Stale workflow comments removed -- no more confusion between source comments and
  branch protection reality

### Negative

- Short-term: PR pipeline remains flaky until action #1 (import audit) lands
- Retry script's exit-code-2 semantic disappears -- any callers that distinguished
  "JIT crash" from "real failure" by exit code will need to be updated (currently
  only the justfile consumes this, and that wiring is being removed)

### Neutral

- Test coverage is unchanged by the import style change
- The upstream Mojo compiler bug is unaffected by these actions; we are reducing
  exposure, not eliminating the underlying bug

## Files Modified

| File | Change |
| --- | --- |
| `docs/adr/ADR-015-flaky-required-checks-jit-crash.md` | New -- this document |
| `docs/adr/ADR-014-jit-crash-retry-mitigation.md` | Cross-link to ADR-015 added |
| `docs/dev/mojo-jit-crash-workaround.md` | New `## 0.26.3 Required Checks Impact` section added |

The following are **scheduled** by corrective actions above (separate PRs, tracked under
issue #5108):

| File | Scheduled Change |
| --- | --- |
| `scripts/test-with-retry.sh` | Delete (action #2) |
| `tests/smoke/test_retry_script.py` | Delete (action #2) |
| `justfile` | Replace retry wrapper calls at lines 664 and 829 with direct mojo invocation (action #2) |
| `tests/core/types/test_*.mojo` | Import audit -- convert package-level to targeted (action #1) |
| `tests/shared/integration/test_*.mojo` | Import audit -- convert package-level to targeted (action #1) |
| `.github/workflows/comprehensive-tests.yml` | Remove stale "non-blocking" comments (action #3) |

## References

- [Issue #5108](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5108) --
  JIT crash comprehensive tracking
- [ADR-009](ADR-009-heap-corruption-workaround.md) -- Heap corruption workaround
  (resolved 2026-03-20)
- [ADR-014](ADR-014-jit-crash-retry-mitigation.md) -- JIT crash retry mitigation
  (SUPERSEDED; retry script still wired pending this ADR's action #2)
- [`docs/dev/mojo-jit-crash-workaround.md`](../dev/mojo-jit-crash-workaround.md) --
  Import explosion root cause, Symbol-to-Submodule Mapping, diagnostic table
- [`repro/issues/jit-compilation-volume-crash.md`](../../repro/issues/jit-compilation-volume-crash.md) --
  0.26.3 minimal reproducer
- [modular/modular#6187](https://github.com/modular/modular/issues/6187) -- Upstream
  Mojo bug (cited in ADR-014 and workaround doc)
- [modular/modular#6413](https://github.com/modular/modular/issues/6413) -- Upstream
  Mojo bug (cited in test-with-retry.sh header; may be same issue as #6187)

---

## Document Metadata

- **Location**: `docs/adr/ADR-015-flaky-required-checks-jit-crash.md`
- **Status**: Accepted -- Corrective Actions Open
- **Review Frequency**: After Mojo upgrade past 0.26.3, or after all corrective actions complete
- **Next Review**: Post-import-audit observation window (two weeks after action #1 lands)
- **Supersedes**: None
- **Superseded By**: N/A

## Revision History

| Version | Date | Author | Changes |
| --- | --- | --- | --- |
| 1.0 | 2026-04-12 | Claude Code | Initial ADR documenting root cause and corrective actions |
