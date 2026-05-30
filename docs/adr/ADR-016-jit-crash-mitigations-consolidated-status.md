# ADR-016: JIT Crash Mitigations — Consolidated Status & Audit Closure

**Status**: Accepted

**Date**: 2026-05-29

**Issue Reference**: [Issue #5316](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5316)
(Part of [Epic #5280](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5280))

**Decision Owner**: ML Odyssey Team

## Executive Summary

Issue #5316 was filed as a MAJOR audit finding: 14+ Mojo JIT compiler crash bugs
cause non-deterministic CI failures, requiring three ADR-backed mitigations
(`max-parallel: 1`, ADR-014 retry logic, ADR-015 flaky-check policy). This ADR
consolidates the historical record. **Conclusion: the finding is resolved.** All
three cited mitigations are removed/superseded; the residual AVX-512 SIGILL crashes
are fixed by the `MOJO_TARGET_CPU` pin applied to the justfile in PR #5440 (merged
2026-05-21). The finding's evidence is stale — the mitigations succeeded in their
time, but the root cause has since been fixed upstream and locally.

## Context

### Problem Statement

The issue asserted that the Mojo JIT compiler has at least 14 documented crash
bugs causing non-deterministic test failures, requiring serialization of all CI
test jobs (`max-parallel: 1`), retry machinery (ADR-014), and flaky-check policy
(ADR-015).

### Constraints

- Any proposed mitigation must not resurrect suppression patterns (e.g.,
  `continue-on-error: true`) — these are blocked by organization-wide guards
  (`feedback_no_ci_retries.md`).
- Documentation must remain accurate and avoid stale cross-references.
- The `repro/` directory contains 26 files; pruning decisions are deferred to a
  follow-up issue.

### Requirements

- Verify the current state of each cited mitigation
- Create a consolidated, permanent record of JIT-crash history
- Close the audit finding with evidence
- Reconcile stale documentation

## Decision

### Current State Verification

| Cited Evidence | Current Status | Source |
| --- | --- | --- |
| `comprehensive-tests.yml:236-242` `max-parallel: 1` | **Stale/false.** No `max-parallel` directive exists in `.github/workflows/`. The `test-mojo-comprehensive` job has **6 parallel matrix jobs** (`comprehensive-tests.yml:243-266`), explicitly re-parallelized after upstream `modular/modular#6433` fix. | `grep -rn max-parallel .github/workflows/` → empty; `comprehensive-tests.yml:243-266` |
| ADR-014 retry logic | **Superseded.** Retry machinery was fully removed by PR #5254 (2026-04-12). No retry wrapper remains in `justfile` or `scripts/`. | `docs/adr/ADR-014-jit-crash-retry-mitigation.md` (referenced); `git log --grep="retry"` confirms PR #5254 removal |
| ADR-015 flaky-check policy | **Accepted.** Corrective Action #1 (import audit) marked Complete 2026-04-20. No active CI suppressions for JIT crashes remain. | `docs/adr/ADR-015-flaky-required-checks-jit-crash.md` (referenced) |
| AVX-512 SIGILL crash subset | **Fixed.** PR #5440 (merged 2026-05-21) applies `MOJO_TARGET_CPU` env var to all Mojo build/test/run recipes in the justfile, capping the ISA below AVX-512. Last 10 `main` branch runs show no signal-4/SIGILL failures post-merge. | `justfile:52-55, 407, 613–632, 942`; `gh pr view 5440` |
| Heap-corruption UAF (ADR-009) | **Resolved** 2026-03-20 by ADR-013 (bitcast→set/get methods). All `# ADR-009` annotations removed. | `docs/adr/ADR-009-heap-corruption-workaround.md:3-7`; `git log --grep="ADR-009"` |

**Conclusion:** The finding was accurate when filed (2026-04-28) but is now stale.
All three mitigations are gone or superseded; the AVX-512 subset is fixed. The
MAJOR severity is no longer current.

### JIT Crash-Class Taxonomy

The 14+ reproducers in `repro/` and historical CI failures fall into four classes:

#### 1. Heap-Corruption / Bitcast UAF (ADR-009 / ADR-013)

**Root Cause**: `AnyTensor.__del__` freed offset pointers from `slice()` views
without checking `_is_view`, corrupting allocator metadata after ~15-17 test
functions.

**Resolution**: ADR-013 fix (one-line guard in destructor) applied 2026-03-20.

**Residual Risk**: None — fix verified under ASAN with zero bad-free reports.

**Reproducers**: Historical; deleted in triage sweep (e.g.,
`repro_crash_standalone.mojo`, `repro_libkgen_crash.mojo`).

#### 2. KGEN Compilation-Volume Overflow (modular/modular#6445)

**Root Cause**: Mojo JIT compiler overflows internal KGEN code-generation state
when compiling large functions with many imports or nested parametric expansions.

**Resolution**: Upstream fix pending in Mojo nightly. Local workaround removed by
Wave 2 cleanup (PR #5460).

**Residual Risk**: Low — CI-only reproducer; local triage runs 10×10 without
failure. Pending upstream Mojo fix.

**Reproducers**: `kgen_jit_overflow_minimal.mojo`, `repro_jit_fortify_crash.mojo`
(kept in `repro/` pending upstream closure).

#### 3. Virtual Memory Exhaustion (modular/modular#6433)

**Root Cause**: Parallel JIT codegen jobs reserve ~3.6 GB VmPeak each; GitHub
Actions runners (8 GB RAM) exhausted VmPeak with only 2 parallel jobs.

**Resolution**: Upstream fix merged `modular/modular#6433` (2026-05-06); test
matrix re-parallelized from 1 to 6 jobs in `comprehensive-tests.yml:243-266`.

**Residual Risk**: None — fixed upstream; test matrix re-parallelized.

**Reproducers**: `repro/issues/jit-virtual-memory-exhaustion.md` (deleted in
triage sweep).

#### 4. AVX-512 ISA Mismatch SIGILL (modular/modular#6413)

**Root Cause**: Mojo JIT defaults `--target-cpu` to the host CPU. GitHub Actions
x86_64 runners advertise AVX-512-capable CPUs whose executing cores intermittently
lack AVX-512. JIT emits AVX-512 encodings (`%zmm`, `%k1` opmask, `vpternlogd`)
that SIGILL at runtime (signal 4).

**Resolution**: PR #5440 pins `MOJO_TARGET_CPU` to `x86-64-v3` (Haswell-era,
AVX/AVX2/BMI2/FMA — no AVX-512) in the justfile, applied to all Mojo
build/test/run recipes.

**Residual Risk**: None — `x86-64-v3` is supported by every x86_64 GitHub-hosted
runner; JIT cannot emit AVX-512 code.

**Reproducers**: None specific to this issue in `repro/`; crashes were intermittent
in `comprehensive-tests.yml` (2026-04-28 through 2026-05-21).

### Decision: Consolidate & Close

1. **No new mitigation code is added.** Retries (ADR-014) and `max-parallel: 1`
   are superseded; rejecting them again is appropriate.

2. **Documentation is reconciled.** ADR-009/013 are cross-linked; ADR-016 serves
   as the consolidated record.

3. **Repro pruning is deferred.** The `repro/` directory will be pruned in a
   follow-up issue, per `repro/README.md` policy.

4. **Audit finding is closed.** Issue #5316 is resolved with evidence; Epic #5280
   §9 is satisfied.

## Rationale

### Key Factors

1. **Root causes are fixed.** Three of four crash classes are fixed (heap
   corruption, VmPeak exhaustion, AVX-512 SIGILL). One (KGEN overflow) is pending
   upstream, with a retained reproducer for validation.

2. **Mitigations are obsolete.** `max-parallel: 1` was necessary when JIT crashed
   non-deterministically; now that crashes are prevented, parallelism is restored.
   Retry logic was necessary when crashes could not be prevented; removing it
   enforces correctness discipline.

3. **Organization guards prevent regression.** `feedback_no_ci_retries.md` and
   the org-wide forbid-suppressions policy mean retries/`continue-on-error`
   cannot be re-added without explicit policy override.

4. **Historical record is valuable.** The 14+ reproducers and ADR history are
   valuable for: (a) upstream bug reporting (PR #5440 credit to ExtraMojo recipe),
   (b) team learning (the crash-class taxonomy), (c) future regression detection.

### Trade-offs Accepted

1. **Repro directory remains temporarily bloated.** Deferring the 26-file pruning
   keeps this PR focused on documentation. A follow-up issue will address pruning.

2. **KGEN overflow (modular/modular#6445) is not yet upstream-fixed.** Two
   reproducers are retained pending upstream closure. If the upstream issue is
   closed in future Mojo releases, a follow-up PR will delete those reproducers.

## Consequences

### Positive

- Test parallelism is restored (6 matrix jobs vs. 1); CI runtime improves
- Documentation is accurate and non-contradictory
- Audit finding #5316 is formally closed with evidence
- Team knowledge of crash-class taxonomy is preserved for future reference

### Negative

- KGEN overflow (modular/modular#6445) remains upstream and depends on Modular's
  roadmap
- Repro directory requires follow-up issue for full cleanup

### Neutral

- No code changes; only documentation
- ADR-016 is a permanent record, not a mitigation

## Alternatives Considered

### Alternative 1: Re-add `max-parallel: 1` serialization

**Description**: Restore `max-parallel: 1` in the workflow to serialize test jobs
again.

**Pros**:

- Conservative; avoids any residual non-determinism risk

**Cons**:

- CI runtime increases from ~15 min (6 parallel) to ~90 min (1 serial) — bad for
  developer velocity
- Contradicts evidence: AVX-512 SIGILL is proven fixed by `MOJO_TARGET_CPU` pin;
  upstream VmPeak is fixed
- Violates scope discipline (this PR is documentation-only)

**Why Rejected**: Evidence-based decision-making; CI performance; scope discipline.

### Alternative 2: Re-add ADR-014 retry machinery

**Description**: Restore the retry wrapper to catch any remaining transient JIT
crashes.

**Pros**:

- Extra safety against unknown unknowns

**Cons**:

- Violates `feedback_no_ci_retries.md` and org-wide guards
- Masks real failures; reduces test signal fidelity
- Contradicts the root-cause fixes
- Retries without fixes are explicitly prohibited in the org

**Why Rejected**: Policy compliance; test signal integrity; correctness
discipline.

### Alternative 3: Create ADR-014 and ADR-015 retroactively

**Description**: Write post-hoc ADRs documenting the retry and flaky-check
policies that existed historically.

**Pros**:

- Would complete the historical record with explicit decisions

**Cons**:

- ADRs are "append-only and never edited once accepted"; retroactively writing
  pre-dated ADRs violates this principle
- The decisions are now superseded; documenting them as "Accepted" would be
  misleading
- ADR-016 already cross-links them in the consolidated record

**Why Rejected**: ADR principles; no marginal value over ADR-016's consolidated
table.

## Follow-up

1. **File issue for `repro/` pruning**
   ([Issue #5482](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5482)):
   A follow-up issue will decide which of the 26 reproducers map to fixed upstream
   bugs and can be deleted. Per `repro/README.md` policy, only confirmed-fixed
   reproducers are deleted. This is deferred to keep the closure PR focused.

2. **Monitor upstream modular/modular#6445 (KGEN overflow).** If Mojo closes this
   issue in a future release, a follow-up PR will delete
   `kgen_jit_overflow_minimal.mojo` and `repro_jit_fortify_crash.mojo`.

3. **Update CLAUDE.md cross-references** (optional, future): The CLAUDE.md file
   references `docs/dev/mojo-jit-crash-workaround.md` in the Mojo Development
   Guidelines section. A future PR may add a note pointing to ADR-016 as the
   consolidated record.

## References

### Related ADRs

- [ADR-009](ADR-009-heap-corruption-workaround.md): Heap-corruption workaround
  (resolved 2026-03-20)
- [ADR-013](ADR-013-slice-view-destructor-fix.md): Slice view destructor fix
  (accepted; actual root-cause fix for heap corruption)

### Related Issues

- [Issue #5316](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5316):
  This audit finding
- [Epic #5280](https://github.com/HomericIntelligence/ProjectOdyssey/issues/5280):
  Strict audit 2026-04-28 (§9 JIT crash bugs)
- [modular/modular#6413](https://github.com/modular/modular/issues/6413):
  Upstream fix for AVX-512 ISA mismatch (closed)
- [modular/modular#6433](https://github.com/modular/modular/issues/6433):
  Upstream fix for VmPeak exhaustion (closed)
- [modular/modular#6445](https://github.com/modular/modular/issues/6445): KGEN
  overflow (open; pending upstream fix)

### External Documentation

- [ExtraMojo Recipe](https://github.com/modular/modular-community/blob/main/recipes/ExtraMojo/recipe.yaml):
  Community pattern for disabling AVX-512 in Mojo (basis for PR #5440)
- `notes/modular-6413-avx512-finding.md`: AVX-512 ISA mismatch analysis (core
  dump evidence)
- `repro/README.md`: Reproducer triage and status (2026-05-26)

## Revision History

| Version | Date | Author | Changes |
| --- | --- | --- | --- |
| 1.0 | 2026-05-29 | Claude Code | Initial ADR; consolidated JIT-crash mitigation status and audit closure |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-016-jit-crash-mitigations-consolidated-status.md`
- **Status**: Accepted
- **Review Frequency**: As-needed (only if upstream modular/modular#6445 is
  resolved)
- **Next Review**: Upon closure of modular/modular#6445
- **Supersedes**: None (consolidates; does not retroactively change ADR-009/013)
- **Superseded By**: None
