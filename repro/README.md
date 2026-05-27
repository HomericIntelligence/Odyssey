# Crash Reproducers — Triage Pending

These minimal reproducers were **restored** on 2026-05-26 after Wave 2 of the
[modular/modular#6413](https://github.com/modular/modular/issues/6413)
demolition (PR #5460) over-deleted them.

The `#6413` issue (AVX-512 mis-emission causing SIGILL via
`libKGENCompilerRTShared.so`) was fixed in Mojo `1.0.0b2.dev2026052506`. Wave 2's
scorched-earth purge correctly removed `#6413`-specific artifacts
(`repro_6413_*.mojo`, `notes/modular-6413-avx512-finding.md`,
`.github/workflows/repro-6413.yml`), but **also deleted reproducers for several
distinct upstream bugs** that share the `libKGEN`/JIT-crash surface but are
independent issues.

> **Status of every file here: TRIAGE-PENDING.** None has been re-validated
> against Mojo `1.0.0b2.dev2026052506`. The `#6413` fix may have incidentally
> resolved some, none, or all of them — that determination requires running each
> reproducer and is tracked in the follow-up issue list below.

## Files in this directory

| Reproducer | Issue file | Suspected bug class | Status |
| --- | --- | --- | --- |
| `repro_libkgen_crash.mojo` | `issues/libkgen-heap-corruption-crash.md` | Allocator heap corruption via AnyTensor + conv2d | TRIAGE-PENDING |
| `repro_libasyncrt_crash.mojo` | `issues/libasyncrt-vgg16-training-crash.md` | `libAsyncRTRuntimeGlobals.so` crash in VGG16 training | TRIAGE-PENDING |
| `repro_jit_volume_crash.mojo` | `issues/jit-virtual-memory-exhaustion.md` | JIT virtual-memory exhaustion at high module volume | TRIAGE-PENDING |
| `kgen_jit_overflow_minimal.mojo` | `issues/jit-compilation-volume-crash.md` | KGEN JIT buffer overflow at compilation volume | TRIAGE-PENDING |
| `repro_jit_fortify_crash.mojo` | `issues/jit-fortify-buffer-overflow.md` | `__fortify_fail_abort` buffer overflow | TRIAGE-PENDING |
| `repro_parametric_monomorphization_crash.mojo` | (no issue file) | Parametric monomorphization crash | TRIAGE-PENDING |
| `repro_module_import_crash.mojo` + `_fixed.mojo` | `issues/jit-targeted-imports-still-crashes.md` | Module-level import chain crash | TRIAGE-PENDING |
| `repro_jit_targeted_imports_crash.mojo` | `issues/jit-targeted-imports-still-crashes.md` | Targeted import chain crash | TRIAGE-PENDING |
| `repro_jit_heavy_import_test.mojo` | (no issue file) | Heavy import volume crash | TRIAGE-PENDING |
| `repro_import_chain_synthetic.mojo` | (no issue file) | Synthetic import-chain crash | TRIAGE-PENDING |
| `repro_crash_standalone.mojo` | (no issue file) | Standalone crash variant | TRIAGE-PENDING |
| `configs_kgen_crash.mojo` | (no issue file) | KGEN crash in configs subsystem | TRIAGE-PENDING |
| `run_all_experiments.sh` | — | Driver script to run all reproducers | — |
| `repro_hello.mojo` | — | Sanity-check (pre-existing, never deleted) | — |
| `spinlock_deadlock.mojo`, `spinlock_race_condition.mojo` | — | Concurrency repros (pre-existing) | — |
| `investigate_import_threshold.sh`, `bug_repro_*.mojo.bug` | — | Pre-existing investigation artifacts | — |

## Triage protocol (per file)

For each `TRIAGE-PENDING` entry, run:

```bash
# Inside the Podman container (matches CI runner GLIBC):
just shell
mojo build --print-effective-target repro/<file>.mojo   # verify codegen
mojo run repro/<file>.mojo                              # run 10× to test determinism
```

Outcomes:

- **Reproduces**: file an upstream issue at `modular/modular` if one isn't already
  open (check the `issues/*.md` file for any prior filing); update this README
  with the issue link and change status to `OPEN-#NNNN`.
- **Fixed by Mojo dev2026052506+**: delete the reproducer in a per-file commit
  citing the validation run hash.
- **Cannot reproduce / nondeterministic on new Mojo**: keep the file with status
  `INTERMITTENT — needs longer soak`; consider whether to file or defer.

## Why these were restored

Wave 2 of the `#6413` demolition (PR #5460, commit `f5de5894`) applied a
blanket delete to everything under `repro/` and `repro/issues/`. The audit-trail
comment posted to issue #5108 listed all 22 deleted paths but did not
distinguish #6413-specific from independent bugs.

Restored in PR #5464 from the last commit that touched all of these files
together: `e8c5609a` (refactor(packaging): rename shared/ → src/projectodyssey/
(PR 1 of 4)).

The same PR also restored `.github/workflows/gradient-soak.yml` (from commit
`7cde386d` — the issue #5170 gradient-checker soak workflow). That workflow
addresses a different intermittent JIT crash (#5170 / #5104, per-element bitcast
elimination), not #6413, and its own header instructs removal only after
"enough consecutive clean runs" — a decision that requires separate validation
on the post-#6413-fix Mojo, not bundling into the #6413 demolition.

**Lesson for future demolitions**: when a "scorched-earth" purge targets a bug
class, partition the affected files into (a) the bug itself, (b) bugs with
overlapping symptoms but distinct root causes. Validate (b) independently
before deleting. Captured as a follow-up amendment to the
[`workaround-demolition-wave-strategy`](https://github.com/HomericIntelligence/ProjectMnemosyne/blob/main/skills/workaround-demolition-wave-strategy.md)
skill.
