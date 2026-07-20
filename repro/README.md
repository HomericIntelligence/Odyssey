# Crash Reproducers — Triage Outcomes (2026-05-26)

Local triage on Mojo `1.0.0b2.dev2026052506` (glibc 2.39 host) ran every
reproducer here 10× consecutively against the post-#6413-fix toolchain.

## Outcome summary

| Reproducer | Upstream | State | Local 10× | Action |
| --- | --- | --- | --- | --- |
| `repro_hello.mojo` | (sanity) | — | 10/10 | KEEP — sanity baseline |
| `kgen_jit_overflow_minimal.mojo` | [modular/modular#6445](https://github.com/modular/modular/issues/6445) | OPEN | 10/10 local; CI-only repro | KEEP — pending CI validation |
| `repro_jit_fortify_crash.mojo` | [modular/modular#6445](https://github.com/modular/modular/issues/6445) | OPEN | 10/10 local; CI-only repro | KEEP — pending CI validation |
| `repro/issues/jit-fortify-buffer-overflow.md` | [modular/modular#6445](https://github.com/modular/modular/issues/6445) | OPEN | — | KEEP — local stub |
| `spinlock_race_condition.mojo` | n/a (our bug) | — | passes | KEEP — didactic |
| `spinlock_deadlock.mojo` | n/a (our bug) | — | passes | KEEP — didactic |
| `bug_repro_lenet5_layers_monolithic.mojo.bug` | historical | — | — | KEEP — deprecated artifact |
| `bug_repro_vgg16_e2e_part1_pre_fix.mojo.bug` + `.merge-conflict-snapshot` | historical | — | — | KEEP — deprecated artifact |
| `run_all_experiments.sh`, `investigate_import_threshold.sh` | (harnesses) | — | — | KEEP — diagnostic scripts |
| `repro_crash_standalone.mojo` | [modular/modular#6187](https://github.com/modular/modular/issues/6187) | CLOSED 2026-04-10 | 10/10 | **DELETED** — fixed |
| `repro_libkgen_crash.mojo` | [modular/modular#6187](https://github.com/modular/modular/issues/6187) | CLOSED | 10/10 | **DELETED** — fixed |
| `repro_libasyncrt_crash.mojo` | [modular/modular#6187](https://github.com/modular/modular/issues/6187) | CLOSED | 10/10 | **DELETED** — fixed |
| `repro_jit_volume_crash.mojo` | (unfiled) | — | 10/10 | **DELETED** — fixed |
| `repro_jit_targeted_imports_crash.mojo` | (unfiled) | — | 10/10 | **DELETED** — fixed |
| `repro_jit_heavy_import_test.mojo` | (unfiled) | — | 10/10 | **DELETED** — fixed |
| `repro_module_import_crash.mojo` + `_fixed.mojo` | (unfiled) | — | 10/10 | **DELETED** — fixed |
| `repro_import_chain_synthetic.mojo` + `repro_heavy_{A,B}.mojo` | (unfiled) | — | 10/10 | **DELETED** — fixed |
| `repro_parametric_monomorphization_crash.mojo` | (unfiled) | — | 10/10 | **DELETED** — fixed |
| `configs_kgen_crash.mojo` | (unfiled) | — | 10/10 | **DELETED** — fixed |
| `repro/issues/libkgen-heap-corruption-crash.md` | [modular/modular#6187](https://github.com/modular/modular/issues/6187) | CLOSED | — | **DELETED** — stub for fixed bug |
| `repro/issues/libasyncrt-vgg16-training-crash.md` | [modular/modular#6187](https://github.com/modular/modular/issues/6187) | CLOSED | — | **DELETED** — stub for fixed bug |
| `repro/issues/jit-virtual-memory-exhaustion.md` | [modular/modular#6433](https://github.com/modular/modular/issues/6433) | CLOSED 2026-05-06 | — | **DELETED** — stub for fixed bug |
| `repro/issues/jit-compilation-volume-crash.md` | (unfiled) | — | — | **DELETED** — stub for fixed-locally bug |
| `repro/issues/jit-targeted-imports-still-crashes.md` | (unfiled) | — | — | **DELETED** — stub for fixed-locally bug |

## What was migrated to Mojo 1.0

Every reproducer remaining or deleted in this sweep was first migrated to
Mojo 1.0.0b2 syntax so the triage measured behavior, not syntax drift. Per
`docs/dev/mojo-1.0-migration-recipe.md`:

- `fn → def` (Recipe: top-level `fn` removed); affected
  `repro_jit_targeted_imports_crash.mojo` and
  `repro_parametric_monomorphization_crash.mojo`.
- `out:` param renamed to `dst:` (`out` is reserved for constructor self);
  `repro_parametric_monomorphization_crash.mojo`.
- `UnsafePointer[T].alloc(N)` → `alloc[T](N)` (Recipe 6); same file.
- `std.os.atomic` → `std.atomic` (Recipe 6); `spinlock_race_condition.mojo`.
- Removed redundant `if self._refcount:` null-checks (Recipe 3 Case B);
  `repro_crash_standalone.mojo`.
- Restored synthetic `repro_heavy_A.mojo` / `repro_heavy_B.mojo`
  (~1357/1359 lines) that the synthetic import-chain reproducer depended on.
- Resolved unresolved git merge conflict in `bug_repro_vgg16_e2e_part1_pre_fix.mojo.bug`
  (preserved verbatim at `.merge-conflict-snapshot`).

## #6445 (KGEN fortify overflow) — still OPEN

Two reproducers self-document as Docker-CI-only (UID 1001, fresh pixi cache,
no TTY). Local 10/10 clean is necessary but not sufficient to declare
fixed. The local result is being posted as a comment on
[modular/modular#6445](https://github.com/modular/modular/issues/6445)
with a CI-validation caveat; the maintainer can decide whether to close
based on the in-flight CI workflow.

## Triage protocol used

```bash
for i in 1..10:
  uv run mojo run -I src -I . repro/<file>.mojo
```

Tally per file: clean exits, nonzero exits, any `execution crashed` /
`libKGENCompilerRTShared.so+` / `libAsyncRT` / `SIGILL` / `fortify_fail`
substring in output. Full logs live under `/tmp/repro-triage/*.log` for
the validation run that produced this sweep.

## Why these were originally restored (2026-05-26 morning)

PR #5460 (Wave 2 of the [#6413](https://github.com/modular/modular/issues/6413)
demolition) over-deleted 22 files in `repro/` and `repro/issues/`. The
restoration in PR #5464 brought them back because their bugs were
independent of #6413, even though they shared the libKGEN/JIT-crash
surface. This sweep now closes the triage: most are confirmed fixed and
deleted; the two CI-only #6445 stress reproducers remain pending CI
validation.
