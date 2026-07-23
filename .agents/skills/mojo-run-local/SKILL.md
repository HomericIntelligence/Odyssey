---
name: mojo-run-local
description: Resolve `mojo: error: unable to locate module 'odyssey'` by passing the canonical 3-flag include triple `-I src -I "$REPO_ROOT" -I .` that the justfile already uses. Use when a raw `uv run mojo run <file>` fails to resolve a project package.
---

# Run Mojo locally outside the justfile

The repo's `mojo.toml` (`[packages]` block at `mojo.toml:13-16`) declares
`odyssey = "src/odyssey"`, but `mojo run` does NOT auto-include `src/`
on the include path. The justfile's `_test-mojo-inner` recipe (around
`justfile:646`) works because it passes the canonical 3-flag triple:

```bash
uv run mojo --Werror -I "$REPO_ROOT/src" -I "$REPO_ROOT" -I . "$test_file"
```

A raw `uv run mojo run tests/...` invocation from the project root
fails with:

```text
error: unable to locate module 'odyssey'
from odyssey.tensor.any_tensor import AnyTensor
```

## Fix

Mirror the justfile's canonical include triple verbatim:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
uv run --locked mojo run \
    -I src \
    -I "$REPO_ROOT" \
    -I . \
    <relative-path>.mojo
```

- `-I src` resolves `odyssey = "src/odyssey"`.
- `-I "$REPO_ROOT"` covers any root-level package (none today, but
  mirrors the justfile).
- `-I .` keeps the cwd's `mojo.toml` and the sibling `examples/`,
  `tests/`, `benchmarks/` packages on the path. The dual inclusion
  of `"$REPO_ROOT"` and `.` is deliberate: it survives runs where
  `cwd != repo_root` (containers, CI runners, worktrees) without
  regressing on the local case.

When the exact `$REPO_ROOT` is awkward (shell-escaping on older
bash, embedded in scripts), the 2-flag `-I src -I .` works on local
host where `cwd == repo_root`. Use the 3-flag version any time
`cwd` may not be the repo root.

## Verification

```bash
# FAILS (pre-existing include-path bug):
uv run mojo run tests/odyssey/conftest.mojo

# SUCCEEDS (the canonical fix, works in any cwd):
REPO_ROOT="$(git rev-parse --show-toplevel)"
uv run --locked mojo run -I src -I "$REPO_ROOT" -I . tests/odyssey/conftest.mojo

# ALSO succeeds (2-flag shortcut — local repo_root only):
uv run --locked mojo run -I src -I . tests/odyssey/conftest.mojo
```

Same fix for the K=3 byte-identical tests on PR #5684:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
uv run --locked mojo run -I src -I "$REPO_ROOT" -I . \
    tests/odyssey/training/optimizers/test_optimizer_delegator_equivalence.mojo
```

## When to use this skill

- Running a single `.mojo` file locally for quick debugging (instead
  of `just test-mojo` which sweeps the whole tree).
- CI smoke before pushing — match the 3-flag invocation so local ↔ CI
  parity is preserved.
- Iterating inside a Podman container whose `cwd` is `/workspace`
  (the justfile's mountpoint) rather than the host repo path.
- `just test-mojo` itself works without this fix because the
  justfile already passes the canonical triple.

## Out of scope

- The K=3 test file raised a separate compile error
  (`cannot transfer out of immutable reference`) on first run that
  hit the canonical include path. The `mut grads` signature fix
  on `_run_oo_k3_steps` addresses this — see the test file. This
  skill resolves only the `unable to locate module 'odyssey'`
  failure.

## Cross-references (line-numbered)

- `justfile:646-660` — `_test-mojo-inner` recipe, the canonical
  working invocation.
- `justfile:574-602` — `_test-group-inner`, identical 3-flag pattern.
- `mojo.toml:13-16` — `[packages]` block that declares
  `odyssey = "src/odyssey"`.
- `AGENTS.md` § "Mojo Test Execution and GLIBC Compatibility" — if the
  failure is GLIBC-related rather than include-path, use
  `just podman-up` instead of this skill.
