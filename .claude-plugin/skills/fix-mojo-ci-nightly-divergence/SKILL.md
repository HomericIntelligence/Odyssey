# Skill: fix-mojo-ci-nightly-divergence

> **HISTORICAL (ADR-018):** This skill described the conda `max-nightly` vs
> `max` channel divergence problem. After the uv migration, Odyssey no longer
> uses conda channels at all — the Mojo compiler is pinned (`mojo==1.0.0b2`) in
> `pyproject.toml` and locked in `uv.lock`, installed from the single Modular
> PyPI index. CI and local dev therefore resolve the exact same Mojo build by
> construction, so channel-divergence can no longer occur. Kept for historical
> reference; the commands below are updated to their uv equivalents.

## Overview

| Field | Value |
| --- | --- |
| **Date** | 2026-03-10 |
| **Category** | ci-cd |
| **Objective** | Fix CI failures caused by Mojo nightly vs stable release API differences |
| **Outcome** | ✅ Success - pinned to stable channel, reverted unnecessary workarounds |

## When to Use

Invoke this skill when:

- CI fails with errors that don't reproduce locally (or vice versa)
- `pixi.toml` uses `max-nightly` channel but local dev uses stable Mojo
- You see API differences like `String[byte=index]` or `Int(py=obj)` working locally but failing in CI
- You're tempted to add compatibility shims for nightly-only API changes

## Verified Workflow

### Step 1: Identify the Root Cause

Check if CI and local use different Mojo versions:

```bash
# Local version
uv run mojo --version

# Check the pinned mojo version (single Modular PyPI index, no channels)
grep '"mojo' pyproject.toml
# Under uv there is only one source, so CI and local always match.
```

### Step 2: Pin the Mojo version (uv)

```bash
# In pyproject.toml [project.dependencies], set the exact pin, e.g.:
#   "mojo==1.0.0b2",
# Then regenerate the lockfile:
uv lock
uv sync --locked
```

This ensures CI uses the exact same Mojo version as local development.

### Step 3: Revert Any Nightly Workarounds

If you previously added workarounds for nightly API differences, revert them:

```bash
git revert <workaround-commit-hash>
```

Common workarounds that should be reverted:

- `chr(Int(s.as_bytes()[i]))` → revert to `s[byte=i]`
- `atol(String(py_obj))` → revert to `Int(py=py_obj)`
- Any `# CI nightly compatibility` comments

## Failed Attempts

### ❌ Patching Code for Nightly Compatibility

**What was tried**: Replace `String[byte=index]` with `chr(Int(s.as_bytes()[index]))` and
`Int(py=obj)` with `atol(String(obj))` to work on both nightly and stable.

**Why it failed**: Creates unnecessary code churn, makes code harder to read, and the
workarounds themselves can introduce bugs (e.g., invalid `result[span=0:...]` syntax). The
real fix is aligning the Mojo version, not patching around differences.

### ❌ Maintaining Dual Compatibility

**What was tried**: Writing code that works on both nightly and stable Mojo.

**Why it failed**: Nightly APIs change frequently. Maintaining compatibility is a moving
target that wastes development time. Pin to one version instead.

## Related CI Fixes

### grep in bash -e Scripts

`grep` returns exit code 1 when no matches are found, which causes `bash -e` (default in
GitHub Actions) to fail the step. Fix:

```yaml
# ❌ Fails when grep finds nothing
violations=$(grep -rn "pattern" .)

# ✅ Correct: suppress exit code
violations=$(grep -rn "pattern" . || true)
```

### Duplicate YAML Keys

YAML silently uses only the last value for duplicate keys. This causes subtle bugs in
`.pre-commit-config.yaml` and workflow files. Always check for duplicates when hooks fail
unexpectedly.

### Mojo 0.26.1 Compilation Patterns

These are genuine fixes needed regardless of nightly vs stable:

- Struct parameter qualification: `T0` → `Self.T0`
- Ownership keyword: `owned other` → `deinit other`
- Removed stdlib function: `is_apple_silicon()` no longer exists
- Docstrings must end with `.` or backtick to avoid compiler warnings

## Results & Parameters

| Metric | Value |
| --- | --- |
| PRs created | 2 (fix-pre-commit #4483, pin-mojo-stable-channel #4484) |
| Files modified | ~40 |
| Workaround commits reverted | 1 (a50db177) |
| Pre-commit hooks passing | 17/17 |
| Channel | `https://conda.modular.com/max` (stable) |
| Mojo version | 0.26.1.0 |
