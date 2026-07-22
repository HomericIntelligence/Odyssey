# Consolidated Build & Installation

From root MDs (backed up `notes/root-backup/`). Links to [phases.md](phases.md).

## Justfile Build Recipes

The project uses [Just](https://just.systems/) as a unified command runner. Four recipes cover
the main build and validation scenarios:

| Recipe | What it does | When to use |
| --- | --- | --- |
| `just build` | Compile entry-point executables (debug mode by default) | Day-to-day development; produces binaries in `build/debug/` |
| `just check` | Type-check the `src/odyssey/` library without producing any artifacts | Fast feedback loop — catches type errors without a full build |
| `just ci-build` | Full CI build: entry points (`just build ci`) + package compilation (`just package ci`) | Pre-push validation; mirrors exactly what CI runs |
| `just validate` | `ci-build` + the full Mojo test suite | Final gate before merging; slowest but most complete |

### Choosing the Right Recipe

```text
Editing src/odyssey/ code?
  └─ just check          ← fastest; no artifacts, type errors only

Editing entry points or want binaries?
  └─ just build          ← compiles executables in build/debug/

Before pushing a branch?
  └─ just ci-build       ← confirms the full build mirrors CI

Before opening a PR / after all changes?
  └─ just validate       ← ci-build + tests; matches the CI pipeline exactly
```

### Details

**`just check`** compiles the `src/odyssey/` library with `mojo package --Werror` into a temporary
directory and immediately discards the output. It is the fastest way to confirm that type
annotations, imports, and function signatures in `src/odyssey/` are correct without waiting for a
full build or test run.

**`just ci-build`** runs two sub-recipes in sequence:

1. `just build ci` — compiles all entry-point executables with CI flags (`-g1 --Werror`)
2. `just package ci` — packages the `src/odyssey/` library into a `.mojopkg` artifact

This is the minimum validation that must pass before CI will accept a PR.

**`just validate`** extends `ci-build` with `just test-mojo` (all Mojo unit tests). Use it
locally only when you need confidence equivalent to a full CI run; the test suite can be slow
(see [Testing Strategy](../dev/testing-strategy.md) for tier details).

## Package Building

**File**: `BUILD_PACKAGE.md`

**Training Module**:

- `mojo package src/odyssey/training -o dist/training-0.1.0.mojopkg`
- Verify: `./scripts/install_verify_training.sh`
- Automated: `./scripts/build_training_package.sh`

**Other Packages**: data/utils via scripts/build_*_package.sh.

## Installation

**File**: `INSTALL.md`

**Docker (Recommended)**:

```bash
git clone https://github.com/HomericIntelligence/Odyssey.git
cd Odyssey
docker-compose up -d Odyssey-dev
docker-compose exec Odyssey-dev bash
uv run pytest tests/
```

**uv Local**:

```bash
uv sync --locked
source .venv/bin/activate
pre-commit install
pytest tests/
```

**Workflow**: pre-commit, pytest, scripts/create_issues.py.

## Docker Details

**File**: `DOCKER.md` (summary): docker-compose.yml services (dev/ci/prod); volumes, ports.

## Build Instructions

**File**: `BUILD_INSTRUCTIONS.md`, `EXECUTE_BUILD.md`: pyproject.toml deps, mojo package steps.

**Troubleshooting**: Docker perms, uv cache clean, mojo 1.0.0b2.

Updated: 2025-11-24
