#!/usr/bin/env bash
#
# Smoke-test the locally-built *conda* package by installing it into a scratch
# pixi/conda env and running a one-line import. Called by `just install-local`.
#
# INTENTIONAL pixi/conda EXCEPTION (ADR-018): the rest of the repo migrated to
# uv, but this script tests the CONDA distributable published to
# modular-community. Installing a `.conda` artifact is inherently a
# conda-ecosystem operation, so it still shells out to `pixi`/conda. This runs
# only at conda-release time via `just install-local` (never a required CI
# check), and requires `pixi` on PATH. It has no bearing on the uv dev workflow.
#
# Why a shell script? `pixi add --channel <local-path>` requires shelling out
# and the surrounding logic is `mkdir`/`pixi init`/`pixi run` — no Mojo or
# Python value-add. If this grows past ~50 lines, promote to Python per
# ADR-001.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RECIPE_OUT="$REPO_ROOT/build/recipe"

if [ ! -d "$RECIPE_OUT" ]; then
    echo "❌ $RECIPE_OUT does not exist — run 'just build-recipe' first" >&2
    exit 1
fi

# Find the .conda package rattler-build produced. There should be exactly one
# under build/recipe/<platform>/.
CONDA_PKG=$(find "$RECIPE_OUT" -name 'odyssey-*.conda' -type f | head -1)
if [ -z "$CONDA_PKG" ]; then
    echo "❌ no odyssey-*.conda found under $RECIPE_OUT" >&2
    echo "   (did 'just build-recipe' succeed?)" >&2
    exit 1
fi

SCRATCH="$(mktemp -d -t odyssey-install-XXXXXX)"
echo "📦 Scratch pixi env: $SCRATCH"

cd "$SCRATCH"
pixi init --quiet

# Use `pixi project` for back-compat with pixi <0.40; newer pixi accepts
# `pixi workspace` as an alias.
pixi project channel add --quiet "$RECIPE_OUT"
pixi project channel add --quiet "https://conda.modular.com/max"

# Let the recipe's `mojo-compiler` run requirement resolve the right version
# from conda.modular.com/max — pinning to a dev build that isn't on the
# public channel breaks reproducibility.
pixi add --quiet odyssey

cat > smoke_test.mojo <<'EOF'
from odyssey.tensor.tensor import Tensor
from odyssey.tensor.any_tensor import zeros


def main() raises:
    var _t = Tensor[DType.float32]([2, 3])
    var _a = zeros([2, 3], DType.float32)
    print("install-local smoke test: OK")
EOF

echo "▶️  uv run mojo run smoke_test.mojo"
uv run mojo run smoke_test.mojo

echo
echo "✅ Local recipe install validated."
echo "   Scratch env left at $SCRATCH for inspection — safe to remove."
