#!/usr/bin/env bash
# Build/compile Mojo files with mode-specific flags.
# Runs INSIDE the container — uses uv run mojo directly.
#
# Usage: bash scripts/build_mojo.sh [mode]
#   mode: debug (default) | release | test | ci

set -euo pipefail

MODE="${1:-debug}"
REPO_ROOT="$(pwd)"
STRICT="--validate-doc-strings"
FAIL_ON_ERROR=0

case "$MODE" in
    debug)
        FLAGS="-g --no-optimization $STRICT"
        ;;
    release)
        FLAGS="-g0 -O3 $STRICT"
        ;;
    test)
        FLAGS="-g1 $STRICT"
        ;;
    ci)
        FLAGS="-g1 $STRICT"
        # Don't fail on linker errors - Mojo doesn't support -lm flag yet
        FAIL_ON_ERROR=0
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Valid modes: debug | release | test | ci"
        exit 1
        ;;
esac

BUILD_DIR="build/$MODE"
mkdir -p "$BUILD_DIR"

echo "🔨 Building Mojo files"
echo "Mode:  $MODE"
echo "Flags: $FLAGS"
echo "Out:   $BUILD_DIR"
echo

FAILED=0

find . -name "*.mojo" \
    -not -path "./.pixi/*" \
    -not -path "./worktrees/*" \
    -not -path "./.claude/*" \
    -not -path "./tests/*" \
    -not -path "./shared/*" \
    -not -path "./benchmarks/*" \
    -not -name "test_*.mojo" \
    -not -name "model.mojo" \
    | while read -r file; do
        out="$BUILD_DIR/$(basename "$file" .mojo)"
        echo "→ Building: $file"

        if ! uv run mojo build $FLAGS -I "$REPO_ROOT" "$file" -o "$out" 2>&1; then
            echo "❌ Failed: $file"
            FAILED=1
            if [[ "$FAIL_ON_ERROR" -eq 1 ]]; then
                exit 1
            fi
        fi
    done

if [[ "$FAILED" -eq 1 ]]; then
    echo
    echo "⚠️  Build completed with errors"
    echo "Outputs in $BUILD_DIR"
    [[ "$FAIL_ON_ERROR" -eq 1 ]] && exit 1
else
    echo
    echo "✅ Build successful"
    echo "Outputs in $BUILD_DIR"
fi
