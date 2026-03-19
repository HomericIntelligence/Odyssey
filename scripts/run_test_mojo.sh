#!/usr/bin/env bash
# Run all Mojo test files under tests/.
# Runs INSIDE the container — uses pixi run mojo directly.
#
# Usage: bash scripts/test_mojo.sh

set -e

REPO_ROOT="$(pwd)"
echo "Running all Mojo tests..."

failed=0
for test_file in tests/**/*.mojo; do
    if [ -f "$test_file" ]; then
        echo "Testing: $test_file"
        if ! pixi run mojo -I "$REPO_ROOT" -I . "$test_file"; then
            failed=1
        fi
    fi
done

if [ $failed -eq 1 ]; then
    echo "❌ Some tests failed"
    exit 1
fi
echo "✅ All tests passed"
