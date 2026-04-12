#!/usr/bin/env bash
# test-with-retry.sh — Run a single Mojo test file with JIT crash retry logic.
#
# Usage: bash scripts/test-with-retry.sh <REPO_ROOT> <test_file>
#
# Exit codes:
#   0 — test passed (on first attempt or after retry)
#   1 — test failed with a real failure (assertion error, compile error) — NOT retried
#   2 — test crashed with JIT fault on both attempts (persisted after retry)
#
# The retry targets ONLY the Mojo 0.26.x JIT crash signature ("execution crashed"
# from libKGENCompilerRTShared.so). Real test failures are never retried.
#
# See: docs/adr/ADR-014-jit-crash-retry-mitigation.md
# See: https://github.com/modular/modular/issues/6413

set -o pipefail

REPO_ROOT="${1:?Usage: test-with-retry.sh <REPO_ROOT> <test_file>}"
TEST_FILE="${2:?Usage: test-with-retry.sh <REPO_ROOT> <test_file>}"
MAX_RETRIES="${TEST_WITH_RETRY_MAX:-1}"
JIT_CRASH_MARKER="execution crashed"

run_test() {
    local output_file
    output_file=$(mktemp)

    # set -o pipefail propagates the mojo exit code through the tee pipeline.
    # Capture it directly from $? — no PIPESTATUS indexing needed.
    pixi run mojo --Werror -I "$REPO_ROOT" -I . "$TEST_FILE" 2>&1 | tee "$output_file"
    local exit_code=$?

    OUTPUT=$(cat "$output_file")
    rm -f "$output_file"
    return $exit_code
}

# --- Attempt 1 ---
OUTPUT=""
run_test
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "PASSED: $TEST_FILE"
    exit 0
fi

# Check if the failure is a JIT crash (not a real test failure)
if echo "$OUTPUT" | grep -q "$JIT_CRASH_MARKER"; then
    echo "JIT crash detected in $TEST_FILE — retrying (attempt 2/$((MAX_RETRIES + 1)))..."
else
    # Real test failure — do NOT retry
    echo "FAILED: $TEST_FILE"
    exit 1
fi

# --- Retry loop ---
attempt=2
while [ $((attempt - 1)) -le "$MAX_RETRIES" ]; do
    OUTPUT=""
    run_test
    RESULT=$?

    if [ $RESULT -eq 0 ]; then
        echo "PASSED on retry (attempt $attempt): $TEST_FILE"
        exit 0
    fi

    if echo "$OUTPUT" | grep -q "$JIT_CRASH_MARKER"; then
        if [ $attempt -le "$MAX_RETRIES" ]; then
            echo "JIT crash persisted — retrying (attempt $((attempt + 1))/$((MAX_RETRIES + 1)))..."
        fi
    else
        # Retry produced a real failure — report as real failure
        echo "FAILED: $TEST_FILE"
        exit 1
    fi

    attempt=$((attempt + 1))
done

echo "FAILED (JIT crash persisted after $((MAX_RETRIES + 1)) attempts): $TEST_FILE"
exit 2
