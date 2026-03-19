#!/usr/bin/env bash
# Run a group of Mojo test files matching a pattern.
# Runs INSIDE the container — uses pixi run mojo directly.
#
# Usage: bash scripts/test_group.sh <path> <pattern>
#   path:    directory containing test files
#   pattern: glob pattern or filename (e.g. "test_*.mojo")

set -e

REPO_ROOT="$(pwd)"
TEST_PATH="${1:?Usage: test_group.sh <path> <pattern>}"
PATTERN="${2:?Usage: test_group.sh <path> <pattern>}"
test_count=0
passed_count=0
failed_count=0
failed_tests=""

echo "=================================================="
echo "Testing: $TEST_PATH"
echo "Pattern: $PATTERN"
echo "=================================================="

# Expand pattern into actual files
test_files=""
for pat in $PATTERN; do
    if [[ "$pat" == *"*"* ]]; then
        # Glob pattern - expand it
        for file in $TEST_PATH/$pat; do
            if [ -f "$file" ]; then
                test_files="$test_files $file"
            fi
        done
    else
        # Direct file or subdirectory pattern
        if [ -f "$TEST_PATH/$pat" ]; then
            test_files="$test_files $TEST_PATH/$pat"
        elif [[ "$pat" == *"/"* ]]; then
            for file in $TEST_PATH/$pat; do
                if [ -f "$file" ]; then
                    test_files="$test_files $file"
                fi
            done
        fi
    fi
done

if [ -z "$test_files" ]; then
    echo "❌ ERROR: No test files found in $TEST_PATH matching $PATTERN"
    echo "   This usually means the directory is empty or was renamed."
    echo "   Fix: update the test group path/pattern in comprehensive-tests.yml"
    exit 1
fi

# Run each test file
for test_file in $test_files; do
    if [ -f "$test_file" ]; then
        echo ""
        echo "Running: $test_file"
        test_count=$((test_count + 1))

        attempt=0
        max_attempts=3
        delay=1
        test_passed=0
        while [ $attempt -lt $max_attempts ]; do
            attempt=$((attempt + 1))
            if pixi run mojo -I "$REPO_ROOT" -I . "$test_file"; then
                test_passed=1
                break
            fi
            if [ $attempt -lt $max_attempts ]; then
                echo "⚠️  FAILED (attempt $attempt), retrying in ${delay}s: $test_file"
                sleep $delay
                delay=$((delay * 2))
            fi
        done
        if [ $test_passed -eq 1 ]; then
            if [ $attempt -eq 1 ]; then
                echo "✅ PASSED: $test_file"
            else
                echo "✅ PASSED on attempt $attempt: $test_file"
            fi
            passed_count=$((passed_count + 1))
        else
            echo "❌ FAILED: $test_file"
            failed_count=$((failed_count + 1))
            failed_tests="$failed_tests\n  - $test_file"
        fi
    fi
done

echo ""
echo "=================================================="
echo "Summary"
echo "=================================================="
echo "Total: $test_count tests"
echo "Passed: $passed_count tests"
echo "Failed: $failed_count tests"

if [ $failed_count -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    echo -e "$failed_tests"
    exit 1
fi
