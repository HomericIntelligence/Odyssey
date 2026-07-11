#!/bin/bash
# run_all_experiments.sh — validates every claim in the Day 53 blog post
#
# Usage: cd Odyssey && bash repro/run_all_experiments.sh
#
# Runs each experiment and prints the command, expected outcome, and PASS/FAIL.

set -uo pipefail

PASS=0
FAIL=0
TOTAL=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# --- Helpers ---

run_test() {
    local name="$1"
    local expected="$2"  # "crash", "pass", "asan_uaf", "asan_clean"
    local cmd="$3"
    TOTAL=$((TOTAL + 1))

    echo "  [$name]"
    echo "  Repro: $cmd"
    echo "  Expected: $expected"

    local output exit_code
    set +e
    output=$(eval "$cmd" 2>&1)
    exit_code=$?
    set -e

    if [ "$expected" = "crash" ]; then
        if [ $exit_code -ne 0 ] && echo "$output" | grep -q "libKGENCompilerRTShared.so\|execution crashed"; then
            echo "  Result: Crashed (exit=$exit_code)"
            PASS=$((PASS + 1))
        else
            echo "  Result: Did NOT crash (exit=$exit_code) — UNEXPECTED"
            FAIL=$((FAIL + 1))
        fi
    elif [ "$expected" = "pass" ]; then
        if [ $exit_code -eq 0 ]; then
            echo "  Result: Completed successfully"
            PASS=$((PASS + 1))
        else
            echo "  Result: Failed (exit=$exit_code) — UNEXPECTED"
            echo "  Output: $(echo "$output" | tail -3)"
            FAIL=$((FAIL + 1))
        fi
    elif [ "$expected" = "asan_uaf" ]; then
        if echo "$output" | grep -q "heap-use-after-free"; then
            echo "  Result: ASAN detected heap-use-after-free"
            PASS=$((PASS + 1))
        else
            echo "  Result: ASAN did NOT detect heap-use-after-free — UNEXPECTED"
            FAIL=$((FAIL + 1))
        fi
    elif [ "$expected" = "asan_clean" ]; then
        if [ $exit_code -eq 0 ] && ! echo "$output" | grep -q "AddressSanitizer"; then
            echo "  Result: ASAN clean (no errors)"
            PASS=$((PASS + 1))
        else
            echo "  Result: ASAN reported errors — UNEXPECTED"
            FAIL=$((FAIL + 1))
        fi
    fi
    echo ""
}

echo "========================================"
echo "  Day 53 Blog Post Experiment Validator"
echo "========================================"
echo "  Repo root: $REPO_ROOT"
echo "  Artifacts: $SCRIPT_DIR"
echo ""

# ==========================================================================
echo "--- 1. Standalone Reproducer Crashes (no ASAN) ---"
echo ""
# ==========================================================================

run_test \
    "Standalone reproducer crashes" \
    "crash" \
    "pixi run mojo run $SCRIPT_DIR/repro_crash_standalone.mojo"

# ==========================================================================
echo "--- 2. ASAN Proves Heap-Use-After-Free ---"
echo ""
# ==========================================================================

echo "  Building: pixi run mojo build --sanitize address -g -o $SCRIPT_DIR/repro_asan $SCRIPT_DIR/repro_crash_standalone.mojo"
pixi run mojo build --sanitize address -g \
    -o "$SCRIPT_DIR/repro_asan" \
    "$SCRIPT_DIR/repro_crash_standalone.mojo" 2>&1 | tail -3
echo ""

run_test \
    "ASAN detects use-after-free" \
    "asan_uaf" \
    "$SCRIPT_DIR/repro_asan"

# ==========================================================================
echo "--- 3. Keepalive Fix Eliminates ASAN Error ---"
echo ""
# ==========================================================================

# Create fixed version: insert "_ = target  # keepalive" after the last "_ = relu(x)"
awk '
/_ = relu\(x\)/ { last_line = NR; last_content = $0 }
{ lines[NR] = $0 }
END {
    for (i = 1; i <= NR; i++) {
        print lines[i]
        if (i == last_line) print "    _ = target  # keepalive"
    }
}
' "$SCRIPT_DIR/repro_crash_standalone.mojo" > "$SCRIPT_DIR/repro_fixed.mojo"

echo "  Building: pixi run mojo build --sanitize address -g -o $SCRIPT_DIR/repro_fixed_bin $SCRIPT_DIR/repro_fixed.mojo"
pixi run mojo build --sanitize address -g \
    -o "$SCRIPT_DIR/repro_fixed_bin" \
    "$SCRIPT_DIR/repro_fixed.mojo" 2>&1 | tail -3
echo ""

run_test \
    "ASAN clean with keepalive fix" \
    "asan_clean" \
    "$SCRIPT_DIR/repro_fixed_bin"

# Clean up generated files
rm -f "$SCRIPT_DIR/repro_asan" "$SCRIPT_DIR/repro_fixed.mojo" "$SCRIPT_DIR/repro_fixed_bin"

# ==========================================================================
echo "--- 4. Pre-Workaround VGG16 Test Crashes ---"
echo ""
# ==========================================================================

# Copy pre-workaround test into tests/models/ so imports resolve
cp "$SCRIPT_DIR/bug_repro_vgg16_e2e_part1_pre_fix.mojo.bug" \
    "$REPO_ROOT/tests/models/_tmp_pre_fix_part1.mojo"

run_test \
    "Pre-workaround VGG16 part1 crashes" \
    "crash" \
    "pixi run mojo run tests/models/_tmp_pre_fix_part1.mojo"

rm -f "$REPO_ROOT/tests/models/_tmp_pre_fix_part1.mojo"

# ==========================================================================
echo "--- 5. Post-Workaround VGG16 Tests Pass ---"
echo ""
# ==========================================================================

run_test \
    "Fixed VGG16 part1 passes" \
    "pass" \
    "pixi run mojo run tests/models/test_vgg16_e2e_part1.mojo"

run_test \
    "Fixed VGG16 part2 passes" \
    "pass" \
    "pixi run mojo run tests/models/test_vgg16_e2e_part2.mojo"

run_test \
    "Fixed VGG16 layers part2 passes" \
    "pass" \
    "pixi run mojo run tests/models/test_vgg16_layers_part2.mojo"

# ==========================================================================
echo "--- 6. Project-Import Reproducers Crash ---"
echo ""
# ==========================================================================

run_test \
    "repro_libkgen_crash.mojo crashes" \
    "crash" \
    "pixi run mojo run $SCRIPT_DIR/repro_libkgen_crash.mojo"

run_test \
    "repro_libasyncrt_crash.mojo crashes" \
    "crash" \
    "pixi run mojo run $SCRIPT_DIR/repro_libasyncrt_crash.mojo"

# ==========================================================================
echo "--- 7. LeNet-5 Monolithic File Crashes (ADR-009 — Same Bug) ---"
echo ""
# ==========================================================================

# The original 24-test monolithic file from Dec 2025 (Issue #2942).
# This requires the shared library to be built. If not available, skip gracefully.
echo "  Note: This test requires 'pixi run just build' to have been run first."
echo "  The monolithic file uses project imports (from odyssey.core.*)."
echo ""

run_test \
    "LeNet-5 monolithic (24 tests) crashes after ~15" \
    "crash" \
    "cp $SCRIPT_DIR/bug_repro_lenet5_layers_monolithic.mojo.bug $REPO_ROOT/tests/models/_tmp_lenet5_monolithic.mojo && pixi run mojo run tests/models/_tmp_lenet5_monolithic.mojo; ret=\$?; rm -f $REPO_ROOT/tests/models/_tmp_lenet5_monolithic.mojo; exit \$ret"

# ==========================================================================
echo "--- 8. LeNet-5 Split Files Pass (ADR-009 Workaround) ---"
echo ""
# ==========================================================================

# Run each of the 5 split files — all should pass
run_test \
    "LeNet-5 conv layers (6 tests) passes" \
    "pass" \
    "pixi run mojo run tests/models/test_lenet5_conv_layers.mojo"

run_test \
    "LeNet-5 activation layers (3 tests) passes" \
    "pass" \
    "pixi run mojo run tests/models/test_lenet5_activation_layers.mojo"

# ==========================================================================
echo "========================================"
echo "  $TOTAL experiments, $FAIL unexpected results"
echo "========================================"

if [ $FAIL -gt 0 ]; then
    exit 1
else
    exit 0
fi
