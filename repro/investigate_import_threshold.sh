#!/usr/bin/env bash
# Investigates the JIT import chain threshold that triggers the crash.
#
# Strategy: Generate synthetic Mojo modules of increasing size and measure
# at what transitive import depth / total lines the crash occurs.
#
# Run this inside the Podman container: just shell, then bash repro/investigate_import_threshold.sh
#
# Expected output:
#   depth=1 size=500   -> PASS
#   depth=1 size=1000  -> PASS
#   depth=1 size=1500  -> CRASH  <-- threshold found
#   depth=2 size=500   -> PASS
#   depth=2 size=1000  -> CRASH  <-- threshold found for 2-level chain
#   ...
#
# Results feed directly into a Mojo bug report with exact reproduction parameters.

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR="$(mktemp -d /tmp/mojo-import-threshold-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "=== Mojo JIT Import Chain Threshold Investigation ==="
echo "Tmp dir: $TMP_DIR"
echo ""

# Generate a synthetic Mojo module with N lines of real (non-trivial) functions
generate_module() {
    local name="$1"
    local n_funcs="$2"
    local imports="$3"  # optional import line to include at module level
    local out="$TMP_DIR/${name}.mojo"

    {
        echo '"""Synthetic heavy module: '"$name"' ('"$n_funcs"' functions)."""'
        echo ""
        echo "from std.collections import List"
        echo "from std.memory import UnsafePointer"
        if [ -n "$imports" ]; then
            echo "$imports"
        fi
        echo ""

        for i in $(seq 1 "$n_funcs"); do
            echo "def ${name}_op_${i}(x: Int) -> Int:"
            echo "    # Function $i: simulate real computation"
            echo "    var a = x * $i"
            echo "    var b = a + $((i * 7))"
            echo "    var c = b - $((i * 3))"
            # Add some local state to increase IR volume
            for j in 1 2 3 4 5; do
                echo "    var v${j} = c + $((i * j))"
            done
            echo "    return v1 + v2 + v3 + v4 + v5"
            echo ""
        done
    } > "$out"
    echo "$out"
}

# Generate a test file that imports from a chain
generate_test() {
    local chain_top="$1"  # module to import from
    local out="$TMP_DIR/test_threshold.mojo"

    {
        echo '"""Test file for import chain threshold."""'
        echo ""
        echo "from ${chain_top} import ${chain_top}_op_1"
        echo ""
        echo "def test_basic() raises:"
        echo "    var result = ${chain_top}_op_1(42)"
        echo "    print('result:', result)"
        echo ""
        echo "def main() raises:"
        echo "    print('Running: test_basic')"
        echo "    test_basic()"
        echo "    print('PASS')"
    } > "$out"
    echo "$out"
}

run_test() {
    local test_file="$1"
    local label="$2"

    # Run with timeout to catch hangs; capture both stdout+stderr
    if timeout 30 mojo run "$test_file" >/dev/null 2>&1; then
        echo "  $label -> PASS"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "  $label -> TIMEOUT (30s)"
        else
            echo "  $label -> CRASH (exit $exit_code)"
        fi
        return 1
    fi
}

# ============================================================
# Experiment 1: Single module at various sizes
# ============================================================
echo "--- Experiment 1: Single-level import at various sizes ---"
for n_funcs in 10 20 50 100 200 500; do
    mod_file=$(generate_module "heavy_single" "$n_funcs" "")
    test_file=$(generate_test "heavy_single")
    run_test "$test_file" "depth=1 funcs=$n_funcs (~$((n_funcs * 8)) lines)"
done

# ============================================================
# Experiment 2: Two-level chain (test -> B -> A)
# ============================================================
echo ""
echo "--- Experiment 2: Two-level import chain ---"
for n_funcs in 10 50 100 200; do
    mod_a=$(generate_module "heavy_a" "$n_funcs" "")
    mod_b=$(generate_module "heavy_b" "$n_funcs" "from heavy_a import heavy_a_op_1")
    test_file=$(generate_test "heavy_b")
    run_test "$test_file" "depth=2 funcs_per_level=$n_funcs (~$((n_funcs * 8 * 2)) lines total)"
done

# ============================================================
# Experiment 3: Three-level chain (test -> C -> B -> A)
# ============================================================
echo ""
echo "--- Experiment 3: Three-level import chain ---"
for n_funcs in 10 50 100 200; do
    mod_a=$(generate_module "heavy_aa" "$n_funcs" "")
    mod_b=$(generate_module "heavy_bb" "$n_funcs" "from heavy_aa import heavy_aa_op_1")
    mod_c=$(generate_module "heavy_cc" "$n_funcs" "from heavy_bb import heavy_bb_op_1")
    test_file=$(generate_test "heavy_cc")
    run_test "$test_file" "depth=3 funcs_per_level=$n_funcs (~$((n_funcs * 8 * 3)) lines total)"
done

# ============================================================
# Experiment 4: Real-world sizes (matching ProjectOdyssey modules)
# ============================================================
echo ""
echo "--- Experiment 4: Real-world module sizes ---"
# Simulates: test -> loss_utils (~335 lines) -> elementwise (1650) -> dtype_dispatch (1520)
mod_dtype=$(generate_module "synth_dtype_dispatch" "150" "")  # ~1500 lines
mod_elem=$(generate_module "synth_elementwise" "130" "from synth_dtype_dispatch import synth_dtype_dispatch_op_1")  # ~1650 lines
mod_loss=$(generate_module "synth_loss_utils" "35" "from synth_elementwise import synth_elementwise_op_1")  # ~335 lines
test_file=$(generate_test "synth_loss_utils")
run_test "$test_file" "real-world: loss_utils->elementwise->dtype_dispatch chain"

# Simulates: test -> reduction (~1255) -> shape (1371)
mod_shape=$(generate_module "synth_shape" "130" "")  # ~1371 lines
mod_reduction=$(generate_module "synth_reduction" "120" "from synth_shape import synth_shape_op_1")  # ~1255 lines
test_file=$(generate_test "synth_reduction")
run_test "$test_file" "real-world: reduction->shape chain"

echo ""
echo "=== Investigation complete ==="
echo ""
echo "Note: If all experiments PASS, the crash depends on:"
echo "  1. Parametric monomorphizations (dtype_dispatch has 176+) — not just line count"
echo "  2. System state / memory fragmentation at test start"
echo "  3. Combination of many test files in same mojo test session (accumulation)"
echo ""
echo "Next step: Run with 'mojo test' instead of 'mojo run' to test the"
echo "test-runner accumulation hypothesis."
