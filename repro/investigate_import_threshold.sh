#!/usr/bin/env bash
# Investigates what triggers the Mojo JIT crash:
#   - Is it monomorphization count (parametric fn × DType variants)?
#   - Is it line count?
#   - Is it both?
#   - Is it import chain depth into real modules?
#
# All Experiment 1/2/3 tests are SINGLE-FILE (no cross-file imports needed).
# Experiment 4 uses real Odyssey modules (requires just build first).
#
# Run from the repo root with mojo in PATH:
#   export PATH=/path/to/.pixi/envs/default/bin:$PATH
#   bash repro/investigate_import_threshold.sh
#
# Expected output shows at what threshold the JIT crashes.

set -euo pipefail

TMP_DIR="$(mktemp -d /tmp/mojo-import-threshold-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "=== Mojo JIT Import Chain Threshold Investigation ==="
echo "Mojo: $(mojo --version 2>&1)"
echo "GLIBC: $(ldd --version 2>&1 | head -1)"
echo "Tmp dir: $TMP_DIR"
echo ""

run_test() {
    local test_file="$1"
    local label="$2"

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
# Experiment 1: Single parametric fn, varying DType variants
# Each DType callsite = one monomorphization
# ============================================================
echo "--- Experiment 1: Monomorphization count (1 fn × N dtypes) ---"

gen_mono_test() {
    local n_dtypes="$1"
    local out="$TMP_DIR/test_mono_${n_dtypes}.mojo"

    # List of dtype/value pairs
    local dtypes=("float32:Float32:1.0" "float64:Float64:2.0" "int32:Int32:3" "int64:Int64:4"
                  "uint8:UInt8:5" "int8:Int8:6" "int16:Int16:7" "uint16:UInt16:8"
                  "uint32:UInt32:9" "uint64:UInt64:10" "float16:Float16:1.0" "bool:Bool:1")

    {
        echo 'from std.memory import UnsafePointer'
        echo ''
        echo 'fn typed_sum[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int) -> Scalar[dtype]:'
        echo '    var total = Scalar[dtype](0)'
        echo '    for i in range(n):'
        echo '        total += ptr[i]'
        echo '    return total'
        echo ''
        echo 'def main() raises:'

        for i in $(seq 0 $((n_dtypes - 1))); do
            local entry="${dtypes[$i]}"
            local mojo_dtype="${entry%%:*}"
            local rest="${entry#*:}"
            local scalar_type="${rest%%:*}"
            local val="${rest##*:}"
            echo "    var a${i} = UnsafePointer[${scalar_type}].alloc(16)"
            echo "    for j in range(16): a${i}[j] = ${scalar_type}(${val})"
            echo "    _ = typed_sum[DType.${mojo_dtype}](a${i}, 16)"
            echo "    a${i}.free()"
        done
        echo '    print("PASS")'
    } > "$out"
    echo "$out"
}

for n in 1 2 4 6 8 10 12; do
    f=$(gen_mono_test "$n")
    run_test "$f" "1 fn × $n DType variants = $n monomorphizations"
done

# ============================================================
# Experiment 2: N parametric fns, fixed DType set
# ============================================================
echo ""
echo "--- Experiment 2: Function count (N fns × 10 dtypes) ---"

gen_nfn_test() {
    local n_fns="$1"
    local out="$TMP_DIR/test_nfn_${n_fns}.mojo"
    local dtypes=("float32:Float32:1.0" "float64:Float64:2.0" "int32:Int32:3" "int64:Int64:4"
                  "uint8:UInt8:5" "int8:Int8:6" "int16:Int16:7" "uint16:UInt16:8"
                  "uint32:UInt32:9" "uint64:UInt64:10")

    {
        echo 'from std.memory import UnsafePointer'
        echo ''

        for i in $(seq 1 "$n_fns"); do
            echo "fn typed_op_${i}[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int) -> Scalar[dtype]:"
            echo '    var total = Scalar[dtype](0)'
            echo '    for j in range(n):'
            echo "        total += ptr[j] * Scalar[dtype]($i)"
            echo '    return total'
            echo ''
        done

        echo 'def main() raises:'

        for i in $(seq 1 "$n_fns"); do
            for entry in "${dtypes[@]}"; do
                local mojo_dtype="${entry%%:*}"
                local rest="${entry#*:}"
                local scalar_type="${rest%%:*}"
                local val="${rest##*:}"
                echo "    var a_fn${i}_${mojo_dtype} = UnsafePointer[${scalar_type}].alloc(8)"
                echo "    for j in range(8): a_fn${i}_${mojo_dtype}[j] = ${scalar_type}(${val})"
                echo "    _ = typed_op_${i}[DType.${mojo_dtype}](a_fn${i}_${mojo_dtype}, 8)"
                echo "    a_fn${i}_${mojo_dtype}.free()"
            done
        done

        echo '    print("PASS")'
    } > "$out"
    echo "$out"
}

for n_fns in 1 2 5 10 14 20 30; do
    f=$(gen_nfn_test "$n_fns")
    local_mono=$((n_fns * 10))
    run_test "$f" "$n_fns fns × 10 dtypes = $local_mono monomorphizations"
done

# ============================================================
# Experiment 3: Large single-file line count (no generics)
# Tests whether raw line count matters independent of generics
# ============================================================
echo ""
echo "--- Experiment 3: Raw line count (no generics) ---"

gen_linecount_test() {
    local n_fns="$1"
    local out="$TMP_DIR/test_lines_${n_fns}.mojo"

    {
        echo 'from std.memory import UnsafePointer'
        echo ''

        for i in $(seq 1 "$n_fns"); do
            echo "def concrete_fn_${i}(x: Float32) -> Float32:"
            echo "    var a = x * Float32($i)"
            echo "    var b = a + Float32($((i + 1)))"
            echo "    var c = b - Float32($((i * 2)))"
            echo "    var d = c * Float32(0.5)"
            echo "    var e = d + Float32(0.1)"
            echo "    return e"
            echo ''
        done

        echo 'def main() raises:'
        for i in $(seq 1 "$n_fns"); do
            echo "    _ = concrete_fn_${i}(Float32($i))"
        done
        echo '    print("PASS")'
    } > "$out"
    echo "$out"
}

for n_fns in 10 50 100 200 500 1000; do
    f=$(gen_linecount_test "$n_fns")
    approx_lines=$((n_fns * 8))
    run_test "$f" "$n_fns concrete fns (~$approx_lines lines, 0 generics)"
done

# ============================================================
# Experiment 4: Real Odyssey module imports
# Tests whether cross-module imports at module level crash
# Requires: MOJO_PACKAGE_PATH pointing to built shared package
# ============================================================
echo ""
echo "--- Experiment 4: Real cross-module import chains ---"
echo "  (Skipping if shared package not built)"

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARED_PKG="$PROJ_ROOT/build/shared.mojopkg"

if [ ! -f "$SHARED_PKG" ]; then
    echo "  SKIP: $SHARED_PKG not found. Run 'just build' first."
else
    # Test 1: Module-level import of reduction (-> shape 1371 lines)
    cat > "$TMP_DIR/test_real_reduction.mojo" << 'MOJO'
from odyssey.core.reduction import sum as reduce_sum
def test() raises:
    from odyssey.tensor.any_tensor import ones
    var t = ones(List[Int](4), DType.float32)
    _ = reduce_sum(t)
    print("PASS")
def main() raises:
    test()
MOJO
    timeout 30 mojo run -I "$PROJ_ROOT/build" "$TMP_DIR/test_real_reduction.mojo" >/dev/null 2>&1 \
        && echo "  reduction module-level import -> PASS" \
        || echo "  reduction module-level import -> CRASH"

    # Test 2: Module-level import of loss_utils (-> elementwise 1650 -> dtype_dispatch 1520)
    cat > "$TMP_DIR/test_real_loss_utils.mojo" << 'MOJO'
from odyssey.core.loss_utils import clip_predictions
def test() raises:
    from odyssey.tensor.any_tensor import ones
    var t = ones(List[Int](4), DType.float32)
    _ = clip_predictions(t)
    print("PASS")
def main() raises:
    test()
MOJO
    timeout 30 mojo run -I "$PROJ_ROOT/build" "$TMP_DIR/test_real_loss_utils.mojo" >/dev/null 2>&1 \
        && echo "  loss_utils module-level import -> PASS" \
        || echo "  loss_utils module-level import -> CRASH"

    # Test 3: Per-function imports (fixed version)
    cat > "$TMP_DIR/test_real_perfn.mojo" << 'MOJO'
def test() raises:
    from odyssey.core.reduction import sum as reduce_sum
    from odyssey.core.loss_utils import clip_predictions
    from odyssey.tensor.any_tensor import ones
    var t = ones(List[Int](4), DType.float32)
    _ = reduce_sum(t)
    _ = clip_predictions(t)
    print("PASS")
def main() raises:
    test()
MOJO
    timeout 30 mojo run -I "$PROJ_ROOT/build" "$TMP_DIR/test_real_perfn.mojo" >/dev/null 2>&1 \
        && echo "  per-function imports (fixed) -> PASS" \
        || echo "  per-function imports (fixed) -> CRASH"
fi

echo ""
echo "=== Investigation complete ==="
echo ""
echo "Key interpretation:"
echo "  Exp 1 crash -> monomorphization count alone triggers crash"
echo "  Exp 2 crash at N fns -> threshold is N*10 monomorphizations"
echo "  Exp 3 crash at N lines -> raw line count triggers crash (no generics needed)"
echo "  Exp 4 shows whether real module imports crash vs per-function imports pass"
