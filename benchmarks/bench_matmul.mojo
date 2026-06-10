"""Progressive Matrix Multiplication Benchmark.

Demonstrates optimization stages from naive O(n³) to optimized GEMM.

Usage:
    # Run all stages on default sizes
    mojo run benchmarks/bench_matmul.mojo

    # Run specific stage
    mojo run benchmarks/bench_matmul.mojo --stage 2

    # Run on custom sizes
    mojo run benchmarks/bench_matmul.mojo --sizes "64,256,1024"

    # Run with specific dtype
    mojo run benchmarks/bench_matmul.mojo --dtype float64

    # Verify correctness only
    mojo run benchmarks/bench_matmul.mojo --verify-only

    # Custom iterations
    mojo run benchmarks/bench_matmul.mojo --iterations 20

Output:
    | Stage | 64x64  | 256x256 | 1024x1024 | GFLOPS | Speedup |
    |-------|--------|---------|-----------|--------|---------|
    | v0    | Xms    | Xms     | Xms       | X      | 1.0x    |
    | v1    | Xms    | Xms     | Xms       | X      | Xx      |
    | v2    | Xms    | Xms     | Xms       | X      | Xx      |
    | v3    | Xms    | Xms     | Xms       | X      | Xx      |
    | v4    | Xms    | Xms     | Xms       | X      | Xx      |
"""

from projectodyssey.utils.arg_parser import ArgumentParser
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from std.time import perf_counter_ns
from std.collections import List


def str_trunc(s: String, n: Int) -> String:
    """Truncate string to n bytes (ASCII-safe, for float display).

    mojo-1.0: String.substr() removed; use byte-slice + String(unsafe_from_utf8=).
    """
    var b = s.as_bytes()
    if len(b) <= n:
        return s
    return String(unsafe_from_utf8=b[:n])


from projectodyssey.core.matmul import (
    matmul_typed as _matmul_typed_impl,
    matmul_simd as _matmul_simd_impl,
    matmul_tiled as _matmul_tiled_impl,
)


# ============================================================================
# Matrix Multiplication Kernels (Baseline + Placeholders)
# ============================================================================


def matmul_naive[
    dtype: DType
](a: AnyTensor, b: AnyTensor, mut result: AnyTensor) raises:
    """Stage 0: Baseline naive implementation (Float64 conversion).

    This is the reference implementation. Slow but simple and correct.
    Uses Float64 conversion for all arithmetic operations.
    """
    var a_rows = a.shape()[0]
    var a_cols = a.shape()[1]
    var b_cols = b.shape()[1]

    # Naive triple nested loop with Float64 conversion
    for i in range(a_rows):
        for j in range(b_cols):
            var sum_val: Float64 = 0.0
            for k in range(a_cols):
                var a_val = a._get_float64(i * a_cols + k)
                var b_val = b._get_float64(k * b_cols + j)
                sum_val += a_val * b_val
            result._set_float64(i * b_cols + j, sum_val)


def matmul_typed[
    dtype: DType
](a: AnyTensor, b: AnyTensor, mut result: AnyTensor) raises:
    """Stage 1: Dtype-specific kernel (eliminates Float64 conversion).

    Uses real implementation from projectodyssey.core.matmul.
    Expected speedup: 3-5x over naive
    """
    _matmul_typed_impl(a, b, result)


def matmul_simd[
    dtype: DType
](a: AnyTensor, b: AnyTensor, mut result: AnyTensor) raises:
    """Stage 2: SIMD vectorization (vectorize J-loop).

    Uses real implementation from projectodyssey.core.matmul.
    Expected speedup: 4-8x over typed (15-40x cumulative)
    """
    _matmul_simd_impl(a, b, result)


def matmul_tiled[
    dtype: DType
](a: AnyTensor, b: AnyTensor, mut result: AnyTensor) raises:
    """Stage 3: Cache-aware blocking/tiling (production kernel).

    Uses real implementation from projectodyssey.core.matmul.
    This is the fastest kernel and is used by matmul() in production.
    Expected speedup: 3-5x over simd (50-150x cumulative)
    """
    _matmul_tiled_impl(a, b, result)


# ============================================================================
# Correctness Verification
# ============================================================================


def verify_stage[
    dtype: DType
](
    stage_id: Int,
    stage_name: String,
    a: AnyTensor,
    b: AnyTensor,
    reference: AnyTensor,
    rtol: Float64 = 1e-5,
    atol: Float64 = 1e-8,
) raises:
    """Verify a single optimization stage produces correct results.

    Args:
        stage_id: Stage number (0-4).
        stage_name: Descriptive stage name.
        a: First input matrix.
        b: Second input matrix.
        reference: Expected result (from baseline).
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        Error if results don't match within tolerance.
    """
    var result_shape = List[Int]()
    result_shape.append(a.shape()[0])
    result_shape.append(b.shape()[1])
    var result = AnyTensor(result_shape, dtype)

    # Run the stage-specific kernel
    if stage_id == 0:
        matmul_naive[dtype](a, b, result)
    elif stage_id == 1:
        matmul_typed[dtype](a, b, result)
    elif stage_id == 2:
        matmul_simd[dtype](a, b, result)
    elif stage_id == 3:
        matmul_tiled[dtype](a, b, result)
    else:
        raise Error("Invalid stage ID: " + String(stage_id))

    # Compare with reference
    for i in range(result.shape()[0]):
        for j in range(result.shape()[1]):
            var idx = i * result.shape()[1] + j
            var ref_val = reference._get_float64(idx)
            var result_val = result._get_float64(idx)
            var diff = abs(result_val - ref_val)
            var tolerance = atol + rtol * abs(ref_val)

            if diff > tolerance:
                raise Error(
                    stage_name
                    + " failed: mismatch at ["
                    + String(i)
                    + ","
                    + String(j)
                    + "] - expected "
                    + String(ref_val)
                    + ", got "
                    + String(result_val)
                    + " (diff="
                    + String(diff)
                    + ", tolerance="
                    + String(tolerance)
                    + ")"
                )


def verify_all_stages[dtype: DType](size: Int) raises:
    """Verify all optimization stages produce identical results.

    Args:
        size: Matrix size (NxN square matrices).

    Raises:
        Error if any stage produces incorrect results.
    """
    # Create test matrices
    var shape = List[Int]()
    shape.append(size)
    shape.append(size)

    var a = ones(shape, dtype)
    var b = ones(shape, dtype)

    # Generate reference result using baseline (v0)
    var reference = AnyTensor(shape, dtype)
    matmul_naive[dtype](a, b, reference)

    # Verify each stage
    verify_stage[dtype](0, "Stage 0 (baseline)", a, b, reference)
    verify_stage[dtype](1, "Stage 1 (dtype-specific)", a, b, reference)
    verify_stage[dtype](2, "Stage 2 (SIMD)", a, b, reference)
    verify_stage[dtype](3, "Stage 3 (cache-tiled)", a, b, reference)


# ============================================================================
# Benchmarking
# ============================================================================


def bench_stage[
    dtype: DType
](stage_id: Int, size: Int, iterations: Int) raises -> Float64:
    """Benchmark a single optimization stage.

    Args:
        stage_id: Stage number (0-4).
        size: Matrix size (NxN).
        iterations: Number of iterations to run.

    Returns:
        Mean execution time in milliseconds.
    """
    # Create test matrices
    var shape = List[Int]()
    shape.append(size)
    shape.append(size)

    var a = ones(shape, dtype)
    var b = ones(shape, dtype)
    var result = AnyTensor(shape, dtype)

    # Warmup (3 iterations)
    for _ in range(3):
        if stage_id == 0:
            matmul_naive[dtype](a, b, result)
        elif stage_id == 1:
            matmul_typed[dtype](a, b, result)
        elif stage_id == 2:
            matmul_simd[dtype](a, b, result)
        elif stage_id == 3:
            matmul_tiled[dtype](a, b, result)

    # Measure
    var total_ns: Int = 0
    for _ in range(iterations):
        var start = perf_counter_ns()

        if stage_id == 0:
            matmul_naive[dtype](a, b, result)
        elif stage_id == 1:
            matmul_typed[dtype](a, b, result)
        elif stage_id == 2:
            matmul_simd[dtype](a, b, result)
        elif stage_id == 3:
            matmul_tiled[dtype](a, b, result)

        var end = perf_counter_ns()
        total_ns += Int(end - start)

    # Return mean time in milliseconds
    return Float64(total_ns) / Float64(iterations) / 1_000_000.0


def calculate_gflops(size: Int, time_ms: Float64) -> Float64:
    """Calculate GFLOPS (billion floating-point operations per second).

    Matrix multiplication requires 2*M*N*K FLOPs for C = A @ B
    where A is MxK, B is KxN, C is MxN.
    For square matrices: 2*N*N*N = 2*N^3

    Args:
        size: Matrix size (NxN).
        time_ms: Execution time in milliseconds.

    Returns:
        GFLOPS (billions of floating-point operations per second).
    """
    var flops = 2.0 * Float64(size) * Float64(size) * Float64(size)
    var time_s = time_ms / 1000.0
    return flops / time_s / 1_000_000_000.0


# ============================================================================
# Argument Parsing
# ============================================================================


def print_help(prog_name: String):
    """Print help message and usage information.

    Args:
        prog_name: Program name from argv[0].
    """
    print(
        """
Matrix Multiplication Optimization Benchmark
=============================================

USAGE:
    pixi run mojo run """
        + prog_name
        + """ [OPTIONS]

OPTIONS:
    --stage <int>           Stage to run: -1 (all, default), 0-4 (specific stage)
                           0 = baseline (naive)
                           1 = dtype-specific (eliminate Float64 conversion)
                           2 = SIMD vectorization
                           3 = cache-aware tiling
                           4 = advanced optimizations (transpose + register blocking)

    --sizes <string>        Comma-separated matrix sizes (default: "64,256,512,1024")
                           Example: --sizes "64,128,256"

    --dtype <string>        Data type: float32 (default), float64, float16
                           Example: --dtype float64

    --iterations <int>      Number of benchmark iterations (default: 10)
                           More iterations = more accurate but slower
                           Example: --iterations 20

    --verify-only           Run correctness verification only (no benchmarking)
                           Fast check that all stages produce identical results

    -h, --help, -?          Show this help message

EXAMPLES:
    # Run all stages on default sizes
    pixi run mojo run """
        + prog_name
        + """

    # Run only Stage 2 (SIMD) on small sizes
    pixi run mojo run """
        + prog_name
        + """ --stage 2 --sizes "64,256"

    # Quick correctness check
    pixi run mojo run """
        + prog_name
        + """ --verify-only

    # High-precision benchmark with float64
    pixi run mojo run """
        + prog_name
        + """ --dtype float64 --iterations 20

OUTPUT:
    Benchmark results table showing:
    - Time per size (ms/μs/s)
    - GFLOPS (billions of floating-point operations per second)
    - Speedup over baseline

EXPECTED SPEEDUPS:
    - Stage 1 (dtype-specific):     3-5x over baseline
    - Stage 2 (SIMD):               15-40x over baseline (cumulative)
    - Stage 3 (cache-tiled):        50-150x over baseline (cumulative)
    - Stage 4 (advanced):           100-400x over baseline (cumulative)
"""
    )


def parse_sizes(sizes_str: String) raises -> List[Int]:
    """Parse comma-separated sizes string into list of integers.

    Args:
        sizes_str: Comma-separated sizes (e.g., "64,256,1024").

    Returns:
        List of integer sizes.

    Raises:
        Error if parsing fails.
    """
    var sizes = List[Int]()
    var parts = sizes_str.split(",")

    for i in range(len(parts)):
        var part = parts[i].strip()
        try:
            var size = Int(part)
            if size <= 0:
                raise Error("Matrix size must be positive, got: " + part)
            sizes.append(size)
        except:
            raise Error("Cannot parse '" + part + "' as integer size")

    return sizes^


def parse_args() raises -> Tuple[Int, List[Int], DType, Int, Bool]:
    """Parse command-line arguments.

    Returns:
        Tuple of (stage, sizes, dtype, iterations, verify_only).

    Raises:
        Error if argument parsing or validation fails.
    """
    var parser = ArgumentParser()

    # Which stage to run (-1 for all stages)
    parser.add_argument("stage", "int", "-1")

    # Matrix sizes (comma-separated, e.g., "64,256,1024")
    parser.add_argument("sizes", "string", "64,256,512,1024")

    # Data type to use
    parser.add_argument("dtype", "string", "float32")

    # Number of benchmark iterations
    parser.add_argument("iterations", "int", "10")

    # Verify correctness only (no benchmarking)
    parser.add_flag("verify-only")

    var args = parser.parse()

    # Parse stage
    var stage = args.get_int("stage", -1)
    if stage < -1 or stage > 4:
        raise Error("Invalid stage: must be -1 (all) or 0-4")

    # Parse sizes from comma-separated string
    var sizes_str = args.get_string("sizes", "64,256,512,1024")
    var sizes = parse_sizes(sizes_str)

    # Parse dtype
    var dtype_str = args.get_string("dtype", "float32")
    var dtype: DType
    if dtype_str == "float32":
        dtype = DType.float32
    elif dtype_str == "float64":
        dtype = DType.float64
    elif dtype_str == "float16":
        dtype = DType.float16
    else:
        raise Error("Invalid dtype: must be float32, float64, or float16")

    # Parse iterations
    var iterations = args.get_int("iterations", 10)
    if iterations < 1:
        raise Error("Invalid iterations: must be >= 1")

    # Parse verify-only flag
    var verify_only = args.get_bool("verify-only")

    return (stage, sizes^, dtype, iterations, verify_only)


# ============================================================================
# Main Benchmark Driver
# ============================================================================


def format_time(time_ms: Float64) -> String:
    """Format time in milliseconds with appropriate precision.

    Args:
        time_ms: Time in milliseconds.

    Returns:
        Formatted string (e.g., "12.34ms").
    """
    # mojo-1.0: String.substr(i, j) removed; use str_trunc() helper instead
    if time_ms < 0.01:
        return str_trunc(String(time_ms * 1000.0), 6) + "us"
    elif time_ms < 1.0:
        return str_trunc(String(time_ms), 6) + "ms"
    elif time_ms < 1000.0:
        return str_trunc(String(time_ms), 7) + "ms"
    else:
        return str_trunc(String(time_ms / 1000.0), 6) + "s"


def format_gflops(gflops: Float64) -> String:
    """Format GFLOPS with appropriate precision.

    Args:
        gflops: GFLOPS value.

    Returns:
        Formatted string (e.g., "12.3").
    """
    # mojo-1.0: String.substr(i, j) removed; use str_trunc() helper instead
    if gflops < 1.0:
        return str_trunc(String(gflops), 5)
    elif gflops < 10.0:
        return str_trunc(String(gflops), 4)
    elif gflops < 100.0:
        return str_trunc(String(gflops), 5)
    else:
        return str_trunc(String(gflops), 6)


def format_speedup(speedup: Float64) -> String:
    """Format speedup with appropriate precision.

    Args:
        speedup: Speedup factor.

    Returns:
        Formatted string (e.g., "12.3x").
    """
    # mojo-1.0: String.substr(i, j) removed; use str_trunc() helper instead
    if speedup < 1.0:
        return str_trunc(String(speedup), 4) + "x"
    elif speedup < 10.0:
        return str_trunc(String(speedup), 4) + "x"
    elif speedup < 100.0:
        return str_trunc(String(speedup), 5) + "x"
    else:
        return str_trunc(String(speedup), 6) + "x"


def run_benchmarks[
    dtype: DType
](stage: Int, sizes: List[Int], iterations: Int) raises:
    """Run benchmarks for specified stages and sizes.

    Args:
        stage: Stage to run (-1 for all).
        sizes: List of matrix sizes to test.
        iterations: Number of iterations per benchmark.
    """
    # Determine which stages to run
    var stages_to_run = List[Int]()
    if stage == -1:
        # Run all stages (0-3)
        for s in range(4):
            stages_to_run.append(s)
    else:
        # Run specific stage
        stages_to_run.append(stage)

    # Print separate table for each matrix size
    print("\nBenchmark Results:")
    print("=" * 90)

    # Store stage names
    var stage_names = List[String]()
    stage_names.append("naive")
    stage_names.append("typed")
    stage_names.append("simd")
    stage_names.append("tiled")

    # Benchmark each size separately
    for size_idx in range(len(sizes)):
        var size = sizes[size_idx]

        print("\n" + String(size) + "x" + String(size) + " Matrix:")
        print("-" * 90)
        print("| Stage       | Time      | GFLOPS | Incremental | Cumulative |")
        print("|-------------|-----------|--------|-------------|------------|")

        var baseline_time: Float64 = 0.0
        var prev_time: Float64 = 0.0

        # Benchmark each stage for this size
        for s in stages_to_run:
            var time_ms = bench_stage[dtype](s, size, iterations)

            # Track baseline and previous stage for speedup
            if s == 0:
                baseline_time = time_ms
                prev_time = time_ms

            # Calculate GFLOPS for this size
            var gflops = calculate_gflops(size, time_ms)

            # Calculate incremental speedup (vs previous stage)
            var incr_speedup: Float64
            if s == 0:
                incr_speedup = 1.0
            else:
                incr_speedup = prev_time / time_ms

            # Calculate cumulative speedup (vs baseline)
            var cum_speedup: Float64
            if s == 0:
                cum_speedup = 1.0
            else:
                cum_speedup = baseline_time / time_ms

            # Format stage name (padded to 11 chars)
            var stage_name = stage_names[s]
            var padding = 11 - stage_name.byte_length()
            for _ in range(padding):
                stage_name += " "

            var row = "| " + stage_name + " | " + format_time(time_ms)
            row += " | " + format_gflops(gflops)
            row += " | " + format_speedup(incr_speedup) + "     "
            row += " | " + format_speedup(cum_speedup) + "    |"
            print(row)

            # Update prev_time for next iteration
            prev_time = time_ms

    print("\n" + "=" * 90)
    print("")


def main() raises:
    """Main benchmark driver."""
    # Check for help flags
    from std.sys import argv

    var args = argv()
    for i in range(len(args)):
        var arg = args[i]
        if arg == "-h" or arg == "--help" or arg == "-?":
            print_help(args[0])
            return

    # Parse command-line arguments
    var args_tuple = parse_args()
    var stage = args_tuple[0]
    var sizes = List[Int]()
    for i in range(len(args_tuple[1])):
        sizes.append(args_tuple[1][i])
    var dtype = args_tuple[2]
    var iterations = args_tuple[3]
    var verify_only = args_tuple[4]

    print("=" * 90)
    print("Matrix Multiplication Optimization Benchmark")
    print("=" * 90)
    print("")
    print("Configuration:")
    print("  Stage: ", "all" if stage == -1 else String(stage))
    print("  Sizes: ", end="")
    for i in range(len(sizes)):
        if i > 0:
            print(", ", end="")
        print(sizes[i], end="")
    print("")

    var dtype_str = "float32"
    if dtype == DType.float64:
        dtype_str = "float64"
    elif dtype == DType.float16:
        dtype_str = "float16"
    print("  DType: ", dtype_str)
    print("  Iterations: ", iterations)
    print("  Verify only: ", verify_only)
    print("")

    # Verify correctness
    if verify_only or stage == -1:
        print("Verifying correctness...")

        # Verify on smaller sizes only (verification is slow)
        for size in sizes:
            if size <= 256:
                print("  Verifying ", size, "x", size, "...", end=" ")
                if dtype == DType.float32:
                    verify_all_stages[DType.float32](size)
                elif dtype == DType.float64:
                    verify_all_stages[DType.float64](size)
                elif dtype == DType.float16:
                    verify_all_stages[DType.float16](size)
                print("PASS")

        print("✅ All stages verified correct.")
        print("")

    # Exit if verify-only mode
    if verify_only:
        return

    # Run benchmarks
    if dtype == DType.float32:
        run_benchmarks[DType.float32](stage, sizes, iterations)
    elif dtype == DType.float64:
        run_benchmarks[DType.float64](stage, sizes, iterations)
    elif dtype == DType.float16:
        run_benchmarks[DType.float16](stage, sizes, iterations)

    print("Summary:")
    print("  - naive: Baseline implementation with Float64 conversion")
    print("  - typed: Dtype-specific kernels (eliminates type conversion)")
    print("  - simd: SIMD vectorization of inner loop")
    print("  - tiled: Cache-aware blocking/tiling (production kernel)")
    print("")
    print("Expected incremental speedups (each stage vs previous):")
    print("  - typed: 1.5-2x over naive")
    print("  - simd: 5-10x over typed")
    print("  - tiled: 3-5x over simd")
    print("")
    print("Expected cumulative speedups (vs naive):")
    print("  - typed: 1.5-2x")
    print("  - simd: 10-20x")
    print("  - tiled: 30-100x")
    print("")
    print("=" * 90)
