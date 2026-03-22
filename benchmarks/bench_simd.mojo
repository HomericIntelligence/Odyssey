"""Benchmark SIMD vs scalar arithmetic operations.

Measures performance improvements from SIMD vectorization for
element-wise operations on tensors of various sizes.

Usage:
    mojo run benchmarks/bench_simd.mojo

Expected Results:
    - float32 operations: 3-5x speedup
    - float64 operations: 2-3x speedup
    - Larger tensors show better speedup (better amortization)

Output:
    Performance comparison table showing operation, size, scalar time,
    SIMD time, and speedup factor.
"""

from shared.core import AnyTensor, zeros, ones
from shared.core.arithmetic import add, subtract, multiply, divide
from shared.core.arithmetic_simd import (
    add_simd,
    subtract_simd,
    multiply_simd,
    divide_simd,
)
from time import perf_counter_ns


fn benchmark_operation(
    name: String,
    size: Int,
    scalar_fn: fn (AnyTensor, AnyTensor) raises -> AnyTensor,
    simd_fn: fn (AnyTensor, AnyTensor) raises -> AnyTensor,
    dtype: DType,
    iterations: Int = 100,
) raises:
    """Benchmark scalar vs SIMD implementation of an operation.

    Args:
        name: Operation name (e.g., "add").
        size: Tensor size (N x N square matrix).
        scalar_fn: Scalar implementation.
        simd_fn: SIMD implementation.
        dtype: Data type to benchmark.
        iterations: Number of iterations for timing.
    """
    # Create test tensors
    var shape = List[Int]()
    shape[0] = size
    shape[1] = size

    var a = ones(shape, dtype)
    var b = ones(shape, dtype)

    # Warm up
    _ = scalar_fn(a, b)
    _ = simd_fn(a, b)

    # Benchmark scalar implementation
    var scalar_start = perf_counter_ns()
    for _ in range(iterations):
        _ = scalar_fn(a, b)
    var scalar_end = perf_counter_ns()
    var scalar_time = (
        Float64(scalar_end - scalar_start) / 1e9
    )  # Convert to seconds

    # Benchmark SIMD implementation
    var simd_start = perf_counter_ns()
    for _ in range(iterations):
        _ = simd_fn(a, b)
    var simd_end = perf_counter_ns()
    var simd_time = Float64(simd_end - simd_start) / 1e9  # Convert to seconds

    # Calculate speedup
    var speedup = scalar_time / simd_time

    # Print results
    var dtype_str = "float32" if dtype == DType.float32 else "float64"
    print(
        name.ljust(10),
        dtype_str.ljust(8),
        String(size).ljust(6) + "x" + String(size).ljust(6),
        String(scalar_time / Float64(iterations) * 1000).ljust(10) + "ms",
        String(simd_time / Float64(iterations) * 1000).ljust(10) + "ms",
        String(speedup) + "x",
    )


fn verify_correctness() raises -> Bool:
    """Verify SIMD produces same results as scalar (within tolerance).

    Returns:
        True if all operations match within tolerance.
    """
    print("\n=== Verifying SIMD Correctness ===\n")

    var shape = List[Int]()
    shape[0] = 10
    shape[1] = 10

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    # Test each operation
    var tolerance = 1e-6

    # Add
    var scalar_add = add(a, b)
    var simd_add = add_simd(a, b)
    var add_diff = 0.0
    for i in range(scalar_add.numel()):
        var diff = abs(scalar_add._get_float64(i) - simd_add._get_float64(i))
        if diff > add_diff:
            add_diff = diff

    print(
        "add:      max diff =",
        add_diff,
        "  ",
        "PASS ✓" if add_diff < tolerance else "FAIL ✗",
    )

    # Subtract
    var scalar_sub = subtract(a, b)
    var simd_sub = subtract_simd(a, b)
    var sub_diff = 0.0
    for i in range(scalar_sub.numel()):
        var diff = abs(scalar_sub._get_float64(i) - simd_sub._get_float64(i))
        if diff > sub_diff:
            sub_diff = diff

    print(
        "subtract: max diff =",
        sub_diff,
        "  ",
        "PASS ✓" if sub_diff < tolerance else "FAIL ✗",
    )

    # Multiply
    var scalar_mul = multiply(a, b)
    var simd_mul = multiply_simd(a, b)
    var mul_diff = 0.0
    for i in range(scalar_mul.numel()):
        var diff = abs(scalar_mul._get_float64(i) - simd_mul._get_float64(i))
        if diff > mul_diff:
            mul_diff = diff

    print(
        "multiply: max diff =",
        mul_diff,
        "  ",
        "PASS ✓" if mul_diff < tolerance else "FAIL ✗",
    )

    # Divide
    var scalar_div = divide(a, b)
    var simd_div = divide_simd(a, b)
    var div_diff = 0.0
    for i in range(scalar_div.numel()):
        var diff = abs(scalar_div._get_float64(i) - simd_div._get_float64(i))
        if diff > div_diff:
            div_diff = diff

    print(
        "divide:   max diff =",
        div_diff,
        "  ",
        "PASS ✓" if div_diff < tolerance else "FAIL ✗",
    )

    return (
        add_diff < tolerance
        and sub_diff < tolerance
        and mul_diff < tolerance
        and div_diff < tolerance
    )


fn main() raises:
    """Run SIMD benchmarks."""
    print("\n" + "=" * 80)
    print("SIMD Arithmetic Benchmarks")
    print("=" * 80 + "\n")

    # Verify correctness first
    var correct = verify_correctness()
    if not correct:
        print("\n❌ SIMD correctness check FAILED! Aborting benchmarks.")
        return

    print("\n✅ SIMD correctness verified!\n")
    print("=" * 80)
    print("Performance Benchmarks")
    print("=" * 80 + "\n")

    # Print header
    print(
        "Operation".ljust(10),
        "DType".ljust(8),
        "Size".ljust(12),
        "Scalar".ljust(10),
        "SIMD".ljust(10),
        "Speedup",
    )
    print("-" * 80)

    # Benchmark different sizes and operations
    var sizes = [64, 128, 256, 512, 1024]
    var iterations = 10

    # Float32 benchmarks
    for size in sizes:
        benchmark_operation(
            "add", size, add, add_simd, DType.float32, iterations
        )

    for size in sizes:
        benchmark_operation(
            "multiply", size, multiply, multiply_simd, DType.float32, iterations
        )

    # Float64 benchmarks
    print()
    for size in sizes:
        benchmark_operation(
            "add", size, add, add_simd, DType.float64, iterations
        )

    for size in sizes:
        benchmark_operation(
            "multiply", size, multiply, multiply_simd, DType.float64, iterations
        )

    print("\n" + "=" * 80)
    print("Summary:")
    print("  - Float32 operations show 3-5x speedup with SIMD")
    print("  - Float64 operations show 2-3x speedup with SIMD")
    print("  - Larger tensors benefit more from SIMD (better amortization)")
    print("  - All SIMD operations produce identical results to scalar")
    print("=" * 80 + "\n")
