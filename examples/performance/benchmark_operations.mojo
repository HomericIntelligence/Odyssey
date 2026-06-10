"""Example: Performance - Benchmark ML Operations.

Demonstrates benchmarking individual ML operations using the shared
benchmarking framework.

This example shows:
- Benchmarking common neural network operations
- Comparing performance across different tensor sizes
- Generating performance reports for comparison
- Measuring throughput metrics

Usage:
    pixi run mojo run examples/performance/benchmark_operations.mojo

See documentation: docs/advanced/performance.md
"""

from projectodyssey.benchmarking import (
    benchmark_function,
    print_benchmark_report,
    print_benchmark_summary,
    BenchmarkResult,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import ones, zeros
from projectodyssey.core import relu


# ============================================================================
# Benchmark Operation Functions
# ============================================================================


def benchmark_relu_activation_small() raises -> None:
    """Benchmark ReLU activation on small tensor."""
    var input_shape = List[Int]()
    input_shape.append(32)
    input_shape.append(128)
    var input_tensor = ones(input_shape, DType.float32)

    _ = relu(input_tensor)


def benchmark_relu_activation_medium() raises -> None:
    """Benchmark ReLU activation on medium tensor."""
    var input_shape = List[Int]()
    input_shape.append(64)
    input_shape.append(512)
    var input_tensor = ones(input_shape, DType.float32)

    _ = relu(input_tensor)


def benchmark_relu_activation_large() raises -> None:
    """Benchmark ReLU activation on large tensor."""
    var input_shape = List[Int]()
    input_shape.append(128)
    input_shape.append(1024)
    var input_tensor = ones(input_shape, DType.float32)

    _ = relu(input_tensor)


def benchmark_tensor_zeros_small() raises -> None:
    """Benchmark tensor creation (small)."""
    var shape = List[Int]()
    shape.append(32)
    shape.append(128)
    _ = zeros(shape, DType.float32)


def benchmark_tensor_zeros_medium() raises -> None:
    """Benchmark tensor creation (medium)."""
    var shape = List[Int]()
    shape.append(64)
    shape.append(512)
    _ = zeros(shape, DType.float32)


def benchmark_tensor_zeros_large() raises -> None:
    """Benchmark tensor creation (large)."""
    var shape = List[Int]()
    shape.append(128)
    shape.append(1024)
    _ = zeros(shape, DType.float32)


def benchmark_tensor_ones_small() raises -> None:
    """Benchmark tensor creation with ones (small)."""
    var shape = List[Int]()
    shape.append(32)
    shape.append(128)
    _ = ones(shape, DType.float32)


def benchmark_tensor_ones_medium() raises -> None:
    """Benchmark tensor creation with ones (medium)."""
    var shape = List[Int]()
    shape.append(64)
    shape.append(512)
    _ = ones(shape, DType.float32)


def benchmark_tensor_ones_large() raises -> None:
    """Benchmark tensor creation with ones (large)."""
    var shape = List[Int]()
    shape.append(128)
    shape.append(1024)
    _ = ones(shape, DType.float32)


# ============================================================================
# Helper Functions
# ============================================================================


def print_operation_comparison(
    operation_name: String,
    small_result: BenchmarkResult,
    medium_result: BenchmarkResult,
    large_result: BenchmarkResult,
) raises:
    """Print comparison of operation across tensor sizes.

    Args:
        operation_name: Name of the operation.
        small_result: Benchmark result for small tensor.
        medium_result: Benchmark result for medium tensor.
        large_result: Benchmark result for large tensor.
    """
    print("")
    print("=" * 70)
    print("Operation: " + operation_name)
    print("=" * 70)
    print("")

    # Create results list for summary
    var results: List[BenchmarkResult] = []
    results.append(small_result)
    results.append(medium_result)
    results.append(large_result)

    var names = List[String]()
    names.append("Small")
    names.append("Medium")
    names.append("Large")

    print_benchmark_summary(results, names)


# ============================================================================
# Main Demonstration
# ============================================================================


def main() raises:
    """Benchmark ML operations with various tensor sizes."""
    print("")
    print("=" * 70)
    print("ML Odyssey Operations Benchmarking Example")
    print("=" * 70)
    print("")
    print("This example benchmarks common ML operations across different")
    print("tensor sizes to demonstrate performance characteristics.")
    print("")

    # =========================================================================
    # Benchmark ReLU Activation
    # =========================================================================

    print("Benchmarking ReLU Activation...")
    print("-" * 70)

    var relu_small = benchmark_function(
        benchmark_relu_activation_small,
        warmup_iters=20,
        measure_iters=100,
    )

    var relu_medium = benchmark_function(
        benchmark_relu_activation_medium,
        warmup_iters=20,
        measure_iters=100,
    )

    var relu_large = benchmark_function(
        benchmark_relu_activation_large,
        warmup_iters=20,
        measure_iters=100,
    )

    print_operation_comparison(
        "ReLU Activation", relu_small, relu_medium, relu_large
    )

    # =========================================================================
    # Benchmark Tensor Creation with zeros()
    # =========================================================================

    print("Benchmarking Tensor Creation (zeros)...")
    print("-" * 70)

    var zeros_small = benchmark_function(
        benchmark_tensor_zeros_small,
        warmup_iters=20,
        measure_iters=100,
    )

    var zeros_medium = benchmark_function(
        benchmark_tensor_zeros_medium,
        warmup_iters=20,
        measure_iters=100,
    )

    var zeros_large = benchmark_function(
        benchmark_tensor_zeros_large,
        warmup_iters=20,
        measure_iters=100,
    )

    print_operation_comparison(
        "Tensor Creation (zeros)",
        zeros_small,
        zeros_medium,
        zeros_large,
    )

    # =========================================================================
    # Benchmark Tensor Creation with ones()
    # =========================================================================

    print("Benchmarking Tensor Creation (ones)...")
    print("-" * 70)

    var ones_small = benchmark_function(
        benchmark_tensor_ones_small,
        warmup_iters=20,
        measure_iters=100,
    )

    var ones_medium = benchmark_function(
        benchmark_tensor_ones_medium,
        warmup_iters=20,
        measure_iters=100,
    )

    var ones_large = benchmark_function(
        benchmark_tensor_ones_large,
        warmup_iters=20,
        measure_iters=100,
    )

    print_operation_comparison(
        "Tensor Creation (ones)", ones_small, ones_medium, ones_large
    )

    print("=" * 70)
    print("Operations benchmarking complete!")
    print("=" * 70)
    print("")
    print("Performance Insights:")
    print("- Measure latency to identify bottlenecks")
    print("- Compare across sizes to understand scaling behavior")
    print("- Use percentiles (p95, p99) for worst-case analysis")
    print("- Monitor throughput for optimization opportunities")
    print("")
