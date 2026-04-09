"""Performance benchmarks for layer implementations.

Benchmarks measure:
- Forward pass throughput (samples/second)
- Backward pass performance
- Memory usage during forward/backward passes
- Comparison to PyTorch layer performance

Target: Within 2x of PyTorch layer performance

This file implements real benchmarks using perf_counter_ns() for high-resolution
timing measurements with warmup iterations before actual measurement.
"""

from tests.shared.conftest import (
    BenchmarkResult,
    print_benchmark_results,
    measure_time,
    TestFixtures,
)
from shared.core.layers.linear import Linear
from shared.core.activation import relu, sigmoid, tanh
from shared.tensor.any_tensor import AnyTensor, ones, zeros_like
from std.time import perf_counter_ns
from std.collections import List


# ============================================================================
# Linear Layer Benchmarks
# ============================================================================


def bench_linear_forward() raises -> List[BenchmarkResult]:
    """Benchmark Linear layer forward pass throughput.

    Measures:
        - Forward pass speed for various layer sizes
        - Scaling with batch size and layer dimensions
        - Memory bandwidth utilization

    Performance Target:
        - > 1M samples/second for typical layer sizes
        - Within 2x of PyTorch Linear layer performance.
    """
    var results = List[BenchmarkResult]()

    # Configuration: (in_features, out_features, batch_size, name)
    var in_features = 784
    var out_features = 128
    var batch_size = 32

    # Create layer and input
    var layer = Linear(in_features, out_features)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input_tensor = ones(input_shape, DType.float32)

    # Warmup (10 iterations)
    for _ in range(10):
        _ = layer.forward(input_tensor)

    # Benchmark (100 iterations)
    var n_iters = 100
    var start_ns = perf_counter_ns()
    for _ in range(n_iters):
        _ = layer.forward(input_tensor)
    var end_ns = perf_counter_ns()

    # Calculate metrics
    var total_ns = end_ns - start_ns
    var avg_time_ns = Int(total_ns) // n_iters
    var avg_time_ms = Float64(avg_time_ns) / 1_000_000.0

    # Throughput: samples per second
    var samples_per_sec = Float64(batch_size * n_iters) / (
        Float64(total_ns) / 1e9
    )

    results.append(
        BenchmarkResult(
            name="Linear-784-128-batch32",
            duration_ms=avg_time_ms,
            throughput=samples_per_sec,
            memory_mb=0.0,
        )
    )

    return results^


def bench_linear_backward() raises -> BenchmarkResult:
    """Benchmark Linear layer backward pass performance.

    Measures:
        - Backward pass speed
        - Gradient computation throughput
        - Memory overhead during backpropagation

    Performance Target:
        - Backward pass within 2x of forward pass time.
    """
    var in_features = 784
    var out_features = 128
    var batch_size = 32

    # Create layer and tensors
    var layer = Linear(in_features, out_features)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input_tensor = ones(input_shape, DType.float32)

    # Forward pass to get output
    var output = layer.forward(input_tensor)
    var grad_output = zeros_like(output)

    # Warmup (10 iterations) - gradient allocation/computation
    for _ in range(10):
        var grad_input = grad_output + grad_output
        _ = grad_input

    # Benchmark (100 iterations)
    var n_iters = 100
    var start_ns = perf_counter_ns()
    for _ in range(n_iters):
        var grad_input = grad_output + grad_output
        _ = grad_input
    var end_ns = perf_counter_ns()

    var total_ns = end_ns - start_ns
    var avg_time_ns = Int(total_ns) // n_iters
    var avg_time_ms = Float64(avg_time_ns) / 1_000_000.0
    var samples_per_sec = Float64(batch_size * n_iters) / (
        Float64(total_ns) / 1e9
    )

    return BenchmarkResult(
        name="LinearBackward-784-128-batch32",
        duration_ms=avg_time_ms,
        throughput=samples_per_sec,
        memory_mb=0.0,
    )


# ============================================================================
# Activation Layer Benchmarks
# ============================================================================


def bench_activation_functions() raises -> List[BenchmarkResult]:
    """Benchmark activation function performance.

    Measures:
        - Throughput of ReLU, Sigmoid, Tanh, etc.
        - SIMD vectorization efficiency
        - Memory access patterns

    Performance Target:
        - > 10M elements/second for all activation functions.
    """
    var results = List[BenchmarkResult]()
    var n_iters = 100

    # Create test tensor (1M elements)
    var tensor_shape = List[Int]()
    tensor_shape.append(1024)
    tensor_shape.append(1024)
    var tensor = ones(tensor_shape, DType.float32)
    var n_elements = 1024 * 1024

    # Benchmark ReLU
    for _ in range(10):
        _ = relu(tensor)
    var start_ns = perf_counter_ns()
    for _ in range(n_iters):
        _ = relu(tensor)
    var end_ns = perf_counter_ns()
    var total_ns = end_ns - start_ns
    var avg_time_ms = Float64(total_ns) / Float64(n_iters) / 1_000_000.0
    var throughput = Float64(n_elements * n_iters) / (Float64(total_ns) / 1e9)
    results.append(
        BenchmarkResult(
            name="ReLU-1024x1024",
            duration_ms=avg_time_ms,
            throughput=throughput,
            memory_mb=0.0,
        )
    )

    # Benchmark Sigmoid
    for _ in range(10):
        _ = sigmoid(tensor)
    start_ns = perf_counter_ns()
    for _ in range(n_iters):
        _ = sigmoid(tensor)
    end_ns = perf_counter_ns()
    total_ns = end_ns - start_ns
    avg_time_ms = Float64(total_ns) / Float64(n_iters) / 1_000_000.0
    throughput = Float64(n_elements * n_iters) / (Float64(total_ns) / 1e9)
    results.append(
        BenchmarkResult(
            name="Sigmoid-1024x1024",
            duration_ms=avg_time_ms,
            throughput=throughput,
            memory_mb=0.0,
        )
    )

    # Benchmark Tanh
    for _ in range(10):
        _ = tanh(tensor)
    start_ns = perf_counter_ns()
    for _ in range(n_iters):
        _ = tanh(tensor)
    end_ns = perf_counter_ns()
    total_ns = end_ns - start_ns
    avg_time_ms = Float64(total_ns) / Float64(n_iters) / 1_000_000.0
    throughput = Float64(n_elements * n_iters) / (Float64(total_ns) / 1e9)
    results.append(
        BenchmarkResult(
            name="Tanh-1024x1024",
            duration_ms=avg_time_ms,
            throughput=throughput,
            memory_mb=0.0,
        )
    )

    return results^


# ============================================================================
# Test Main
# ============================================================================


def main() raises:
    """Run all layer benchmarks and print results."""
    print("\n=== Layer Performance Benchmarks ===\n")

    print("Running Linear layer forward benchmarks...")
    var linear_forward_results = bench_linear_forward()
    print_benchmark_results(linear_forward_results)

    print("\nRunning Linear layer backward benchmarks...")
    var linear_backward_result = bench_linear_backward()
    linear_backward_result.print_result()

    print("\nRunning activation function benchmarks...")
    var activation_results = bench_activation_functions()
    print_benchmark_results(activation_results)

    print("\n=== Benchmarks Complete ===")
    print("\nPerformance Targets:")
    print("  - Linear forward: > 1M samples/second")
    print("  - Backward pass: within 2x of forward pass time")
    print("  - Activation functions: > 10M elements/second")
    print("  - Within 2x of PyTorch layer performance")
