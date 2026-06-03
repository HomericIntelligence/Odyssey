"""Example: Performance - Benchmark Model Inference.

Demonstrates how to use the shared benchmarking framework to measure
and report model inference performance.

This example shows:
- Benchmarking a neural network forward pass
- Measuring latency, throughput, and statistical metrics
- Generating formatted reports
- Comparing performance across different configurations

Usage:
    pixi run mojo run examples/performance/benchmark_model_inference.mojo

See documentation: docs/advanced/performance.md
"""

from projectodyssey.benchmarking import (
    benchmark_function,
    print_benchmark_report,
    print_benchmark_summary,
    BenchmarkResult,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import ones


# ============================================================================
# Simple Neural Network for Demonstration
# ============================================================================


@fieldwise_init
struct SimpleNetwork(Copyable, Movable):
    """Simple neural network for benchmarking demonstration.

    A minimal network with one hidden layer to demonstrate benchmarking
    different forward pass configurations.

    Attributes:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output classes.
    """

    var input_size: Int
    var hidden_size: Int
    var output_size: Int


def simple_forward(
    mut network: SimpleNetwork, input_data: AnyTensor
) raises -> AnyTensor:
    """Perform a simple forward pass.

    Args:
        network: Neural network struct.
        input_data: Input tensor.

    Returns:
        Output tensor from forward pass.
    """
    # Simplified forward pass for demonstration
    # In real code, this would include matrix multiplications
    # and activation functions
    var output_shape = List[Int]()
    output_shape.append(input_data.shape()[0])
    output_shape.append(network.output_size)

    var output = ones(output_shape, DType.float32)
    return output


# ============================================================================
# Benchmark Functions
# ============================================================================


def benchmark_inference_small() raises -> None:
    """Benchmark forward pass with small input."""
    var network = SimpleNetwork(784, 128, 10)
    var input_shape = List[Int]()
    input_shape.append(32)  # batch size
    input_shape.append(784)  # input features
    var input_data = ones(input_shape, DType.float32)

    # Perform forward pass
    _ = simple_forward(network, input_data)


def benchmark_inference_medium() raises -> None:
    """Benchmark forward pass with medium input."""
    var network = SimpleNetwork(3072, 512, 10)
    var input_shape = List[Int]()
    input_shape.append(64)  # batch size
    input_shape.append(3072)  # input features
    var input_data = ones(input_shape, DType.float32)

    # Perform forward pass
    _ = simple_forward(network, input_data)


def benchmark_inference_large() raises -> None:
    """Benchmark forward pass with large input."""
    var network = SimpleNetwork(25088, 4096, 1000)
    var input_shape = List[Int]()
    input_shape.append(32)  # batch size
    input_shape.append(25088)  # input features
    var input_data = ones(input_shape, DType.float32)

    # Perform forward pass
    _ = simple_forward(network, input_data)


# ============================================================================
# Main Demonstration
# ============================================================================


def main() raises:
    """Benchmark different neural network configurations."""
    print("")
    print("=" * 70)
    print("ML Odyssey Benchmarking Example")
    print("=" * 70)
    print("")
    print("This example demonstrates how to use the shared benchmarking")
    print("framework to measure neural network performance.")
    print("")

    # Benchmark small network inference
    print("Benchmarking small network inference...")
    var result_small = benchmark_function(
        benchmark_inference_small,
        warmup_iters=10,
        measure_iters=100,
    )
    print_benchmark_report(result_small, "Small Network Inference (32x784)")
    print("")

    # Benchmark medium network inference
    print("Benchmarking medium network inference...")
    var result_medium = benchmark_function(
        benchmark_inference_medium,
        warmup_iters=10,
        measure_iters=100,
    )
    print_benchmark_report(result_medium, "Medium Network Inference (64x3072)")
    print("")

    # Benchmark large network inference
    print("Benchmarking large network inference...")
    var result_large = benchmark_function(
        benchmark_inference_large,
        warmup_iters=5,  # Fewer warmups for larger network
        measure_iters=50,  # Fewer iterations for speed
    )
    print_benchmark_report(result_large, "Large Network Inference (32x25088)")
    print("")

    # Summary comparison
    var results: List[BenchmarkResult] = []
    results.append(result_small)
    results.append(result_medium)
    results.append(result_large)

    var names = List[String]()
    names.append("Small Network")
    names.append("Medium Network")
    names.append("Large Network")

    print_benchmark_summary(results, names)

    print("=" * 70)
    print("Benchmarking complete!")
    print("=" * 70)
    print("")
    print("Key observations:")
    print("- Latency increases with network size")
    print("- Standard deviation indicates measurement consistency")
    print("- Throughput (ops/sec) inversely correlates with latency")
    print("- P50/P95/P99 percentiles show latency distribution")
    print("")
