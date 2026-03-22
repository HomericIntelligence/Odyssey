"""Example: Getting Started - Benchmark Neural Network Training

Extends the quickstart example with performance benchmarking using
the shared benchmarking framework.

This example demonstrates:
- Benchmarking model forward passes
- Measuring training step latency
- Comparing performance metrics
- Generating performance reports

Usage:
    pixi run mojo run examples/getting-started/benchmark_quickstart.mojo

See documentation: docs/getting-started/quickstart.md
"""

from shared.benchmarking import (
    benchmark_function,
    print_benchmark_report,
    BenchmarkResult,
)
from shared.core import AnyTensor, ones, zeros


# ============================================================================
# Simple Neural Network
# ============================================================================


@fieldwise_init
struct SimpleNN(Copyable, Movable):
    """Simple neural network for demonstration.

    A minimal 2-layer network suitable for benchmarking.

    Attributes:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output classes.
    """

    var input_size: Int
    var hidden_size: Int
    var output_size: Int


fn forward_pass(network: SimpleNN, input_data: AnyTensor) raises -> AnyTensor:
    """Perform a forward pass through the network.

    Args:
        network: Neural network.
        input_data: Input tensor (batch_size, input_size).

    Returns:
        Output tensor (batch_size, output_size).
    """
    # Create output tensor
    var batch_size = input_data.shape()[0]
    var output_shape = List[Int]()
    output_shape.append(batch_size)
    output_shape.append(network.output_size)

    var output = ones(output_shape, DType.float32)
    return output


fn training_step(
    network: SimpleNN, input_data: AnyTensor, targets: AnyTensor
) raises -> Float32:
    """Perform a training step.

    Args:
        network: Neural network.
        input_data: Input tensor.
        targets: Target labels.

    Returns:
        Loss value.
    """
    # Forward pass
    var logits = forward_pass(network, input_data)

    # Placeholder loss computation
    var batch_size = logits.shape()[0]

    # Simplified loss (in real code: cross-entropy, MSE, etc.)
    var loss = Float32(batch_size) / 100.0

    # In real training:
    # - Backward pass computes gradients
    # - Optimizer updates weights
    # - Return actual loss value

    return loss


# ============================================================================
# Benchmarking Functions
# ============================================================================


fn benchmark_forward_batch32() raises -> None:
    """Benchmark forward pass with batch size 32."""
    var network = SimpleNN(784, 256, 10)
    var input_shape = List[Int]()
    input_shape.append(32)  # batch size
    input_shape.append(784)  # input features
    var input_data = ones(input_shape, DType.float32)

    _ = forward_pass(network, input_data)


fn benchmark_forward_batch64() raises -> None:
    """Benchmark forward pass with batch size 64."""
    var network = SimpleNN(784, 256, 10)
    var input_shape = List[Int]()
    input_shape.append(64)  # batch size
    input_shape.append(784)  # input features
    var input_data = ones(input_shape, DType.float32)

    _ = forward_pass(network, input_data)


fn benchmark_forward_batch128() raises -> None:
    """Benchmark forward pass with batch size 128."""
    var network = SimpleNN(784, 256, 10)
    var input_shape = List[Int]()
    input_shape.append(128)  # batch size
    input_shape.append(784)  # input features
    var input_data = ones(input_shape, DType.float32)

    _ = forward_pass(network, input_data)


fn benchmark_training_step() raises -> None:
    """Benchmark a full training step."""
    var network = SimpleNN(784, 256, 10)
    var input_shape = List[Int]()
    input_shape.append(32)  # batch size
    input_shape.append(784)  # input features
    var input_data = ones(input_shape, DType.float32)

    var target_shape = List[Int]()
    target_shape.append(32)  # batch size
    var targets = zeros(target_shape, DType.float32)

    _ = training_step(network, input_data, targets)


# ============================================================================
# Analysis and Comparison
# ============================================================================


fn analyze_batch_scaling(
    batch32_result: BenchmarkResult,
    batch64_result: BenchmarkResult,
    batch128_result: BenchmarkResult,
) raises:
    """Analyze how performance scales with batch size.

    Args:
        batch32_result: Forward pass benchmark with batch size 32.
        batch64_result: Forward pass benchmark with batch size 64.
        batch128_result: Forward pass benchmark with batch size 128.
    """
    print("")
    print("=" * 70)
    print("Batch Scaling Analysis")
    print("=" * 70)
    print("")

    var batch32_latency = batch32_result.mean_latency_ms
    var batch64_latency = batch64_result.mean_latency_ms
    var batch128_latency = batch128_result.mean_latency_ms

    print("Batch Size 32:  ", batch32_latency, " ms")
    print("Batch Size 64:  ", batch64_latency, " ms")
    print("Batch Size 128: ", batch128_latency, " ms")
    print("")

    if batch32_latency > 0.0:
        var scaling_64 = batch64_latency / batch32_latency
        var scaling_128 = batch128_latency / batch32_latency
        print("Scaling factors (relative to batch 32):")
        print("  Batch 64:  ", scaling_64, "x")
        print("  Batch 128: ", scaling_128, "x")
        print("")

        if scaling_64 < 2.0:
            print("Good scaling: Batch size 64 has reasonable latency increase")
        if scaling_128 < 4.0:
            print(
                "Good scaling: Batch size 128 has reasonable latency increase"
            )

    print("")


# ============================================================================
# Main Demonstration
# ============================================================================


fn main() raises:
    """Benchmark neural network training components."""
    print("")
    print("=" * 70)
    print("ML Odyssey Quickstart Benchmarking")
    print("=" * 70)
    print("")
    print("This example demonstrates how to benchmark neural network")
    print("training components using the shared benchmarking framework.")
    print("")

    # Benchmark forward pass with different batch sizes
    print("Benchmarking forward passes with different batch sizes...")
    print("-" * 70)

    var batch32_result = benchmark_function(
        benchmark_forward_batch32,
        warmup_iters=10,
        measure_iters=100,
    )
    print_benchmark_report(batch32_result, "Forward Pass (Batch 32)")

    var batch64_result = benchmark_function(
        benchmark_forward_batch64,
        warmup_iters=10,
        measure_iters=100,
    )
    print_benchmark_report(batch64_result, "Forward Pass (Batch 64)")

    var batch128_result = benchmark_function(
        benchmark_forward_batch128,
        warmup_iters=10,
        measure_iters=100,
    )
    print_benchmark_report(batch128_result, "Forward Pass (Batch 128)")

    # Analyze scaling
    analyze_batch_scaling(batch32_result, batch64_result, batch128_result)

    # Benchmark training step
    print("Benchmarking complete training step...")
    print("-" * 70)

    var training_result = benchmark_function(
        benchmark_training_step,
        warmup_iters=10,
        measure_iters=100,
    )
    print_benchmark_report(training_result, "Training Step")

    print("=" * 70)
    print("Benchmarking complete!")
    print("=" * 70)
    print("")
    print("Key Performance Metrics:")
    print("- Latency: Time per operation (milliseconds)")
    print("- Throughput: Operations per second")
    print("- Std Dev: Consistency of measurements")
    print("- Percentiles: Latency distribution (p50, p95, p99)")
    print("")
    print("Next steps:")
    print("1. Compare results across different hardware")
    print("2. Identify bottlenecks using profiling")
    print("3. Optimize hot paths with SIMD or kernel fusion")
    print("4. Re-benchmark to validate optimizations")
    print("")
