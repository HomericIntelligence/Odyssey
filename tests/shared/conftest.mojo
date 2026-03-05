"""Shared test fixtures and utilities for the shared library test suite.

This module provides:
- Common assertion functions for testing (imported from shared.testing.assertions)
- Test fixtures for creating test data
- Utilities for test setup and teardown
"""

from random import seed
from time import perf_counter_ns
from shared.core.extensor import ExTensor
from shared.testing import SimpleMLP

# Re-export all assertions from shared.testing.assertions for backward compatibility
from shared.testing.assertions import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_not_none,
    assert_almost_equal,
    assert_dtype_equal,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
    assert_greater,
    assert_less,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_shape_equal,
    assert_not_equal_tensor,
    assert_tensor_equal,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_type,
    assert_contiguous,
)


# ============================================================================
# Test Fixtures
# ============================================================================


struct TestFixtures:
    """Collection of reusable test fixtures and utilities.

    This struct provides static methods for creating common test data,
    models, and scenarios used across multiple tests.
    """

    @staticmethod
    fn deterministic_seed() -> Int:
        """Get deterministic random seed for reproducible tests.

        Returns:
            Fixed random seed value (42).
        """
        return 42

    @staticmethod
    fn set_seed():
        """Set random seed for deterministic test execution."""
        seed(Self.deterministic_seed())

    @staticmethod
    fn small_tensor() raises -> ExTensor:
        """Create small 3x3 tensor for unit tests.

        Returns:
            3x3 ExTensor with deterministic values (0.1).
        """
        from shared.core import full

        var shape = List[Int]()
        shape.append(3)
        shape.append(3)
        return full(shape, Float32(0.1), DType.float32)

    @staticmethod
    fn medium_tensor() raises -> ExTensor:
        """Create medium 10x10 tensor for unit tests.

        Returns:
            10x10 ExTensor with zeros.
        """
        from shared.core import zeros

        var shape = List[Int]()
        shape.append(10)
        shape.append(10)
        return zeros(shape, DType.float32)

    @staticmethod
    fn simple_weights() raises -> ExTensor:
        """Create simple weight tensor for linear layer tests.

        Returns:
            5x3 ExTensor with deterministic small values.
        """
        from shared.core import full

        var shape = List[Int]()
        shape.append(5)
        shape.append(3)
        return full(shape, Float32(0.01), DType.float32)

    @staticmethod
    fn simple_bias() raises -> ExTensor:
        """Create simple bias tensor for linear layer tests.

        Returns:
            5-element ExTensor with zeros.
        """
        from shared.core import zeros

        var shape = List[Int]()
        shape.append(5)
        return zeros(shape, DType.float32)

    @staticmethod
    fn synthetic_data(
        n_samples: Int = 100, n_features: Int = 10
    ) raises -> ExTensor:
        """Create synthetic data tensor for testing.

        Args:
            n_samples: Number of samples (rows).
            n_features: Number of features (columns).

        Returns:
            ExTensor of shape (n_samples, n_features) with random values.
        """
        from shared.core import randn

        var shape = List[Int]()
        shape.append(n_samples)
        shape.append(n_features)
        return randn(shape, DType.float32)

    @staticmethod
    fn synthetic_labels(n_samples: Int = 100) raises -> ExTensor:
        """Create synthetic label tensor for testing.

        Args:
            n_samples: Number of samples.

        Returns:
            ExTensor of shape (n_samples,) with binary labels (0 or 1).
        """
        from shared.core import zeros

        var shape = List[Int]()
        shape.append(n_samples)
        return zeros(shape, DType.float32)


# ============================================================================
# Benchmark Utilities
# ============================================================================


struct BenchmarkResult(Copyable, Movable):
    """Results from a performance benchmark.

    Attributes:
        name: Benchmark name/description.
        duration_ms: Execution duration in milliseconds.
        throughput: Operations per second.
        memory_mb: Memory used in megabytes.
    """

    var name: String
    var duration_ms: Float64
    var throughput: Float64
    var memory_mb: Float64

    fn __init__(
        out self,
        name: String,
        duration_ms: Float64,
        throughput: Float64,
        memory_mb: Float64 = 0.0,
    ):
        """Initialize benchmark result.

        Args:
            name: Benchmark name.
            duration_ms: Duration in milliseconds.
            throughput: Operations per second.
            memory_mb: Memory usage in MB.
        """
        self.name = name
        self.duration_ms = duration_ms
        self.throughput = throughput
        self.memory_mb = memory_mb

    fn print_result(self):
        """Print benchmark result in readable format."""
        print("Benchmark:", self.name)
        print("  Duration:", self.duration_ms, "ms")
        print("  Throughput:", self.throughput, "ops/s")
        if self.memory_mb > 0:
            print("  Memory:", self.memory_mb, "MB")


# DEPRECATED: Use BenchmarkResult directly instead of BenchmarkStatistics
# This comptime is maintained for backward compatibility during type consolidation.
comptime BenchmarkStatistics = BenchmarkResult


fn print_benchmark_results(results: List[BenchmarkResult]):
    """Print all benchmark results.

    Args:
        results: List of benchmark results to print.
    """
    print("\n=== Benchmark Results ===")
    for i in range(len(results)):
        results[i].print_result()
        print()


# ============================================================================
# Test Helpers
# ============================================================================


fn measure_time[func: fn () raises -> None]() raises -> Float64:
    """Measure execution time of a function.

    Parameters:
        func: Function to measure (must be fn, not def).

    Returns:
        Execution time in milliseconds.

    Uses perf_counter_ns() for high-resolution timing.
    """
    var start = perf_counter_ns()
    func()
    var end = perf_counter_ns()
    # Convert nanoseconds to milliseconds
    return Float64(end - start) / 1_000_000.0


fn measure_throughput[
    func: fn () raises -> None
](n_iterations: Int) raises -> Float64:
    """Measure throughput of a function.

    Parameters:
        func: Function to measure.

    Args:
        n_iterations: Number of iterations to run.

    Returns:
        Operations per second.

    Runs the function multiple times and calculates throughput as
    iterations per second using high-resolution timing.
    """
    var start = perf_counter_ns()
    for _ in range(n_iterations):
        func()
    var end = perf_counter_ns()
    # Convert nanoseconds to seconds and compute throughput
    var duration_s = Float64(end - start) / 1_000_000_000.0
    if duration_s > 0:
        return Float64(n_iterations) / duration_s
    return 0.0


# ============================================================================
# Test Data Generators
# ============================================================================


fn create_test_vector(size: Int, value: Float32 = 1.0) -> List[Float32]:
    """Create test vector filled with specific value.

    Args:
        size: Vector size.
        value: Fill value.

    Returns:
        List of Float32 values.
    """
    var vec = List[Float32]()
    for _ in range(size):
        vec.append(value)
    return vec^


fn create_test_matrix(
    rows: Int, cols: Int, value: Float32 = 1.0
) -> List[List[Float32]]:
    """Create test matrix filled with specific value.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        value: Fill value.

    Returns:
        2D list of Float32 values.
    """
    var matrix = List[List[Float32]]()
    for _ in range(rows):
        matrix.append(create_test_vector(cols, value))
    return matrix^


fn create_sequential_vector(size: Int, start: Float32 = 0.0) -> List[Float32]:
    """Create vector with sequential values [start, start+1, ...].

    Args:
        size: Vector size.
        start: Starting value.

    Returns:
        List of sequential Float32 values.
    """
    var vec = List[Float32]()
    for i in range(size):
        vec.append(start + Float32(i))
    return vec^


# ============================================================================
# Training Test Fixtures
# ============================================================================


fn create_simple_model() -> SimpleMLP:
    """Create a simple 2-layer neural network for testing training loops.

    Returns a mock neural network with:
    - Input dimension: 10
    - Hidden dimension: 5
    - Output dimension: 1
    - Uses ReLU activation between layers

    The model uses constant initialization for predictable testing behavior.

    Returns:
        SimpleMLP instance configured for testing.

    Example:
        ```mojo
        var model = create_simple_model()
        var input = create_test_vector(10, value=1.0)
        var output = model.forward(input)
        ```
    """
    var model = SimpleMLP(
        input_dim=10,
        hidden_dim=5,
        output_dim=1,
        num_hidden_layers=1,
        init_value=0.1,
    )
    return model^


fn create_simple_dataset(
    n_samples: Int = 100,
    input_dim: Int = 10,
    output_dim: Int = 1,
    seed_value: Int = 42,
) -> List[Tuple[List[Float32], List[Float32]]]:
    """Create a simple dataset for testing data loading.

    Generates synthetic data with:
    - Configurable number of samples
    - Configurable input and output dimensions
    - Deterministic seeding for reproducible tests

    Returns a list of (input, output) tuples for each sample.

    Args:
        n_samples: Number of samples in dataset (default: 100).
        input_dim: Dimension of input features (default: 10).
        output_dim: Dimension of output labels (default: 1).
        seed_value: Random seed for reproducibility (default: 42).

    Returns:
        List of tuples containing (data, label) for each sample.

    Example:
        ```mojo
        var dataset = create_simple_dataset(n_samples=50, input_dim=10, output_dim=5)
        var item = dataset[0]
        var data = item[0]
        var label = item[1]
        ```
    """
    # Set seed for reproducibility
    seed(seed_value)

    var samples = List[Tuple[List[Float32], List[Float32]]]()

    # Generate samples
    for i in range(n_samples):
        var item_seed = seed_value + i

        # Generate input features
        var input_features = List[Float32]()
        for j in range(input_dim):
            # Use deterministic values based on seed and index
            var val = Float32((seed_value + i + j) % 100) / 100.0
            input_features.append(val)

        # Generate output labels
        var output_labels = List[Float32]()
        for j in range(output_dim):
            # Use deterministic values based on seed and index
            var val = Float32((seed_value + i + j + 1000) % 100) / 100.0
            output_labels.append(val)

        samples.append((input_features^, output_labels^))

    return samples^


struct MockDataLoader(Copyable, Movable):
    """Mock data loader for testing training loops.

    Provides a simple data loader that yields batches of data.

    Attributes:
        samples: List of (input, output) tuples.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples.
    """

    var samples: List[Tuple[List[Float32], List[Float32]]]
    var batch_size: Int
    var shuffle: Bool

    fn __init__(
        out self,
        var samples: List[Tuple[List[Float32], List[Float32]]],
        batch_size: Int = 32,
        shuffle: Bool = False,
    ):
        """Initialize mock data loader.

        Args:
            samples: List of (input, output) tuples.
            batch_size: Samples per batch (default: 32).
            shuffle: Whether to shuffle (default: False).
        """
        self.samples = samples^
        self.batch_size = batch_size
        self.shuffle = shuffle

    fn __len__(self) -> Int:
        """Get number of batches."""
        var n_samples = len(self.samples)
        return (n_samples + self.batch_size - 1) // self.batch_size


fn create_mock_dataloader(
    n_batches: Int = 10,
    batch_size: Int = 32,
    n_samples: Int = 320,
    input_dim: Int = 10,
    output_dim: Int = 1,
    seed_value: Int = 42,
    shuffle: Bool = False,
) -> MockDataLoader:
    """Create mock data loader with specified number of batches.

    Generates a synthetic dataset and wraps it in a MockDataLoader
    for testing training loops and data pipelines.

    Args:
        n_batches: Desired number of batches (default: 10).
        batch_size: Samples per batch (default: 32).
        n_samples: Total number of samples (default: 320 = 10 batches * 32).
        input_dim: Input feature dimension (default: 10).
        output_dim: Output label dimension (default: 1).
        seed_value: Random seed (default: 42).
        shuffle: Whether to shuffle data (default: False).

    Returns:
        MockDataLoader with synthetic samples.

    Example:
        ```mojo
        var loader = create_mock_dataloader(n_batches=5, batch_size=32)
        var num_batches = loader.__len__()
        ```

    Note:
        - If n_samples != n_batches * batch_size, the last batch may be partial.
        - All data is synthetically generated and deterministic.
    """
    # Create the simple dataset
    var dataset = create_simple_dataset(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        seed_value=seed_value,
    )

    # Create and return mock loader
    return MockDataLoader(dataset^, batch_size=batch_size, shuffle=shuffle)
