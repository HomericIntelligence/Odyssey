"""Core Fuzzing Utilities for ML Odyssey

Provides core infrastructure for property-based testing and fuzzing of tensor
operations. This module enables discovering edge cases and unexpected behaviors
in tensor implementations through random input generation and invariant checking.

Fuzzing Strategy:
    1. Generate random inputs within configurable bounds
    2. Execute operations under test
    3. Verify invariants (no crashes, valid outputs, numeric properties)
    4. Report failures with reproducible seeds

Key Components:
    - FuzzConfig: Configuration for fuzzing parameters
    - FuzzResult: Results from a fuzzing run
    - Random tensor generators with edge case support
    - Property-based testing helpers
    - Invariant checkers for numeric operations

Usage:
    from shared.testing.fuzz_core import (
        FuzzConfig,
        FuzzResult,
        fuzz_tensor_creation,
        fuzz_binary_op,
        check_numeric_invariants,
    )

    # Configure fuzzing
    var config = FuzzConfig(seed=42, iterations=1000, max_dim=100)

    # Run fuzzing
    var result = fuzz_tensor_creation(config)

    # Check results
    if result.failures > 0:
        print("Found", result.failures, "failures")
        for failure in result.failure_details:
            print(failure)

Example:
    ```mojo
    # Fuzz matrix multiplication
    var config = FuzzConfig(seed=42, iterations=100)
    var result = fuzz_binary_op["matmul"](config)
    assert result.failures == 0, "matmul should handle all inputs"
    ```
"""

from random import random_float64, seed as random_seed
from math import isnan, isinf
from shared.core import ExTensor, zeros, ones, full


# ============================================================================
# Fuzzing Configuration
# ============================================================================


struct FuzzConfig(Copyable, Movable):
    """Configuration for fuzzing parameters.

    Controls the behavior of fuzzing runs including randomness, iteration counts,
    and bounds for generated values.

    Attributes:
        seed: Random seed for reproducible fuzzing (default: 42).
        iterations: Number of fuzzing iterations to run (default: 100).
        max_dim: Maximum dimension size for generated shapes (default: 100).
        max_ndim: Maximum number of dimensions (default: 6).
        include_edge_cases: Whether to include edge cases like NaN, Inf (default: True).
        include_empty: Whether to include empty tensors (default: True).
        include_scalar: Whether to include 0D scalar tensors (default: True).
        min_value: Minimum value for random floats (default: -1e6).
        max_value: Maximum value for random floats (default: 1e6).
        verbose: Whether to print progress during fuzzing (default: False).

    Example:
        ```mojo
        var config = FuzzConfig(seed=42, iterations=1000)
        config.max_dim = 50  # Limit dimension sizes
        config.include_edge_cases = True
        ```
    """

    var seed: Int
    var iterations: Int
    var max_dim: Int
    var max_ndim: Int
    var include_edge_cases: Bool
    var include_empty: Bool
    var include_scalar: Bool
    var min_value: Float64
    var max_value: Float64
    var verbose: Bool

    fn __init__(
        out self,
        seed: Int = 42,
        iterations: Int = 100,
        max_dim: Int = 100,
        max_ndim: Int = 6,
        include_edge_cases: Bool = True,
        include_empty: Bool = True,
        include_scalar: Bool = True,
        min_value: Float64 = -1e6,
        max_value: Float64 = 1e6,
        verbose: Bool = False,
    ):
        """Initialize fuzzing configuration.

        Args:
            seed: Random seed for reproducibility.
            iterations: Number of iterations to run.
            max_dim: Maximum size for any dimension.
            max_ndim: Maximum number of dimensions.
            include_edge_cases: Include NaN, Inf, etc.
            include_empty: Include empty (0-element) tensors.
            include_scalar: Include 0D scalar tensors.
            min_value: Minimum random float value.
            max_value: Maximum random float value.
            verbose: Print progress during fuzzing.
        """
        self.seed = seed
        self.iterations = iterations
        self.max_dim = max_dim
        self.max_ndim = max_ndim
        self.include_edge_cases = include_edge_cases
        self.include_empty = include_empty
        self.include_scalar = include_scalar
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose


struct FuzzResult(Copyable, Movable):
    """Results from a fuzzing run.

    Captures the outcome of fuzzing including success/failure counts
    and details about any failures encountered.

    Attributes:
        total_runs: Total number of iterations executed.
        successes: Number of successful iterations.
        failures: Number of failed iterations.
        expected_errors: Number of expected/handled errors.
        failure_details: List of failure descriptions.
        failure_seeds: Seeds that triggered failures for reproduction.

    Example:
        ```mojo
        var result = fuzz_tensor_creation(config)
        if result.failures > 0:
            print("Failures:", result.failures)
            # Reproduce failures using result.failure_seeds
        ```
    """

    var total_runs: Int
    var successes: Int
    var failures: Int
    var expected_errors: Int
    var failure_details: List[String]
    var failure_seeds: List[Int]

    fn __init__(out self):
        """Initialize empty fuzz result."""
        self.total_runs = 0
        self.successes = 0
        self.failures = 0
        self.expected_errors = 0
        self.failure_details = List[String]()
        self.failure_seeds = List[Int]()

    fn add_success(mut self):
        """Record a successful iteration."""
        self.total_runs += 1
        self.successes += 1

    fn add_failure(mut self, detail: String, iteration_seed: Int):
        """Record a failed iteration.

        Args:
            detail: Description of the failure.
            iteration_seed: Seed that triggered this failure.
        """
        self.total_runs += 1
        self.failures += 1
        self.failure_details.append(detail)
        self.failure_seeds.append(iteration_seed)

    fn add_expected_error(mut self):
        """Record an expected/handled error (not a failure)."""
        self.total_runs += 1
        self.expected_errors += 1

    fn is_success(self) -> Bool:
        """Check if fuzzing run was successful (no failures).

        Returns:
            True if no failures occurred.
        """
        return self.failures == 0

    fn print_summary(self):
        """Print summary of fuzzing results."""
        print("=== Fuzzing Results ===")
        print("  Total runs:", self.total_runs)
        print("  Successes:", self.successes)
        print("  Failures:", self.failures)
        print("  Expected errors:", self.expected_errors)

        if self.failures > 0:
            print("\n  Failure details:")
            var num_to_show = min(10, len(self.failure_details))
            for i in range(num_to_show):
                print("    -", self.failure_details[i])
            if len(self.failure_details) > 10:
                print("    ... and", len(self.failure_details) - 10, "more")


# ============================================================================
# Random Number Generation with Seeds
# ============================================================================


struct SeededRNG(Copyable, Movable):
    """Seeded random number generator for reproducible fuzzing.

    Provides random number generation with explicit seed tracking for
    reproducing failures. Uses Mojo's built-in random module internally.

    Attributes:
        seed: Current seed value.
        iteration: Current iteration number.

    Example:
        ```mojo
        var rng = SeededRNG(42)
        var val = rng.random_float(-1.0, 1.0)
        var idx = rng.random_int(0, 100)
        ```
    """

    var seed: Int
    var iteration: Int

    fn __init__(out self, seed: Int = 42):
        """Initialize RNG with seed.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.iteration = 0
        random_seed(seed)

    fn reseed(mut self, new_seed: Int):
        """Reset RNG with new seed.

        Args:
            new_seed: New seed value.
        """
        self.seed = new_seed
        self.iteration = 0
        random_seed(new_seed)

    fn next_iteration(mut self):
        """Advance to next iteration with new seed."""
        self.iteration += 1
        random_seed(self.seed + self.iteration)

    fn random_float(self, low: Float64 = 0.0, high: Float64 = 1.0) -> Float64:
        """Generate random float in range [low, high).

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).

        Returns:
            Random float in [low, high).
        """
        var rand = random_float64()
        return low + rand * (high - low)

    fn random_int(self, low: Int, high: Int) -> Int:
        """Generate random integer in range [low, high).

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).

        Returns:
            Random integer in [low, high).
        """
        if high <= low:
            return low
        var rand = random_float64()
        var range_size = high - low
        return low + Int(rand * Float64(range_size))

    fn random_bool(self) -> Bool:
        """Generate random boolean.

        Returns:
            Random True or False.
        """
        return random_float64() < 0.5

    fn get_current_seed(self) -> Int:
        """Get seed for current iteration (for reproduction).

        Returns:
            Seed value that can reproduce current random state.
        """
        return self.seed + self.iteration


# ============================================================================
# Random Tensor Generation
# ============================================================================


fn create_random_tensor(
    rng: SeededRNG,
    shape: List[Int],
    dtype: DType,
    low: Float64 = -1.0,
    high: Float64 = 1.0,
) raises -> ExTensor:
    """Create tensor with random values.

    Generates a tensor filled with random values uniformly distributed
    in [low, high).

    Args:
        rng: Seeded random number generator.
        shape: Shape of the tensor.
        dtype: Data type of the tensor.
        low: Minimum value (inclusive).
        high: Maximum value (exclusive).

    Returns:
        ExTensor filled with random values.

    Raises:
        Error: If tensor creation fails.

    Example:
        ```mojo
        var rng = SeededRNG(42)
        var shape = List[Int]()
        shape.append(3)
        shape.append(4)
        var tensor = create_random_tensor(rng, shape, DType.float32, -1.0, 1.0)
        ```
    """
    var tensor = zeros(shape, dtype)
    var numel = tensor.numel()
    var range_size = high - low

    for i in range(numel):
        var rand_val = random_float64()
        var scaled_val = low + rand_val * range_size
        tensor._set_float64(i, scaled_val)

    return tensor^


fn create_edge_case_tensor(
    shape: List[Int], dtype: DType, edge_type: String
) raises -> ExTensor:
    """Create tensor filled with edge case values.

    Generates tensors with special numeric values for testing edge cases.

    Args:
        shape: Shape of the tensor.
        dtype: Data type of the tensor.
        edge_type: Type of edge case:
            - "nan": All NaN values
            - "inf": All positive infinity
            - "neg_inf": All negative infinity
            - "zero": All zeros
            - "epsilon": All epsilon (smallest positive)
            - "max": All maximum representable values
            - "min": All minimum representable values
            - "mixed": Mix of edge case values.

    Returns:
        ExTensor filled with edge case values.

    Raises:
        Error: If tensor creation fails or unknown edge_type.

    Example:
        ```mojo
        var shape = List[Int]()
        shape.append(3)
        var nan_tensor = create_edge_case_tensor(shape, DType.float32, "nan")
        var inf_tensor = create_edge_case_tensor(shape, DType.float32, "inf")
        ```
    """
    var tensor = zeros(shape, dtype)
    var numel = tensor.numel()

    if edge_type == "nan":
        # Create NaN via 0.0 / 0.0
        var nan_val = Float64(0.0) / Float64(0.0)
        for i in range(numel):
            tensor._set_float64(i, nan_val)

    elif edge_type == "inf":
        # Create positive infinity via 1.0 / 0.0
        var inf_val = Float64(1.0) / Float64(0.0)
        for i in range(numel):
            tensor._set_float64(i, inf_val)

    elif edge_type == "neg_inf":
        # Create negative infinity via -1.0 / 0.0
        var neg_inf_val = Float64(-1.0) / Float64(0.0)
        for i in range(numel):
            tensor._set_float64(i, neg_inf_val)

    elif edge_type == "zero":
        # Already zeros, nothing to do
        pass

    elif edge_type == "epsilon":
        # Smallest positive value (approximate)
        var epsilon_val = Float64(1e-38)
        for i in range(numel):
            tensor._set_float64(i, epsilon_val)

    elif edge_type == "max":
        # Large positive value (approximate max for float32)
        var max_val = Float64(3.4e38)
        for i in range(numel):
            tensor._set_float64(i, max_val)

    elif edge_type == "min":
        # Large negative value (approximate min for float32)
        var min_val = Float64(-3.4e38)
        for i in range(numel):
            tensor._set_float64(i, min_val)

    elif edge_type == "mixed":
        # Mix of edge case values in pattern
        var edge_values = List[Float64]()
        edge_values.append(Float64(0.0) / Float64(0.0))  # NaN
        edge_values.append(Float64(1.0) / Float64(0.0))  # Inf
        edge_values.append(Float64(-1.0) / Float64(0.0))  # -Inf
        edge_values.append(Float64(0.0))  # Zero
        edge_values.append(Float64(1e-38))  # Epsilon
        edge_values.append(Float64(3.4e38))  # Max

        for i in range(numel):
            var val = edge_values[i % len(edge_values)]
            tensor._set_float64(i, val)

    else:
        raise Error("Unknown edge case type: " + edge_type)

    return tensor^


# ============================================================================
# Numeric Invariant Checking
# ============================================================================


fn has_nan(tensor: ExTensor) -> Bool:
    """Check if tensor contains any NaN values.

    Args:
        tensor: Tensor to check.

    Returns:
        True if any element is NaN.

    Example:
        ```mojo
        var tensor = create_edge_case_tensor(shape, DType.float32, "nan")
        assert has_nan(tensor), "Should contain NaN"
        ```
    """
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        if isnan(val):
            return True
    return False


fn has_inf(tensor: ExTensor) -> Bool:
    """Check if tensor contains any infinity values.

    Args:
        tensor: Tensor to check.

    Returns:
        True if any element is positive or negative infinity.

    Example:
        ```mojo
        var tensor = create_edge_case_tensor(shape, DType.float32, "inf")
        assert has_inf(tensor), "Should contain Inf"
        ```
    """
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        if isinf(val):
            return True
    return False


fn is_finite(tensor: ExTensor) -> Bool:
    """Check if all tensor values are finite (not NaN or Inf).

    Args:
        tensor: Tensor to check.

    Returns:
        True if all elements are finite numbers.

    Example:
        ```mojo
        var tensor = ones(shape, DType.float32)
        assert is_finite(tensor), "Should be all finite"
        ```
    """
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        if isnan(val) or isinf(val):
            return False
    return True


fn all_values_in_range(tensor: ExTensor, low: Float64, high: Float64) -> Bool:
    """Check if all tensor values are within specified range.

    Args:
        tensor: Tensor to check.
        low: Minimum allowed value (inclusive).
        high: Maximum allowed value (inclusive).

    Returns:
        True if all elements are in [low, high].

    Example:
        ```mojo
        var tensor = create_random_tensor(rng, shape, DType.float32, 0.0, 1.0)
        assert all_values_in_range(tensor, 0.0, 1.0), "Should be in [0, 1]"
        ```
    """
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        if isnan(val) or val < low or val > high:
            return False
    return True


struct NumericInvariants(Copyable, Movable):
    """Result of numeric invariant checks.

    Captures the results of checking various numeric properties
    on a tensor.

    Attributes:
        has_nan: Whether tensor contains NaN.
        has_inf: Whether tensor contains infinity.
        has_negative: Whether tensor contains negative values.
        min_value: Minimum value in tensor.
        max_value: Maximum value in tensor.
        is_valid: Overall validity (no NaN/Inf unless expected).

    Example:
        ```mojo
        var invariants = check_numeric_invariants(tensor)
        if not invariants.is_valid:
            print("Invalid tensor:", invariants.has_nan, invariants.has_inf)
        ```
    """

    var has_nan: Bool
    var has_inf: Bool
    var has_negative: Bool
    var min_value: Float64
    var max_value: Float64
    var is_valid: Bool

    fn __init__(out self):
        """Initialize with default values."""
        self.has_nan = False
        self.has_inf = False
        self.has_negative = False
        self.min_value = Float64(1.0) / Float64(0.0)  # +Inf
        self.max_value = Float64(-1.0) / Float64(0.0)  # -Inf
        self.is_valid = True


fn check_numeric_invariants(
    tensor: ExTensor, allow_nan: Bool = False, allow_inf: Bool = False
) -> NumericInvariants:
    """Check numeric invariants on a tensor.

    Analyzes a tensor for various numeric properties and returns
    a summary of findings.

    Args:
        tensor: Tensor to check.
        allow_nan: Whether NaN values are allowed (default: False).
        allow_inf: Whether Inf values are allowed (default: False).

    Returns:
        NumericInvariants with analysis results.

    Example:
        ```mojo
        var result = create_random_tensor(rng, shape, DType.float32)
        var invariants = check_numeric_invariants(result)
        assert invariants.is_valid, "Result should be valid"
        ```
    """
    var invariants = NumericInvariants()
    var numel = tensor.numel()

    if numel == 0:
        return invariants^

    # Check first value to initialize min/max
    var first_val = tensor._get_float64(0)
    if not isnan(first_val) and not isinf(first_val):
        invariants.min_value = first_val
        invariants.max_value = first_val

    for i in range(numel):
        var val = tensor._get_float64(i)

        if isnan(val):
            invariants.has_nan = True
            if not allow_nan:
                invariants.is_valid = False
        elif isinf(val):
            invariants.has_inf = True
            if not allow_inf:
                invariants.is_valid = False
        else:
            if val < 0:
                invariants.has_negative = True
            if val < invariants.min_value:
                invariants.min_value = val
            if val > invariants.max_value:
                invariants.max_value = val

    return invariants^


# ============================================================================
# Property-Based Testing Helpers
# ============================================================================


fn verify_shape_preserved(original: ExTensor, result: ExTensor) -> Bool:
    """Verify that operation preserved tensor shape.

    Args:
        original: Original tensor before operation.
        result: Result tensor after operation.

    Returns:
        True if shapes match.

    Example:
        ```mojo
        var tensor = ones(shape, DType.float32)
        var result = relu(tensor)
        assert verify_shape_preserved(tensor, result), "Shape should be preserved"
        ```
    """
    var orig_shape = original.shape()
    var res_shape = result.shape()

    if len(orig_shape) != len(res_shape):
        return False

    for i in range(len(orig_shape)):
        if orig_shape[i] != res_shape[i]:
            return False

    return True


fn verify_dtype_preserved(original: ExTensor, result: ExTensor) -> Bool:
    """Verify that operation preserved tensor dtype.

    Args:
        original: Original tensor before operation.
        result: Result tensor after operation.

    Returns:
        True if dtypes match.

    Example:
        ```mojo
        var tensor = ones(shape, DType.float32)
        var result = relu(tensor)
        assert verify_dtype_preserved(tensor, result), "DType should be preserved"
        ```
    """
    return original.dtype() == result.dtype()


fn verify_numel_preserved(original: ExTensor, result: ExTensor) -> Bool:
    """Verify that operation preserved number of elements.

    Args:
        original: Original tensor before operation.
        result: Result tensor after operation.

    Returns:
        True if element counts match.

    Example:
        ```mojo
        var tensor = ones(shape, DType.float32)
        var result = reshape(tensor, new_shape)
        assert verify_numel_preserved(tensor, result), "Numel should be preserved"
        ```
    """
    return original.numel() == result.numel()


# ============================================================================
# Fuzzing Runner Utilities
# ============================================================================


fn format_failure_message(
    operation: String,
    iteration: Int,
    seed: Int,
    shape: List[Int],
    dtype: DType,
    error_msg: String,
) -> String:
    """Format a failure message for reporting.

    Args:
        operation: Name of the operation that failed.
        iteration: Iteration number when failure occurred.
        seed: Random seed for reproduction.
        shape: Shape of input tensor(s).
        dtype: Data type of tensor(s).
        error_msg: Error message from the failure.

    Returns:
        Formatted failure message string.

    Example:
        ```mojo
        var msg = format_failure_message(
            "matmul", 42, 12345, shape, DType.float32, "Shape mismatch"
        )
        print(msg)
        ```
    """
    var shape_str = String("[")
    for i in range(len(shape)):
        if i > 0:
            shape_str += ", "
        shape_str += String(shape[i])
    shape_str += "]"

    return (
        "Operation: "
        + operation
        + ", Iteration: "
        + String(iteration)
        + ", Seed: "
        + String(seed)
        + ", Shape: "
        + shape_str
        + ", DType: "
        + dtype_to_string(dtype)
        + ", Error: "
        + error_msg
    )
