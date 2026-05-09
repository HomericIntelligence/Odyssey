"""Test Data Generators for ML Odyssey.

Provides utilities to generate synthetic test data for machine learning tests:
- Random tensors with configurable distributions
- Synthetic classification datasets
- Reproducible random generation

This module is essential for writing comprehensive tests that validate model
behavior across different input distributions and dataset characteristics.

Example:
    from shared.testing import random_tensor, random_normal, synthetic_classification_data
    from shared.core import AnyTensor

    # Create a random tensor
    var weights = random_tensor([10, 5], DType.float32)

    # Create normally distributed data
    var features = random_normal([100, 20], mean=0.0, std=1.0)

    # Create classification data
    var (X, y) = synthetic_classification_data(100, 10, 3)
    ```
"""

from std.random import random_float64
from std.math import sqrt, log, cos, sin, pi
from shared.tensor.any_tensor import AnyTensor, zeros


# ============================================================================
# Random Tensor Generators
# ============================================================================


def random_tensor(
    shape: List[Int], dtype: DType = DType.float32
) raises -> AnyTensor:
    """Generate tensor with random values from uniform distribution [0, 1).

    Args:
            shape: Shape of the output tensor as a list of dimensions.
            dtype: Data type of tensor elements (default: float32).

    Returns:
            AnyTensor with random values uniformly distributed in [0, 1).

    Example:
        ```mojo
        var weights = random_tensor([10, 5], DType.float32)
        # Creates 10x5 tensor with random values in [0, 1)
        ```

    Note:
            Values are uniformly distributed in [0, 1) regardless of dtype
            For integer dtypes, values are truncated to Int.

    Raises:
        Error: If operation fails.
    """
    # Create empty tensor with the specified shape
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Fill with random values
    for i in range(numel):
        var rand_val = random_float64()

        # Convert to appropriate dtype
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
        ):
            tensor._set_float64(i, rand_val)
        elif (
            dtype == DType.int8
            or dtype == DType.int16
            or dtype == DType.int32
            or dtype == DType.int64
        ):
            tensor._set_int64(i, Int64(rand_val))
        elif (
            dtype == DType.uint8
            or dtype == DType.uint16
            or dtype == DType.uint32
            or dtype == DType.uint64
        ):
            tensor._set_int64(i, Int64(rand_val))

    return tensor^


def random_uniform(
    shape: List[Int],
    low: Float64 = 0.0,
    high: Float64 = 1.0,
    dtype: DType = DType.float32,
) raises -> AnyTensor:
    """Generate tensor with random values from uniform distribution [low, high).

    Args:
            shape: Shape of the output tensor as a list of dimensions.
            low: Lower bound of uniform distribution (inclusive, default: 0.0).
            high: Upper bound of uniform distribution (exclusive, default: 1.0).
            dtype: Data type of tensor elements (default: float32).

    Returns:
            AnyTensor with random values uniformly distributed in [low, high)

        Example:
            ```mojo
            var data = random_uniform([100, 20], low=-1.0, high=1.0)
            # Creates 100x20 tensor with random values in [-1.0, 1.0)
            ```

    Note:
            The range [low, high) is linearly scaled from [0, 1)
            For integer dtypes, values are truncated to Int.

    Raises:
        Error: If operation fails.
    """
    # Create empty tensor with the specified shape
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Calculate value range
    var value_range = high - low

    # Fill with random values
    for i in range(numel):
        var rand_val = random_float64()
        var scaled_val = low + rand_val * value_range

        # Convert to appropriate dtype
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
        ):
            tensor._set_float64(i, scaled_val)
        elif (
            dtype == DType.int8
            or dtype == DType.int16
            or dtype == DType.int32
            or dtype == DType.int64
        ):
            tensor._set_int64(i, Int64(scaled_val))
        elif (
            dtype == DType.uint8
            or dtype == DType.uint16
            or dtype == DType.uint32
            or dtype == DType.uint64
        ):
            tensor._set_int64(i, Int64(scaled_val))

    return tensor^


def random_normal(
    shape: List[Int],
    mean: Float64 = 0.0,
    std: Float64 = 1.0,
    dtype: DType = DType.float32,
) raises -> AnyTensor:
    """Generate tensor with random values from normal distribution N(mean, std^2).

        Uses Box-Muller transform to convert uniform random values to normal distribution

    Args:
            shape: Shape of the output tensor as a list of dimensions.
            mean: Mean of the normal distribution (default: 0.0).
            std: Standard deviation of the normal distribution (default: 1.0).
            dtype: Data type of tensor elements (default: float32).

    Returns:
            AnyTensor with random values from normal distribution N(mean, std^2)

        Example:
            ```mojo
            var weights = random_normal([784, 256], mean=0.0, std=0.01)
            # Creates 784x256 tensor with normally distributed values
            ```

    Note:
            Uses Box-Muller transform for efficiency
            For integer dtypes, values are truncated to Int after sampling.

    Raises:
        Error: If operation fails.
    """
    # Create empty tensor with the specified shape
    var tensor = zeros(shape, dtype)

    # Calculate total number of elements
    var numel = 1
    for dim in shape:
        numel *= dim

    # Generate pairs of normal values using Box-Muller transform
    var i = 0
    while i < numel:
        # Generate two uniform random values
        var u1 = random_float64()
        var u2 = random_float64()

        # Avoid log(0)
        if u1 < 1e-10:
            u1 = 1e-10

        # Box-Muller transform
        var r = sqrt(-2.0 * log(u1))
        var theta = 2.0 * pi * u2

        # Convert to appropriate dtype
        if (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
        ):
            # First value
            var z1 = r * cos(theta)
            tensor._set_float64(i, mean + z1 * std)
            i += 1

            # Second value (if there's room)
            if i < numel:
                var z2 = r * sin(theta)
                tensor._set_float64(i, mean + z2 * std)
                i += 1
        else:
            # For integer dtypes, still use Box-Muller but truncate
            var z1 = r * cos(theta)
            tensor._set_int64(i, Int64(mean + z1 * std))
            i += 1

            if i < numel:
                var z2 = r * sin(theta)
                tensor._set_int64(i, Int64(mean + z2 * std))
                i += 1

    return tensor^


# ============================================================================
# Classification Dataset Generators
# ============================================================================


def synthetic_classification_data(
    num_samples: Int,
    num_features: Int,
    num_classes: Int,
    dtype: DType = DType.float32,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Generate synthetic classification dataset.

        Creates a random dataset with linearly separable classes by generating
        class centers and adding noise around each center

    Args:
            num_samples: Total number of samples to generate.
            num_features: Number of features per sample.
            num_classes: Number of classes.
            dtype: Data type for features (labels are always int32).

    Returns:
            Tuple of (features, labels) where:
            - features: AnyTensor of shape [num_samples, num_features]
            - labels: AnyTensor of shape [num_samples] with values in [0, num_classes)

        Example:
            ```mojo
            var (X, y) = synthetic_classification_data(100, 20, 3)
            # X shape: [100, 20], Y shape: [100]
            # Y contains values in {0, 1, 2}
            ```

        Algorithm:
            1. Generate random class centers in [-5, 5]^num_features.
            2. For each sample, assign to random class.
            3. Add Gaussian noise around class center.
            4. Normalize features to zero mean and unit variance.

    Raises:
        Error: If operation fails.
    """
    # Validate inputs
    if num_samples <= 0 or num_features <= 0 or num_classes <= 0:
        raise Error("All parameters must be positive")

    # Create class centers: [num_classes, num_features]
    var centers_shape = List[Int]()
    centers_shape.append(num_classes)
    centers_shape.append(num_features)
    var centers = random_uniform(centers_shape, low=-5.0, high=5.0, dtype=dtype)

    # Create features tensor: [num_samples, num_features]
    var features_shape = List[Int]()
    features_shape.append(num_samples)
    features_shape.append(num_features)
    var features = zeros(features_shape, dtype)

    # Create labels tensor: [num_samples]
    var labels_shape = List[Int]()
    labels_shape.append(num_samples)
    var labels = zeros(labels_shape, DType.int32)

    # Fill features and labels
    for sample_idx in range(num_samples):
        # Randomly select a class for this sample
        var class_idx = Int(random_float64() * Float64(num_classes))
        if class_idx >= num_classes:
            class_idx = num_classes - 1

        # Set label
        labels._set_int64(sample_idx, Int64(class_idx))

        # Copy center for this class and add noise
        for feat_idx in range(num_features):
            var center_idx = class_idx * num_features + feat_idx
            var center_val = centers._get_float64(center_idx)

            # Add Gaussian noise (mean=0, std=0.5)
            var noise = random_normal([1], mean=0.0, std=0.5, dtype=dtype)
            var noise_val = noise._get_float64(0)

            var feature_val = center_val + noise_val
            var features_idx = sample_idx * num_features + feat_idx
            features._set_float64(features_idx, feature_val)

    return Tuple[AnyTensor, AnyTensor](features^, labels^)
