"""Weight initialization methods for neural networks.

Provides common initialization strategies including Xavier/Glorot for sigmoid/tanh
networks, Kaiming/He for ReLU networks, and basic uniform/normal distributions.
Proper initialization is crucial for avoiding vanishing/exploding gradients.

Mathematical foundations:
- Xavier: Var(W) = 2/(fan_in + fan_out) for symmetric activations
- Kaiming: Var(W) = 2/fan_in for ReLU activations
- Uniform/Normal: Basic distributions with configurable parameters

Type support:
- All initializers: float16, float32, float64

Naming conventions:
- kaiming_uniform/kaiming_normal: Standard names
- he_uniform/he_normal: Aliases (Kaiming He is the author)

Issues covered:
- #258-260: Xavier/Glorot initialization (uniform and normal variants)
- #268-272: Uniform/Normal basic distributions
"""

from random import random_float64, random_si64, seed as random_seed
from math import sqrt, log, cos, sin
from collections import List
from .extensor import ExTensor


# ============================================================================
# Internal Helper Functions (Dtype-Generic)
# ============================================================================


@always_inline
fn _fill_uniform_scaled[
    dtype: DType
](result: ExTensor, scale: Float64, offset: Float64) raises:
    """Fill tensor with scaled uniform random values: offset + random() * scale.

        This is a dtype-generic helper that eliminates dtype branching.
        random_float64() returns [0, 1), which is transformed to [offset, offset+scale).

    Args:
            result: Tensor to fill (must be pre-allocated).
            scale: Scale factor for random values.
            offset: Offset to add to scaled values.

    Raises:
            Error: If operation fails.
    """
    var ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(result._numel):
        var rand_val = random_float64()
        ptr[i] = Scalar[dtype](offset + rand_val * scale)


@always_inline
fn _fill_normal_boxmuller[
    dtype: DType
](result: ExTensor, mean: Float64, std: Float64) raises:
    """Fill tensor with normal random values using Box-Muller transform.

        This is a dtype-generic helper that eliminates dtype branching.
        Generates pairs of normal random values using Box-Muller transform.

    Args:
            result: Tensor to fill (must be pre-allocated).
            mean: Mean of normal distribution.
            std: Standard deviation of normal distribution.

    Raises:
            Error: If operation fails.
    """
    var ptr = result._data.bitcast[Scalar[dtype]]()
    var i = 0
    comptime PI = 3.14159265359

    while i < result._numel:
        # Box-Muller transform to generate normal distribution
        var u1 = random_float64()
        var u2 = random_float64()

        # Avoid log(0)
        if u1 < 1e-10:
            u1 = 1e-10

        var mag = std * sqrt(-2.0 * log(u1))
        var z0 = mean + mag * cos(2.0 * PI * u2)
        var z1 = mean + mag * sin(2.0 * PI * u2)

        ptr[i] = Scalar[dtype](z0)
        i += 1

        if i < result._numel:
            ptr[i] = Scalar[dtype](z1)
            i += 1


@always_inline
fn _fill_constant[dtype: DType](result: ExTensor, value: Float64) raises:
    """Fill tensor with constant value.

        This is a dtype-generic helper that eliminates dtype branching.

    Args:
            result: Tensor to fill (must be pre-allocated).
            value: Constant value to fill with.

    Raises:
            Error: If operation fails.
    """
    var ptr = result._data.bitcast[Scalar[dtype]]()
    var val = Scalar[dtype](value)
    for i in range(result._numel):
        ptr[i] = val


# ============================================================================
# DType Dispatch Helpers (Issue #2582)
# ============================================================================


fn _dispatch_fill_uniform_scaled(
    result: ExTensor, scale: Float64, offset: Float64
) raises:
    """Dispatch uniform fill based on tensor dtype.

    Centralizes dtype branching for _fill_uniform_scaled to avoid
    repetitive if/elif chains throughout the codebase.

    Args:
        result: Tensor to fill (pre-allocated, dtype determines dispatch).
        scale: Scale factor for random values.
        offset: Offset to add to scaled values.

    Raises:
        Error: If tensor dtype is not float16, float32, or float64.
    """
    var dtype = result.dtype()
    if dtype == DType.float16:
        _fill_uniform_scaled[DType.float16](result, scale, offset)
    elif dtype == DType.float32:
        _fill_uniform_scaled[DType.float32](result, scale, offset)
    elif dtype == DType.float64:
        _fill_uniform_scaled[DType.float64](result, scale, offset)
    else:
        raise Error(
            "Unsupported dtype for uniform fill: only float16, float32,"
            " float64 supported"
        )


fn _dispatch_fill_normal_boxmuller(
    result: ExTensor, mean: Float64, std: Float64
) raises:
    """Dispatch normal fill based on tensor dtype.

    Centralizes dtype branching for _fill_normal_boxmuller to avoid
    repetitive if/elif chains throughout the codebase.

    Args:
        result: Tensor to fill (pre-allocated, dtype determines dispatch).
        mean: Mean of normal distribution.
        std: Standard deviation of normal distribution.

    Raises:
        Error: If tensor dtype is not float16, float32, or float64.
    """
    var dtype = result.dtype()
    if dtype == DType.float16:
        _fill_normal_boxmuller[DType.float16](result, mean, std)
    elif dtype == DType.float32:
        _fill_normal_boxmuller[DType.float32](result, mean, std)
    elif dtype == DType.float64:
        _fill_normal_boxmuller[DType.float64](result, mean, std)
    else:
        raise Error(
            "Unsupported dtype for normal fill: only float16, float32,"
            " float64 supported"
        )


fn _dispatch_fill_constant(result: ExTensor, value: Float64) raises:
    """Dispatch constant fill based on tensor dtype.

    Centralizes dtype branching for _fill_constant to avoid
    repetitive if/elif chains throughout the codebase.

    Args:
        result: Tensor to fill (pre-allocated, dtype determines dispatch).
        value: Constant value to fill with.

    Raises:
        Error: If tensor dtype is not float16, float32, or float64.
    """
    var dtype = result.dtype()
    if dtype == DType.float16:
        _fill_constant[DType.float16](result, value)
    elif dtype == DType.float32:
        _fill_constant[DType.float32](result, value)
    elif dtype == DType.float64:
        _fill_constant[DType.float64](result, value)
    else:
        raise Error(
            "Unsupported dtype for constant fill: only float16, float32,"
            " float64 supported"
        )


# ============================================================================
# Xavier/Glorot Initialization (#258-260)
# ============================================================================


fn xavier_uniform(
    fan_in: Int,
    fan_out: Int,
    shape: List[Int],
    dtype: DType = DType.float32,
    seed_val: Int = -1,
) raises -> ExTensor:
    """Initialize weights using Xavier/Glorot uniform distribution.

        Draws samples from uniform distribution U(-a, a) where:
            a = sqrt(6 / (fan_in + fan_out)).

        This initialization maintains variance of activations and gradients across
        layers for networks using sigmoid or tanh activations.

        Mathematical derivation:
        - For uniform U(-a, a), variance is a²/3.
        - Target variance: 2/(fan_in + fan_out).
        - Therefore: a²/3 = 2/(fan_in + fan_out).
        - Solving: a = sqrt(6/(fan_in + fan_out)).

        Supported dtypes: float16, float32, float64.

    Args:
            fan_in: Number of input units to the layer.
            fan_out: Number of output units from the layer.
            shape: Shape of weight tensor to initialize.
            dtype: Data type (default: float32).
            seed_val: Random seed for reproducibility (-1 for random seed).

    Returns:
            Initialized weight tensor with Xavier uniform distribution.

    Raises:
            Error: If fan_in or fan_out are not positive.

    Examples:
            # Fully connected layer: 784 inputs -> 128 outputs
            var weights = xavier_uniform(784, 128, [784, 128])

            # With fixed seed for reproducibility
            var w = xavier_uniform(100, 50, [100, 50], seed_val=42)

        References:
            Glorot & Bengio (2010): "Understanding the difficulty of training
            deep feedforward neural networks".
    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error("xavier_uniform: fan_in and fan_out must be positive")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Calculate bound: sqrt(6 / (fan_in + fan_out))
    var bound = sqrt(6.0 / Float64(fan_in + fan_out))

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with uniform random values in [-bound, bound]
    _dispatch_fill_uniform_scaled(result, 2.0 * bound, -bound)

    return result^


fn xavier_normal(
    fan_in: Int,
    fan_out: Int,
    shape: List[Int],
    dtype: DType = DType.float32,
    seed_val: Int = -1,
) raises -> ExTensor:
    """Initialize weights using Xavier/Glorot normal distribution.

        Draws samples from normal distribution N(0, std²) where:
            std = sqrt(2 / (fan_in + fan_out)).

        This initialization maintains variance of activations and gradients across
        layers for networks using sigmoid or tanh activations.

        Mathematical derivation:
        - For normal N(0, σ²), variance is σ².
        - Target variance: 2/(fan_in + fan_out).
        - Therefore: σ² = 2/(fan_in + fan_out).
        - Standard deviation: σ = sqrt(2/(fan_in + fan_out)).

        Supported dtypes: float16, float32, float64.

    Args:
            fan_in: Number of input units to the layer.
            fan_out: Number of output units from the layer.
            shape: Shape of weight tensor to initialize.
            dtype: Data type (default: float32).
            seed_val: Random seed for reproducibility (-1 for random seed).

    Returns:
            Initialized weight tensor with Xavier normal distribution.

    Raises:
            Error: If fan_in or fan_out are not positive.

    Examples:
            # Fully connected layer: 784 inputs -> 128 outputs
            var weights = xavier_normal(784, 128, [784, 128])

            # With fixed seed for reproducibility
            var w = xavier_normal(100, 50, [100, 50], seed_val=42)

        References:
            Glorot & Bengio (2010): "Understanding the difficulty of training
            deep feedforward neural networks".
    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error("xavier_normal: fan_in and fan_out must be positive")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Calculate standard deviation: sqrt(2 / (fan_in + fan_out))
    var std = sqrt(2.0 / Float64(fan_in + fan_out))

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with normal random values using Box-Muller transform
    _dispatch_fill_normal_boxmuller(result, 0.0, std)

    return result^


# Helper functions for Box-Muller transform (using math.log, math.cos, math.sin from imports)


# ============================================================================
# Kaiming/He Initialization (#263-267)
# ============================================================================


fn kaiming_uniform(
    fan_in: Int,
    fan_out: Int,
    shape: List[Int],
    fan_mode: String = "fan_in",
    dtype: DType = DType.float32,
    seed_val: Int = -1,
) raises -> ExTensor:
    """Initialize weights using Kaiming/He uniform distribution.

        Draws samples from uniform distribution U(-a, a) where:
            a = sqrt(6 / fan)  (fan depends on fan_mode).

        This initialization is designed for networks with ReLU activations, which
        kill half the activations (output 0 for negative inputs). The gain factor
        accounts for this to maintain variance.

        Mathematical derivation:
        - For uniform U(-a, a), variance is a²/3.
        - Target variance for ReLU: 2/fan (not 2/(fan_in + fan_out) like Xavier).
        - Therefore: a²/3 = 2/fan.
        - Solving: a = sqrt(6/fan).

        Supported dtypes: float16, float32, float64.

    Args:
            fan_in: Number of input units to the layer.
            fan_out: Number of output units from the layer.
            shape: Shape of weight tensor to initialize.
            fan_mode: "fan_in" (default) or "fan_out" for fan calculation.
            dtype: Data type (default: float32).
            seed_val: Random seed for reproducibility (-1 for random seed).

    Returns:
            Initialized weight tensor with Kaiming uniform distribution.

    Raises:
            Error: If fan_in or fan_out are not positive.
            Error: If fan_mode is not "fan_in" or "fan_out".

    Examples:
            # Fully connected layer: 784 inputs -> 128 outputs (using fan_in)
            var weights = kaiming_uniform(784, 128, [784, 128])

            # Using fan_out mode
            var w = kaiming_uniform(784, 128, [784, 128], fan_mode="fan_out")

            # With fixed seed for reproducibility
            var w_repro = kaiming_uniform(100, 50, [100, 50], seed_val=42)

        References:
            He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level
            Performance on ImageNet Classification".

    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error("kaiming_uniform: fan_in and fan_out must be positive")

    # Determine which fan to use
    var fan: Int
    if fan_mode == "fan_in":
        fan = fan_in
    elif fan_mode == "fan_out":
        fan = fan_out
    else:
        raise Error("kaiming_uniform: fan_mode must be 'fan_in' or 'fan_out'")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Calculate bound: sqrt(6 / fan)
    var bound = sqrt(6.0 / Float64(fan))

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with uniform random values in [-bound, bound]
    _dispatch_fill_uniform_scaled(result, 2.0 * bound, -bound)

    return result^


fn kaiming_normal(
    fan_in: Int,
    fan_out: Int,
    shape: List[Int],
    fan_mode: String = "fan_in",
    dtype: DType = DType.float32,
    seed_val: Int = -1,
) raises -> ExTensor:
    """Initialize weights using Kaiming/He normal distribution.

        Draws samples from normal distribution N(0, std²) where:
            std = sqrt(2 / fan)  (fan depends on fan_mode).

        This initialization is designed for networks with ReLU activations, which
        kill half the activations. The variance scaling accounts for this effect.

        Mathematical derivation:
        - For normal N(0, σ²), variance is σ².
        - Target variance for ReLU: 2/fan.
        - Therefore: σ² = 2/fan.
        - Standard deviation: σ = sqrt(2/fan).

        Supported dtypes: float16, float32, float64.

    Args:
            fan_in: Number of input units to the layer.
            fan_out: Number of output units from the layer.
            shape: Shape of weight tensor to initialize.
            fan_mode: "fan_in" (default) or "fan_out" for fan calculation.
            dtype: Data type (default: float32).
            seed_val: Random seed for reproducibility (-1 for random seed).

    Returns:
            Initialized weight tensor with Kaiming normal distribution.

    Raises:
            Error: If fan_in or fan_out are not positive.
            Error: If fan_mode is not "fan_in" or "fan_out".

    Examples:
            # Fully connected layer: 784 inputs -> 128 outputs
            var weights = kaiming_normal(784, 128, [784, 128])

            # Using fan_out mode
            var w = kaiming_normal(784, 128, [784, 128], fan_mode="fan_out")

            # With fixed seed for reproducibility
            var w_repro = kaiming_normal(100, 50, [100, 50], seed_val=42)

        References:
            He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level
            Performance on ImageNet Classification".

    """
    if fan_in <= 0 or fan_out <= 0:
        raise Error("kaiming_normal: fan_in and fan_out must be positive")

    # Determine which fan to use
    var fan: Int
    if fan_mode == "fan_in":
        fan = fan_in
    elif fan_mode == "fan_out":
        fan = fan_out
    else:
        raise Error("kaiming_normal: fan_mode must be 'fan_in' or 'fan_out'")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Calculate standard deviation: sqrt(2 / fan)
    var std = sqrt(2.0 / Float64(fan))

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with normal random values using Box-Muller transform
    _dispatch_fill_normal_boxmuller(result, 0.0, std)

    return result^


# ============================================================================
# Uniform/Normal Initialization (#268-272)
# ============================================================================


fn uniform(
    shape: List[Int],
    low: Float64 = -0.1,
    high: Float64 = 0.1,
    dtype: DType = DType.float32,
    seed_val: Int = -1,
) raises -> ExTensor:
    """Initialize weights using uniform distribution.

        Draws samples from uniform distribution U(low, high) with configurable bounds.
        This is a basic initializer useful for biases, embeddings, or custom schemes.

    Args:
            shape: Shape of tensor to initialize.
            low: Lower bound of uniform distribution (default: -0.1).
            high: Upper bound of uniform distribution (default: 0.1).
            dtype: Data type (default: float32).
            seed_val: Random seed for reproducibility (-1 for random seed).

    Returns:
            Initialized tensor with uniform distribution.

    Raises:
            Error: If low >= high.

    Examples:
    ```
            # Default range [-0.1, 0.1]
            var weights = uniform([100, 50])

            # Custom range [0, 1]
            var w = uniform([10, 10], low=0.0, high=1.0)

            # With fixed seed
            var w_repro = uniform([50, 50], seed_val=42)
    ```
    """
    if low >= high:
        raise Error("uniform: low must be less than high")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with uniform random values in [low, high]
    var range_val = high - low
    _dispatch_fill_uniform_scaled(result, range_val, low)

    return result^


fn normal(
    shape: List[Int],
    mean: Float64 = 0.0,
    std: Float64 = 0.01,
    dtype: DType = DType.float32,
    seed_val: Int = -1,
) raises -> ExTensor:
    """Initialize weights using normal (Gaussian) distribution.

        Draws samples from normal distribution N(mean, std²) with configurable parameters.
        This is a basic initializer useful for biases, embeddings, or custom schemes.

        Uses Box-Muller transform to generate normal distribution from uniform samples.

    Args:
            shape: Shape of tensor to initialize.
            mean: Mean of normal distribution (default: 0.0).
            std: Standard deviation of normal distribution (default: 0.01).
            dtype: Data type (default: float32).
            seed_val: Random seed for reproducibility (-1 for random seed).

    Returns:
            Initialized tensor with normal distribution.

    Raises:
            Error: If std <= 0.

    Examples:
    ```
            # Default: N(0, 0.01)
            var weights = normal([100, 50])

            # Custom: N(0.5, 0.1)
            var w = normal([10, 10], mean=0.5, std=0.1)

            # With fixed seed
            var w_repro = normal([50, 50], seed_val=42)
    ```
    """
    if std <= 0.0:
        raise Error("normal: standard deviation must be positive")

    # Set random seed if provided
    if seed_val >= 0:
        random_seed(seed_val)

    # Create tensor
    var result = ExTensor(shape, dtype)

    # Fill with normal random values using Box-Muller transform
    _dispatch_fill_normal_boxmuller(result, mean, std)

    return result^


fn constant(
    shape: List[Int], value: Float64, dtype: DType = DType.float32
) raises -> ExTensor:
    """Initialize tensor with constant value.

        Fills all elements with the specified constant value.
        Useful for specific initialization strategies (ones, custom bias values, etc.).

    Args:
            shape: Shape of tensor to initialize.
            value: Constant value to fill tensor with.
            dtype: Data type (default: float32).

    Returns:
            Tensor filled with constant value.

    Raises:
        Error: If operation fails.

    Examples:
    ```
            # Initialize with ones
            var ones = constant([10, 10], 1.0)

            # Initialize with custom value
            var custom = constant([5, 5], 0.5)

            # Initialize bias with 0.01
            var bias = constant(List[Int](), 0.01)
    ```
    """
    var result = ExTensor(shape, dtype)
    _dispatch_fill_constant(result, value)
    return result^


# ============================================================================
# Aliases for Compatibility
# ============================================================================


fn he_uniform(
    fan_in: Int,
    fan_out: Int,
    shape: List[Int],
    fan_mode: String = "fan_in",
    dtype: DType = DType.float32,
    seed_val: Int = -1,
) raises -> ExTensor:
    """Alias for kaiming_uniform.

    He and Kaiming refer to the same initialization method (Kaiming He is the author).
    This comptime provides compatibility with code that uses 'he_uniform' naming.

    See kaiming_uniform() for full documentation.

    Raises:
        Error: If operation fails.
    """
    return kaiming_uniform(fan_in, fan_out, shape, fan_mode, dtype, seed_val)


fn he_normal(
    fan_in: Int,
    fan_out: Int,
    shape: List[Int],
    fan_mode: String = "fan_in",
    dtype: DType = DType.float32,
    seed_val: Int = -1,
) raises -> ExTensor:
    """Alias for kaiming_normal.

    He and Kaiming refer to the same initialization method (Kaiming He is the author).
    This comptime provides compatibility with code that uses 'he_normal' naming.

    See kaiming_normal() for full documentation.

    Raises:
        Error: If operation fails.
    """
    return kaiming_normal(fan_in, fan_out, shape, fan_mode, dtype, seed_val)


# ============================================================================
# Convenience Overloads (Auto-compute fan_in/fan_out)
# ============================================================================


fn _compute_fan_from_shape(shape: List[Int]) raises -> Tuple[Int, Int]:
    """Compute (fan_in, fan_out) from tensor shape for weight initialization.

    Args:
            shape: Tensor shape - must be 2D [out_features, in_features] for linear
                   or 4D [out_channels, in_channels, kH, kW] for conv.

    Returns:
            Tuple of (fan_in, fan_out).

    Raises:
            Error if shape is not 2D or 4D.
    """
    if len(shape) == 2:
        # Linear layer: [out_features, in_features]
        return (shape[1], shape[0])
    elif len(shape) == 4:
        # Conv layer: [out_channels, in_channels, kH, kW]
        var out_channels = shape[0]
        var in_channels = shape[1]
        var kH = shape[2]
        var kW = shape[3]
        return (in_channels * kH * kW, out_channels * kH * kW)
    else:
        raise Error(
            "Shape must be 2D (linear) or 4D (conv) for auto fan computation"
        )


fn he_uniform(
    shape: List[Int], dtype: DType = DType.float32
) raises -> ExTensor:
    """Convenience overload that computes fan_in/fan_out from shape.

        For conv weights [out_channels, in_channels, kH, kW]:
            fan_in = in_channels * kH * kW.
            fan_out = out_channels * kH * kW.

        For linear weights [out_features, in_features]:
            fan_in = in_features.
            fan_out = out_features.

    Args:
            shape: Tensor shape (2D for linear, 4D for conv).
            dtype: Data type for the tensor.

    Returns:
            Initialized tensor.

    Raises:
            Error: If shape is not 2D or 4D.
    """
    var fans = _compute_fan_from_shape(shape)
    var fan_in = fans[0]
    var fan_out = fans[1]

    return kaiming_uniform(fan_in, fan_out, shape, "fan_in", dtype, -1)


fn xavier_uniform(
    shape: List[Int], dtype: DType = DType.float32
) raises -> ExTensor:
    """Convenience overload that computes fan_in/fan_out from shape.

        For conv weights [out_channels, in_channels, kH, kW]:
            fan_in = in_channels * kH * kW.
            fan_out = out_channels * kH * kW.

        For linear weights [out_features, in_features]:
            fan_in = in_features.
            fan_out = out_features.

    Args:
            shape: Tensor shape (2D for linear, 4D for conv).
            dtype: Data type for the tensor.

    Returns:
            Initialized tensor.

    Raises:
            Error: If shape is not 2D or 4D.
    """
    var fans = _compute_fan_from_shape(shape)
    var fan_in = fans[0]
    var fan_out = fans[1]

    return xavier_uniform(fan_in, fan_out, shape, dtype, -1)
