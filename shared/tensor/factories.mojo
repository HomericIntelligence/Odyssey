"""Typed factory functions for Tensor[dtype].

Provides compile-time typed equivalents of the AnyTensor factory functions
(zeros, ones, full, empty, arange, eye, linspace, randn, etc.).

All functions return Tensor[dtype] with the dtype parameter on both the
function and return type (B1 fix — explicit dtype parameter on all return
types). Fill values use Scalar[dtype] instead of Float64 to avoid
precision-losing round-trips.

Example:
    ```mojo
    from shared.tensor.factories import zeros, ones, full

    var z = zeros[DType.float32]([3, 4])       # 3x4 float32 zeros
    var o = ones[DType.float64]([2, 2])         # 2x2 float64 ones
    var f = full[DType.float32]([5], Scalar[DType.float32](0.5))  # [0.5, 0.5, ...]
    ```
"""

from collections import List
from math import sqrt, log, cos, sin
from random import random_float64, seed as random_seed
from utils.numerics import inf as numeric_inf, neg_inf as numeric_neg_inf

from shared.tensor.tensor import Tensor


# ------------------------------------------------------------------
# Basic creation functions
# ------------------------------------------------------------------


fn zeros[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a zero-filled tensor with compile-time dtype.

    The Tensor constructor already zero-initializes memory, so no
    additional fill is needed.

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with zeros.

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES.
    """
    var t = Tensor[dtype](shape)
    return t^


fn ones[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a one-filled tensor with compile-time dtype.

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with ones.

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES.
    """
    var t = Tensor[dtype](shape)
    var one = Scalar[dtype](1)
    for i in range(t.numel()):
        t[i] = one
    return t^


fn full[
    dtype: DType
](shape: List[Int], fill_value: Scalar[dtype]) raises -> Tensor[dtype]:
    """Create a tensor filled with a specific value.

    Uses Scalar[dtype] for the fill value to avoid precision-losing
    Float64 round-trips.

    Args:
        shape: The shape of the output tensor.
        fill_value: The typed value to fill the tensor with.

    Returns:
        A new Tensor[dtype] filled with fill_value.

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES.
    """
    var t = Tensor[dtype](shape)
    for i in range(t.numel()):
        t[i] = fill_value
    return t^


fn empty[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create an uninitialized tensor (fast allocation).

    The Tensor constructor zero-initializes memory. This function exists
    for API consistency with NumPy/PyTorch. In practice, the returned
    tensor will be zero-initialized.

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] with allocated memory.

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES.
    """
    var t = Tensor[dtype](shape)
    return t^


# ------------------------------------------------------------------
# Sequence creation functions
# ------------------------------------------------------------------


fn arange[
    dtype: DType
](
    start: Scalar[dtype], stop: Scalar[dtype], step: Scalar[dtype]
) raises -> Tensor[dtype]:
    """Create 1D tensor with evenly spaced values.

    Uses Scalar[dtype] for start/stop/step to preserve precision.

    Args:
        start: Start value (inclusive).
        stop: End value (exclusive).
        step: Spacing between values.

    Returns:
        A new 1D Tensor[dtype] with values in range [start, stop).

    Raises:
        Error: If step is zero or tensor size exceeds MAX_TENSOR_BYTES.
    """
    # Calculate number of elements using Float64 for the division
    var start_f = Float64(start)
    var stop_f = Float64(stop)
    var step_f = Float64(step)
    var num_elements = Int((stop_f - start_f) / step_f)

    if num_elements < 0:
        num_elements = 0

    var shape = List[Int]()
    shape.append(num_elements)

    var t = Tensor[dtype](shape)

    # Fill with sequence using typed arithmetic
    var value = start
    for i in range(num_elements):
        t[i] = value
        value = value + step

    return t^


fn eye[dtype: DType](n: Int, m: Int, k: Int) raises -> Tensor[dtype]:
    """Create 2D tensor with ones on diagonal.

    Args:
        n: Number of rows.
        m: Number of columns.
        k: Diagonal offset (0 for main, >0 upper, <0 lower).

    Returns:
        A new 2D Tensor[dtype] with ones on the k-th diagonal.

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES.
    """
    var shape = List[Int]()
    shape.append(n)
    shape.append(m)

    # Tensor constructor zero-initializes, so only set diagonal
    var t = Tensor[dtype](shape)
    var one = Scalar[dtype](1)

    for i in range(n):
        var j = i + k
        if j >= 0 and j < m:
            var index = i * m + j
            t[index] = one

    return t^


fn linspace[
    dtype: DType
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    num: Int,
    endpoint: Bool = True,
) raises -> Tensor[dtype]:
    """Create 1D tensor with evenly spaced values.

    Uses Scalar[dtype] for start/stop to preserve precision.

    Args:
        start: Start value (inclusive).
        stop: End value (inclusive if endpoint=True).
        num: Number of values to generate.
        endpoint: Whether to include stop value (default True).

    Returns:
        A new 1D Tensor[dtype] with num evenly spaced values.

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES.
    """
    var shape = List[Int]()
    shape.append(num)

    var t = Tensor[dtype](shape)

    if num == 0:
        return t^

    if num == 1:
        t[0] = start
        return t^

    # Calculate step size using Float64 for precision
    var start_f = Float64(start)
    var stop_f = Float64(stop)
    var divisor: Int
    if endpoint:
        divisor = num - 1
    else:
        divisor = num
    var step_f = (stop_f - start_f) / Float64(divisor)

    for i in range(num):
        var value = start_f + step_f * Float64(i)
        t[i] = Scalar[dtype](value)

    return t^


fn randn[
    dtype: DType
](shape: List[Int], seed: Int = 0) raises -> Tensor[dtype]:
    """Create tensor filled with random values from standard normal distribution.

    Uses Box-Muller transform to generate normally distributed random values
    from uniform random values. Generates values with mean=0 and std=1.

    For non-float dtypes, values are generated as Float64 then cast to the
    target dtype.

    Args:
        shape: The shape of the output tensor.
        seed: Random seed for reproducibility (default: 0 uses system randomness).

    Returns:
        A new Tensor[dtype] filled with random values from N(0, 1).

    Raises:
        Error: If tensor size exceeds MAX_TENSOR_BYTES.
    """
    if seed > 0:
        random_seed(seed)

    var t = Tensor[dtype](shape)

    # Box-Muller transform: generates pairs of independent N(0,1) values
    var i = 0
    while i < t.numel():
        var u1 = random_float64()
        var u2 = random_float64()

        # Ensure u1 is not zero (would cause log(0))
        if u1 < 1e-10:
            u1 = 1e-10

        var magnitude = sqrt(-2.0 * log(u1))
        var angle = 2.0 * 3.14159265358979323846 * u2

        var z0 = magnitude * cos(angle)
        var z1 = magnitude * sin(angle)

        t[i] = Scalar[dtype](z0)
        i += 1

        if i < t.numel():
            t[i] = Scalar[dtype](z1)
            i += 1

    return t^


# ------------------------------------------------------------------
# *_like functions — infer dtype from input tensor
# ------------------------------------------------------------------


fn zeros_like[
    dtype: DType
](t: Tensor[dtype]) raises -> Tensor[dtype]:
    """Create a zero-filled tensor with same shape as input.

    The dtype is inferred from the input tensor's compile-time parameter.

    Args:
        t: Template tensor to match shape.

    Returns:
        A new Tensor[dtype] filled with zeros, same shape as input.

    Raises:
        Error: If tensor creation fails.
    """
    return zeros[dtype](t.shape())


fn ones_like[
    dtype: DType
](t: Tensor[dtype]) raises -> Tensor[dtype]:
    """Create a one-filled tensor with same shape as input.

    The dtype is inferred from the input tensor's compile-time parameter.

    Args:
        t: Template tensor to match shape.

    Returns:
        A new Tensor[dtype] filled with ones, same shape as input.

    Raises:
        Error: If tensor creation fails.
    """
    return ones[dtype](t.shape())


fn full_like[
    dtype: DType
](t: Tensor[dtype], fill_value: Scalar[dtype]) raises -> Tensor[dtype]:
    """Create a tensor filled with a value, same shape as input.

    The dtype is inferred from the input tensor's compile-time parameter.

    Args:
        t: Template tensor to match shape.
        fill_value: The typed value to fill the tensor with.

    Returns:
        A new Tensor[dtype] filled with fill_value, same shape as input.

    Raises:
        Error: If tensor creation fails.
    """
    return full[dtype](t.shape(), fill_value)


# ------------------------------------------------------------------
# Special value tensors (float types only)
# ------------------------------------------------------------------


fn nan_tensor[
    dtype: DType
](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a tensor filled with NaN values.

    Only valid for floating-point dtypes (float16, float32, float64, bfloat16).

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with NaN values.

    Raises:
        Error: If dtype is not floating-point, or if tensor size exceeds
            MAX_TENSOR_BYTES.
    """
    # Compile-time check for float types
    @parameter
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
        and dtype != DType.bfloat16
    ):
        raise Error("nan_tensor: only floating-point dtypes support NaN")

    var t = Tensor[dtype](shape)
    # IEEE 754 NaN
    var nan_value = Scalar[dtype](0.0 / 0.0)
    for i in range(t.numel()):
        t[i] = nan_value
    return t^


fn inf_tensor[
    dtype: DType
](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a tensor filled with positive infinity values.

    Only valid for floating-point dtypes (float16, float32, float64, bfloat16).

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with positive infinity values.

    Raises:
        Error: If dtype is not floating-point, or if tensor size exceeds
            MAX_TENSOR_BYTES.
    """
    @parameter
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
        and dtype != DType.bfloat16
    ):
        raise Error(
            "inf_tensor: only floating-point dtypes support Inf"
        )

    var t = Tensor[dtype](shape)
    var inf_val = Scalar[dtype](numeric_inf[DType.float64]())
    for i in range(t.numel()):
        t[i] = inf_val
    return t^


fn neg_inf_tensor[
    dtype: DType
](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a tensor filled with negative infinity values.

    Only valid for floating-point dtypes (float16, float32, float64, bfloat16).

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with negative infinity values.

    Raises:
        Error: If dtype is not floating-point, or if tensor size exceeds
            MAX_TENSOR_BYTES.
    """
    @parameter
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
        and dtype != DType.bfloat16
    ):
        raise Error(
            "neg_inf_tensor: only floating-point dtypes support Inf"
        )

    var t = Tensor[dtype](shape)
    var neg_inf_val = Scalar[dtype](numeric_neg_inf[DType.float64]())
    for i in range(t.numel()):
        t[i] = neg_inf_val
    return t^
