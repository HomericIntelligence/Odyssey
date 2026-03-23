"""Typed factory functions for Tensor[dtype].

Provides compile-time typed factory functions that return Tensor[dtype] instead
of runtime-typed AnyTensor. Each function delegates to the corresponding
AnyTensor factory and converts via as_tensor[dtype]().

Factory functions:
- zeros, ones, full, empty: Shape-based creation
- arange, eye, linspace: Sequence/pattern creation
- randn: Random normal distribution
- zeros_like, ones_like, full_like: Clone shape from existing tensor
- nan_tensor, inf_tensor, neg_inf_tensor: Special value tensors
"""

from collections import List
from .tensor import Tensor
from .any_tensor import (
    zeros as _zeros,
    ones as _ones,
    full as _full,
    empty as _empty,
    arange as _arange,
    eye as _eye,
    linspace as _linspace,
    randn as _randn,
    zeros_like as _zeros_like,
    ones_like as _ones_like,
    full_like as _full_like,
    nan_tensor as _nan_tensor,
    inf_tensor as _inf_tensor,
    neg_inf_tensor as _neg_inf_tensor,
)


fn zeros[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a zero-filled Tensor[dtype].

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with zeros.
    """
    return _zeros(shape, dtype).as_tensor[dtype]()


fn ones[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a one-filled Tensor[dtype].

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with ones.
    """
    return _ones(shape, dtype).as_tensor[dtype]()


fn full[dtype: DType](
    shape: List[Int], fill_value: Float64
) raises -> Tensor[dtype]:
    """Create a Tensor[dtype] filled with a constant value.

    Args:
        shape: The shape of the output tensor.
        fill_value: The value to fill the tensor with.

    Returns:
        A new Tensor[dtype] filled with fill_value.
    """
    return _full(shape, fill_value, dtype).as_tensor[dtype]()


fn empty[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create an uninitialized Tensor[dtype].

    Warning: The tensor contains uninitialized memory. Values are undefined
    until written.

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] with uninitialized values.
    """
    return _empty(shape, dtype).as_tensor[dtype]()


fn arange[dtype: DType](
    start: Float64, stop: Float64, step: Float64
) raises -> Tensor[dtype]:
    """Create a 1D Tensor[dtype] with evenly spaced values.

    Args:
        start: Start value (inclusive).
        stop: End value (exclusive).
        step: Step size between values.

    Returns:
        A new 1D Tensor[dtype] with values [start, start+step, ...).
    """
    return _arange(start, stop, step, dtype).as_tensor[dtype]()


fn eye[dtype: DType](n: Int, m: Int, k: Int = 0) raises -> Tensor[dtype]:
    """Create a 2D Tensor[dtype] with ones on the diagonal.

    Args:
        n: Number of rows.
        m: Number of columns.
        k: Diagonal offset (0=main, positive=above, negative=below).

    Returns:
        A new 2D Tensor[dtype] with ones on the k-th diagonal.
    """
    return _eye(n, m, k, dtype).as_tensor[dtype]()


fn linspace[dtype: DType](
    start: Float64, stop: Float64, num: Int
) raises -> Tensor[dtype]:
    """Create a 1D Tensor[dtype] with evenly spaced values (inclusive).

    Args:
        start: Start value (inclusive).
        stop: End value (inclusive).
        num: Number of values.

    Returns:
        A new 1D Tensor[dtype] with num evenly spaced values.
    """
    return _linspace(start, stop, num, dtype).as_tensor[dtype]()


fn randn[dtype: DType](
    shape: List[Int], seed: Int = 0
) raises -> Tensor[dtype]:
    """Create a Tensor[dtype] filled with random normal values.

    Uses Box-Muller transform for normally distributed random values
    with mean=0 and std=1.

    Args:
        shape: The shape of the output tensor.
        seed: Random seed for reproducibility (default: 0 uses system random).

    Returns:
        A new Tensor[dtype] filled with random values from N(0, 1).
    """
    return _randn(shape, dtype, seed).as_tensor[dtype]()


fn zeros_like[dtype: DType](tensor: Tensor[dtype]) raises -> Tensor[dtype]:
    """Create a zero-filled tensor with the same shape as the input.

    Args:
        tensor: The tensor whose shape to copy.

    Returns:
        A new Tensor[dtype] filled with zeros, same shape as input.
    """
    return _zeros_like(tensor.as_any()).as_tensor[dtype]()


fn ones_like[dtype: DType](tensor: Tensor[dtype]) raises -> Tensor[dtype]:
    """Create a one-filled tensor with the same shape as the input.

    Args:
        tensor: The tensor whose shape to copy.

    Returns:
        A new Tensor[dtype] filled with ones, same shape as input.
    """
    return _ones_like(tensor.as_any()).as_tensor[dtype]()


fn full_like[dtype: DType](
    tensor: Tensor[dtype], fill_value: Float64
) raises -> Tensor[dtype]:
    """Create a constant-filled tensor with the same shape as the input.

    Args:
        tensor: The tensor whose shape to copy.
        fill_value: The value to fill the tensor with.

    Returns:
        A new Tensor[dtype] filled with fill_value, same shape as input.
    """
    return _full_like(tensor.as_any(), fill_value).as_tensor[dtype]()


fn nan_tensor[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a Tensor[dtype] filled with NaN values.

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with NaN values.

    Raises:
        Error: If dtype is not floating-point.
    """
    return _nan_tensor(shape, dtype).as_tensor[dtype]()


fn inf_tensor[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a Tensor[dtype] filled with positive infinity.

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with +inf.

    Raises:
        Error: If dtype is not floating-point.
    """
    return _inf_tensor(shape, dtype).as_tensor[dtype]()


fn neg_inf_tensor[dtype: DType](shape: List[Int]) raises -> Tensor[dtype]:
    """Create a Tensor[dtype] filled with negative infinity.

    Args:
        shape: The shape of the output tensor.

    Returns:
        A new Tensor[dtype] filled with -inf.

    Raises:
        Error: If dtype is not floating-point.
    """
    return _neg_inf_tensor(shape, dtype).as_tensor[dtype]()
