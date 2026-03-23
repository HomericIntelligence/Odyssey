"""Typed Tensor[dtype] numerical safety cores.

Internal module -- not part of the public API.
"""

from math import isnan, isinf, sqrt
from shared.tensor.tensor import Tensor

fn _has_nan_core[dtype: DType](tensor: Tensor[dtype]) -> Bool:
    """Check if typed tensor contains any NaN values (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor to check.

    Returns:
        True if any element is NaN, False otherwise.
    """
    # Integer/unsigned types cannot have NaN
    @parameter
    if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return False

    var size = tensor.numel()
    var ptr = tensor._data
    for i in range(size):
        @parameter
        if dtype == DType.float16:
            if isnan(Float32(ptr[i])):
                return True
        else:
            if isnan(ptr[i]):
                return True
    return False



fn _has_inf_core[dtype: DType](tensor: Tensor[dtype]) -> Bool:
    """Check if typed tensor contains any Inf values (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor to check.

    Returns:
        True if any element is Inf or -Inf, False otherwise.
    """
    @parameter
    if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return False

    var size = tensor.numel()
    var ptr = tensor._data
    for i in range(size):
        @parameter
        if dtype == DType.float16:
            if isinf(Float32(ptr[i])):
                return True
        else:
            if isinf(ptr[i]):
                return True
    return False



fn _count_nan_core[dtype: DType](tensor: Tensor[dtype]) -> Int:
    """Count NaN values in typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Number of NaN elements.
    """
    @parameter
    if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return 0

    var size = tensor.numel()
    var count = 0
    var ptr = tensor._data
    for i in range(size):
        @parameter
        if dtype == DType.float16:
            if isnan(Float32(ptr[i])):
                count += 1
        else:
            if isnan(ptr[i]):
                count += 1
    return count



fn _count_inf_core[dtype: DType](tensor: Tensor[dtype]) -> Int:
    """Count Inf values in typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Number of Inf/-Inf elements.
    """
    @parameter
    if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return 0

    var size = tensor.numel()
    var count = 0
    var ptr = tensor._data
    for i in range(size):
        @parameter
        if dtype == DType.float16:
            if isinf(Float32(ptr[i])):
                count += 1
        else:
            if isinf(ptr[i]):
                count += 1
    return count



fn _tensor_min_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Find minimum value in typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Minimum value as Float64.
    """
    var size = tensor.numel()
    if size == 0:
        return 0.0

    var min_val = Float64(1e308)
    var ptr = tensor._data
    for i in range(size):
        var val = Float64(ptr[i])
        if val < min_val:
            min_val = val
    return min_val



fn _tensor_max_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Find maximum value in typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Maximum value as Float64.
    """
    var size = tensor.numel()
    if size == 0:
        return 0.0

    var max_val = Float64(-1e308)
    var ptr = tensor._data
    for i in range(size):
        var val = Float64(ptr[i])
        if val > max_val:
            max_val = val
    return max_val



fn _compute_l2_norm_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Compute L2 norm of typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        L2 norm as Float64.
    """
    var size = tensor.numel()
    var sum_sq = Float64(0.0)
    var ptr = tensor._data
    for i in range(size):
        var val = Float64(ptr[i])
        sum_sq += val * val
    return sqrt(sum_sq)
