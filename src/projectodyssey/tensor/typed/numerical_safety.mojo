"""Typed Tensor[dtype] numerical safety cores.

Internal module -- not part of the public API.

SIMD-vectorized implementations for hot-path NaN/Inf detection.
has_nan/has_inf use manual SIMD loops with early exit.
count_nan/count_inf use vectorize[] with lane reduction.
"""

from std.math import isnan, isinf, sqrt, nan
from std.sys import simd_width_of
from std.algorithm import vectorize
from odyssey.tensor.tensor import Tensor


def _has_nan_core[dtype: DType](tensor: Tensor[dtype]) -> Bool:
    """Check if typed tensor contains any NaN values (SIMD-vectorized).

    Uses manual SIMD loop with early exit for maximum throughput.
    Falls back to scalar loop for tail elements.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor to check.

    Returns:
        True if any element is NaN, False otherwise.
    """
    # Integer/unsigned types cannot have NaN
    comptime if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return False

    var size = tensor.numel()
    var ptr = tensor._data
    comptime simd_w = simd_width_of[dtype]()

    # SIMD loop with early exit
    var i = 0
    while i + simd_w <= size:
        var vec = ptr.load[width=simd_w](i)
        if isnan(vec).reduce_or():
            return True
        i += simd_w

    # Scalar tail
    while i < size:
        if isnan(ptr[i]):
            return True
        i += 1
    return False


def _has_inf_core[dtype: DType](tensor: Tensor[dtype]) -> Bool:
    """Check if typed tensor contains any Inf values (SIMD-vectorized).

    Uses manual SIMD loop with early exit for maximum throughput.
    Falls back to scalar loop for tail elements.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor to check.

    Returns:
        True if any element is Inf or -Inf, False otherwise.
    """
    comptime if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return False

    var size = tensor.numel()
    var ptr = tensor._data
    comptime simd_w = simd_width_of[dtype]()

    # SIMD loop with early exit
    var i = 0
    while i + simd_w <= size:
        var vec = ptr.load[width=simd_w](i)
        if isinf(vec).reduce_or():
            return True
        i += simd_w

    # Scalar tail
    while i < size:
        if isinf(ptr[i]):
            return True
        i += 1
    return False


def _count_nan_core[dtype: DType](tensor: Tensor[dtype]) -> Int:
    """Count NaN values in typed tensor (SIMD-vectorized).

    Uses vectorize[] with SIMD lane reduction for parallel counting.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Number of NaN elements.
    """
    comptime if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return 0

    var size = tensor.numel()
    var count = 0
    var ptr = tensor._data
    comptime simd_w = simd_width_of[dtype]()

    @always_inline
    def _count[width: Int](idx: Int) {var ptr, mut count}:
        var vec = ptr.load[width=width](idx)
        count += Int(isnan(vec).cast[DType.uint8]().reduce_add())

    vectorize[simd_w](size, _count)
    return count


def _count_inf_core[dtype: DType](tensor: Tensor[dtype]) -> Int:
    """Count Inf values in typed tensor (SIMD-vectorized).

    Uses vectorize[] with SIMD lane reduction for parallel counting.

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Number of Inf/-Inf elements.
    """
    comptime if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return 0

    var size = tensor.numel()
    var count = 0
    var ptr = tensor._data
    comptime simd_w = simd_width_of[dtype]()

    @always_inline
    def _count[width: Int](idx: Int) {var ptr, mut count}:
        var vec = ptr.load[width=width](idx)
        count += Int(isinf(vec).cast[DType.uint8]().reduce_add())

    vectorize[simd_w](size, _count)
    return count


def _tensor_min_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Find minimum value in typed tensor (SIMD-vectorized).

    Uses manual SIMD loop with reduce_min for parallel computation.
    Falls back to scalar loop for tail elements.

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
    comptime simd_w = simd_width_of[dtype]()

    # SIMD loop
    var i = 0
    while i + simd_w <= size:
        var vec = ptr.load[width=simd_w](i)
        var vec_min = Float64(vec.reduce_min())
        if vec_min < min_val:
            min_val = vec_min
        i += simd_w

    # Scalar tail
    while i < size:
        var val = Float64(ptr[i])
        if val < min_val:
            min_val = val
        i += 1
    return min_val


def _tensor_max_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Find maximum value in typed tensor (SIMD-vectorized).

    Uses manual SIMD loop with reduce_max for parallel computation.
    Falls back to scalar loop for tail elements.

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
    comptime simd_w = simd_width_of[dtype]()

    # SIMD loop
    var i = 0
    while i + simd_w <= size:
        var vec = ptr.load[width=simd_w](i)
        var vec_max = Float64(vec.reduce_max())
        if vec_max > max_val:
            max_val = vec_max
        i += simd_w

    # Scalar tail
    while i < size:
        var val = Float64(ptr[i])
        if val > max_val:
            max_val = val
        i += 1
    return max_val


def _compute_l2_norm_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Compute L2 norm of typed tensor (SIMD-vectorized).

    Uses manual SIMD loop with reduce_add for parallel squaring and summation.
    Falls back to scalar loop for tail elements.

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
    comptime simd_w = simd_width_of[dtype]()

    # SIMD loop
    var i = 0
    while i + simd_w <= size:
        var vec = ptr.load[width=simd_w](i)
        var squared = vec * vec
        sum_sq += Float64(squared.reduce_add())
        i += simd_w

    # Scalar tail
    while i < size:
        var val = Float64(ptr[i])
        sum_sq += val * val
        i += 1
    return sqrt(sum_sq)
