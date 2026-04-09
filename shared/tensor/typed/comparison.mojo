"""Typed Tensor[dtype] comparison dispatch cores.

Internal module -- not part of the public API.
"""

from std.collections import List
from shared.tensor.tensor import Tensor
from shared.tensor.any_tensor import AnyTensor
from shared.base.broadcasting import broadcast_shapes, compute_broadcast_strides
from shared.base.dtype_ordinal import (
    dtype_to_ordinal,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
)


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] Comparison Implementations
# ============================================================================


def _equal_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise equality on native Tensor[dtype] (core)."""
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] == b_ptr[idx_b]

    return result^


def _not_equal_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise inequality on native Tensor[dtype] (core)."""
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] != b_ptr[idx_b]

    return result^


def _less_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise less-than on native Tensor[dtype] (core)."""
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] < b_ptr[idx_b]

    return result^


def _less_equal_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise less-equal on native Tensor[dtype] (core)."""
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] <= b_ptr[idx_b]

    return result^


def _greater_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise greater-than on native Tensor[dtype] (core)."""
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] > b_ptr[idx_b]

    return result^


def _greater_equal_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[DType.bool]:
    """Element-wise greater-equal on native Tensor[dtype] (core)."""
    var a_shape = a.shape()
    var b_shape = b.shape()
    var result_shape = broadcast_shapes(a_shape, b_shape)
    var result = Tensor[DType.bool](result_shape)

    var strides_a = compute_broadcast_strides(a_shape, result_shape)
    var strides_b = compute_broadcast_strides(b_shape, result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var a_ptr = a._data
    var b_ptr = b._data
    var out_ptr = result._data

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        out_ptr[result_idx] = a_ptr[idx_a] >= b_ptr[idx_b]

    return result^


# ============================================================================
# Layer 2: AnyTensor dispatch helpers (as_tensor -> typed core -> as_any)
# ============================================================================


def _equal_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _equal_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


def _not_equal_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _not_equal_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


def _less_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _less_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


def _less_equal_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _less_equal_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


def _greater_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _greater_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


def _greater_equal_dispatch[
    dtype: DType
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return _greater_equal_typed[dtype](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()
