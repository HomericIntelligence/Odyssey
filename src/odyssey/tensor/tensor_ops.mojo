"""Tensor broadcasting and element-wise operation helpers for AnyTensor.

Extracted from any_tensor.mojo per issue #5182 (SRP reduction).
Cross-module private-field access (_data, _dtype, _shape, _numel) is valid
in Mojo (package-scoped privacy). See tensor_io.mojo for precedent.
"""

from std.collections import List
from odyssey.base.broadcasting import (
    broadcast_shapes,
    compute_broadcast_strides,
    are_shapes_broadcastable,
)
from odyssey.base.dtype_ordinal import (
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
from .any_tensor import AnyTensor


def broadcast_to_impl(
    tensor: AnyTensor, target_shape: List[Int]
) raises -> AnyTensor:
    """Broadcast tensor to target shape (extracted from AnyTensor.broadcast_to per #5182).
    """
    # Inline broadcast_to to avoid circular import via odyssey.core.shape.
    # Uses module-level are_shapes_broadcastable and compute_broadcast_strides.
    # See Issue #4513.
    var shape = tensor.shape()

    if len(target_shape) < len(shape):
        raise Error("broadcast_to: cannot broadcast to fewer dimensions")

    if not are_shapes_broadcastable(shape, target_shape):
        raise Error("broadcast_to: shapes are not broadcast-compatible")

    var broadcast_strides = compute_broadcast_strides(shape, target_shape)
    var result = AnyTensor(target_shape, tensor.dtype())
    var result_numel = result.numel()

    for i in range(result_numel):
        var coords = List[Int]()
        var temp_i = i
        for j in range(len(target_shape)):
            var stride = 1
            for k in range(j + 1, len(target_shape)):
                stride *= target_shape[k]
            var coord = temp_i // stride
            coords.append(coord)
            temp_i = temp_i % stride

        var src_idx = 0
        for j in range(len(target_shape)):
            src_idx += coords[j] * broadcast_strides[j]

        var val = tensor._get_float64(src_idx)
        result._set_float64(i, val)

    return result^


# ============================================================================
# Private Broadcasting Helpers
# ============================================================================
# Binary, unary, and comparison helpers implement element-wise operations with
# NumPy-style broadcasting for use by AnyTensor's operator overloads.
# Defined here (rather than in arithmetic/comparison modules) to break circular
# import chains. Both files are now siblings in src/odyssey/tensor/.
# See Issue #4513.


def _anytensor_binary_op[
    op: def[T: DType](Scalar[T], Scalar[T]) thin -> Scalar[T]
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Apply a compile-time-typed binary arithmetic op with broadcasting."""
    if a._dtype != b._dtype:
        raise Error("Cannot operate on tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var result = AnyTensor(result_shape, a._dtype)
    var ordinal = dtype_to_ordinal(a._dtype)

    @parameter
    def _apply[dtype: DType]():
        var a_ptr = a._data.bitcast[Scalar[dtype]]()
        var b_ptr = b._data.bitcast[Scalar[dtype]]()
        var r_ptr = result._data.bitcast[Scalar[dtype]]()
        var result_strides = List[Int]()
        var s = 1
        for i in range(len(result_shape) - 1, -1, -1):
            result_strides.append(s)
            s *= result_shape[i]
        var result_strides_final = List[Int]()
        for i in range(len(result_strides) - 1, -1, -1):
            result_strides_final.append(result_strides[i])
        for result_idx in range(total_elems):
            var idx_a = 0
            var idx_b = 0
            var remaining = result_idx
            for d in range(len(result_shape) - 1, -1, -1):
                var coord = remaining % result_shape[d]
                remaining //= result_shape[d]
                idx_a += coord * strides_a[d]
                idx_b += coord * strides_b[d]
            r_ptr[result_idx] = op[dtype](a_ptr[idx_a], b_ptr[idx_b])

    if ordinal == DTYPE_FLOAT16:
        _apply[DType.float16]()
    elif ordinal == DTYPE_FLOAT32:
        _apply[DType.float32]()
    elif ordinal == DTYPE_FLOAT64:
        _apply[DType.float64]()
    elif ordinal == DTYPE_INT8:
        _apply[DType.int8]()
    elif ordinal == DTYPE_INT16:
        _apply[DType.int16]()
    elif ordinal == DTYPE_INT32:
        _apply[DType.int32]()
    elif ordinal == DTYPE_INT64:
        _apply[DType.int64]()
    elif ordinal == DTYPE_UINT8:
        _apply[DType.uint8]()
    elif ordinal == DTYPE_UINT16:
        _apply[DType.uint16]()
    elif ordinal == DTYPE_UINT32:
        _apply[DType.uint32]()
    elif ordinal == DTYPE_UINT64:
        _apply[DType.uint64]()

    return result^


def _anytensor_unary_op[
    op: def[T: DType](Scalar[T]) thin -> Scalar[T]
](tensor: AnyTensor) raises -> AnyTensor:
    """Apply a compile-time-typed unary op element-wise."""
    var shape = tensor.shape()
    var result = AnyTensor(shape, tensor._dtype)
    var ordinal = dtype_to_ordinal(tensor._dtype)

    @parameter
    def _apply[dtype: DType]():
        var src_ptr = tensor._data.bitcast[Scalar[dtype]]()
        var dst_ptr = result._data.bitcast[Scalar[dtype]]()
        for i in range(tensor._numel):
            dst_ptr[i] = op[dtype](src_ptr[i])

    if ordinal == DTYPE_FLOAT16:
        _apply[DType.float16]()
    elif ordinal == DTYPE_FLOAT32:
        _apply[DType.float32]()
    elif ordinal == DTYPE_FLOAT64:
        _apply[DType.float64]()
    elif ordinal == DTYPE_INT8:
        _apply[DType.int8]()
    elif ordinal == DTYPE_INT16:
        _apply[DType.int16]()
    elif ordinal == DTYPE_INT32:
        _apply[DType.int32]()
    elif ordinal == DTYPE_INT64:
        _apply[DType.int64]()
    elif ordinal == DTYPE_UINT8:
        _apply[DType.uint8]()
    elif ordinal == DTYPE_UINT16:
        _apply[DType.uint16]()
    elif ordinal == DTYPE_UINT32:
        _apply[DType.uint32]()
    elif ordinal == DTYPE_UINT64:
        _apply[DType.uint64]()

    return result^


def _anytensor_compare_op[
    op: def[T: DType](Scalar[T], Scalar[T]) thin -> Bool
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Apply a compile-time-typed binary comparison op with broadcasting."""
    if a._dtype != b._dtype:
        raise Error("Cannot compare tensors with different dtypes")

    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    var result = AnyTensor(result_shape, DType.bool)
    var ordinal = dtype_to_ordinal(a._dtype)

    @parameter
    def _apply[dtype: DType]():
        var a_ptr = a._data.bitcast[Scalar[dtype]]()
        var b_ptr = b._data.bitcast[Scalar[dtype]]()
        var r_ptr = result._data.bitcast[Scalar[DType.bool]]()
        var result_strides = List[Int]()
        var s = 1
        for i in range(len(result_shape) - 1, -1, -1):
            result_strides.append(s)
            s *= result_shape[i]
        var result_strides_final = List[Int]()
        for i in range(len(result_strides) - 1, -1, -1):
            result_strides_final.append(result_strides[i])
        for result_idx in range(total_elems):
            var idx_a = 0
            var idx_b = 0
            var remaining = result_idx
            for d in range(len(result_shape) - 1, -1, -1):
                var coord = remaining % result_shape[d]
                remaining //= result_shape[d]
                idx_a += coord * strides_a[d]
                idx_b += coord * strides_b[d]
            r_ptr[result_idx] = op[dtype](a_ptr[idx_a], b_ptr[idx_b])

    if ordinal == DTYPE_FLOAT16:
        _apply[DType.float16]()
    elif ordinal == DTYPE_FLOAT32:
        _apply[DType.float32]()
    elif ordinal == DTYPE_FLOAT64:
        _apply[DType.float64]()
    elif ordinal == DTYPE_INT8:
        _apply[DType.int8]()
    elif ordinal == DTYPE_INT16:
        _apply[DType.int16]()
    elif ordinal == DTYPE_INT32:
        _apply[DType.int32]()
    elif ordinal == DTYPE_INT64:
        _apply[DType.int64]()
    elif ordinal == DTYPE_UINT8:
        _apply[DType.uint8]()
    elif ordinal == DTYPE_UINT16:
        _apply[DType.uint16]()
    elif ordinal == DTYPE_UINT32:
        _apply[DType.uint32]()
    elif ordinal == DTYPE_UINT64:
        _apply[DType.uint64]()

    return result^


def _anytensor_matmul(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Basic matrix multiplication (2D x 2D) for AnyTensor.__matmul__.

    Note: For full matmul with batching and contiguity handling, use
    odyssey.core.matrix.matmul. This implementation handles the common 2D case
    to avoid the circular import: any_tensor <- matrix <- shape <- any_tensor.
    """
    var a_ndim = len(a._shape)
    var b_ndim = len(b._shape)

    # 2D x 2D: (m, k) @ (k, n) -> (m, n)
    if a_ndim == 2 and b_ndim == 2:
        var m = a._shape[0]
        var k = a._shape[1]
        var n = b._shape[1]
        if k != b._shape[0]:
            raise Error(
                "matmul: incompatible dimensions "
                + String(k)
                + " vs "
                + String(b._shape[0])
            )
        if a._dtype != b._dtype:
            raise Error("matmul: tensors must have the same dtype")
        var result = AnyTensor([m, n], a._dtype)
        var ordinal = dtype_to_ordinal(a._dtype)

        @parameter
        def _mm[dtype: DType]():
            var a_ptr = a._data.bitcast[Scalar[dtype]]()
            var b_ptr = b._data.bitcast[Scalar[dtype]]()
            var r_ptr = result._data.bitcast[Scalar[dtype]]()
            for i in range(m):
                for j in range(n):
                    var acc = Scalar[dtype](0)
                    for p in range(k):
                        acc += a_ptr[i * k + p] * b_ptr[p * n + j]
                    r_ptr[i * n + j] = acc

        if ordinal == DTYPE_FLOAT16:
            _mm[DType.float16]()
        elif ordinal == DTYPE_FLOAT32:
            _mm[DType.float32]()
        elif ordinal == DTYPE_FLOAT64:
            _mm[DType.float64]()
        elif ordinal == DTYPE_INT8:
            _mm[DType.int8]()
        elif ordinal == DTYPE_INT16:
            _mm[DType.int16]()
        elif ordinal == DTYPE_INT32:
            _mm[DType.int32]()
        elif ordinal == DTYPE_INT64:
            _mm[DType.int64]()
        else:
            raise Error("matmul: unsupported dtype")
        return result^

    # 1D x 2D or 2D x 1D: delegate to the local arithmetic for now
    # by raising a helpful error pointing to matrix.matmul
    raise Error(
        "AnyTensor.__matmul__ only supports 2D x 2D. "
        "For 1D/batched matmul use odyssey.core.matrix.matmul directly."
    )
