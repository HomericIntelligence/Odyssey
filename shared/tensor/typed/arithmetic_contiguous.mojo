"""Typed Tensor[dtype] contiguous arithmetic dispatch cores.

Internal module -- not part of the public API.
"""

from algorithm import vectorize
from sys.info import simd_width_of
from shared.tensor.tensor import Tensor
from shared.core.any_tensor import AnyTensor


@always_inline
fn _add_contiguous_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Optimized addition for contiguous same-shape Tensor[dtype]."""
    var result = Tensor[dtype](a.shape())
    var size = a.numel()
    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_add[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec + b_vec)

        vectorize[simd_width](size, vectorized_add)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] + b_ptr[i]

    return result^


@always_inline
fn _subtract_contiguous_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Optimized subtraction for contiguous same-shape Tensor[dtype]."""
    var result = Tensor[dtype](a.shape())
    var size = a.numel()
    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_sub[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec - b_vec)

        vectorize[simd_width](size, vectorized_sub)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] - b_ptr[i]

    return result^


@always_inline
fn _multiply_contiguous_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Optimized multiplication for contiguous same-shape Tensor[dtype]."""
    var result = Tensor[dtype](a.shape())
    var size = a.numel()
    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_mul[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec * b_vec)

        vectorize[simd_width](size, vectorized_mul)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] * b_ptr[i]

    return result^


@always_inline
fn _divide_contiguous_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Optimized division for contiguous same-shape Tensor[dtype]."""
    var result = Tensor[dtype](a.shape())
    var size = a.numel()
    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_div[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec / b_vec)

        vectorize[simd_width](size, vectorized_div)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] / b_ptr[i]

    return result^
