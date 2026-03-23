"""Typed Tensor[dtype] SIMD arithmetic dispatch cores.

Internal module -- not part of the public API.
"""

from algorithm import vectorize
from sys.info import simd_width_of
from shared.tensor.tensor import Tensor
from shared.core.any_tensor import AnyTensor


@always_inline
fn _add_simd_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """SIMD addition for typed Tensor[dtype]. Zero bitcasts."""
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
fn _subtract_simd_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """SIMD subtraction for typed Tensor[dtype]. Zero bitcasts."""
    var result = Tensor[dtype](a.shape())
    var size = a.numel()
    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_subtract[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec - b_vec)

        vectorize[simd_width](size, vectorized_subtract)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] - b_ptr[i]

    return result^


@always_inline
fn _multiply_simd_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """SIMD multiplication for typed Tensor[dtype]. Zero bitcasts."""
    var result = Tensor[dtype](a.shape())
    var size = a.numel()
    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_multiply[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec * b_vec)

        vectorize[simd_width](size, vectorized_multiply)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] * b_ptr[i]

    return result^


@always_inline
fn _divide_simd_typed[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """SIMD division for typed Tensor[dtype]. Zero bitcasts."""
    var result = Tensor[dtype](a.shape())
    var size = a.numel()
    var a_ptr = a._data
    var b_ptr = b._data
    var result_ptr = result._data

    @parameter
    if dtype == DType.float32 or dtype == DType.float64:
        comptime simd_width = simd_width_of[dtype]()

        @parameter
        fn vectorized_divide[width: Int](idx: Int) unified {mut}:
            var a_vec = a_ptr.load[width=width](idx)
            var b_vec = b_ptr.load[width=width](idx)
            result_ptr.store[width=width](idx, a_vec / b_vec)

        vectorize[simd_width](size, vectorized_divide)
    else:
        for i in range(size):
            result_ptr[i] = a_ptr[i] / b_ptr[i]

    return result^
