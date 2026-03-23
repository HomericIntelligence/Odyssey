"""Typed Tensor[dtype] SIMD activation dispatch cores.

Internal module -- not part of the public API.
"""

from sys import simd_width_of
from algorithm import vectorize
from shared.tensor.tensor import Tensor


# ============================================================================
# Typed Tensor[dtype] SIMD Overloads
# ============================================================================
# These accept Tensor[dtype] directly, avoiding AnyTensor bitcasts.
# Tensor[dtype]._data is already UnsafePointer[Scalar[dtype]].


fn _relu_simd_typed[dt: DType](input: Tensor[dt], mut result: Tensor[dt]):
    """SIMD ReLU for native Tensor[dtype] -- zero bitcasts."""
    comptime simd_width = simd_width_of[dt]()
    var size = input.numel()
    var in_ptr = input._data
    var out_ptr = result._data

    @parameter
    fn vectorized_relu[width: Int](idx: Int) unified {mut}:
        var vec = in_ptr.load[width=width](idx)
        var zero_vec = SIMD[dt, width](0)
        out_ptr.store[width=width](idx, max(zero_vec, vec))

    vectorize[simd_width](size, vectorized_relu)
