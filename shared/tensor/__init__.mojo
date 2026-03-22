"""Tensor package — parametric and type-erased tensor types.

Provides:
- Tensor[dtype: DType]: Compile-time typed tensor with SIMD-like element access
- TensorLike: Shared trait interface for all tensor types
- Factory functions: zeros, ones, full, empty, arange, eye, linspace, randn,
  zeros_like, ones_like, full_like, nan_tensor, inf_tensor, neg_inf_tensor
"""

from shared.tensor.tensor_traits import TensorLike
from shared.tensor.tensor import Tensor
from shared.tensor.factories import (
    zeros,
    ones,
    full,
    empty,
    arange,
    eye,
    linspace,
    randn,
    zeros_like,
    ones_like,
    full_like,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
)
