"""Tensor package — parametric and type-erased tensor types.

Provides:
- Tensor[dtype: DType]: Compile-time typed tensor with SIMD-like element access
- TensorLike: Shared trait interface for all tensor types
"""

from shared.tensor.tensor_traits import TensorLike
from shared.tensor.tensor import Tensor
