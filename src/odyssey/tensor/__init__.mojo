"""Tensor package — parametric and type-erased tensor types.

Provides:
- Tensor[dtype: DType]: Compile-time typed tensor with SIMD-like element access
- AnyTensor: Runtime-typed tensor (import from odyssey.tensor.any_tensor)
- TensorLike: Shared trait interface for all tensor types
- Factory functions: Available via odyssey.tensor.factories
"""

from odyssey.tensor.tensor_traits import TensorLike
