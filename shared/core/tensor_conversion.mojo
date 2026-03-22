"""Standalone conversion functions between AnyTensor (ExTensor) and Tensor[dt].

Avoids circular import: extensor.mojo cannot import Tensor at module level
because tensor.mojo imports ExTensor from extensor.mojo. This module breaks
the cycle by importing both and providing free functions.

Usage:
    ```mojo
    from shared.core.tensor_conversion import any_to_tensor

    var any_t = zeros([3, 4], DType.float32)
    var typed = any_to_tensor[DType.float32](any_t)
    ```
"""

from shared.core.extensor import ExTensor
from shared.tensor.tensor import Tensor


fn any_to_tensor[dt: DType](any_t: ExTensor) raises -> Tensor[dt]:
    """Convert ExTensor to Tensor[dt] via zero-copy shared refcount.

    Uses Tensor's internal constructor which increments the shared refcount
    (B4 protocol). Both the source ExTensor and returned Tensor share the
    same underlying data buffer and refcount pointer.

    Parameters:
        dt: The compile-time DType to bitcast the data pointer to.

    Args:
        any_t: The ExTensor to convert (must have matching dtype).

    Returns:
        A Tensor[dt] sharing the same data and refcount.

    Raises:
        Error: If any_t's runtime dtype doesn't match dt.
    """
    if any_t._dtype != dt:
        raise Error(
            "any_to_tensor: dtype mismatch — ExTensor has "
            + String(any_t._dtype)
            + " but requested "
            + String(dt)
        )
    return Tensor[dt](
        any_t._data.bitcast[Scalar[dt]](),
        any_t._shape,
        any_t._strides,
        any_t._refcount,
        any_t._numel,
        any_t._is_view,
        any_t._allocated_size,
        any_t._original_numel_quantized,
    )
