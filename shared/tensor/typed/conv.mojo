"""Typed Tensor[dtype] convolution dispatch cores.

Internal module -- not part of the public API.
"""

from collections import List
from shared.tensor.tensor import Tensor
from shared.tensor.any_tensor import AnyTensor
from shared.base.dtype_ordinal import (
    dtype_to_ordinal,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
)


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] Typed Implementations
# ============================================================================


fn _conv2d_typed[dt: DType](
    x: Tensor[dt],
    kernel: Tensor[dt],
    bias: Tensor[dt],
    stride: Int = 1,
    padding: Int = 0,
) raises -> Tensor[dt]:
    """Native typed conv2d (Layer 3 core).

    Accepts Tensor[dt] inputs, delegates to existing parametric kernel,
    and returns Tensor[dt] result.

    Args:
        x: Input tensor of shape (batch, in_channels, height, width).
        kernel: Convolution kernels of shape (out_channels, in_channels, kH, kW).
        bias: Bias vector of shape (out_channels,).
        stride: Stride for convolution.
        padding: Zero-padding added to input.

    Returns:
        A new Tensor[dt] with the convolution result.
    """
    from shared.core.conv import conv2d

    var x_any = x.as_any()
    var k_any = kernel.as_any()
    var b_any = bias.as_any()
    var result_any = conv2d(x_any, k_any, b_any, stride, padding)
    return result_any.as_tensor[dt]()


fn _conv2d_no_bias_typed[dt: DType](
    x: Tensor[dt],
    kernel: Tensor[dt],
    stride: Int = 1,
    padding: Int = 0,
) raises -> Tensor[dt]:
    """Native typed conv2d without bias (Layer 3 core).

    Args:
        x: Input tensor of shape (batch, in_channels, height, width).
        kernel: Convolution kernels of shape (out_channels, in_channels, kH, kW).
        stride: Stride for convolution.
        padding: Zero-padding added to input.

    Returns:
        A new Tensor[dt] with the convolution result.
    """
    from shared.core.conv import conv2d_no_bias

    var x_any = x.as_any()
    var k_any = kernel.as_any()
    var result_any = conv2d_no_bias(x_any, k_any, stride, padding)
    return result_any.as_tensor[dt]()


# ============================================================================
# Layer 2: Ordinal-Based Dispatch for Typed Conv Operations
# ============================================================================


fn _dispatch_conv2d_typed(
    x: AnyTensor,
    kernel: AnyTensor,
    bias: AnyTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> AnyTensor:
    """Runtime dispatch to typed conv2d via ordinal-based lookup.

    Args:
        x: Input tensor.
        kernel: Convolution kernels.
        bias: Bias vector.
        stride: Stride for convolution.
        padding: Zero-padding added to input.

    Returns:
        Convolution result.
    """
    var ordinal = dtype_to_ordinal(x.dtype())
    if ordinal == DTYPE_FLOAT16:
        return _conv2d_typed[DType.float16](
            x.as_tensor[DType.float16](),
            kernel.as_tensor[DType.float16](),
            bias.as_tensor[DType.float16](),
            stride,
            padding,
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _conv2d_typed[DType.float32](
            x.as_tensor[DType.float32](),
            kernel.as_tensor[DType.float32](),
            bias.as_tensor[DType.float32](),
            stride,
            padding,
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _conv2d_typed[DType.float64](
            x.as_tensor[DType.float64](),
            kernel.as_tensor[DType.float64](),
            bias.as_tensor[DType.float64](),
            stride,
            padding,
        ).as_any()
    else:
        raise Error(
            "conv2d: unsupported dtype, only float16/float32/float64 supported"
        )
