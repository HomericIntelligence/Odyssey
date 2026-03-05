"""Functional multi-dimensional convolution operations.

This module provides pure functional implementations of multi-dimensional
convolution operations using direct convolution (not im2col).
The caller manages all state (kernels, biases).
"""

from algorithm import parallelize
from collections import List

from shared.core.extensor import ExTensor, zeros
from shared.core.arithmetic import add
from shared.core.reduction import sum as reduce_sum
from shared.core.shape import conv2d_output_shape
from shared.core.gradient_types import (
    GradientPair,
    GradientTriple,
    GradientQuad,
)
from shared.core.parallel_utils import should_parallelize

# max is now a builtin in Mojo - no import needed



fn _conv2d_kernel[
    dtype: DType
](
    x: ExTensor,
    kernel: ExTensor,
    bias: ExTensor,
    stride: Int,
    padding: Int,
    batch: Int,
    in_channels: Int,
    in_height: Int,
    in_width: Int,
    out_channels: Int,
    out_height: Int,
    out_width: Int,
    kH: Int,
    kW: Int,
) raises -> ExTensor:
    """Dtype-generic conv2d kernel - handles accumulation in correct precision.

    This helper eliminates hardcoded Float32 accumulation, preventing overflow
    when using float16 or other lower-precision dtypes.

    For Float16, uses FP32 accumulation internally (industry-standard mixed precision).
    This matches NVIDIA TensorCores, PyTorch, and TensorFlow behavior.
    See GitHub issue #3009 for rationale.
    """
    # Create output tensor with correct dtype
    var out_shape = List[Int]()
    out_shape.append(batch)
    out_shape.append(out_channels)
    out_shape.append(out_height)
    out_shape.append(out_width)
    var output = zeros(out_shape, dtype)

    # Direct convolution algorithm with dtype-correct accumulation
    if should_parallelize(batch):
        # Parallel convolution over batch dimension
        @parameter
        fn conv_batch(b: Int) capturing:
            # For each output channel
            for oc in range(out_channels):
                # For each output position
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Use FP32 accumulation for FP16, native dtype for others
                        @parameter
                        if dtype == DType.float16:
                            var sum_val = Float32(0.0)

                            # Compute input position
                            var in_h_start = oh * stride - padding
                            var in_w_start = ow * stride - padding

                            # Convolve over input channels and kernel
                            for ic in range(in_channels):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        # Input position with padding
                                        var in_h = in_h_start + kh
                                        var in_w = in_w_start + kw

                                        # Check bounds (zero padding)
                                        if (
                                            in_h >= 0
                                            and in_h < in_height
                                            and in_w >= 0
                                            and in_w < in_width
                                        ):
                                            # Get input and kernel values
                                            var in_idx = (
                                                b
                                                * (
                                                    in_channels
                                                    * in_height
                                                    * in_width
                                                )
                                                + ic * (in_height * in_width)
                                                + in_h * in_width
                                                + in_w
                                            )
                                            var k_idx = (
                                                oc * (in_channels * kH * kW)
                                                + ic * (kH * kW)
                                                + kh * kW
                                                + kw
                                            )

                                            var in_val = x._data.bitcast[
                                                Scalar[dtype]
                                            ]()[in_idx]
                                            var k_val = kernel._data.bitcast[
                                                Scalar[dtype]
                                            ]()[k_idx]

                                            sum_val += Float32(
                                                in_val
                                            ) * Float32(k_val)

                            # Add bias
                            var b_val = bias._data.bitcast[Scalar[dtype]]()[oc]
                            sum_val += Float32(b_val)

                            # Write to output
                            var out_idx = (
                                b * (out_channels * out_height * out_width)
                                + oc * (out_height * out_width)
                                + oh * out_width
                                + ow
                            )
                            output._data.bitcast[Scalar[DType.float16]]()[
                                out_idx
                            ] = Float16(sum_val)
                        else:
                            var sum_val = Scalar[dtype](0.0)

                            # Compute input position
                            var in_h_start = oh * stride - padding
                            var in_w_start = ow * stride - padding

                            # Convolve over input channels and kernel
                            for ic in range(in_channels):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        # Input position with padding
                                        var in_h = in_h_start + kh
                                        var in_w = in_w_start + kw

                                        # Check bounds (zero padding)
                                        if (
                                            in_h >= 0
                                            and in_h < in_height
                                            and in_w >= 0
                                            and in_w < in_width
                                        ):
                                            # Get input and kernel values
                                            var in_idx = (
                                                b
                                                * (
                                                    in_channels
                                                    * in_height
                                                    * in_width
                                                )
                                                + ic * (in_height * in_width)
                                                + in_h * in_width
                                                + in_w
                                            )
                                            var k_idx = (
                                                oc * (in_channels * kH * kW)
                                                + ic * (kH * kW)
                                                + kh * kW
                                                + kw
                                            )

                                            var in_val = x._data.bitcast[
                                                Scalar[dtype]
                                            ]()[in_idx]
                                            var k_val = kernel._data.bitcast[
                                                Scalar[dtype]
                                            ]()[k_idx]

                                            sum_val += in_val * k_val

                            # Add bias
                            var b_val = bias._data.bitcast[Scalar[dtype]]()[oc]
                            sum_val += b_val

                            # Write to output
                            var out_idx = (
                                b * (out_channels * out_height * out_width)
                                + oc * (out_height * out_width)
                                + oh * out_width
                                + ow
                            )
                            output._data.bitcast[Scalar[dtype]]()[
                                out_idx
                            ] = sum_val

        parallelize[conv_batch](batch)
    else:
        # Sequential convolution for small batches
        for b in range(batch):
            # For each output channel
            for oc in range(out_channels):
                # For each output position
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Use FP32 accumulation for FP16, native dtype for others
                        @parameter
                        if dtype == DType.float16:
                            var sum_val = Float32(0.0)

                            # Compute input position
                            var in_h_start = oh * stride - padding
                            var in_w_start = ow * stride - padding

                            # Convolve over input channels and kernel
                            for ic in range(in_channels):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        # Input position with padding
                                        var in_h = in_h_start + kh
                                        var in_w = in_w_start + kw

                                        # Check bounds (zero padding)
                                        if (
                                            in_h >= 0
                                            and in_h < in_height
                                            and in_w >= 0
                                            and in_w < in_width
                                        ):
                                            # Get input and kernel values
                                            var in_idx = (
                                                b
                                                * (
                                                    in_channels
                                                    * in_height
                                                    * in_width
                                                )
                                                + ic * (in_height * in_width)
                                                + in_h * in_width
                                                + in_w
                                            )
                                            var k_idx = (
                                                oc * (in_channels * kH * kW)
                                                + ic * (kH * kW)
                                                + kh * kW
                                                + kw
                                            )

                                            var in_val = x._data.bitcast[
                                                Scalar[dtype]
                                            ]()[in_idx]
                                            var k_val = kernel._data.bitcast[
                                                Scalar[dtype]
                                            ]()[k_idx]

                                            sum_val += Float32(
                                                in_val
                                            ) * Float32(k_val)

                            # Add bias
                            var b_val = bias._data.bitcast[Scalar[dtype]]()[oc]
                            sum_val += Float32(b_val)

                            # Write to output
                            var out_idx = (
                                b * (out_channels * out_height * out_width)
                                + oc * (out_height * out_width)
                                + oh * out_width
                                + ow
                            )
                            output._data.bitcast[Scalar[DType.float16]]()[
                                out_idx
                            ] = Float16(sum_val)
                        else:
                            var sum_val = Scalar[dtype](0.0)

                            # Compute input position
                            var in_h_start = oh * stride - padding
                            var in_w_start = ow * stride - padding

                            # Convolve over input channels and kernel
                            for ic in range(in_channels):
                                for kh in range(kH):
                                    for kw in range(kW):
                                        # Input position with padding
                                        var in_h = in_h_start + kh
                                        var in_w = in_w_start + kw

                                        # Check bounds (zero padding)
                                        if (
                                            in_h >= 0
                                            and in_h < in_height
                                            and in_w >= 0
                                            and in_w < in_width
                                        ):
                                            # Get input and kernel values
                                            var in_idx = (
                                                b
                                                * (
                                                    in_channels
                                                    * in_height
                                                    * in_width
                                                )
                                                + ic * (in_height * in_width)
                                                + in_h * in_width
                                                + in_w
                                            )
                                            var k_idx = (
                                                oc * (in_channels * kH * kW)
                                                + ic * (kH * kW)
                                                + kh * kW
                                                + kw
                                            )

                                            var in_val = x._data.bitcast[
                                                Scalar[dtype]
                                            ]()[in_idx]
                                            var k_val = kernel._data.bitcast[
                                                Scalar[dtype]
                                            ]()[k_idx]

                                            sum_val += in_val * k_val

                            # Add bias
                            var b_val = bias._data.bitcast[Scalar[dtype]]()[oc]
                            sum_val += b_val

                            # Write to output
                            var out_idx = (
                                b * (out_channels * out_height * out_width)
                                + oc * (out_height * out_width)
                                + oh * out_width
                                + ow
                            )
                            output._data.bitcast[Scalar[dtype]]()[
                                out_idx
                            ] = sum_val

    return output^


fn conv2d(
    x: ExTensor,
    kernel: ExTensor,
    bias: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> ExTensor:
    """Functional 2D convolution using direct convolution: y = conv2d(x, kernel) + bias.

        Pure function - caller manages kernel and bias. No internal state.
        Uses direct convolution algorithm (not im2col).

    Args:
            x: Input tensor of shape (batch, in_channels, height, width).
            kernel: Convolution kernels of shape (out_channels, in_channels, kH, kW).
            bias: Bias vector of shape (out_channels,).
            stride: Stride for convolution (default: 1).
            padding: Zero-padding added to input (default: 0).

    Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width).
            where:
                out_height = (height + 2*padding - kH) // stride + 1.
                out_width = (width + 2*padding - kW) // stride + 1.

        Example:
            ```mojo
            from shared.core import ExTensor, conv2d, zeros, he_uniform

            # Caller manages state
            var kernel = he_uniform((16, 3, 3, 3), DType.float32)  # 16 filters, 3x3
            var bias = zeros(16, DType.float32)

            # Pure function call
            var output = conv2d(input, kernel, bias, stride=1, padding=1)
            ```

    Raises:
            Error: If tensor shapes are incompatible.
    """
    # Get input dimensions
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error("Input must be 4D tensor (batch, channels, height, width)")

    var batch = x_shape[0]
    var in_channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    # Get kernel dimensions
    var k_shape = kernel.shape()
    if len(k_shape) != 4:
        raise Error(
            "Kernel must be 4D tensor (out_channels, in_channels, kH, kW)"
        )

    var out_channels = k_shape[0]
    var kernel_in_channels = k_shape[1]
    var kH = k_shape[2]
    var kW = k_shape[3]

    if kernel_in_channels != in_channels:
        raise Error("Kernel in_channels must match input in_channels")

    # Compute output dimensions using shape computation helper
    var (out_h, out_w) = conv2d_output_shape(
        in_height, in_width, kH, kW, stride, padding
    )
    var out_height = out_h
    var out_width = out_w

    # Dispatch to dtype-generic kernel based on input dtype
    var input_dtype = x.dtype()
    if input_dtype == DType.float16:
        return _conv2d_kernel[DType.float16](
            x,
            kernel,
            bias,
            stride,
            padding,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kH,
            kW,
        )
    elif input_dtype == DType.float32:
        return _conv2d_kernel[DType.float32](
            x,
            kernel,
            bias,
            stride,
            padding,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kH,
            kW,
        )
    elif input_dtype == DType.float64:
        return _conv2d_kernel[DType.float64](
            x,
            kernel,
            bias,
            stride,
            padding,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kH,
            kW,
        )
    else:
        raise Error(
            "Unsupported dtype for conv2d: only float16, float32, float64"
            " supported"
        )


fn conv2d_no_bias(
    x: ExTensor, kernel: ExTensor, stride: Int = 1, padding: Int = 0
) raises -> ExTensor:
    """Functional 2D convolution without bias: y = conv2d(x, kernel).

        Pure function for convolution with no bias term.
        Uses direct convolution algorithm.

    Args:
            x: Input tensor of shape (batch, in_channels, height, width).
            kernel: Convolution kernels of shape (out_channels, in_channels, kH, kW).
            stride: Stride for convolution (default: 1).
            padding: Zero-padding added to input (default: 0).

    Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width).

    Raises:
            Error: If tensor shapes are incompatible.

    Note:
            This is a convenience wrapper around conv2d with zero bias.
    """
    # Create zero bias
    var k_shape = kernel.shape()
    var out_channels = k_shape[0]
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, x.dtype())

    return conv2d(x, kernel, bias, stride, padding)


fn _conv2d_backward_kernel[
    dtype: DType
](
    grad_output: ExTensor,
    x: ExTensor,
    kernel: ExTensor,
    stride: Int,
    padding: Int,
    batch: Int,
    in_channels: Int,
    in_height: Int,
    in_width: Int,
    out_channels: Int,
    out_height: Int,
    out_width: Int,
    kH: Int,
    kW: Int,
) raises -> GradientTriple:
    """Dtype-generic conv2d backward kernel - handles gradient accumulation in correct precision.
    """
    # Get shapes for gradient tensors
    var x_shape = x.shape()
    var k_shape = kernel.shape()

    # Initialize gradients
    var grad_input = zeros(x_shape, dtype)
    var grad_kernel = zeros(k_shape, dtype)

    # Compute grad_input
    # For each input position, sum contributions from all output positions it affected
    #
    # Derivation:
    #   Forward: in_h = oh * stride - padding + kh
    #   Backward: For input position ih, find (oh, kh) pairs where ih = oh * stride - padding + kh
    #   Solving: kh = ih - (oh * stride - padding) = ih - oh * stride + padding
    #
    # This correctly handles all stride values including stride > 1
    for b in range(batch):
        for ic in range(in_channels):
            for ih in range(in_height):
                for iw in range(in_width):
                    var grad_sum = Scalar[dtype](0.0)

                    # This input position (ih, iw) contributed to output positions (oh, ow)
                    # where: ih = oh * stride - padding + kh
                    # Solving for kh: kh = ih - oh * stride + padding
                    for oh in range(out_height):
                        for ow in range(out_width):
                            # Compute kernel offsets that would access this input position
                            # from this output position in the forward pass
                            var kh = ih - oh * stride + padding
                            var kw = iw - ow * stride + padding

                            # Check if kernel offsets are valid
                            if kh >= 0 and kh < kH and kw >= 0 and kw < kW:
                                # Sum over output channels
                                for oc in range(out_channels):
                                    # Get grad_output value
                                    var grad_out_idx = (
                                        b
                                        * (
                                            out_channels
                                            * out_height
                                            * out_width
                                        )
                                        + oc * (out_height * out_width)
                                        + oh * out_width
                                        + ow
                                    )
                                    var grad_out_val = (
                                        grad_output._data.bitcast[
                                            Scalar[dtype]
                                        ]()[grad_out_idx]
                                    )

                                    # Get kernel value
                                    var k_idx = (
                                        oc * (in_channels * kH * kW)
                                        + ic * (kH * kW)
                                        + kh * kW
                                        + kw
                                    )
                                    var k_val = kernel._data.bitcast[
                                        Scalar[dtype]
                                    ]()[k_idx]

                                    grad_sum += grad_out_val * k_val

                    # Write to grad_input
                    var grad_in_idx = (
                        b * (in_channels * in_height * in_width)
                        + ic * (in_height * in_width)
                        + ih * in_width
                        + iw
                    )
                    grad_input._data.bitcast[Scalar[dtype]]()[
                        grad_in_idx
                    ] = grad_sum

    # Compute grad_kernel
    # For each kernel position, sum input * grad_output over all valid positions
    for oc in range(out_channels):
        for ic in range(in_channels):
            for kh in range(kH):
                for kw in range(kW):
                    var grad_sum = Scalar[dtype](0.0)

                    # For each batch element
                    for b in range(batch):
                        # For each output position
                        for oh in range(out_height):
                            for ow in range(out_width):
                                # Compute input position
                                var in_h = oh * stride - padding + kh
                                var in_w = ow * stride - padding + kw

                                # Check bounds
                                if (
                                    in_h >= 0
                                    and in_h < in_height
                                    and in_w >= 0
                                    and in_w < in_width
                                ):
                                    # Get input value
                                    var in_idx = (
                                        b * (in_channels * in_height * in_width)
                                        + ic * (in_height * in_width)
                                        + in_h * in_width
                                        + in_w
                                    )
                                    var in_val = x._data.bitcast[
                                        Scalar[dtype]
                                    ]()[in_idx]

                                    # Get grad_output value
                                    var grad_out_idx = (
                                        b
                                        * (
                                            out_channels
                                            * out_height
                                            * out_width
                                        )
                                        + oc * (out_height * out_width)
                                        + oh * out_width
                                        + ow
                                    )
                                    var grad_out_val = (
                                        grad_output._data.bitcast[
                                            Scalar[dtype]
                                        ]()[grad_out_idx]
                                    )

                                    grad_sum += in_val * grad_out_val

                    # Write to grad_kernel
                    var grad_k_idx = (
                        oc * (in_channels * kH * kW)
                        + ic * (kH * kW)
                        + kh * kW
                        + kw
                    )
                    grad_kernel._data.bitcast[Scalar[dtype]]()[
                        grad_k_idx
                    ] = grad_sum

    # Compute grad_bias: sum over batch, height, width
    var grad_bias_shape = List[Int]()
    grad_bias_shape.append(out_channels)
    var grad_bias = zeros(grad_bias_shape, dtype)

    for oc in range(out_channels):
        var bias_grad_sum = Scalar[dtype](0.0)

        for b in range(batch):
            for oh in range(out_height):
                for ow in range(out_width):
                    var grad_out_idx = (
                        b * (out_channels * out_height * out_width)
                        + oc * (out_height * out_width)
                        + oh * out_width
                        + ow
                    )
                    var grad_out_val = grad_output._data.bitcast[
                        Scalar[dtype]
                    ]()[grad_out_idx]
                    bias_grad_sum += grad_out_val

        grad_bias._data.bitcast[Scalar[dtype]]()[oc] = bias_grad_sum

    return GradientTriple(grad_input^, grad_kernel^, grad_bias^)


fn conv2d_backward(
    grad_output: ExTensor,
    x: ExTensor,
    kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> GradientTriple:
    """Backward pass for 2D convolution.

        Computes gradients with respect to input, kernel, and bias.

        Math:
            Given: y = conv2d(x, kernel, bias, stride, padding).
            This function computes:
            - grad_input: Gradient w.r.t. input.
            - grad_kernel: Gradient w.r.t. kernel.
            - grad_bias: Gradient w.r.t. bias.

    Args:
            grad_output: Gradient w.r.t. output, shape (batch, out_channels, out_H, out_W).
            x: Input from forward pass, shape (batch, in_channels, in_H, in_W).
            kernel: Kernel from forward pass, shape (out_channels, in_channels, kH, kW).
            stride: Stride used in forward pass.
            padding: Padding used in forward pass.

    Returns:
            GradientTriple containing:
                - grad_input: Gradient w.r.t. input, shape (batch, in_channels, in_H, in_W).
                - grad_kernel: Gradient w.r.t. kernel, shape (out_channels, in_channels, kH, kW).
                - grad_bias: Gradient w.r.t. bias, shape (out_channels,).

        Example:
            ```mojo
            from shared.core import conv2d, conv2d_backward

            # Forward pass
            var output = conv2d(x, kernel, bias, stride, padding)
            # ... compute loss and grad_output ...

            # Backward pass
            var result = conv2d_backward(grad_output, x, kernel, stride, padding)
            var grad_x = result.grad_input
            var grad_k = result.grad_weights
            var grad_b = result.grad_bias
            ```

    Raises:
            Error: If tensor shapes are incompatible.
    """
    # Get dimensions
    var x_shape = x.shape()
    var k_shape = kernel.shape()
    var grad_out_shape = grad_output.shape()

    var batch = x_shape[0]
    var in_channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    var out_channels = k_shape[0]
    var kH = k_shape[2]
    var kW = k_shape[3]

    var out_height = grad_out_shape[2]
    var out_width = grad_out_shape[3]

    # Dispatch to dtype-generic kernel based on input dtype
    var input_dtype = x.dtype()
    if input_dtype == DType.float16:
        return _conv2d_backward_kernel[DType.float16](
            grad_output,
            x,
            kernel,
            stride,
            padding,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kH,
            kW,
        )
    elif input_dtype == DType.float32:
        return _conv2d_backward_kernel[DType.float32](
            grad_output,
            x,
            kernel,
            stride,
            padding,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kH,
            kW,
        )
    elif input_dtype == DType.float64:
        return _conv2d_backward_kernel[DType.float64](
            grad_output,
            x,
            kernel,
            stride,
            padding,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kH,
            kW,
        )
    else:
        raise Error(
            "Unsupported dtype for conv2d_backward: only float16, float32,"
            " float64 supported"
        )


fn conv2d_no_bias_backward(
    grad_output: ExTensor,
    x: ExTensor,
    kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> GradientPair:
    """Backward pass for 2D convolution without bias.

    Args:
            grad_output: Gradient w.r.t. output.
            x: Input from forward pass.
            kernel: Kernel from forward pass.
            stride: Stride used in forward pass.
            padding: Padding used in forward pass.

    Returns:
            GradientPair containing grad_input and grad_kernel.

    Raises:
            Error: If tensor shapes are incompatible.
    """
    var result = conv2d_backward(grad_output, x, kernel, stride, padding)
    # Copy needed fields before result is destroyed (ExTensor is ImplicitlyCopyable)
    var grad_input_copy = result.grad_input
    var grad_kernel_copy = result.grad_weights
    return GradientPair(grad_input_copy^, grad_kernel_copy^)


fn depthwise_conv2d(
    x: ExTensor,
    kernel: ExTensor,
    bias: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> ExTensor:
    """Functional depthwise 2D convolution: y = depthwise_conv2d(x, kernel) + bias.

        Each input channel is convolved with its own filter (no cross-channel mixing).
        Used in efficient architectures like MobileNet and EfficientNet.

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            kernel: Depthwise kernels of shape (channels, 1, kH, kW).
            bias: Bias vector of shape (channels,).
            stride: Stride for convolution (default: 1).
            padding: Zero-padding added to input (default: 0).

    Returns:
            Output tensor of shape (batch, channels, out_height, out_width).
            where:
                out_height = (height + 2*padding - kH) // stride + 1.
                out_width = (width + 2*padding - kW) // stride + 1.

        Example:
            ```mojo
            from shared.core import depthwise_conv2d, zeros, he_uniform

            # Depthwise kernel: one 3x3 filter per channel
            var kernel = he_uniform((32, 1, 3, 3), DType.float32)  # 32 channels
            var bias = zeros(32, DType.float32)

            var output = depthwise_conv2d(input, kernel, bias, stride=1, padding=1)
            ```

    Raises:
            Error: If tensor shapes are incompatible.
    """
    # Get input dimensions
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error(
            "depthwise_conv2d: Input must be 4D tensor (batch, channels,"
            " height, width)"
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    # Get kernel dimensions
    var k_shape = kernel.shape()
    if len(k_shape) != 4:
        raise Error(
            "depthwise_conv2d: Kernel must be 4D tensor (channels, 1, kH, kW)"
        )

    var kernel_channels = k_shape[0]
    var kernel_depth = k_shape[1]
    var kH = k_shape[2]
    var kW = k_shape[3]

    if kernel_channels != channels:
        raise Error(
            "depthwise_conv2d: Kernel channels must match input channels"
        )

    if kernel_depth != 1:
        raise Error(
            "depthwise_conv2d: Kernel depth must be 1 for depthwise convolution"
        )

    # Compute output dimensions
    var (out_h, out_w) = conv2d_output_shape(
        in_height, in_width, kH, kW, stride, padding
    )
    var out_height = out_h
    var out_width = out_w

    # Create output tensor
    var out_shape = List[Int]()
    out_shape.append(batch)
    out_shape.append(channels)
    out_shape.append(out_height)
    out_shape.append(out_width)
    var output = zeros(out_shape, x.dtype())

    # Depthwise convolution: each channel convolved independently
    for b in range(batch):
        for c in range(channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    var sum_val = Float32(0.0)

                    # Compute input position
                    var in_h_start = oh * stride - padding
                    var in_w_start = ow * stride - padding

                    # Convolve with this channel's kernel only
                    for kh in range(kH):
                        for kw in range(kW):
                            var in_h = in_h_start + kh
                            var in_w = in_w_start + kw

                            # Check bounds (zero padding)
                            if (
                                in_h >= 0
                                and in_h < in_height
                                and in_w >= 0
                                and in_w < in_width
                            ):
                                # Get input value
                                var in_idx = (
                                    b * (channels * in_height * in_width)
                                    + c * (in_height * in_width)
                                    + in_h * in_width
                                    + in_w
                                )
                                # Get kernel value (kernel shape is [channels, 1, kH, kW])
                                var k_idx = c * (1 * kH * kW) + kh * kW + kw

                                var in_val = x._data.bitcast[Float32]()[in_idx]
                                var k_val = kernel._data.bitcast[Float32]()[
                                    k_idx
                                ]

                                sum_val += in_val * k_val

                    # Add bias
                    var b_val = bias._data.bitcast[Float32]()[c]
                    sum_val += b_val

                    # Write to output
                    var out_idx = (
                        b * (channels * out_height * out_width)
                        + c * (out_height * out_width)
                        + oh * out_width
                        + ow
                    )
                    output._data.bitcast[Float32]()[out_idx] = sum_val

    return output^


fn depthwise_conv2d_no_bias(
    x: ExTensor, kernel: ExTensor, stride: Int = 1, padding: Int = 0
) raises -> ExTensor:
    """Functional depthwise 2D convolution without bias.

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            kernel: Depthwise kernels of shape (channels, 1, kH, kW).
            stride: Stride for convolution (default: 1).
            padding: Zero-padding added to input (default: 0).

    Returns:
            Output tensor of shape (batch, channels, out_height, out_width).

    Raises:
            Error: If tensor shapes are incompatible.

    Note:
            This is a convenience wrapper around depthwise_conv2d with zero bias.
    """
    var x_shape = x.shape()
    var channels = x_shape[1]
    var bias_shape = List[Int]()
    bias_shape.append(channels)
    var bias = zeros(bias_shape, x.dtype())

    return depthwise_conv2d(x, kernel, bias, stride, padding)


fn depthwise_conv2d_backward(
    grad_output: ExTensor,
    x: ExTensor,
    kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> DepthwiseGradientTriple:
    """Backward pass for depthwise 2D convolution.

        Computes gradients with respect to input, kernel, and bias.

    Args:
            grad_output: Gradient w.r.t. output, shape (batch, channels, out_H, out_W).
            x: Input from forward pass, shape (batch, channels, in_H, in_W).
            kernel: Kernel from forward pass, shape (channels, 1, kH, kW).
            stride: Stride used in forward pass.
            padding: Padding used in forward pass.

    Returns:
            DepthwiseGradientTriple containing:
                - grad_input: Gradient w.r.t. input, shape (batch, channels, in_H, in_W).
                - grad_kernel: Gradient w.r.t. kernel, shape (channels, 1, kH, kW).
                - grad_bias: Gradient w.r.t. bias, shape (channels,).

        Example:
            ```mojo
            from shared.core import depthwise_conv2d, depthwise_conv2d_backward

            # Forward pass
            var output = depthwise_conv2d(x, kernel, bias, stride, padding)
            # ... compute loss and grad_output ...

            # Backward pass
            var result = depthwise_conv2d_backward(grad_output, x, kernel, stride, padding)
            var grad_x = result.grad_input
            var grad_k = result.grad_weights
            var grad_b = result.grad_bias
            ```

    Raises:
            Error: If tensor shapes are incompatible.
    """
    # Get dimensions
    var x_shape = x.shape()
    var k_shape = kernel.shape()
    var grad_out_shape = grad_output.shape()

    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    var kH = k_shape[2]
    var kW = k_shape[3]

    var out_height = grad_out_shape[2]
    var out_width = grad_out_shape[3]

    # Initialize gradients
    var grad_input = zeros(x_shape, x.dtype())
    var grad_kernel = zeros(k_shape, kernel.dtype())

    # Compute grad_input
    # For depthwise conv, each channel's gradient only depends on its own kernel
    for b in range(batch):
        for c in range(channels):
            for ih in range(in_height):
                for iw in range(in_width):
                    var grad_sum = Float32(0.0)

                    for oh in range(out_height):
                        for ow in range(out_width):
                            # Compute kernel offset
                            var kh = ih - oh * stride + padding
                            var kw = iw - ow * stride + padding

                            # Check if kernel offset is valid
                            if kh >= 0 and kh < kH and kw >= 0 and kw < kW:
                                # Get grad_output value
                                var grad_out_idx = (
                                    b * (channels * out_height * out_width)
                                    + c * (out_height * out_width)
                                    + oh * out_width
                                    + ow
                                )
                                var grad_out_val = grad_output._data.bitcast[
                                    Float32
                                ]()[grad_out_idx]

                                # Get kernel value (shape: [channels, 1, kH, kW])
                                var k_idx = c * (1 * kH * kW) + kh * kW + kw
                                var k_val = kernel._data.bitcast[Float32]()[
                                    k_idx
                                ]

                                grad_sum += grad_out_val * k_val

                    # Write to grad_input
                    var grad_in_idx = (
                        b * (channels * in_height * in_width)
                        + c * (in_height * in_width)
                        + ih * in_width
                        + iw
                    )
                    grad_input._data.bitcast[Float32]()[grad_in_idx] = grad_sum

    # Compute grad_kernel
    for c in range(channels):
        for kh in range(kH):
            for kw in range(kW):
                var grad_sum = Float32(0.0)

                for b in range(batch):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            # Compute input position
                            var in_h = oh * stride - padding + kh
                            var in_w = ow * stride - padding + kw

                            # Check bounds
                            if (
                                in_h >= 0
                                and in_h < in_height
                                and in_w >= 0
                                and in_w < in_width
                            ):
                                # Get input value
                                var in_idx = (
                                    b * (channels * in_height * in_width)
                                    + c * (in_height * in_width)
                                    + in_h * in_width
                                    + in_w
                                )
                                var in_val = x._data.bitcast[Float32]()[in_idx]

                                # Get grad_output value
                                var grad_out_idx = (
                                    b * (channels * out_height * out_width)
                                    + c * (out_height * out_width)
                                    + oh * out_width
                                    + ow
                                )
                                var grad_out_val = grad_output._data.bitcast[
                                    Float32
                                ]()[grad_out_idx]

                                grad_sum += in_val * grad_out_val

                # Write to grad_kernel (shape: [channels, 1, kH, kW])
                var grad_k_idx = c * (1 * kH * kW) + kh * kW + kw
                grad_kernel._data.bitcast[Float32]()[grad_k_idx] = grad_sum

    # Compute grad_bias: sum over batch, height, width
    var grad_bias_shape = List[Int]()
    grad_bias_shape.append(channels)
    var grad_bias = zeros(grad_bias_shape, grad_output.dtype())

    for c in range(channels):
        var bias_grad_sum = Float32(0.0)

        for b in range(batch):
            for oh in range(out_height):
                for ow in range(out_width):
                    var grad_out_idx = (
                        b * (channels * out_height * out_width)
                        + c * (out_height * out_width)
                        + oh * out_width
                        + ow
                    )
                    var grad_out_val = grad_output._data.bitcast[Float32]()[
                        grad_out_idx
                    ]
                    bias_grad_sum += grad_out_val

        grad_bias._data.bitcast[Float32]()[c] = bias_grad_sum

    return DepthwiseGradientTriple(grad_input^, grad_kernel^, grad_bias^)


fn depthwise_conv2d_no_bias_backward(
    grad_output: ExTensor,
    x: ExTensor,
    kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> DepthwiseGradientPair:
    """Backward pass for depthwise 2D convolution without bias.

    Args:
            grad_output: Gradient w.r.t. output.
            x: Input from forward pass.
            kernel: Kernel from forward pass.
            stride: Stride used in forward pass.
            padding: Padding used in forward pass.

    Returns:
            DepthwiseGradientPair containing grad_input and grad_kernel.

    Raises:
            Error: If tensor shapes are incompatible.
    """
    var result = depthwise_conv2d_backward(
        grad_output, x, kernel, stride, padding
    )
    # Copy needed fields before result is destroyed (ExTensor is ImplicitlyCopyable)
    var grad_input_copy = result.grad_input
    var grad_kernel_copy = result.grad_weights
    return DepthwiseGradientPair(
        grad_input_copy^, grad_kernel_copy^
    )


# ============================================================================
# Depthwise Separable Convolution
# ============================================================================


fn depthwise_separable_conv2d(
    x: ExTensor,
    depthwise_kernel: ExTensor,
    pointwise_kernel: ExTensor,
    bias: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> ExTensor:
    """Depthwise separable 2D convolution.

        Combines depthwise and pointwise convolutions for efficient mobile architectures.
        Used extensively in MobileNet, EfficientNet, and other efficient networks.

        The operation consists of two stages:
        1. Depthwise conv: Each input channel is convolved with its own filter.
        2. Pointwise conv: 1x1 convolution to combine/project channels.

    Args:
            x: Input tensor of shape (batch, in_channels, height, width).
            depthwise_kernel: Depthwise filter of shape (in_channels, 1, kH, kW).
            pointwise_kernel: Pointwise filter of shape (out_channels, in_channels, 1, 1).
            bias: Bias tensor of shape (out_channels,).
            stride: Stride for depthwise convolution (default: 1).
            padding: Padding for depthwise convolution (default: 0).

    Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import depthwise_separable_conv2d

            # Input: (batch=1, channels=32, H=28, W=28)
            # Depthwise: (32, 1, 3, 3) - one 3x3 filter per channel
            # Pointwise: (64, 32, 1, 1) - project to 64 channels

            var output = depthwise_separable_conv2d(
                x, depthwise_kernel, pointwise_kernel, bias,
                stride=1, padding=1
            )
            # output shape: (1, 64, 28, 28)
            ```

        Formula:
            intermediate = depthwise_conv2d(x, depthwise_kernel).
            output = conv2d_1x1(intermediate, pointwise_kernel) + bias.

    Note:
            This is more efficient than standard convolution:
            - Standard: out_channels * in_channels * kH * kW multiplications.
            - Separable: in_channels * kH * kW + out_channels * in_channels multiplications.
    """
    # Stage 1: Depthwise convolution
    var depthwise_output = depthwise_conv2d_no_bias(
        x, depthwise_kernel, stride, padding
    )

    # Stage 2: Pointwise (1x1) convolution with bias
    var output = conv2d(
        depthwise_output, pointwise_kernel, bias, stride=1, padding=0
    )

    return output


fn depthwise_separable_conv2d_no_bias(
    x: ExTensor,
    depthwise_kernel: ExTensor,
    pointwise_kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> ExTensor:
    """Depthwise separable 2D convolution without bias.

    Args:
            x: Input tensor of shape (batch, in_channels, height, width).
            depthwise_kernel: Depthwise filter of shape (in_channels, 1, kH, kW).
            pointwise_kernel: Pointwise filter of shape (out_channels, in_channels, 1, 1).
            stride: Stride for depthwise convolution (default: 1).
            padding: Padding for depthwise convolution (default: 0).

    Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width).

    Raises:
            Error: If operation fails.
    """
    # Stage 1: Depthwise convolution
    var depthwise_output = depthwise_conv2d_no_bias(
        x, depthwise_kernel, stride, padding
    )

    # Stage 2: Pointwise (1x1) convolution without bias
    var output = conv2d_no_bias(
        depthwise_output, pointwise_kernel, stride=1, padding=0
    )

    return output


fn depthwise_separable_conv2d_backward(
    grad_output: ExTensor,
    x: ExTensor,
    depthwise_kernel: ExTensor,
    pointwise_kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> DepthwiseSeparableGradientTriple:
    """Backward pass for depthwise separable 2D convolution.

        Computes gradients with respect to input and both kernels.

    Args:
            grad_output: Gradient w.r.t. output (batch, out_channels, out_H, out_W).
            x: Original input tensor (batch, in_channels, H, W).
            depthwise_kernel: Depthwise filter (in_channels, 1, kH, kW).
            pointwise_kernel: Pointwise filter (out_channels, in_channels, 1, 1).
            stride: Stride used in forward pass.
            padding: Padding used in forward pass.

    Returns:
            DepthwiseSeparableGradientTriple containing:
                - grad_input: Gradient w.r.t. input.
                - grad_depthwise_kernel: Gradient w.r.t. depthwise kernel.
                - grad_pointwise_kernel: Gradient w.r.t. pointwise kernel.
                - grad_bias: Gradient w.r.t. bias (sum over batch and spatial dims).

    Raises:
        Error: If operation fails.

    Note:
            Pure functional: returns new tensors, does not modify inputs.
    """
    # Recompute intermediate activation for backward
    var depthwise_output = depthwise_conv2d_no_bias(
        x, depthwise_kernel, stride, padding
    )

    # Backward through pointwise (1x1) convolution
    var pointwise_result = conv2d_backward(
        grad_output, depthwise_output, pointwise_kernel, stride=1, padding=0
    )
    var grad_depthwise_output = pointwise_result.grad_input
    var grad_pointwise_kernel = pointwise_result.grad_weights
    var grad_bias = pointwise_result.grad_bias

    # Backward through depthwise convolution
    var depthwise_result = depthwise_conv2d_no_bias_backward(
        grad_depthwise_output, x, depthwise_kernel, stride, padding
    )
    var grad_input = depthwise_result.grad_a
    var grad_depthwise_kernel = depthwise_result.grad_b

    return DepthwiseSeparableGradientTriple(
        grad_input^, grad_depthwise_kernel^, grad_pointwise_kernel^, grad_bias^
    )


fn depthwise_separable_conv2d_no_bias_backward(
    grad_output: ExTensor,
    x: ExTensor,
    depthwise_kernel: ExTensor,
    pointwise_kernel: ExTensor,
    stride: Int = 1,
    padding: Int = 0,
) raises -> DepthwiseSeparableGradientPair:
    """Backward pass for depthwise separable 2D convolution without bias.

    Args:
            grad_output: Gradient w.r.t. output.
            x: Original input tensor.
            depthwise_kernel: Depthwise filter.
            pointwise_kernel: Pointwise filter.
            stride: Stride used in forward pass.
            padding: Padding used in forward pass.

    Returns:
            DepthwiseSeparableGradientPair containing gradients.

    Raises:
        Error: If operation fails.
    """
    # Recompute intermediate activation
    var depthwise_output = depthwise_conv2d_no_bias(
        x, depthwise_kernel, stride, padding
    )

    # Backward through pointwise convolution
    var pointwise_result = conv2d_no_bias_backward(
        grad_output, depthwise_output, pointwise_kernel, stride=1, padding=0
    )
    var grad_depthwise_output = pointwise_result.grad_a
    var grad_pointwise_kernel = pointwise_result.grad_b

    # Backward through depthwise convolution
    var depthwise_result = depthwise_conv2d_no_bias_backward(
        grad_depthwise_output, x, depthwise_kernel, stride, padding
    )

    # Extract fields before constructing result
    var grad_input = depthwise_result.grad_a
    var grad_depthwise_kernel = depthwise_result.grad_b

    return DepthwiseSeparableGradientPair(
        grad_input^,
        grad_depthwise_kernel^,
        grad_pointwise_kernel^,
    )
