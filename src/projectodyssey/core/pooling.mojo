"""Functional pooling operations for 2D inputs.

This module provides pure functional implementations of pooling operations.
All operations are stateless - caller provides all inputs.
"""

from std.algorithm import parallelize
from std.collections import List

from projectodyssey.tensor.any_tensor import AnyTensor, zeros
from projectodyssey.core.parallel_utils import should_parallelize

# max and min are now builtins in Mojo - no import needed


def maxpool2d(
    x: AnyTensor,
    kernel_size: Int,
    stride: Int = 0,  # 0 means use kernel_size
    padding: Int = 0,
    method: String = "direct",
) raises -> AnyTensor:
    """Functional 2D max pooling with selectable implementation.

        Pure function - no internal state. Downsamples spatial dimensions by
        taking maximum value in each kernel_size x kernel_size window.

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            kernel_size: Size of the pooling window.
            stride: Stride for pooling (default: kernel_size if 0).
            padding: Zero-padding added to input (default: 0).
            method: Implementation method - "direct" (default), "optimized" (future).

    Returns:
            Output tensor of shape (batch, channels, out_height, out_width).
            where:
                stride_actual = kernel_size if stride == 0 else stride.
                out_height = (height + 2*padding - kernel_size) / stride_actual + 1.
                out_width = (width + 2*padding - kernel_size) / stride_actual + 1.

        Example:
            ```mojo
            from projectodyssey.core import AnyTensor, maxpool2d

            # Pure function call - no state to manage
            var pooled = maxpool2d(input, kernel_size=2, stride=2)

            # Or select implementation method
            var pooled = maxpool2d(input, kernel_size=2, stride=2, method="direct")
            ```

    Raises:
            Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct" and method != "optimized":
        raise Error("maxpool2d: method must be 'direct' or 'optimized'")

    # Get input dimensions
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error("Input must be 4D tensor (batch, channels, height, width)")

    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    # Determine actual stride
    var actual_stride = stride if stride > 0 else kernel_size

    # Compute output dimensions using shape computation helper
    from projectodyssey.core.shape import pool_output_shape

    var (out_h, out_w) = pool_output_shape(
        in_height, in_width, kernel_size, actual_stride, padding
    )
    var out_height = out_h
    var out_width = out_w

    # Handle empty output case (kernel larger than input with no padding)
    if out_height <= 0 or out_width <= 0:
        var empty_shape = List[Int](capacity=4)
        empty_shape.append(batch)
        empty_shape.append(channels)
        empty_shape.append(0)
        empty_shape.append(0)
        return zeros(empty_shape, x.dtype())

    # Create output tensor
    var out_shape = List[Int](capacity=4)
    out_shape.append(batch)
    out_shape.append(channels)
    out_shape.append(out_height)
    out_shape.append(out_width)
    var output = zeros(out_shape, x.dtype())

    if method == "optimized":
        # Optimized algorithm: precompute row-wise maximums
        # For each row, we compute max over kernel_size consecutive elements
        # This reduces inner loop from O(k²) to O(k) per output
        _maxpool2d_optimized(
            x,
            output,
            batch,
            channels,
            in_height,
            in_width,
            out_height,
            out_width,
            kernel_size,
            actual_stride,
            padding,
        )
        return output^

    # Direct max pooling algorithm
    if should_parallelize(batch):
        # Parallel maxpool over batch dimension
        @parameter
        def maxpool_batch(b: Int) capturing:
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Compute input window bounds
                        var in_h_start = oh * actual_stride - padding
                        var in_w_start = ow * actual_stride - padding

                        # Find maximum in window
                        # Use Float64(-65504.0) — smallest finite float16 value, safe
                        # initial sentinel for all supported dtypes (float16/float32/float64).
                        # Reading via _get_float64 is required to handle float16 correctly:
                        # bitcast[Float32] reads wrong byte offsets for 2-byte float16 elements.
                        var max_val = Float64(-65504.0)

                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                var in_h = in_h_start + kh
                                var in_w = in_w_start + kw

                                # Check bounds (zero padding treated as -inf for max)
                                if (
                                    in_h >= 0
                                    and in_h < in_height
                                    and in_w >= 0
                                    and in_w < in_width
                                ):
                                    var in_idx = (
                                        b * (channels * in_height * in_width)
                                        + c * (in_height * in_width)
                                        + in_h * in_width
                                        + in_w
                                    )
                                    var val = x._get_float64(in_idx)
                                    if val > max_val:
                                        max_val = val

                        # Write maximum to output
                        var out_idx = (
                            b * (channels * out_height * out_width)
                            + c * (out_height * out_width)
                            + oh * out_width
                            + ow
                        )
                        output._set_float64(out_idx, max_val)

        parallelize[maxpool_batch](batch)
    else:
        # Sequential maxpool for small batches
        for b in range(batch):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Compute input window bounds
                        var in_h_start = oh * actual_stride - padding
                        var in_w_start = ow * actual_stride - padding

                        # Find maximum in window
                        # Use Float64(-65504.0) — smallest finite float16 value, safe
                        # initial sentinel for all supported dtypes (float16/float32/float64).
                        # Reading via _get_float64 is required to handle float16 correctly:
                        # bitcast[Float32] reads wrong byte offsets for 2-byte float16 elements.
                        var max_val = Float64(-65504.0)

                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                var in_h = in_h_start + kh
                                var in_w = in_w_start + kw

                                # Check bounds (zero padding treated as -inf for max)
                                if (
                                    in_h >= 0
                                    and in_h < in_height
                                    and in_w >= 0
                                    and in_w < in_width
                                ):
                                    var in_idx = (
                                        b * (channels * in_height * in_width)
                                        + c * (in_height * in_width)
                                        + in_h * in_width
                                        + in_w
                                    )
                                    var val = x._get_float64(in_idx)
                                    if val > max_val:
                                        max_val = val

                        # Write maximum to output
                        var out_idx = (
                            b * (channels * out_height * out_width)
                            + c * (out_height * out_width)
                            + oh * out_width
                            + ow
                        )
                        output._set_float64(out_idx, max_val)

    return output^


def _maxpool2d_optimized(
    x: AnyTensor,
    mut output: AnyTensor,
    batch: Int,
    channels: Int,
    in_height: Int,
    in_width: Int,
    out_height: Int,
    out_width: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
):
    """Optimized maxpool using row-wise precomputation.

    Algorithm:
    1. For each input row, precompute max over horizontal windows
    2. For each output position, compute max over precomputed row values

    This reduces complexity from O(k²) to O(k) per output when windows overlap.
    """
    # Allocate row-wise max buffer for current batch/channel.
    # Use Float64 to correctly handle float16/float32/float64 input dtypes.
    # bitcast[Float32] would read wrong byte offsets for float16 elements.
    # row_max[h][w] = max over [w, w+kernel_size) for row h
    var row_max_size = in_height * out_width
    var row_max = List[Float64]()
    for _ in range(row_max_size):
        row_max.append(Float64(-65504.0))

    for b in range(batch):
        for c in range(channels):
            # Step 1: Precompute row-wise maximums for this channel
            for ih in range(in_height):
                for ow in range(out_width):
                    # Window starts at w_start = ow * stride - padding
                    var w_start = ow * stride - padding
                    var row_max_val = Float64(-65504.0)

                    for kw in range(kernel_size):
                        var iw = w_start + kw
                        if iw >= 0 and iw < in_width:
                            var in_idx = (
                                b * (channels * in_height * in_width)
                                + c * (in_height * in_width)
                                + ih * in_width
                                + iw
                            )
                            var val = x._get_float64(in_idx)
                            if val > row_max_val:
                                row_max_val = val

                    row_max[ih * out_width + ow] = row_max_val

            # Step 2: For each output position, compute max over kernel_size rows
            for oh in range(out_height):
                var h_start = oh * stride - padding

                for ow in range(out_width):
                    var max_val = Float64(-65504.0)

                    for kh in range(kernel_size):
                        var ih = h_start + kh
                        if ih >= 0 and ih < in_height:
                            var row_val = row_max[ih * out_width + ow]
                            if row_val > max_val:
                                max_val = row_val

                    var out_idx = (
                        b * (channels * out_height * out_width)
                        + c * (out_height * out_width)
                        + oh * out_width
                        + ow
                    )
                    output._set_float64(out_idx, max_val)


def avgpool2d(
    x: AnyTensor,
    kernel_size: Int,
    stride: Int = 0,  # 0 means use kernel_size
    padding: Int = 0,
    method: String = "direct",
) raises -> AnyTensor:
    """Functional 2D average pooling with selectable implementation.

        Pure function - no internal state. Downsamples spatial dimensions by
        taking average value in each kernel_size x kernel_size window.

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            kernel_size: Size of the pooling window.
            stride: Stride for pooling (default: kernel_size if 0).
            padding: Zero-padding added to input (default: 0).
            method: Implementation method - "direct" (default), "optimized" (future).

    Returns:
            Output tensor of shape (batch, channels, out_height, out_width).

        Example:
            ```mojo
            from projectodyssey.core import AnyTensor, avgpool2d

            # Pure function call - no state to manage
            var pooled = avgpool2d(input, kernel_size=2, stride=2)

            # Or select implementation method
            var pooled = avgpool2d(input, kernel_size=2, stride=2, method="direct")
            ```

    Raises:
            Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct":
        raise Error("Only 'direct' method is currently supported for avgpool2d")

    # Get input dimensions
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error("Input must be 4D tensor (batch, channels, height, width)")

    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    # Determine actual stride
    var actual_stride = stride if stride > 0 else kernel_size

    # Compute output dimensions using shape computation helper
    from projectodyssey.core.shape import pool_output_shape

    var (out_h, out_w) = pool_output_shape(
        in_height, in_width, kernel_size, actual_stride, padding
    )
    var out_height = out_h
    var out_width = out_w

    # Create output tensor
    var out_shape = List[Int](capacity=4)
    out_shape.append(batch)
    out_shape.append(channels)
    out_shape.append(out_height)
    out_shape.append(out_width)
    var output = zeros(out_shape, x.dtype())

    # Direct average pooling algorithm
    # Use Float64 accumulators and _get_float64/_set_float64 to handle all
    # dtypes correctly. bitcast[Float32] reads wrong byte offsets for float16.
    for b in range(batch):
        for c in range(channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    # Compute input window bounds
                    var in_h_start = oh * actual_stride - padding
                    var in_w_start = ow * actual_stride - padding

                    # Compute sum and count in window
                    var sum_val = Float64(0.0)
                    var count = 0

                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            var in_h = in_h_start + kh
                            var in_w = in_w_start + kw

                            # Check bounds (zero padding counted in divisor)
                            if (
                                in_h >= 0
                                and in_h < in_height
                                and in_w >= 0
                                and in_w < in_width
                            ):
                                var in_idx = (
                                    b * (channels * in_height * in_width)
                                    + c * (in_height * in_width)
                                    + in_h * in_width
                                    + in_w
                                )
                                var val = x._get_float64(in_idx)
                                sum_val += val
                                count += 1

                    # Compute average
                    var avg_val = sum_val / Float64(
                        count
                    ) if count > 0 else Float64(0.0)

                    # Write average to output
                    var out_idx = (
                        b * (channels * out_height * out_width)
                        + c * (out_height * out_width)
                        + oh * out_width
                        + ow
                    )
                    output._set_float64(out_idx, avg_val)

    return output^


def global_avgpool2d(
    x: AnyTensor, method: String = "direct"
) raises -> AnyTensor:
    """Functional global average pooling with selectable implementation.

        Pure function that reduces spatial dimensions (H, W) to (1, 1) by
        averaging all values in each channel.

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            method: Implementation method - "direct" (default), "optimized" (future).

    Returns:
            Output tensor of shape (batch, channels, 1, 1).

        Example:
            ```mojo
            from projectodyssey.core import AnyTensor, global_avgpool2d

            # Pure function call
            var pooled = global_avgpool2d(input)  # (B, C, H, W) -> Tuple[B, C, 1, 1]

            # Or select implementation method
            var pooled = global_avgpool2d(input, method="direct")
            ```

    Raises:
            Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct":
        raise Error(
            "Only 'direct' method is currently supported for global_avgpool2d"
        )

    # Get input dimensions
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error("Input must be 4D tensor (batch, channels, height, width)")

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    # Create output tensor (B, C, 1, 1)
    var out_shape = List[Int](capacity=4)
    out_shape.append(batch)
    out_shape.append(channels)
    out_shape.append(1)
    out_shape.append(1)
    var output = zeros(out_shape, x.dtype())

    # Compute global average for each channel.
    # Use Float64 accumulators and _get_float64/_set_float64 to handle all
    # dtypes correctly. bitcast[Float32] reads wrong byte offsets for float16.
    for b in range(batch):
        for c in range(channels):
            # Sum all spatial values
            var sum_val = Float64(0.0)

            for h in range(height):
                for w in range(width):
                    var in_idx = (
                        b * (channels * height * width)
                        + c * (height * width)
                        + h * width
                        + w
                    )
                    var val = x._get_float64(in_idx)
                    sum_val += val

            # Compute average
            var avg_val = sum_val / Float64(height * width)

            # Write to output
            var out_idx = b * channels + c
            output._set_float64(out_idx, avg_val)

    return output^


def maxpool2d_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    kernel_size: Int,
    stride: Int = 0,
    padding: Int = 0,
    method: String = "direct",
) raises -> AnyTensor:
    """Backward pass for 2D max pooling.

        Computes gradient with respect to input. Routes gradients only to the
        positions that had the maximum value in the forward pass.

    Args:
            grad_output: Gradient w.r.t. output, shape (batch, channels, out_H, out_W).
            x: Input from forward pass, shape (batch, channels, in_H, in_W).
            kernel_size: Size of the pooling window used in forward pass.
            stride: Stride used in forward pass (0 means use kernel_size).
            padding: Padding used in forward pass.
            method: Implementation method (must match forward pass).

    Returns:
            Gradient w.r.t. input, shape (batch, channels, in_H, in_W).

    Example:
        ```mojo
        from projectodyssey.core import maxpool2d, maxpool2d_backward

        # Forward pass
        var output = maxpool2d(x, kernel_size=2, stride=2)
        # ... compute loss and grad_output ...

        # Backward pass
        var grad_x = maxpool2d_backward(grad_output, x, kernel_size=2, stride=2)
        ```

    Note:
            This implementation recomputes the argmax positions from the forward pass.
            In a stateful implementation, these would be cached.

    Raises:
            Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct":
        raise Error(
            "Only 'direct' method is currently supported for maxpool2d_backward"
        )

    # Get dimensions
    var x_shape = x.shape()
    var grad_out_shape = grad_output.shape()

    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    var out_height = grad_out_shape[2]
    var out_width = grad_out_shape[3]

    # Determine actual stride
    var actual_stride = stride if stride > 0 else kernel_size

    # Initialize grad_input
    var grad_input = zeros(x_shape, x.dtype())

    # For each batch and channel
    # Use Float64 accumulators and _get_float64/_set_float64 to handle all
    # dtypes correctly. bitcast[Float32] reads wrong byte offsets for float16.
    for b in range(batch):
        for c in range(channels):
            # For each output position
            for oh in range(out_height):
                for ow in range(out_width):
                    # Compute input window bounds
                    var in_h_start = oh * actual_stride - padding
                    var in_w_start = ow * actual_stride - padding

                    # Find the position of maximum value in the window
                    var max_val = Float64(-65504.0)
                    var max_h = -1
                    var max_w = -1

                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            var in_h = in_h_start + kh
                            var in_w = in_w_start + kw

                            # Check bounds
                            if (
                                in_h >= 0
                                and in_h < in_height
                                and in_w >= 0
                                and in_w < in_width
                            ):
                                var in_idx = (
                                    b * (channels * in_height * in_width)
                                    + c * (in_height * in_width)
                                    + in_h * in_width
                                    + in_w
                                )
                                var val = x._get_float64(in_idx)

                                if val > max_val:
                                    max_val = val
                                    max_h = in_h
                                    max_w = in_w

                    # Route gradient to the max position
                    if max_h >= 0 and max_w >= 0:
                        var grad_out_idx = (
                            b * (channels * out_height * out_width)
                            + c * (out_height * out_width)
                            + oh * out_width
                            + ow
                        )
                        var grad_out_val = grad_output._get_float64(
                            grad_out_idx
                        )

                        var grad_in_idx = (
                            b * (channels * in_height * in_width)
                            + c * (in_height * in_width)
                            + max_h * in_width
                            + max_w
                        )
                        grad_input._set_float64(
                            grad_in_idx,
                            grad_input._get_float64(grad_in_idx) + grad_out_val,
                        )

    return grad_input^


def avgpool2d_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    kernel_size: Int,
    stride: Int = 0,
    padding: Int = 0,
    method: String = "direct",
) raises -> AnyTensor:
    """Backward pass for 2D average pooling.

        Computes gradient with respect to input. Distributes gradients equally
        to all positions in the pooling window.

    Args:
            grad_output: Gradient w.r.t. output, shape (batch, channels, out_H, out_W).
            x: Input from forward pass, shape (batch, channels, in_H, in_W).
            kernel_size: Size of the pooling window used in forward pass.
            stride: Stride used in forward pass (0 means use kernel_size).
            padding: Padding used in forward pass.
            method: Implementation method (must match forward pass).

    Returns:
            Gradient w.r.t. input, shape (batch, channels, in_H, in_W).

    Example:
        ```mojo
        from projectodyssey.core import avgpool2d, avgpool2d_backward

        # Forward pass
        var output = avgpool2d(x, kernel_size=2, stride=2)
        # ... compute loss and grad_output ...

        # Backward pass
        var grad_x = avgpool2d_backward(grad_output, x, kernel_size=2, stride=2)
        ```

    Raises:
            Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct":
        raise Error(
            "Only 'direct' method is currently supported for avgpool2d_backward"
        )

    # Get dimensions
    var x_shape = x.shape()
    var grad_out_shape = grad_output.shape()

    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    var out_height = grad_out_shape[2]
    var out_width = grad_out_shape[3]

    # Determine actual stride
    var actual_stride = stride if stride > 0 else kernel_size

    # Initialize grad_input
    var grad_input = zeros(x_shape, x.dtype())

    # For each batch and channel
    # Use Float64 accumulators and _get_float64/_set_float64 to handle all
    # dtypes correctly. bitcast[Float32] reads wrong byte offsets for float16.
    for b in range(batch):
        for c in range(channels):
            # For each output position
            for oh in range(out_height):
                for ow in range(out_width):
                    # Get grad_output value
                    var grad_out_idx = (
                        b * (channels * out_height * out_width)
                        + c * (out_height * out_width)
                        + oh * out_width
                        + ow
                    )
                    var grad_out_val = grad_output._get_float64(grad_out_idx)

                    # Compute input window bounds
                    var in_h_start = oh * actual_stride - padding
                    var in_w_start = ow * actual_stride - padding

                    # Count valid positions in window
                    var count = 0
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            var in_h = in_h_start + kh
                            var in_w = in_w_start + kw

                            if (
                                in_h >= 0
                                and in_h < in_height
                                and in_w >= 0
                                and in_w < in_width
                            ):
                                count += 1

                    # Distribute gradient equally to all positions
                    var grad_per_position = grad_out_val / Float64(
                        count
                    ) if count > 0 else Float64(0.0)

                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            var in_h = in_h_start + kh
                            var in_w = in_w_start + kw

                            if (
                                in_h >= 0
                                and in_h < in_height
                                and in_w >= 0
                                and in_w < in_width
                            ):
                                var grad_in_idx = (
                                    b * (channels * in_height * in_width)
                                    + c * (in_height * in_width)
                                    + in_h * in_width
                                    + in_w
                                )
                                grad_input._set_float64(
                                    grad_in_idx,
                                    grad_input._get_float64(grad_in_idx)
                                    + grad_per_position,
                                )

    return grad_input^


def global_avgpool2d_backward(
    grad_output: AnyTensor, x: AnyTensor, method: String = "direct"
) raises -> AnyTensor:
    """Backward pass for global average pooling.

        Computes gradient with respect to input. Distributes gradients equally
        to all spatial positions.

    Args:
            grad_output: Gradient w.r.t. output, shape (batch, channels, 1, 1).
            x: Input from forward pass, shape (batch, channels, height, width).
            method: Implementation method (must match forward pass).

    Returns:
           Gradient w.r.t. input, shape (batch, channels, height, width).

    Example:
        ```mojo
        from projectodyssey.core import global_avgpool2d, global_avgpool2d_backward

        # Forward pass
        var output = global_avgpool2d(x)
        # ... compute loss and grad_output ...

        # Backward pass
        var grad_x = global_avgpool2d_backward(grad_output, x)
        ```

    Raises:
            Error: If tensor shapes are incompatible or method is unsupported.
    """
    if method != "direct":
        raise Error(
            "Only 'direct' method is currently supported for"
            " global_avgpool2d_backward"
        )

    # Get dimensions
    var x_shape = x.shape()

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    # Initialize grad_input
    var grad_input = zeros(x_shape, x.dtype())

    # Total number of spatial elements
    # Use Float64 accumulators and _get_float64/_set_float64 to handle all
    # dtypes correctly. bitcast[Float32] reads wrong byte offsets for float16.
    var spatial_size = Float64(height * width)

    # For each batch and channel
    for b in range(batch):
        for c in range(channels):
            # Get grad_output value at position (b, c, 0, 0)
            # Since grad_output shape is (B, C, 1, 1), linear index is b*C + c
            var grad_out_idx = b * channels + c
            var grad_out_val = grad_output._get_float64(grad_out_idx)

            # Distribute equally to all spatial positions
            var grad_per_position = grad_out_val / spatial_size

            for h in range(height):
                for w in range(width):
                    var grad_in_idx = (
                        b * (channels * height * width)
                        + c * (height * width)
                        + h * width
                        + w
                    )
                    grad_input._set_float64(grad_in_idx, grad_per_position)

    return grad_input^
