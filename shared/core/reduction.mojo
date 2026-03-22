"""Reduction operations for ExTensor.

Implements operations that reduce tensors along specified axes
"""

from collections import List
from .extensor import ExTensor
from shared.tensor.tensor import Tensor, from_any_tensor
from .shape import as_contiguous
from .reduction_utils import (
    compute_strides,
    linear_to_coords,
    coords_to_linear,
    map_result_to_input_coords,
    create_result_coords,
    compute_axis_strides,
    build_reduced_shape,
)
from .reduction_ops import (
    ReduceOp,
    ReduceBackwardOp,
    SumOp,
    MeanOp,
    MaxOp,
    MinOp,
    SumBackwardOp,
    MeanBackwardOp,
    MaxBackwardOp,
    MinBackwardOp,
)


# ============================================================================
# Generic Reduction Templates
# ============================================================================


fn _reduce_all_impl[
    dtype: DType, Op: ReduceOp
](result: ExTensor, tensor: ExTensor, numel: Int):
    """Generic dtype-specialized reduction over all elements."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var op = Op()
    var acc: Scalar[dtype] = Scalar[dtype](op.init_value())
    for i in range(numel):
        var val = Float64(in_ptr[i])
        var acc_float = Float64(acc)
        var result_float = op.apply(acc_float, val)
        acc = Scalar[dtype](result_float)
    var final_val = op.finalize(Float64(acc), numel)
    out_ptr[0] = Scalar[dtype](final_val)


fn _dispatch_reduce_all[
    Op: ReduceOp
](result: ExTensor, tensor: ExTensor, numel: Int) raises:
    """Generic runtime dispatch for reduction over all elements."""
    # Ensure input is contiguous before flat-buffer kernel access.
    var t = tensor if tensor.is_contiguous() else as_contiguous(tensor)
    var dt = t.dtype()
    if dt == DType.float16:
        _reduce_all_impl[DType.float16, Op](result, t, numel)
    elif dt == DType.float32:
        _reduce_all_impl[DType.float32, Op](result, t, numel)
    elif dt == DType.float64:
        _reduce_all_impl[DType.float64, Op](result, t, numel)
    elif dt == DType.int32:
        _reduce_all_impl[DType.int32, Op](result, t, numel)
    elif dt == DType.int64:
        _reduce_all_impl[DType.int64, Op](result, t, numel)
    else:
        raise Error("reduce_all: unsupported dtype")


fn _reduce_axis_impl[
    dtype: DType, Op: ReduceOp
](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
):
    """Generic dtype-specialized reduction along axis."""
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    var op = Op()

    for outer in range(outer_size):
        for inner in range(inner_size):
            var acc: Scalar[dtype] = Scalar[dtype](op.init_value())
            for k in range(axis_size):
                var input_idx = (
                    outer * axis_size * inner_size + k * inner_size + inner
                )
                var val = Float64(in_ptr[input_idx])
                var acc_float = Float64(acc)
                var result_float = op.apply(acc_float, val)
                acc = Scalar[dtype](result_float)
            var result_idx = outer * inner_size + inner
            var final_val = op.finalize(Float64(acc), axis_size)
            out_ptr[result_idx] = Scalar[dtype](final_val)


fn _dispatch_reduce_axis[
    Op: ReduceOp
](
    result: ExTensor,
    tensor: ExTensor,
    outer_size: Int,
    axis_size: Int,
    inner_size: Int,
) raises:
    """Generic runtime dispatch for reduction along axis."""
    # Ensure input is contiguous before flat-buffer kernel access.
    var t = tensor if tensor.is_contiguous() else as_contiguous(tensor)
    var dt = t.dtype()
    if dt == DType.float16:
        _reduce_axis_impl[DType.float16, Op](
            result, t, outer_size, axis_size, inner_size
        )
    elif dt == DType.float32:
        _reduce_axis_impl[DType.float32, Op](
            result, t, outer_size, axis_size, inner_size
        )
    elif dt == DType.float64:
        _reduce_axis_impl[DType.float64, Op](
            result, t, outer_size, axis_size, inner_size
        )
    elif dt == DType.int32:
        _reduce_axis_impl[DType.int32, Op](
            result, t, outer_size, axis_size, inner_size
        )
    elif dt == DType.int64:
        _reduce_axis_impl[DType.int64, Op](
            result, t, outer_size, axis_size, inner_size
        )
    else:
        raise Error("reduce_axis: unsupported dtype")


fn sum(
    tensor: ExTensor, axis: Int = -1, keepdims: Bool = False
) raises -> ExTensor:
    """Sum tensor elements along an axis.

    Args:
            tensor: Input tensor.
            axis: Axis to reduce (-1 for all axes).
            keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
            A new tensor with sum along specified axis.

    Examples:
        ```
            var t = ones([3, 4], DType.float32)
            var s = sum(t, axis=-1)  # Sum all elements -> scalar 12.0
            var row_sums = sum(t, axis=1)  # Sum along rows -> shape (3,)
        ```
    """
    if axis == -1:
        # Sum all elements
        var result_shape = List[Int](capacity=tensor.dim() if keepdims else 0)
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        _dispatch_reduce_all[SumOp](result, tensor, tensor.numel())
        return result^
    else:
        # Sum along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        # Use direct index computation (no coordinate allocations)
        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        _dispatch_reduce_axis[SumOp](
            result, tensor, outer_size, axis_size, inner_size
        )
        return result^


fn mean(
    tensor: ExTensor, axis: Int = -1, keepdims: Bool = False
) raises -> ExTensor:
    """Compute mean of tensor elements along an axis.

    Args:
            tensor: Input tensor.
            axis: Axis to reduce (-1 for all axes).
            keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
            A new tensor with mean along specified axis.

    Examples:
        ```
            var t = ones([3, 4], DType.float32)
            var m = mean(t)  # Mean of all elements -> scalar 1.0
        ```
    """
    if axis == -1:
        # Mean of all elements
        var result_shape = List[Int](capacity=tensor.dim() if keepdims else 0)
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        _dispatch_reduce_all[MeanOp](result, tensor, tensor.numel())
        return result^
    else:
        # Mean along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        _dispatch_reduce_axis[MeanOp](
            result, tensor, outer_size, axis_size, inner_size
        )
        return result^


fn max_reduce(
    tensor: ExTensor, axis: Int = -1, keepdims: Bool = False
) raises -> ExTensor:
    """Find maximum of tensor elements along an axis.

    Args:
            tensor: Input tensor.
            axis: Axis to reduce (-1 for all axes).
            keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
            A new tensor with maximum along specified axis.

    Examples:
        ```
            var t = arange(0.0, 12.0, 1.0, DType.float32)
            var m = max_reduce(t)  # Maximum element -> scalar 11.0
        ```
    """
    if axis == -1:
        # Max of all elements
        var result_shape = List[Int](capacity=tensor.dim() if keepdims else 0)
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        _dispatch_reduce_all[MaxOp](result, tensor, tensor.numel())
        return result^
    else:
        # Max along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        _dispatch_reduce_axis[MaxOp](
            result, tensor, outer_size, axis_size, inner_size
        )
        return result^


fn min_reduce(
    tensor: ExTensor, axis: Int = -1, keepdims: Bool = False
) raises -> ExTensor:
    """Find minimum of tensor elements along an axis.

    Args:
            tensor: Input tensor.
            axis: Axis to reduce (-1 for all axes).
            keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
            A new tensor with minimum along specified axis.

    Examples:
        ```
            var t = arange(0.0, 12.0, 1.0, DType.float32)
            var m = min_reduce(t)  # Minimum element -> scalar 0.0
        ```
    """
    if axis == -1:
        # Min of all elements
        var result_shape = List[Int](capacity=tensor.dim() if keepdims else 0)
        if keepdims:
            for _ in range(tensor.dim()):
                result_shape.append(1)
        var result = ExTensor(result_shape, tensor.dtype())

        _dispatch_reduce_all[MinOp](result, tensor, tensor.numel())
        return result^
    else:
        # Min along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        var result_shape = build_reduced_shape(tensor.shape(), axis, keepdims)
        var result = ExTensor(result_shape, tensor.dtype())
        result._fill_zero()

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        _dispatch_reduce_axis[MinOp](
            result, tensor, outer_size, axis_size, inner_size
        )
        return result^


# ============================================================================
# Generic Backward Pass (Gradient Computation)
# ============================================================================


fn reduce_backward[
    Op: ReduceBackwardOp
](grad_output: ExTensor, x: ExTensor, axis: Int = -1) raises -> ExTensor:
    """Generic backward pass for reduction operations.

    Consolidates the coordinate transformation logic used by all backward passes.
    The operation is determined by the Op template parameter.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Original input tensor before reduction.
        axis: Axis along which reduction was computed (-1 for all axes).

    Returns:
        Gradient w.r.t. input (∂L/∂X) - broadcast back to input_shape.
    """
    var input_shape = x.shape()
    var result = ExTensor(input_shape, grad_output.dtype())
    result._fill_zero()
    var op = Op()

    if axis == -1:
        # Reduction over all elements - collect all values for extremum ops
        var grad_val = grad_output._get_float64(0)
        var all_values = List[Float64]()
        for i in range(x.numel()):
            all_values.append(x._get_float64(i))

        for i in range(result.numel()):
            var input_val = x._get_float64(i)
            var grad = op.compute_gradient(
                grad_val, input_val, all_values, x.numel()
            )
            result._set_float64(i, grad)
    else:
        # Reduction along specific axis - shared coordinate transformation
        var ndim = len(input_shape)
        var normalized_axis = axis if axis >= 0 else ndim + axis

        # Validate axis
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(ndim)
                + " dimensions"
            )

        var strides = compute_strides(input_shape)
        var axis_size = input_shape[normalized_axis]

        for result_idx in range(x.numel()):
            # Convert linear index to coordinates
            var coords = linear_to_coords(result_idx, input_shape)

            # Map to grad_output coordinates (remove axis dimension)
            var grad_dim = grad_output.dim()
            var grad_coords = List[Int](capacity=grad_dim)
            for _ in range(grad_dim):
                grad_coords.append(0)
            var coord_idx = 0
            for i in range(ndim):
                if i != normalized_axis:
                    grad_coords[coord_idx] = coords[i]
                    coord_idx += 1

            # Convert to linear index in grad_output
            var grad_strides = compute_strides(grad_output.shape())
            var grad_idx = coords_to_linear(grad_coords, grad_strides)
            var grad_val = grad_output._get_float64(grad_idx)

            # Collect values along the reduction axis
            var axis_values = List[Float64]()
            for k in range(axis_size):
                var test_coords = List[Int](coords)
                test_coords[normalized_axis] = k
                var test_idx = coords_to_linear(test_coords, strides)
                axis_values.append(x._get_float64(test_idx))

            # Use Op-specific gradient computation
            var input_val = x._get_float64(result_idx)
            var grad = op.compute_gradient(
                grad_val, input_val, axis_values, axis_size
            )
            result._set_float64(result_idx, grad)

    return result^


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================


fn sum_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for sum reduction.

        For Y = sum(X, axis), given ∂L/∂Y, computes:
            ∂L/∂X = broadcast(∂L/∂Y, input_shape)

        The gradient broadcasts the reduced gradient back to the original input shape
        Each element of the input contributes equally to the sum, so gradient is 1

    Args:
            grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
            x: Original input tensor before reduction.
            axis: Axis along which sum was computed (-1 for all axes).

    Returns:
            Gradient w.r.t. input (∂L/∂X) - broadcast back to input_shape.

    Examples:
        ```
            # Sum all elements
            var x = ones([3, 4], DType.float32)
            var y = sum(x, axis=-1)  # Scalar
            var grad_y = ones(List[Int](), DType.float32)  # Scalar gradient
            var grad_x = sum_backward(grad_y, x, axis=-1)  # Shape (3, 4)

            # Sum along specific axis
            var x2 = ones([3, 4], DType.float32)
            var y2 = sum(x2, axis=1)  # Shape (3,)
            var grad_y2 = ones(List[Int](), DType.float32)
            var grad_x2 = sum_backward(grad_y2, x2, axis=1)  # Shape (3, 4)
        ```
    """
    return reduce_backward[SumBackwardOp](grad_output, x, axis)


fn mean_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for mean reduction.

        For Y = mean(X, axis), given ∂L/∂Y, computes:
            ∂L/∂X = broadcast(∂L/∂Y, input_shape) / N

        where N is the number of elements that were averaged

        Similar to sum_backward, but scaled by 1/N since each input element
        contributes 1/N to the mean

    Args:
            grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
            x: Original input tensor before reduction.
            axis: Axis along which mean was computed (-1 for all axes).

    Returns:
            Gradient w.r.t. input (∂L/∂X) - broadcast and scaled.

    Examples:
        ```
            var x = ones([3, 4], DType.float32)
            var y = mean(x, axis=-1)  # Scalar mean
            var grad_y = ones(List[Int](), DType.float32)
            var grad_x = mean_backward(grad_y, x, axis=-1)
            # Each element gets gradient / 12
        ```
    """
    return reduce_backward[MeanBackwardOp](grad_output, x, axis)


fn max_reduce_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for max reduction.

        For Y = max_reduce(X, axis), given ∂L/∂Y, computes:
            ∂L/∂X - Gradient flows only to maximum element(s)

        If multiple elements are maximum, gradient is split equally among them
        This is the standard behavior for max pooling backward pass

    Args:
            grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
            x: Input from forward pass (before reduction).
            axis: Axis along which max was computed (-1 for all axes).

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
        ```
            # Max over all elements
            var x = tensor([1.0, 3.0, 2.0, 3.0])  # Two max values at indices 1, 3
            var y = max_reduce(x, axis=-1)  # Scalar: 3.0
            var grad_y = ones([])  # Gradient: 1.0
            var grad_x = max_reduce_backward(grad_y, x, axis=-1)
            # grad_x = [0.0, 0.5, 0.0, 0.5]  # Split equally between the two 3.0s

            # Max along axis
            var x2 = tensor([[1.0, 3.0], [2.0, 1.0]])
            var y2 = max_reduce(x2, axis=1)  # [3.0, 2.0]
            var grad_y2 = ones([2])
            var grad_x2 = max_reduce_backward(grad_y2, x2, axis=1)
            # grad_x2 = [[0.0, 1.0], [1.0, 0.0]]
        ```
    """
    return reduce_backward[MaxBackwardOp](grad_output, x, axis)


fn min_reduce_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for min reduction.

        For Y = min_reduce(X, axis), given ∂L/∂Y, computes:
            ∂L/∂X - Gradient flows only to minimum element(s)

        If multiple elements are minimum, gradient is split equally among them
        This is analogous to max pooling but for minimum values

    Args:
            grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
            x: Input from forward pass (before reduction).
            axis: Axis along which min was computed (-1 for all axes).

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
        ```
            var x = tensor([3.0, 1.0, 2.0, 1.0])  # Two min values at indices 1, 3
            var y = min_reduce(x, axis=-1)  # Scalar: 1.0
            var grad_y = ones([])  # Gradient: 1.0
            var grad_x = min_reduce_backward(grad_y, x, axis=-1)
            # grad_x = [0.0, 0.5, 0.0, 0.5]  # Split equally between the two 1.0s
        ```
    """
    return reduce_backward[MinBackwardOp](grad_output, x, axis)


# ============================================================================
# Statistical Reduction Operations
# ============================================================================


fn variance(tensor: ExTensor, axis: Int = -1, ddof: Int = 0) raises -> ExTensor:
    """Compute variance of tensor elements along an axis.

    Variance measures how spread out values are from the mean.
    Formula: var = sum((x - mean)^2) / (N - ddof)

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        ddof: Delta degrees of freedom (0=population variance, 1=sample variance).

    Returns:
        A new tensor with variance along specified axis.

    Examples:
        ```
            var t = tensor([1.0, 2.0, 3.0])
            var v = variance(t, axis=-1, ddof=0)  # Population variance: 2/3
            var s = variance(t, axis=-1, ddof=1)  # Sample variance: 1.0
        ```
    """
    # Compute mean using existing mean() function
    var mu = mean(tensor, axis)

    if axis == -1:
        # Variance of all elements
        var result_shape = List[Int]()
        var result = ExTensor(result_shape, tensor.dtype())

        var mean_val = mu._get_float64(0)
        var N = tensor.numel()
        var sum_sq_diff = 0.0

        for i in range(N):
            var diff = tensor._get_float64(i) - mean_val
            sum_sq_diff += diff * diff

        var var_val = sum_sq_diff / Float64(N - ddof)
        result._set_float64(0, var_val)
        return result^
    else:
        # Variance along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        var result_shape = build_reduced_shape(tensor.shape(), axis, False)
        var result = ExTensor(result_shape, tensor.dtype())

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        for outer in range(outer_size):
            for inner in range(inner_size):
                # Get mean for this slice
                var mean_idx = outer * inner_size + inner
                var mean_val = mu._get_float64(mean_idx)

                # Compute sum of squared differences
                var sum_sq_diff = 0.0
                for k in range(axis_size):
                    var input_idx = (
                        outer * axis_size * inner_size + k * inner_size + inner
                    )
                    var diff = tensor._get_float64(input_idx) - mean_val
                    sum_sq_diff += diff * diff

                var var_val = sum_sq_diff / Float64(axis_size - ddof)
                result._set_float64(mean_idx, var_val)

        return result^


fn variance_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1, ddof: Int = 0
) raises -> ExTensor:
    """Compute gradient for variance reduction.

    For Y = variance(X, axis, ddof), given ∂L/∂Y, computes:
        ∂L/∂X = 2 * (x - mean) / (N - ddof) * ∂L/∂Y

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Original input tensor before reduction.
        axis: Axis along which variance was computed (-1 for all axes).
        ddof: Delta degrees of freedom used in variance.

    Returns:
        Gradient w.r.t. input (∂L/∂X).
    """
    var input_shape = x.shape()
    var result = ExTensor(input_shape, grad_output.dtype())

    # Compute mean (needed for gradient)
    var mu = mean(x, axis)

    if axis == -1:
        # All elements reduced
        var mean_val = mu._get_float64(0)
        var grad_val = grad_output._get_float64(0)
        var N = x.numel()
        var scale = 2.0 / Float64(N - ddof)

        for i in range(x.numel()):
            var diff = x._get_float64(i) - mean_val
            var grad = diff * scale * grad_val
            result._set_float64(i, grad)
    else:
        # Per-axis gradient computation
        var sizes = compute_axis_strides(input_shape, axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]
        var scale = 2.0 / Float64(axis_size - ddof)

        for outer in range(outer_size):
            for inner in range(inner_size):
                var grad_idx = outer * inner_size + inner
                var mean_val = mu._get_float64(grad_idx)
                var grad_val = grad_output._get_float64(grad_idx)

                for k in range(axis_size):
                    var input_idx = (
                        outer * axis_size * inner_size + k * inner_size + inner
                    )
                    var diff = x._get_float64(input_idx) - mean_val
                    var grad = diff * scale * grad_val
                    result._set_float64(input_idx, grad)

    return result^


fn std(tensor: ExTensor, axis: Int = -1, ddof: Int = 0) raises -> ExTensor:
    """Compute standard deviation of tensor elements along an axis.

    Standard deviation is the square root of variance.
    Formula: std = sqrt(variance)

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        ddof: Delta degrees of freedom (0=population, 1=sample).

    Returns:
        A new tensor with standard deviation along specified axis.

    Examples:
        ```
            var t = tensor([1.0, 2.0, 3.0])
            var s = std(t, axis=-1, ddof=0)  # sqrt(2/3) ≈ 0.8165
        ```
    """
    from math import sqrt

    var var_result = variance(tensor, axis, ddof)

    # Apply sqrt element-wise
    for i in range(var_result.numel()):
        var val = var_result._get_float64(i)
        var std_val = sqrt(val) if val >= 0.0 else 0.0
        var_result._set_float64(i, std_val)

    return var_result^


fn std_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1, ddof: Int = 0
) raises -> ExTensor:
    """Compute gradient for std reduction.

    For Y = std(X, axis, ddof), given ∂L/∂Y, computes:
        ∂L/∂X = (x - mean) / ((N - ddof) * std) * ∂L/∂Y

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Original input tensor before reduction.
        axis: Axis along which std was computed (-1 for all axes).
        ddof: Delta degrees of freedom used in std.

    Returns:
        Gradient w.r.t. input (∂L/∂X).
    """
    var input_shape = x.shape()
    var result = ExTensor(input_shape, grad_output.dtype())
    var mu = mean(x, axis)
    var sigma = std(x, axis, ddof)

    comptime EPSILON = 1e-8  # Prevent division by zero

    if axis == -1:
        var mean_val = mu._get_float64(0)
        var std_val = sigma._get_float64(0)
        var grad_val = grad_output._get_float64(0)
        var N = x.numel()
        var denom = Float64(N - ddof) * (std_val + EPSILON)

        for i in range(x.numel()):
            var diff = x._get_float64(i) - mean_val
            var grad = (diff / denom) * grad_val
            result._set_float64(i, grad)
    else:
        var sizes = compute_axis_strides(input_shape, axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        for outer in range(outer_size):
            for inner in range(inner_size):
                var grad_idx = outer * inner_size + inner
                var mean_val = mu._get_float64(grad_idx)
                var std_val = sigma._get_float64(grad_idx)
                var grad_val = grad_output._get_float64(grad_idx)
                var denom = Float64(axis_size - ddof) * (std_val + EPSILON)

                for k in range(axis_size):
                    var input_idx = (
                        outer * axis_size * inner_size + k * inner_size + inner
                    )
                    var diff = x._get_float64(input_idx) - mean_val
                    var grad = (diff / denom) * grad_val
                    result._set_float64(input_idx, grad)

    return result^


fn median(tensor: ExTensor, axis: Int = -1) raises -> ExTensor:
    """Compute median of tensor elements along an axis.

    For odd count: returns the middle value after sorting.
    For even count: returns the average of the two middle values.

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).

    Returns:
        A new tensor with median along specified axis.

    Examples:
        ```
            var t = tensor([3.0, 1.0, 4.0, 2.0, 5.0])
            var m = median(t, axis=-1)  # 3.0 (middle of sorted)
        ```
    """
    if axis == -1:
        # Median of all elements
        var result_shape = List[Int]()
        var result = ExTensor(result_shape, tensor.dtype())

        # Collect all values
        var N = tensor.numel()
        var values = List[Float64]()
        for i in range(N):
            values.append(tensor._get_float64(i))

        # Sort using bubble sort
        for i in range(N):
            for j in range(0, N - i - 1):
                if values[j] > values[j + 1]:
                    var temp = values[j]
                    values[j] = values[j + 1]
                    values[j + 1] = temp

        var median_val: Float64
        if N % 2 == 1:
            median_val = values[N // 2]
        else:
            median_val = (values[N // 2 - 1] + values[N // 2]) / 2.0

        result._set_float64(0, median_val)
        return result^
    else:
        # Median along specific axis
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        var result_shape = build_reduced_shape(tensor.shape(), axis, False)
        var result = ExTensor(result_shape, tensor.dtype())

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        for outer in range(outer_size):
            for inner in range(inner_size):
                # Collect values along axis
                var values = List[Float64]()
                for k in range(axis_size):
                    var input_idx = (
                        outer * axis_size * inner_size + k * inner_size + inner
                    )
                    values.append(tensor._get_float64(input_idx))

                # Sort
                for i in range(axis_size):
                    for j in range(0, axis_size - i - 1):
                        if values[j] > values[j + 1]:
                            var temp = values[j]
                            values[j] = values[j + 1]
                            values[j + 1] = temp

                var median_val: Float64
                if axis_size % 2 == 1:
                    median_val = values[axis_size // 2]
                else:
                    median_val = (
                        values[axis_size // 2 - 1] + values[axis_size // 2]
                    ) / 2.0

                var result_idx = outer * inner_size + inner
                result._set_float64(result_idx, median_val)

        return result^


fn median_backward(
    grad_output: ExTensor, x: ExTensor, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for median (subgradient).

    Gradient flows only to the median element(s).
    For even count, gradient is split between two middle elements.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Original input tensor before reduction.
        axis: Axis along which median was computed (-1 for all axes).

    Returns:
        Gradient w.r.t. input (∂L/∂X).
    """
    var input_shape = x.shape()
    var result = ExTensor(input_shape, x.dtype())
    result._fill_zero()  # Initialize all gradients to 0

    if axis == -1:
        var grad_val = grad_output._get_float64(0)
        var N = x.numel()

        # Collect values with their indices and sort
        var values = List[Float64]()
        var indices = List[Int]()
        for i in range(N):
            values.append(x._get_float64(i))
            indices.append(i)

        # Sort by value (bubble sort, keeping indices aligned)
        for i in range(N):
            for j in range(0, N - i - 1):
                if values[j] > values[j + 1]:
                    var temp_val = values[j]
                    values[j] = values[j + 1]
                    values[j + 1] = temp_val
                    var temp_idx = indices[j]
                    indices[j] = indices[j + 1]
                    indices[j + 1] = temp_idx

        if N % 2 == 1:
            # Odd count: gradient to single median element
            var mid_input_idx = indices[N // 2]
            result._set_float64(mid_input_idx, grad_val)
        else:
            # Even count: split gradient between two middle elements
            var lower_input_idx = indices[N // 2 - 1]
            var upper_input_idx = indices[N // 2]
            result._set_float64(lower_input_idx, grad_val / 2.0)
            result._set_float64(upper_input_idx, grad_val / 2.0)
    else:
        # Per-axis backward
        var sizes = compute_axis_strides(input_shape, axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        for outer in range(outer_size):
            for inner in range(inner_size):
                var grad_idx = outer * inner_size + inner
                var grad_val = grad_output._get_float64(grad_idx)

                # Collect values with their indices along axis
                var values = List[Float64]()
                var indices = List[Int]()
                for k in range(axis_size):
                    var input_idx = (
                        outer * axis_size * inner_size + k * inner_size + inner
                    )
                    values.append(x._get_float64(input_idx))
                    indices.append(input_idx)

                # Sort by value
                for i in range(axis_size):
                    for j in range(0, axis_size - i - 1):
                        if values[j] > values[j + 1]:
                            var temp_val = values[j]
                            values[j] = values[j + 1]
                            values[j + 1] = temp_val
                            var temp_idx = indices[j]
                            indices[j] = indices[j + 1]
                            indices[j + 1] = temp_idx

                if axis_size % 2 == 1:
                    var mid_input_idx = indices[axis_size // 2]
                    result._set_float64(mid_input_idx, grad_val)
                else:
                    var lower_input_idx = indices[axis_size // 2 - 1]
                    var upper_input_idx = indices[axis_size // 2]
                    result._set_float64(lower_input_idx, grad_val / 2.0)
                    result._set_float64(upper_input_idx, grad_val / 2.0)

    return result^


fn percentile(tensor: ExTensor, q: Float64, axis: Int = -1) raises -> ExTensor:
    """Compute percentile of tensor elements along an axis.

    Uses linear interpolation between adjacent ranked values.
    q=0 returns minimum, q=50 returns median, q=100 returns maximum.

    Args:
        tensor: Input tensor.
        q: Percentile to compute (must be in range [0, 100]).
        axis: Axis to reduce (-1 for all axes).

    Returns:
        A new tensor with percentile along specified axis.

    Examples:
        ```
            var t = tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            var p50 = percentile(t, 50.0, axis=-1)  # 3.0 (median)
            var p0 = percentile(t, 0.0, axis=-1)    # 1.0 (min)
            var p100 = percentile(t, 100.0, axis=-1) # 5.0 (max)
        ```
    """
    if q < 0.0 or q > 100.0:
        raise Error("percentile: q must be in range [0, 100], got " + String(q))

    if axis == -1:
        var result_shape = List[Int]()
        var result = ExTensor(result_shape, tensor.dtype())

        var N = tensor.numel()
        var values = List[Float64]()
        for i in range(N):
            values.append(tensor._get_float64(i))

        # Sort
        for i in range(N):
            for j in range(0, N - i - 1):
                if values[j] > values[j + 1]:
                    var temp = values[j]
                    values[j] = values[j + 1]
                    values[j + 1] = temp

        # Linear interpolation
        var position = Float64(N - 1) * q / 100.0
        var lower_idx = Int(position)
        var upper_idx = lower_idx + 1
        if upper_idx >= N:
            upper_idx = N - 1
        var fraction = position - Float64(lower_idx)

        var pct_val = (
            values[lower_idx] * (1.0 - fraction) + values[upper_idx] * fraction
        )
        result._set_float64(0, pct_val)
        return result^
    else:
        if axis < 0 or axis >= tensor.dim():
            raise Error(
                "Axis "
                + String(axis)
                + " is out of bounds for tensor with "
                + String(tensor.dim())
                + " dimensions"
            )

        var result_shape = build_reduced_shape(tensor.shape(), axis, False)
        var result = ExTensor(result_shape, tensor.dtype())

        var sizes = compute_axis_strides(tensor.shape(), axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        for outer in range(outer_size):
            for inner in range(inner_size):
                var values = List[Float64]()
                for k in range(axis_size):
                    var input_idx = (
                        outer * axis_size * inner_size + k * inner_size + inner
                    )
                    values.append(tensor._get_float64(input_idx))

                # Sort
                for i in range(axis_size):
                    for j in range(0, axis_size - i - 1):
                        if values[j] > values[j + 1]:
                            var temp = values[j]
                            values[j] = values[j + 1]
                            values[j + 1] = temp

                var position = Float64(axis_size - 1) * q / 100.0
                var lower_idx = Int(position)
                var upper_idx = lower_idx + 1
                if upper_idx >= axis_size:
                    upper_idx = axis_size - 1
                var fraction = position - Float64(lower_idx)

                var pct_val = (
                    values[lower_idx] * (1.0 - fraction)
                    + values[upper_idx] * fraction
                )
                var result_idx = outer * inner_size + inner
                result._set_float64(result_idx, pct_val)

        return result^


fn percentile_backward(
    grad_output: ExTensor, x: ExTensor, q: Float64, axis: Int = -1
) raises -> ExTensor:
    """Compute gradient for percentile.

    Distributes gradient to the interpolated elements proportionally.

    Args:
        grad_output: Gradient from upstream (∂L/∂Y) - reduced tensor.
        x: Original input tensor before reduction.
        q: Percentile that was computed.
        axis: Axis along which percentile was computed (-1 for all axes).

    Returns:
        Gradient w.r.t. input (∂L/∂X).
    """
    var input_shape = x.shape()
    var result = ExTensor(input_shape, x.dtype())
    result._fill_zero()

    if axis == -1:
        var grad_val = grad_output._get_float64(0)
        var N = x.numel()

        # Sort with indices
        var values = List[Float64]()
        var indices = List[Int]()
        for i in range(N):
            values.append(x._get_float64(i))
            indices.append(i)

        for i in range(N):
            for j in range(0, N - i - 1):
                if values[j] > values[j + 1]:
                    var temp_val = values[j]
                    values[j] = values[j + 1]
                    values[j + 1] = temp_val
                    var temp_idx = indices[j]
                    indices[j] = indices[j + 1]
                    indices[j + 1] = temp_idx

        var position = Float64(N - 1) * q / 100.0
        var lower_sorted_idx = Int(position)
        var upper_sorted_idx = lower_sorted_idx + 1
        if upper_sorted_idx >= N:
            upper_sorted_idx = N - 1
        var fraction = position - Float64(lower_sorted_idx)

        var lower_input_idx = indices[lower_sorted_idx]
        var upper_input_idx = indices[upper_sorted_idx]

        result._set_float64(lower_input_idx, (1.0 - fraction) * grad_val)
        result._set_float64(upper_input_idx, fraction * grad_val)
    else:
        var sizes = compute_axis_strides(input_shape, axis)
        var outer_size = sizes[0]
        var axis_size = sizes[1]
        var inner_size = sizes[2]

        for outer in range(outer_size):
            for inner in range(inner_size):
                var grad_idx = outer * inner_size + inner
                var grad_val = grad_output._get_float64(grad_idx)

                var values = List[Float64]()
                var indices = List[Int]()
                for k in range(axis_size):
                    var input_idx = (
                        outer * axis_size * inner_size + k * inner_size + inner
                    )
                    values.append(x._get_float64(input_idx))
                    indices.append(input_idx)

                for i in range(axis_size):
                    for j in range(0, axis_size - i - 1):
                        if values[j] > values[j + 1]:
                            var temp_val = values[j]
                            values[j] = values[j + 1]
                            values[j + 1] = temp_val
                            var temp_idx = indices[j]
                            indices[j] = indices[j + 1]
                            indices[j + 1] = temp_idx

                var position = Float64(axis_size - 1) * q / 100.0
                var lower_sorted_idx = Int(position)
                var upper_sorted_idx = lower_sorted_idx + 1
                if upper_sorted_idx >= axis_size:
                    upper_sorted_idx = axis_size - 1
                var fraction = position - Float64(lower_sorted_idx)

                var lower_input_idx = indices[lower_sorted_idx]
                var upper_input_idx = indices[upper_sorted_idx]

                result._set_float64(
                    lower_input_idx, (1.0 - fraction) * grad_val
                )
                result._set_float64(upper_input_idx, fraction * grad_val)

    return result^


# ============================================================================
# Tensor[dtype] typed overloads (wrapping AnyTensor implementations)
# ============================================================================


fn sum[dt: DType](
    tensor: Tensor[dt], axis: Int = -1, keepdims: Bool = False
) raises -> Tensor[dt]:
    """Sum tensor elements along an axis (typed version).

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        A new Tensor[dt] with sum along specified axis.
    """
    return from_any_tensor[dt](sum(tensor.as_any(), axis, keepdims))


fn mean[dt: DType](
    tensor: Tensor[dt], axis: Int = -1, keepdims: Bool = False
) raises -> Tensor[dt]:
    """Compute mean of tensor elements along an axis (typed version).

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        A new Tensor[dt] with mean along specified axis.
    """
    return from_any_tensor[dt](mean(tensor.as_any(), axis, keepdims))


fn max_reduce[dt: DType](
    tensor: Tensor[dt], axis: Int = -1, keepdims: Bool = False
) raises -> Tensor[dt]:
    """Find maximum of tensor elements along an axis (typed version).

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        A new Tensor[dt] with max along specified axis.
    """
    return from_any_tensor[dt](
        max_reduce(tensor.as_any(), axis, keepdims)
    )


fn min_reduce[dt: DType](
    tensor: Tensor[dt], axis: Int = -1, keepdims: Bool = False
) raises -> Tensor[dt]:
    """Find minimum of tensor elements along an axis (typed version).

    Args:
        tensor: Input tensor.
        axis: Axis to reduce (-1 for all axes).
        keepdims: Whether to keep reduced dimensions as size 1.

    Returns:
        A new Tensor[dt] with min along specified axis.
    """
    return from_any_tensor[dt](
        min_reduce(tensor.as_any(), axis, keepdims)
    )
