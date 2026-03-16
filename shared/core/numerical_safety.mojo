"""Numerical safety utilities for detecting and preventing numerical instabilities.

This module provides tools for:
- NaN/Inf detection in tensors
- Gradient explosion/vanishing detection
- Numerical range checking
- Compile-time optional safety checks (zero overhead when disabled)

All safety checks use @parameter for compile-time enable/disable, ensuring zero
runtime overhead when safety mode is disabled.

Example:
    ```mojo
    from shared.core import ExTensor, check_tensor_safety

    # Enable safety checks at compile time
    var output = model_forward(x)
    check_tensor_safety[enable=True](output, "model_output")

    # Disabled by default (zero overhead)
    check_tensor_safety(output)  # Compiles to nothing
    ```
"""

from .extensor import ExTensor
from math import isnan, isinf, sqrt
from collections import List


fn has_nan(tensor: ExTensor) -> Bool:
    """Check if tensor contains any NaN values.

    Args:
            tensor: Input tensor to check.

    Returns:
            True if any element is NaN, False otherwise.

        Example:
            ```mojo
            var x = ExTensor([[1.0, float("nan"), 3.0]])
            assert_true(has_nan(x))
            ```

    Note:
            Checks all elements regardless of dtype. Supports all floating-point types.
    """
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            if isnan(ptr[i]):
                return True
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            if isnan(ptr[i]):
                return True
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            if isnan(Float32(ptr[i])):  # Convert to float32 for isnan check
                return True
    # Integer types cannot have NaN
    return False


fn has_inf(tensor: ExTensor) -> Bool:
    """Check if tensor contains any Inf values (positive or negative).

    Args:
            tensor: Input tensor to check.

    Returns:
            True if any element is Inf or -Inf, False otherwise.

        Example:
            ```mojo
            var x = ExTensor([[1.0, float("inf"), 3.0]])
            assert_true(has_inf(x))
            ```

    Note:
            Checks all elements regardless of dtype. Supports all floating-point types.
    """
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            if isinf(ptr[i]):
                return True
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            if isinf(ptr[i]):
                return True
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            if isinf(Float32(ptr[i])):
                return True
    # Integer types cannot have Inf
    return False


fn count_nan(tensor: ExTensor) -> Int:
    """Count number of NaN values in tensor.

    Args:
            tensor: Input tensor to check.

    Returns:
            Number of NaN elements.

        Example:
            ```mojo
            var x = ExTensor([[1.0, float("nan"), float("nan")]])
            assert_equal(count_nan(x), 2)
            ```
    """
    var size = tensor.numel()
    var count = 0

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            if isnan(ptr[i]):
                count += 1
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            if isnan(ptr[i]):
                count += 1
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            if isnan(Float32(ptr[i])):
                count += 1

    return count


fn count_inf(tensor: ExTensor) -> Int:
    """Count number of Inf values in tensor.

    Args:
            tensor: Input tensor to check.

    Returns:
            Number of Inf/-Inf elements.

        Example:
            ```mojo
            var x = ExTensor([[1.0, float("inf"), float("-inf")]])
            assert_equal(count_inf(x), 2)
            ```
    """
    var size = tensor.numel()
    var count = 0

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            if isinf(ptr[i]):
                count += 1
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            if isinf(ptr[i]):
                count += 1
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            if isinf(Float32(ptr[i])):
                count += 1

    return count


@parameter
fn check_tensor_safety[
    enable: Bool = False
](tensor: ExTensor, name: String = "tensor") raises:
    """Check tensor for NaN/Inf values with compile-time optional behavior.

        When enable=True, raises Error if NaN or Inf found.
        When enable=False, compiles to nothing (zero overhead).

    Args:
            tensor: Tensor to check.
            name: Name for error message (default: "tensor").

    Raises:
            Error: If tensor contains NaN or Inf values (only when enable=True).

        Example:
            ```mojo
            # Production mode: safety disabled (zero overhead)
            check_tensor_safety(output)  # Compiles to nothing.

            # Debug mode: safety enabled
            check_tensor_safety[enable=True](output, "linear_output")
            # Raises if output contains NaN/Inf
            ```

    Note:
            Use @parameter to enable/disable at compile time for zero runtime cost.
    """

    @parameter
    if enable:
        if has_nan(tensor):
            var count = count_nan(tensor)
            raise Error(name + " contains " + String(count) + " NaN values")
        if has_inf(tensor):
            var count = count_inf(tensor)
            raise Error(name + " contains " + String(count) + " Inf values")
    # If enable=False, this entire function body is eliminated at compile time


fn tensor_min(tensor: ExTensor) -> Float64:
    """Find minimum value in tensor.

    Args:
            tensor: Input tensor.

    Returns:
            Minimum value as Float64.

        Example:
            ```mojo
            var x = ExTensor([[1.0, -5.0, 3.0]])
            assert_equal(tensor_min(x), -5.0)
            ```
    """
    var size = tensor.numel()
    if size == 0:
        return 0.0

    var min_val = Float64(1e308)  # Very large positive number

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            var val = Float64(ptr[i])
            if val < min_val:
                min_val = val
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            var val = ptr[i]
            if val < min_val:
                min_val = val
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            var val = Float64(Float32(ptr[i]))
            if val < min_val:
                min_val = val

    return min_val


fn tensor_max(tensor: ExTensor) -> Float64:
    """Find maximum value in tensor.

    Args:
            tensor: Input tensor.

    Returns:
            Maximum value as Float64.

        Example:
            ```mojo
            var x = ExTensor([[1.0, 10.0, 3.0]])
            assert_equal(tensor_max(x), 10.0)
            ```
    """
    var size = tensor.numel()
    if size == 0:
        return 0.0

    var max_val = Float64(-1e308)  # Very large negative number

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            var val = Float64(ptr[i])
            if val > max_val:
                max_val = val
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            var val = ptr[i]
            if val > max_val:
                max_val = val
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            var val = Float64(Float32(ptr[i]))
            if val > max_val:
                max_val = val

    return max_val


fn check_tensor_range(
    tensor: ExTensor,
    min_val: Float64,
    max_val: Float64,
    name: String = "tensor",
) raises:
    """Check if all tensor values are within [min_val, max_val].

    Args:
            tensor: Tensor to check.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.
            name: Name for error message.

    Raises:
            Error: If any value is outside the range.

        Example:
            ```mojo
            var probs = sigmoid(logits)
            check_tensor_range(probs, 0.0, 1.0, "probabilities")
            # Raises if probs contains values outside [0, 1]
            ```
    """
    var t_min = tensor_min(tensor)
    var t_max = tensor_max(tensor)

    if t_min < min_val or t_max > max_val:
        raise Error(
            name
            + " values out of range: ["
            + String(t_min)
            + ", "
            + String(t_max)
            + "], expected ["
            + String(min_val)
            + ", "
            + String(max_val)
            + "]"
        )


fn compute_tensor_l2_norm(tensor: ExTensor) -> Float64:
    """Compute L2 norm of tensor: sqrt(sum(x^2)).

    Args:
            tensor: Input tensor.

    Returns:
            L2 norm as Float64.

        Example:
            ```mojo
            var x = ExTensor([[3.0, 4.0]])
            assert_equal(compute_tensor_l2_norm(x), 5.0)  # sqrt(9 + 16)
            ```
    """
    var size = tensor.numel()
    var sum_sq = Float64(0.0)

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            var val = Float64(ptr[i])
            sum_sq += val * val
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            var val = ptr[i]
            sum_sq += val * val
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            var val = Float64(Float32(ptr[i]))
            sum_sq += val * val

    return sqrt(sum_sq)


fn check_gradient_norm(
    gradient: ExTensor, max_norm: Float64 = 1000.0, name: String = "gradient"
) raises:
    """Check if gradient L2 norm exceeds threshold (gradient explosion detection).

    Args:
            gradient: Gradient tensor to check.
            max_norm: Maximum allowed L2 norm.
            name: Name for error message.

    Raises:
            Error: If gradient norm exceeds max_norm.

        Example:
            ```mojo
            var grad_w = linear_backward(grad_out, x, w)[1]
            check_gradient_norm(grad_w, max_norm=100.0, "weight_gradient")
            # Raises if ||grad_w||_2 > 100
            ```

    Note:
            Use this to detect gradient explosion during training.
            Common thresholds: 10.0 (strict), 100.0 (moderate), 1000.0 (lenient).
    """
    var norm = compute_tensor_l2_norm(gradient)

    if norm > max_norm:
        raise Error(
            name
            + " norm too large: "
            + String(norm)
            + " > max_norm="
            + String(max_norm)
            + " (gradient explosion)"
        )


fn check_gradient_vanishing(
    gradient: ExTensor, min_norm: Float64 = 1e-7, name: String = "gradient"
) raises:
    """Check if gradient L2 norm is too small (gradient vanishing detection).

    Args:
            gradient: Gradient tensor to check.
            min_norm: Minimum expected L2 norm.
            name: Name for error message.

    Raises:
            Error: If gradient norm is below min_norm.

        Example:
            ```mojo
            var grad_w = linear_backward(grad_out, x, w)[1]
            check_gradient_vanishing(grad_w, min_norm=1e-6, "weight_gradient")
            # Raises if ||grad_w||_2 < 1e-6
            ```

    Note:
            Use this to detect gradient vanishing in deep networks.
            Common thresholds: 1e-7 (lenient), 1e-5 (moderate), 1e-3 (strict).
    """
    var norm = compute_tensor_l2_norm(gradient)

    if norm < min_norm:
        raise Error(
            name
            + " norm too small: "
            + String(norm)
            + " < min_norm="
            + String(min_norm)
            + " (gradient vanishing)"
        )


@parameter
fn check_gradient_safety[
    enable: Bool = False
](
    gradient: ExTensor,
    max_norm: Float64 = 1000.0,
    min_norm: Float64 = 1e-7,
    name: String = "gradient",
) raises:
    """Combined gradient safety check with compile-time optional behavior.

        Checks for:
        - NaN/Inf values.
        - Gradient explosion (norm > max_norm).
        - Gradient vanishing (norm < min_norm).

    Args:
            gradient: Gradient tensor to check.
            max_norm: Maximum allowed L2 norm.
            min_norm: Minimum expected L2 norm.
            name: Name for error message.

    Raises:
            Error: If any safety check fails (only when enable=True).

        Example:
            ```mojo
            # Debug mode: full gradient safety
            var grad = backward_pass(loss)
            check_gradient_safety[enable=True](grad, max_norm=100.0)

            # Production mode: disabled (zero overhead)
            check_gradient_safety(grad)  # Compiles to nothing
            ```
    """

    @parameter
    if enable:
        # Check NaN/Inf
        check_tensor_safety[enable=True](gradient, name)

        # Check gradient explosion
        check_gradient_norm(gradient, max_norm, name)

        # Check gradient vanishing
        check_gradient_vanishing(gradient, min_norm, name)
