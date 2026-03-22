"""Numerical safety utilities with native Tensor[dtype] implementations.

This module provides tools for:
- NaN/Inf detection in tensors
- Gradient explosion/vanishing detection
- Numerical range checking
- Compile-time optional safety checks (zero overhead when disabled)

Architecture: Tensor[dtype] typed implementations are the core.
AnyTensor versions dispatch to typed implementations via ordinal-based table.

All safety checks use @parameter for compile-time enable/disable, ensuring zero
runtime overhead when safety mode is disabled.

Example:
    ```mojo
    from shared.core import AnyTensor, check_tensor_safety

    # Enable safety checks at compile time
    var output = model_forward(x)
    check_tensor_safety[enable=True](output, "model_output")

    # Disabled by default (zero overhead)
    check_tensor_safety(output)  # Compiles to nothing
    ```
"""

from .any_tensor import AnyTensor
from math import isnan, isinf, sqrt
from collections import List
from shared.tensor.tensor import Tensor
from shared.base.dtype_ordinal import (
    dtype_to_ordinal,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
)


# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] implementations
# ============================================================================


fn _has_nan_core[dtype: DType](tensor: Tensor[dtype]) -> Bool:
    """Check if typed tensor contains any NaN values (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor to check.

    Returns:
        True if any element is NaN, False otherwise.
    """
    # Integer/unsigned types cannot have NaN
    @parameter
    if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return False

    var size = tensor.numel()
    var ptr = tensor._data
    for i in range(size):
        @parameter
        if dtype == DType.float16:
            if isnan(Float32(ptr[i])):
                return True
        else:
            if isnan(ptr[i]):
                return True
    return False


fn _has_inf_core[dtype: DType](tensor: Tensor[dtype]) -> Bool:
    """Check if typed tensor contains any Inf values (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor to check.

    Returns:
        True if any element is Inf or -Inf, False otherwise.
    """
    @parameter
    if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return False

    var size = tensor.numel()
    var ptr = tensor._data
    for i in range(size):
        @parameter
        if dtype == DType.float16:
            if isinf(Float32(ptr[i])):
                return True
        else:
            if isinf(ptr[i]):
                return True
    return False


fn _count_nan_core[dtype: DType](tensor: Tensor[dtype]) -> Int:
    """Count NaN values in typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Number of NaN elements.
    """
    @parameter
    if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return 0

    var size = tensor.numel()
    var count = 0
    var ptr = tensor._data
    for i in range(size):
        @parameter
        if dtype == DType.float16:
            if isnan(Float32(ptr[i])):
                count += 1
        else:
            if isnan(ptr[i]):
                count += 1
    return count


fn _count_inf_core[dtype: DType](tensor: Tensor[dtype]) -> Int:
    """Count Inf values in typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Number of Inf/-Inf elements.
    """
    @parameter
    if dtype == DType.int8 or dtype == DType.int16 or dtype == DType.int32 or dtype == DType.int64 or dtype == DType.uint8 or dtype == DType.uint16 or dtype == DType.uint32 or dtype == DType.uint64 or dtype == DType.bool:
        return 0

    var size = tensor.numel()
    var count = 0
    var ptr = tensor._data
    for i in range(size):
        @parameter
        if dtype == DType.float16:
            if isinf(Float32(ptr[i])):
                count += 1
        else:
            if isinf(ptr[i]):
                count += 1
    return count


fn _tensor_min_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Find minimum value in typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Minimum value as Float64.
    """
    var size = tensor.numel()
    if size == 0:
        return 0.0

    var min_val = Float64(1e308)
    var ptr = tensor._data
    for i in range(size):
        var val = Float64(ptr[i])
        if val < min_val:
            min_val = val
    return min_val


fn _tensor_max_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Find maximum value in typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        Maximum value as Float64.
    """
    var size = tensor.numel()
    if size == 0:
        return 0.0

    var max_val = Float64(-1e308)
    var ptr = tensor._data
    for i in range(size):
        var val = Float64(ptr[i])
        if val > max_val:
            max_val = val
    return max_val


fn _compute_l2_norm_core[dtype: DType](tensor: Tensor[dtype]) -> Float64:
    """Compute L2 norm of typed tensor (core implementation).

    Parameters:
        dtype: Compile-time dtype parameter.

    Args:
        tensor: Input typed tensor.

    Returns:
        L2 norm as Float64.
    """
    var size = tensor.numel()
    var sum_sq = Float64(0.0)
    var ptr = tensor._data
    for i in range(size):
        var val = Float64(ptr[i])
        sum_sq += val * val
    return sqrt(sum_sq)


# ============================================================================
# Layer 2: AnyTensor dispatch (ordinal-based)
# ============================================================================
# For Bool-returning functions, we dispatch via ordinal to typed cores.
# Only float types are relevant for NaN/Inf checks but we dispatch all
# for consistency (integer types return False/0 via compile-time guard).


fn has_nan(tensor: AnyTensor) -> Bool:
    """Check if tensor contains any NaN values.

    Args:
            tensor: Input tensor to check.

    Returns:
            True if any element is NaN, False otherwise.

        Example:
            ```mojo
            var x = AnyTensor([[1.0, float("nan"), 3.0]])
            assert_true(has_nan(x))
            ```

    Note:
            Checks all elements regardless of dtype. Supports all floating-point types.
    """
    # Only float types can have NaN - fast path for integers
    var dtype = tensor.dtype()
    if dtype == DType.float32:
        try:
            return _has_nan_core[DType.float32](tensor.as_tensor[DType.float32]())
        except:
            return False
    elif dtype == DType.float64:
        try:
            return _has_nan_core[DType.float64](tensor.as_tensor[DType.float64]())
        except:
            return False
    elif dtype == DType.float16:
        try:
            return _has_nan_core[DType.float16](tensor.as_tensor[DType.float16]())
        except:
            return False
    # Integer types cannot have NaN
    return False


fn has_inf(tensor: AnyTensor) -> Bool:
    """Check if tensor contains any Inf values (positive or negative).

    Args:
            tensor: Input tensor to check.

    Returns:
            True if any element is Inf or -Inf, False otherwise.

        Example:
            ```mojo
            var x = AnyTensor([[1.0, float("inf"), 3.0]])
            assert_true(has_inf(x))
            ```

    Note:
            Checks all elements regardless of dtype. Supports all floating-point types.
    """
    var dtype = tensor.dtype()
    if dtype == DType.float32:
        try:
            return _has_inf_core[DType.float32](tensor.as_tensor[DType.float32]())
        except:
            return False
    elif dtype == DType.float64:
        try:
            return _has_inf_core[DType.float64](tensor.as_tensor[DType.float64]())
        except:
            return False
    elif dtype == DType.float16:
        try:
            return _has_inf_core[DType.float16](tensor.as_tensor[DType.float16]())
        except:
            return False
    return False


fn count_nan(tensor: AnyTensor) -> Int:
    """Count number of NaN values in tensor.

    Args:
            tensor: Input tensor to check.

    Returns:
            Number of NaN elements.

        Example:
            ```mojo
            var x = AnyTensor([[1.0, float("nan"), float("nan")]])
            assert_equal(count_nan(x), 2)
            ```
    """
    var dtype = tensor.dtype()
    if dtype == DType.float32:
        try:
            return _count_nan_core[DType.float32](tensor.as_tensor[DType.float32]())
        except:
            return 0
    elif dtype == DType.float64:
        try:
            return _count_nan_core[DType.float64](tensor.as_tensor[DType.float64]())
        except:
            return 0
    elif dtype == DType.float16:
        try:
            return _count_nan_core[DType.float16](tensor.as_tensor[DType.float16]())
        except:
            return 0
    return 0


fn count_inf(tensor: AnyTensor) -> Int:
    """Count number of Inf values in tensor.

    Args:
            tensor: Input tensor to check.

    Returns:
            Number of Inf/-Inf elements.

        Example:
            ```mojo
            var x = AnyTensor([[1.0, float("inf"), float("-inf")]])
            assert_equal(count_inf(x), 2)
            ```
    """
    var dtype = tensor.dtype()
    if dtype == DType.float32:
        try:
            return _count_inf_core[DType.float32](tensor.as_tensor[DType.float32]())
        except:
            return 0
    elif dtype == DType.float64:
        try:
            return _count_inf_core[DType.float64](tensor.as_tensor[DType.float64]())
        except:
            return 0
    elif dtype == DType.float16:
        try:
            return _count_inf_core[DType.float16](tensor.as_tensor[DType.float16]())
        except:
            return 0
    return 0


@parameter
fn check_tensor_safety[
    enable: Bool = False
](tensor: AnyTensor, name: String = "tensor") raises:
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


fn tensor_min(tensor: AnyTensor) -> Float64:
    """Find minimum value in tensor.

    Args:
            tensor: Input tensor.

    Returns:
            Minimum value as Float64.

        Example:
            ```mojo
            var x = AnyTensor([[1.0, -5.0, 3.0]])
            assert_equal(tensor_min(x), -5.0)
            ```
    """
    var dtype = tensor.dtype()
    if dtype == DType.float32:
        try:
            return _tensor_min_core[DType.float32](tensor.as_tensor[DType.float32]())
        except:
            return 0.0
    elif dtype == DType.float64:
        try:
            return _tensor_min_core[DType.float64](tensor.as_tensor[DType.float64]())
        except:
            return 0.0
    elif dtype == DType.float16:
        try:
            return _tensor_min_core[DType.float16](tensor.as_tensor[DType.float16]())
        except:
            return 0.0
    return 0.0


fn tensor_max(tensor: AnyTensor) -> Float64:
    """Find maximum value in tensor.

    Args:
            tensor: Input tensor.

    Returns:
            Maximum value as Float64.

        Example:
            ```mojo
            var x = AnyTensor([[1.0, 10.0, 3.0]])
            assert_equal(tensor_max(x), 10.0)
            ```
    """
    var dtype = tensor.dtype()
    if dtype == DType.float32:
        try:
            return _tensor_max_core[DType.float32](tensor.as_tensor[DType.float32]())
        except:
            return 0.0
    elif dtype == DType.float64:
        try:
            return _tensor_max_core[DType.float64](tensor.as_tensor[DType.float64]())
        except:
            return 0.0
    elif dtype == DType.float16:
        try:
            return _tensor_max_core[DType.float16](tensor.as_tensor[DType.float16]())
        except:
            return 0.0
    return 0.0


fn check_tensor_range(
    tensor: AnyTensor,
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


fn compute_tensor_l2_norm(tensor: AnyTensor) -> Float64:
    """Compute L2 norm of tensor: sqrt(sum(x^2)).

    Args:
            tensor: Input tensor.

    Returns:
            L2 norm as Float64.

        Example:
            ```mojo
            var x = AnyTensor([[3.0, 4.0]])
            assert_equal(compute_tensor_l2_norm(x), 5.0)  # sqrt(9 + 16)
            ```
    """
    var dtype = tensor.dtype()
    if dtype == DType.float32:
        try:
            return _compute_l2_norm_core[DType.float32](tensor.as_tensor[DType.float32]())
        except:
            return 0.0
    elif dtype == DType.float64:
        try:
            return _compute_l2_norm_core[DType.float64](tensor.as_tensor[DType.float64]())
        except:
            return 0.0
    elif dtype == DType.float16:
        try:
            return _compute_l2_norm_core[DType.float16](tensor.as_tensor[DType.float16]())
        except:
            return 0.0
    return 0.0


fn check_gradient_norm(
    gradient: AnyTensor, max_norm: Float64 = 1000.0, name: String = "gradient"
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
    gradient: AnyTensor, min_norm: Float64 = 1e-7, name: String = "gradient"
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
    gradient: AnyTensor,
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


# ============================================================================
# Gradient Clipping Utilities
# ============================================================================
# Moved from shared/autograd/grad_utils.mojo to break the circular type
# resolution between shared.core and shared.autograd that prevented
# `mojo package shared` from compiling. (Issue #4513)


fn clip_grad_value_(mut grad: AnyTensor, max_value: Float64) raises:
    """Clip each gradient element to [-max_value, max_value].

    This is the simplest form of gradient clipping. Each element is
    independently clipped to stay within the specified range.

    Args:
        grad: The gradient tensor to clip (modified in-place).
        max_value: Maximum absolute value allowed. Elements outside
                   [-max_value, max_value] are clipped.

    Raises:
        Error: If max_value is negative.
    """
    if max_value < 0.0:
        raise Error("max_value must be non-negative, got: " + String(max_value))

    for i in range(grad.numel()):
        var val = grad._get_float64(i)
        if val > max_value:
            grad._set_float64(i, max_value)
        elif val < -max_value:
            grad._set_float64(i, -max_value)


fn clip_grad_norm_(mut grad: AnyTensor, max_norm: Float64) raises -> Float64:
    """Clip gradient if its L2 norm exceeds max_norm.

    Computes the L2 norm of the gradient: norm = sqrt(sum(grad^2)).
    If norm > max_norm, scales the gradient by (max_norm / norm).
    This preserves the direction of the gradient while limiting its magnitude.

    Args:
        grad: The gradient tensor to clip (modified in-place if norm exceeds max_norm).
        max_norm: Maximum allowed L2 norm.

    Returns:
        The original L2 norm of the gradient (before clipping).

    Raises:
        Error: If max_norm is negative.
    """
    if max_norm < 0.0:
        raise Error("max_norm must be non-negative, got: " + String(max_norm))

    var norm_squared = 0.0
    for i in range(grad.numel()):
        var val = grad._get_float64(i)
        norm_squared += val * val

    var norm = sqrt(norm_squared)

    if norm > max_norm and norm > 0.0:
        var scale_factor = max_norm / norm
        for i in range(grad.numel()):
            var val = grad._get_float64(i)
            grad._set_float64(i, val * scale_factor)

    return norm


fn clip_grad_global_norm_(
    mut grads: List[AnyTensor], max_norm: Float64
) raises -> Float64:
    """Clip gradients based on their global L2 norm across all parameters.

    Computes a single norm across all gradient tensors and clips all gradients
    uniformly if the global norm exceeds max_norm.

    Args:
        grads: List of gradient tensors (modified in-place if global norm exceeds max_norm).
        max_norm: Maximum allowed global L2 norm.

    Returns:
        The original global L2 norm (before clipping).

    Raises:
        Error: If max_norm is negative or grads list is empty.
    """
    if max_norm < 0.0:
        raise Error("max_norm must be non-negative, got: " + String(max_norm))

    if len(grads) == 0:
        raise Error("grads list cannot be empty")

    var total_norm_squared = 0.0
    for grad_idx in range(len(grads)):
        var grad = grads[grad_idx]
        for elem_idx in range(grad.numel()):
            var val = grad._get_float64(elem_idx)
            total_norm_squared += val * val

    var global_norm = sqrt(total_norm_squared)

    if global_norm > max_norm and global_norm > 0.0:
        var scale_factor = max_norm / global_norm
        for grad_idx in range(len(grads)):
            var grad = grads[grad_idx]
            for elem_idx in range(grad.numel()):
                var val = grad._get_float64(elem_idx)
                grad._set_float64(elem_idx, val * scale_factor)

    return global_norm


# ============================================================================
# Typed overloads for Tensor[dtype] — delegate to typed cores directly
# ============================================================================


fn has_nan_typed[dt: DType](tensor: Tensor[dt]) raises -> Bool:
    """Typed overload of has_nan for Tensor[dtype].

    Args:
        tensor: Input tensor to check.

    Returns:
        True if any element is NaN.

    Raises:
        Error if conversion fails.
    """
    return _has_nan_core[dt](tensor)


fn has_inf_typed[dt: DType](tensor: Tensor[dt]) raises -> Bool:
    """Typed overload of has_inf for Tensor[dtype].

    Args:
        tensor: Input tensor to check.

    Returns:
        True if any element is infinity.

    Raises:
        Error if conversion fails.
    """
    return _has_inf_core[dt](tensor)


fn tensor_min_typed[dt: DType](tensor: Tensor[dt]) raises -> Float64:
    """Typed overload of tensor_min for Tensor[dtype].

    Args:
        tensor: Input tensor.

    Returns:
        Minimum value as Float64.

    Raises:
        Error if conversion fails.
    """
    return _tensor_min_core[dt](tensor)


fn tensor_max_typed[dt: DType](tensor: Tensor[dt]) raises -> Float64:
    """Typed overload of tensor_max for Tensor[dtype].

    Args:
        tensor: Input tensor.

    Returns:
        Maximum value as Float64.

    Raises:
        Error if conversion fails.
    """
    return _tensor_max_core[dt](tensor)
