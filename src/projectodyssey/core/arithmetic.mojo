"""Arithmetic operations for AnyTensor.

Implements element-wise arithmetic operations following NumPy-style broadcasting.
Typed Tensor[dtype] implementations live in src/projectodyssey/tensor/typed/arithmetic.mojo.
This file provides the AnyTensor public API only.
"""

from std.collections import List
from std.math import nan
from projectodyssey.tensor.any_tensor import AnyTensor, full
from projectodyssey.core.gradient_types import GradientPair


def add(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise addition with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new tensor containing a + b.

    Raises:
        Error: If shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        ```
        var a = zeros(List[Int](), DType.float32)
        var b = ones(List[Int](), DType.float32)
        var c = add(a, b)  # Shape (3, 4), all ones

        # Broadcasting example
        var x = ones([3, 1, 5], DType.float32)
        var y = ones([3, 4, 5], DType.float32)
        var z = add(x, y)  # Shape (3, 4, 5)
        ```
    """
    from projectodyssey.tensor.typed.arithmetic import (
        _dispatch_broadcast_binary,
    )

    # Define add operation
    @always_inline
    def _add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        return x + y

    # Use generic broadcasting dispatcher (eliminates 60 lines and conversion overhead!)
    return _dispatch_broadcast_binary[_add_op](a, b)


def subtract(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise subtraction with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new tensor containing a - b.

    Raises:
        Error: If shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        ```
        var a = ones(List[Int](), DType.float32)
        var b = ones(List[Int](), DType.float32)
        var c = subtract(a, b)  # Shape (3, 4), all zeros

        # Broadcasting example
        var x = ones([3, 1, 5], DType.float32)
        var y = ones([3, 4, 5], DType.float32)
        var z = subtract(x, y)  # Shape (3, 4, 5)
        ```
    """
    from projectodyssey.tensor.typed.arithmetic import (
        _dispatch_broadcast_binary,
    )

    # Define subtract operation
    @always_inline
    def _sub_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        return x - y

    return _dispatch_broadcast_binary[_sub_op](a, b)


def multiply(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise multiplication with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        A new tensor containing a * b.

    Raises:
        Error: If shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        ```
        var a = full(List[Int](), 2.0, DType.float32)
        var b = full(List[Int](), 3.0, DType.float32)
        var c = multiply(a, b)  # Shape (3, 4), all 6.0

        # Broadcasting example
        var x = full([3, 1, 5], 2.0, DType.float32)
        var y = full([3, 4, 5], 3.0, DType.float32)
        var z = multiply(x, y)  # Shape (3, 4, 5), all 6.0
        ```
    """
    from projectodyssey.tensor.typed.arithmetic import (
        _dispatch_broadcast_binary,
    )

    @always_inline
    def _mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        return x * y

    return _dispatch_broadcast_binary[_mul_op](a, b)


def divide(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise division with broadcasting.

    Args:
        a: First tensor (numerator).
        b: Second tensor (denominator).

    Returns:
        A new tensor containing a / b.

    Raises:
        Error: If shapes are not broadcast-compatible or dtypes don't match.

    Note on Floating-Point Division:
        Division by zero follows IEEE 754 semantics for floating-point types:
        - x / 0.0 where x > 0 -> +inf
        - x / 0.0 where x < 0 -> -inf
        - 0.0 / 0.0 -> NaN

    Note on Integer Division:
        For integer dtypes, division by zero results in undefined behavior
        (may produce errors, wrap-around, or saturate depending on the
        underlying platform and Mojo implementation). Users should validate
        that divisors are non-zero before dividing integer tensors.

    Examples:
        ```
        var a = full(List[Int](), 6.0, DType.float32)
        var b = full(List[Int](), 2.0, DType.float32)
        var c = divide(a, b)  # Shape (3, 4), all 3.0

        # Broadcasting example
        var x = full([3, 1, 5], 6.0, DType.float32)
        var y = full([3, 4, 5], 2.0, DType.float32)
        var z = divide(x, y)  # Shape (3, 4, 5), all 3.0
        ```
    """
    from projectodyssey.tensor.typed.arithmetic import (
        _dispatch_broadcast_binary,
    )

    @always_inline
    def _div_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        return x / y

    return _dispatch_broadcast_binary[_div_op](a, b)


def multiply_scalar(tensor: AnyTensor, scalar: Float32) raises -> AnyTensor:
    """Multiply tensor by a scalar value efficiently.

    Dispatches to native Tensor[dtype] implementation via ordinal-based table.
    Useful for operations like negation (multiply by -1.0) or scaling in
    gradient computations.

    Args:
        tensor: Input tensor to multiply.
        scalar: Scalar value to multiply by.

    Returns:
        A new tensor with each element multiplied by the scalar.

    Raises:
        Error: If tensor dtype is unsupported.

    Examples:
        ```
        var a = full([2, 3], 5.0, DType.float32)
        var result = multiply_scalar(a, 2.0)  # Shape (2, 3), all 10.0

        var b = full([2, 3], 3.0, DType.float32)
        var negated = multiply_scalar(b, -1.0)  # Shape (2, 3), all -3.0
        ```
    """
    from projectodyssey.tensor.typed.arithmetic import _dispatch_multiply_scalar

    return _dispatch_multiply_scalar(tensor, scalar)


def floor_divide(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise floor division with broadcasting.

    Args:
        a: First tensor (numerator).
        b: Second tensor (denominator).

    Returns:
        A new tensor containing a // b (floor division).

    Raises:
        Error: If shapes are not broadcast-compatible or dtypes don't match.

    Note:
        Division by zero follows IEEE 754 semantics for floating-point types:
        - x // 0.0 where x > 0 -> +inf
        - x // 0.0 where x < 0 -> -inf
        - 0.0 // 0.0 -> NaN

    Examples:
        ```
        var a = full(List[Int](), 7.0, DType.float32)
        var b = full(List[Int](), 2.0, DType.float32)
        var c = floor_divide(a, b)  # Shape (3, 4), all 3.0

        # Broadcasting example
        var x = full([3, 1, 5], 7.0, DType.float32)
        var y = full([3, 4, 5], 2.0, DType.float32)
        var z = floor_divide(x, y)  # Shape (3, 4, 5), all 3.0
        ```
    """
    from projectodyssey.tensor.typed.arithmetic import (
        _dispatch_broadcast_binary,
    )

    @always_inline
    def _floor_div_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        # Check for division by zero - return inf per IEEE 754 (for floating-point types)
        comptime if T.is_floating_point():
            if y == Scalar[T](0):
                # For floating point, follow IEEE 754: x / 0 = inf or -inf based on sign
                return x / y  # Let hardware handle the division by zero

        # Floor division: floor(x / y)
        # For correct negative handling, use: Int(div) if div >= 0 else Int(div) - 1
        var div_result = x / y
        var as_int = Int(div_result)
        var floored = Scalar[T](as_int) if div_result >= Scalar[T](
            0
        ) else Scalar[T](as_int - 1)
        return floored

    return _dispatch_broadcast_binary[_floor_div_op](a, b)


def modulo(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise modulo with broadcasting.

    Args:
        a: First tensor.
        b: Second tensor (modulus).

    Returns:
        A new tensor containing a % b.

    Raises:
        Error: If shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        ```
        var a = full(List[Int](), 7.0, DType.int32)
        var b = full(List[Int](), 3.0, DType.int32)
        var c = modulo(a, b)  # Shape (3, 4), all 1

        # Broadcasting example
        var x = full([3, 1, 5], 7.0, DType.float32)
        var y = full([3, 4, 5], 3.0, DType.float32)
        var z = modulo(x, y)  # Shape (3, 4, 5), all 1.0
        ```
    """
    from projectodyssey.tensor.typed.arithmetic import (
        _dispatch_broadcast_binary,
    )

    @always_inline
    def _mod_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        # Check for modulo by zero - return NaN per IEEE 754 (for floating-point types)
        comptime if T.is_floating_point():
            if y == Scalar[T](0):
                return Scalar[T](nan[T]())

        # Modulo: a % b = a - floor(a/b) * b
        var div_result = x / y
        var as_int = Int(div_result)
        var floored = Scalar[T](as_int) if div_result >= Scalar[T](
            0
        ) else Scalar[T](as_int - 1)
        return x - floored * y

    return _dispatch_broadcast_binary[_mod_op](a, b)


def power(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Element-wise exponentiation with broadcasting.

    Args:
        a: Base tensor.
        b: Exponent tensor.

    Returns:
        A new tensor containing a ** b.

    Raises:
        Error: If shapes are not broadcast-compatible or dtypes don't match.

    Examples:
        ```
        var a = full(List[Int](), 2.0, DType.float32)
        var b = full(List[Int](), 3.0, DType.float32)
        var c = power(a, b)  # Shape (3, 4), all 8.0

        # Broadcasting example
        var x = full([3, 1, 5], 2.0, DType.float32)
        var y = full([3, 4, 5], 3.0, DType.float32)
        var z = power(x, y)  # Shape (3, 4, 5), all 8.0
        ```

    Note:
        Uses ** operator which delegates to Mojo's built-in power implementation.
        For integer exponents, this uses efficient repeated squaring.
        For fractional exponents, this uses exp(b * log(a)).
    """
    from projectodyssey.tensor.typed.arithmetic import (
        _dispatch_broadcast_binary,
    )

    @always_inline
    def _pow_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        # Use Mojo's built-in ** operator (handles all cases correctly)
        return x**y

    return _dispatch_broadcast_binary[_pow_op](a, b)


# ==============================================================================
# Backward Pass (Gradient Computation)
# ==============================================================================


def _reduce_broadcast_dims(
    grad: AnyTensor, original_shape: List[Int]
) raises -> AnyTensor:
    """Reduce gradient from broadcast shape back to original shape.

    When forward pass broadcasts input from original_shape to grad.shape(),
    backward pass must sum gradient back to original_shape.

    Args:
        grad: Gradient tensor (broadcast shape).
        original_shape: Original input shape before broadcasting.

    Returns:
        Reduced gradient matching original_shape.

    Examples:
        ```
        # Broadcasting (3, 1, 5) -> (3, 4, 5)
        var grad = ones([3, 4, 5])  # Gradient from loss
        var original = [3, 1, 5]    # Original input shape
        var reduced = _reduce_broadcast_dims(grad, original)  # Shape (3, 1, 5)

        # Prepended dimensions: (5,) -> (3, 4, 5)
        var grad2 = ones([3, 4, 5])
        var original2 = [5]
        var reduced2 = _reduce_broadcast_dims(grad2, original2)  # Shape (5,)
        ```
    """
    from projectodyssey.core.reduction import sum

    var result = grad
    var grad_shape = grad.shape()
    var grad_ndim = len(grad_shape)
    var orig_ndim = len(original_shape)

    # Handle prepended dimensions (when original had fewer dims)
    # Example: original (5,) broadcast to (3, 4, 5)
    # Need to sum over first (grad_ndim - orig_ndim) dimensions
    if orig_ndim < grad_ndim:
        var dims_to_sum = grad_ndim - orig_ndim
        for _ in range(dims_to_sum):
            # Always sum over axis 0 since shape shrinks each time
            result = sum(result, axis=0, keepdims=False)

    # Now handle dimensions that were size 1 and got broadcast
    # Example: (3, 1, 5) -> (3, 4, 5), sum over axis 1 keeping dims
    # After reducing prepended dims, result has same ndim as original_shape
    for i in range(orig_ndim):
        if (
            original_shape[i] == 1
            and i < len(result.shape())
            and result.shape()[i] > 1
        ):
            result = sum(result, axis=i, keepdims=True)

    return result


def add_backward(
    grad_output: AnyTensor, a: AnyTensor, b: AnyTensor
) raises -> GradientPair:
    """Compute gradients for element-wise addition.

    For C = A + B, given dL/dC, computes:
        dL/dA = dL/dC (summed over broadcasted dimensions)
        dL/dB = dL/dC (summed over broadcasted dimensions)

    Handles broadcasting: If input was broadcast to output shape, gradient
    is summed back to the original input shape.

    Args:
        grad_output: Gradient from upstream (dL/dC).
        a: First input from forward pass (A).
        b: Second input from forward pass (B).

    Returns:
        GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs.
    """
    # For addition, gradient passes through but must be reduced for broadcasting
    var grad_a = _reduce_broadcast_dims(grad_output, a.shape())
    var grad_b = _reduce_broadcast_dims(grad_output, b.shape())

    return GradientPair(grad_a^, grad_b^)


def subtract_backward(
    grad_output: AnyTensor, a: AnyTensor, b: AnyTensor
) raises -> GradientPair:
    """Compute gradients for element-wise subtraction.

    For C = A - B, given dL/dC, computes:
        dL/dA = dL/dC (reduced for broadcasting)
        dL/dB = -dL/dC (negated and reduced for broadcasting)

    The gradient for B is negated since d(A-B)/dB = -1.

    Args:
        grad_output: Gradient from upstream (dL/dC).
        a: First input from forward pass (A).
        b: Second input from forward pass (B).

    Returns:
        GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs.
    """
    # Gradient for A passes through unchanged (but reduced for broadcasting)
    var grad_a = _reduce_broadcast_dims(grad_output, a.shape())

    # Gradient for B is negated using optimized scalar multiplication
    var neg_grad = multiply_scalar(grad_output, -1.0)

    # Reduce for broadcasting
    var grad_b = _reduce_broadcast_dims(neg_grad, b.shape())

    return GradientPair(grad_a^, grad_b^)


def multiply_backward(
    grad_output: AnyTensor, a: AnyTensor, b: AnyTensor
) raises -> GradientPair:
    """Compute gradients for element-wise multiplication.

    For C = A * B, given dL/dC, computes:
        dL/dA = dL/dC * B  (product rule, reduced for broadcasting)
        dL/dB = dL/dC * A  (reduced for broadcasting)

    Args:
        grad_output: Gradient from upstream (dL/dC).
        a: First input from forward pass (A).
        b: Second input from forward pass (B).

    Returns:
        GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs.
    """
    # grad_a = grad_output * b (then reduce for broadcasting)
    var grad_a_unreduced = multiply(grad_output, b)
    var grad_a = _reduce_broadcast_dims(grad_a_unreduced, a.shape())

    # grad_b = grad_output * a (then reduce for broadcasting)
    var grad_b_unreduced = multiply(grad_output, a)
    var grad_b = _reduce_broadcast_dims(grad_b_unreduced, b.shape())

    return GradientPair(grad_a^, grad_b^)


def divide_backward(
    grad_output: AnyTensor, a: AnyTensor, b: AnyTensor
) raises -> GradientPair:
    """Compute gradients for element-wise division.

    For C = A / B, given dL/dC, computes:
        dL/dA = dL/dC / B  (quotient rule numerator)
        dL/dB = -dL/dC * A / B^2  (quotient rule denominator)

    Includes numerical stability: adds small epsilon to prevent division by zero.

    Args:
        grad_output: Gradient from upstream (dL/dC).
        a: First input from forward pass (A).
        b: Second input from forward pass (B).

    Returns:
        GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs.

    Note:
        Uses epsilon = 1e-10 to prevent division by zero in b^2.
    """
    comptime EPSILON = 1e-10

    # grad_a = grad_output / b (then reduce for broadcasting)
    var grad_a_unreduced = divide(grad_output, b)
    var grad_a = _reduce_broadcast_dims(grad_a_unreduced, a.shape())

    # grad_b = -grad_output * a / b^2
    # Add epsilon to b^2 for numerical stability
    var b_squared = multiply(b, b)

    # Add epsilon to prevent division by zero using tensor operations
    var epsilon_tensor = full(b_squared.shape(), EPSILON, b_squared.dtype())
    var b_squared_safe = add(b_squared, epsilon_tensor)

    # Compute -grad_output * a / b^2
    var temp = multiply(grad_output, a)
    var grad_b_positive = divide(temp, b_squared_safe)

    # Negate it using optimized scalar multiplication
    var grad_b_unreduced = multiply_scalar(grad_b_positive, -1.0)

    # Reduce for broadcasting
    var grad_b = _reduce_broadcast_dims(grad_b_unreduced, b.shape())

    return GradientPair(grad_a^, grad_b^)


# ==============================================================================
# FUTURE WORK: Operator Overloading (out of scope for issues #219-220)
# ==============================================================================
#
# The following dunder methods should be implemented on the AnyTensor struct
# to enable natural operator syntax (e.g., a + b instead of add(a, b)):
#
# Basic operators:
#   fn __add__(self, other: AnyTensor) -> AnyTensor
#   fn __sub__(self, other: AnyTensor) -> AnyTensor
#   fn __mul__(self, other: AnyTensor) -> AnyTensor
#   fn __truediv__(self, other: AnyTensor) -> AnyTensor
#   fn __floordiv__(self, other: AnyTensor) -> AnyTensor
#   fn __mod__(self, other: AnyTensor) -> AnyTensor
#   fn __pow__(self, other: AnyTensor) -> AnyTensor
#
# Reflected variants (for operations like: 2 + tensor):
#   fn __radd__, __rsub__, __rmul__, __rtruediv__, etc.
#
# In-place variants (for operations like: tensor += 2):
#   fn __iadd__, __isub__, __imul__, __itruediv__, etc.
#
# These implementations should delegate to the functions in this module.
