"""Arithmetic operations with native Tensor[dtype] implementations.

Implements element-wise arithmetic operations following NumPy-style broadcasting.
Architecture: Tensor[dtype] typed implementations are the core (zero dtype branches).
AnyTensor versions dispatch to typed implementations via ordinal-based table.

Layer 1 (outer): AnyTensor public API (add, subtract, etc.)
Layer 2: dtype dispatch table (ordinal-based)
Layer 3 (core): Tensor[dtype] native implementation (_broadcast_binary_typed)
"""

from collections import List
from math import nan
from .any_tensor import AnyTensor, full
from shared.base.broadcasting import broadcast_shapes, compute_broadcast_strides
from .shape import as_contiguous
from .gradient_types import GradientPair
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
# Layer 3 (Core): Native Tensor[dtype] Broadcasting Implementation
# ============================================================================


fn _broadcast_binary_typed[
    dtype: DType, op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    """Apply binary operation with broadcasting on native Tensor[dtype].

    This is the core implementation -- zero dtype branches, zero bitcasts.
    Tensor[dtype]._data is already typed as UnsafePointer[Scalar[dtype]].

    Parameters:
        dtype: Compile-time dtype parameter.
        op: Binary operation function (e.g., add, subtract, multiply, divide).

    Args:
        a: First tensor (typed).
        b: Second tensor (typed).

    Returns:
        Result Tensor[dtype] with operation applied element-wise with broadcasting.
    """
    # Ensure inputs are contiguous before flat-buffer kernel access.
    # Non-contiguous views (e.g. from transpose) have non-unit strides that
    # are not reflected in flat index arithmetic, causing silent wrong results.
    # Tensor[dtype] lacks as_contiguous(), so we round-trip via AnyTensor only
    # when needed (rare path -- most tensors are contiguous).
    var a_any = a.as_any()
    var b_any = b.as_any()
    var a_cont = a_any if a.is_contiguous() else as_contiguous(a_any)
    var b_cont = b_any if b.is_contiguous() else as_contiguous(b_any)

    # Compute broadcast shape
    var result_shape = broadcast_shapes(a_cont.shape(), b_cont.shape())
    var result = Tensor[dtype](result_shape)

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a_cont.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b_cont.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    # Precompute row-major strides for result shape
    var result_strides = List[Int]()
    var stride = 1
    for i in range(len(result_shape) - 1, -1, -1):
        result_strides.append(stride)
        stride *= result_shape[i]

    # Reverse to get correct order (left-to-right)
    var result_strides_final = List[Int]()
    for i in range(len(result_strides) - 1, -1, -1):
        result_strides_final.append(result_strides[i])

    # Get typed pointers -- Tensor[dtype]._data is already typed, but
    # a_cont/b_cont are AnyTensor (from contiguity check), so bitcast from
    # their UInt8 storage. Result uses native typed pointer directly.
    var a_ptr = a_cont._data.bitcast[Scalar[dtype]]()
    var b_ptr = b_cont._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data

    # Iterate over all result elements
    for result_idx in range(total_elems):
        var idx_a = 0
        var idx_b = 0
        var temp_idx = result_idx

        # Convert flat index to multi-dimensional coordinates, then compute source indices
        for dim in range(len(result_shape)):
            var coord = temp_idx // result_strides_final[dim]
            temp_idx = temp_idx % result_strides_final[dim]

            idx_a += coord * strides_a[dim]
            idx_b += coord * strides_b[dim]

        # Perform operation with zero overhead (no dtype conversion!)
        result_ptr[result_idx] = op[dtype](a_ptr[idx_a], b_ptr[idx_b])

    return result^


# ============================================================================
# Layer 2: AnyTensor Broadcasting Helper (delegates to typed core)
# ============================================================================


fn _broadcast_binary[
    dtype: DType, op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Apply binary operation with broadcasting via typed core.

    Converts AnyTensor inputs to Tensor[dtype], calls the typed implementation,
    and converts the result back to AnyTensor.

    Parameters:
        dtype: Compile-time dtype parameter.
        op: Binary operation function (e.g., add, subtract, multiply, divide).

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Result tensor with operation applied element-wise with broadcasting.
    """
    return _broadcast_binary_typed[dtype, op](
        a.as_tensor[dtype](), b.as_tensor[dtype]()
    ).as_any()


# ============================================================================
# Layer 2: Runtime Dtype Dispatch (ordinal-based jump table)
# ============================================================================


fn _dispatch_broadcast_binary[
    op: fn[T: DType] (Scalar[T], Scalar[T]) -> Scalar[T]
](a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to compile-time specialized Tensor[dtype] implementation.

    Performs runtime dtype checking and dispatches to the typed core via
    ordinal-based lookup (compiler can generate jump table).

    Parameters:
        op: Binary operation function pointer.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Result tensor with operation applied with broadcasting.

    Raises:
        Error: If dtypes don't match or are unsupported.
    """
    # Validate dtypes match
    if a._dtype != b._dtype:
        raise Error("Cannot operate on tensors with different dtypes")

    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(a._dtype)

    # Dispatch based on ordinal - compiler generates jump table for consecutive integers
    if ordinal == DTYPE_FLOAT16:
        return _broadcast_binary[DType.float16, op](a, b)
    elif ordinal == DTYPE_FLOAT32:
        return _broadcast_binary[DType.float32, op](a, b)
    elif ordinal == DTYPE_FLOAT64:
        return _broadcast_binary[DType.float64, op](a, b)
    elif ordinal == DTYPE_INT8:
        return _broadcast_binary[DType.int8, op](a, b)
    elif ordinal == DTYPE_INT16:
        return _broadcast_binary[DType.int16, op](a, b)
    elif ordinal == DTYPE_INT32:
        return _broadcast_binary[DType.int32, op](a, b)
    elif ordinal == DTYPE_INT64:
        return _broadcast_binary[DType.int64, op](a, b)
    elif ordinal == DTYPE_UINT8:
        return _broadcast_binary[DType.uint8, op](a, b)
    elif ordinal == DTYPE_UINT16:
        return _broadcast_binary[DType.uint16, op](a, b)
    elif ordinal == DTYPE_UINT32:
        return _broadcast_binary[DType.uint32, op](a, b)
    elif ordinal == DTYPE_UINT64:
        return _broadcast_binary[DType.uint64, op](a, b)
    else:
        raise Error("Unsupported dtype for binary operation")


fn add(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
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

    # Define add operation
    @always_inline
    fn _add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        return x + y

    # Use generic broadcasting dispatcher (eliminates 60 lines and conversion overhead!)
    return _dispatch_broadcast_binary[_add_op](a, b)


fn subtract(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
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

    # Define subtract operation
    @always_inline
    fn _sub_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        return x - y

    return _dispatch_broadcast_binary[_sub_op](a, b)


fn multiply(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
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

    @always_inline
    fn _mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        return x * y

    return _dispatch_broadcast_binary[_mul_op](a, b)


fn divide(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
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

    @always_inline
    fn _div_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        return x / y

    return _dispatch_broadcast_binary[_div_op](a, b)


fn _multiply_scalar_typed[
    dt: DType
](tensor: Tensor[dt], scalar: Float32) raises -> Tensor[dt]:
    """Native typed scalar multiplication (Layer 3 core).

    Multiplies each element by a scalar without creating an intermediate
    full tensor. Zero dtype branches -- pointer is already typed.

    Args:
        tensor: Input tensor (typed).
        scalar: Scalar value to multiply by.

    Returns:
        A new Tensor[dt] with each element multiplied by the scalar.
    """
    # Ensure input is contiguous before flat-buffer kernel access.
    # Tensor[dt] lacks as_contiguous(), so round-trip via AnyTensor if needed.
    var t_any = tensor.as_any()
    var t_cont = t_any if tensor.is_contiguous() else as_contiguous(t_any)

    var result = Tensor[dt](t_cont.shape())
    var numel = result.numel()

    var input_ptr = t_cont._data.bitcast[Scalar[dt]]()
    var result_ptr = result._data
    var scalar_cast = Scalar[dt](scalar)
    for i in range(numel):
        result_ptr[i] = input_ptr[i] * scalar_cast

    return result^


fn multiply_scalar(tensor: AnyTensor, scalar: Float32) raises -> AnyTensor:
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
    # Get ordinal for dispatch (compiler can optimize to efficient lookup)
    var ordinal = dtype_to_ordinal(tensor.dtype())

    # Dispatch to typed core via ordinal-based jump table
    if ordinal == DTYPE_FLOAT16:
        return _multiply_scalar_typed[DType.float16](
            tensor.as_tensor[DType.float16](), scalar
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _multiply_scalar_typed[DType.float32](
            tensor.as_tensor[DType.float32](), scalar
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _multiply_scalar_typed[DType.float64](
            tensor.as_tensor[DType.float64](), scalar
        ).as_any()
    elif ordinal == DTYPE_INT8:
        return _multiply_scalar_typed[DType.int8](
            tensor.as_tensor[DType.int8](), scalar
        ).as_any()
    elif ordinal == DTYPE_INT16:
        return _multiply_scalar_typed[DType.int16](
            tensor.as_tensor[DType.int16](), scalar
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _multiply_scalar_typed[DType.int32](
            tensor.as_tensor[DType.int32](), scalar
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _multiply_scalar_typed[DType.int64](
            tensor.as_tensor[DType.int64](), scalar
        ).as_any()
    elif ordinal == DTYPE_UINT8:
        return _multiply_scalar_typed[DType.uint8](
            tensor.as_tensor[DType.uint8](), scalar
        ).as_any()
    elif ordinal == DTYPE_UINT16:
        return _multiply_scalar_typed[DType.uint16](
            tensor.as_tensor[DType.uint16](), scalar
        ).as_any()
    elif ordinal == DTYPE_UINT32:
        return _multiply_scalar_typed[DType.uint32](
            tensor.as_tensor[DType.uint32](), scalar
        ).as_any()
    elif ordinal == DTYPE_UINT64:
        return _multiply_scalar_typed[DType.uint64](
            tensor.as_tensor[DType.uint64](), scalar
        ).as_any()
    else:
        raise Error("Unsupported dtype for multiply_scalar operation")


fn floor_divide(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
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

    @always_inline
    fn _floor_div_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        # Check for division by zero - return inf per IEEE 754 (for floating-point types)
        @parameter
        if T.is_floating_point():
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


fn modulo(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
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

    @always_inline
    fn _mod_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        # Check for modulo by zero - return NaN per IEEE 754 (for floating-point types)
        @parameter
        if T.is_floating_point():
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


fn power(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
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

    @always_inline
    fn _pow_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
        # Use Mojo's built-in ** operator (handles all cases correctly)
        return x**y

    return _dispatch_broadcast_binary[_pow_op](a, b)


# ==============================================================================
# Backward Pass (Gradient Computation)
# ==============================================================================


fn _reduce_broadcast_dims(
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
        # Broadcasting (3, 1, 5) → (3, 4, 5)
        var grad = ones([3, 4, 5])  # Gradient from loss
        var original = [3, 1, 5]    # Original input shape
        var reduced = _reduce_broadcast_dims(grad, original)  # Shape (3, 1, 5)

        # Prepended dimensions: (5,) → (3, 4, 5)
        var grad2 = ones([3, 4, 5])
        var original2 = [5]
        var reduced2 = _reduce_broadcast_dims(grad2, original2)  # Shape (5,)
        ```
    """
    from .reduction import sum

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
    # Example: (3, 1, 5) → (3, 4, 5), sum over axis 1 keeping dims
    # After reducing prepended dims, result has same ndim as original_shape
    for i in range(orig_ndim):
        if (
            original_shape[i] == 1
            and i < len(result.shape())
            and result.shape()[i] > 1
        ):
            result = sum(result, axis=i, keepdims=True)

    return result


fn add_backward(
    grad_output: AnyTensor, a: AnyTensor, b: AnyTensor
) raises -> GradientPair:
    """Compute gradients for element-wise addition.

    For C = A + B, given ∂L/∂C, computes:
        ∂L/∂A = ∂L/∂C (summed over broadcasted dimensions)
        ∂L/∂B = ∂L/∂C (summed over broadcasted dimensions)

    Handles broadcasting: If input was broadcast to output shape, gradient
    is summed back to the original input shape.

    Args:
        grad_output: Gradient from upstream (∂L/∂C).
        a: First input from forward pass (A).
        b: Second input from forward pass (B).

    Returns:
        GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs.

    Examples:
        ```
        # No broadcasting
        var a = ones(List[Int](), DType.float32)
        var b = ones(List[Int](), DType.float32)
        var c = add(a, b)
        var grad_c = ones(List[Int](), DType.float32)
        var grads = add_backward(grad_c, a, b)
        var grad_a = grads.grad_a
        var grad_b = grads.grad_b

        # With broadcasting
        var x = ones(List[Int](), DType.float32)
        var y = ones(List[Int](), DType.float32)
        var z = add(x, y)  # Shape (3, 4)
        var grad_z = ones(List[Int](), DType.float32)
        var grads = add_backward(grad_z, x, y)
        # grads.grad_a will be shape (3, 1) - summed over broadcast dimension
        ```
    """
    # For addition, gradient passes through but must be reduced for broadcasting
    var grad_a = _reduce_broadcast_dims(grad_output, a.shape())
    var grad_b = _reduce_broadcast_dims(grad_output, b.shape())

    return GradientPair(grad_a^, grad_b^)


fn subtract_backward(
    grad_output: AnyTensor, a: AnyTensor, b: AnyTensor
) raises -> GradientPair:
    """Compute gradients for element-wise subtraction.

    For C = A - B, given ∂L/∂C, computes:
        ∂L/∂A = ∂L/∂C (reduced for broadcasting)
        ∂L/∂B = -∂L/∂C (negated and reduced for broadcasting)

    The gradient for B is negated since ∂(A-B)/∂B = -1.

    Args:
        grad_output: Gradient from upstream (∂L/∂C).
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


fn multiply_backward(
    grad_output: AnyTensor, a: AnyTensor, b: AnyTensor
) raises -> GradientPair:
    """Compute gradients for element-wise multiplication.

    For C = A * B, given ∂L/∂C, computes:
        ∂L/∂A = ∂L/∂C * B  (product rule, reduced for broadcasting)
        ∂L/∂B = ∂L/∂C * A  (reduced for broadcasting)

    Args:
        grad_output: Gradient from upstream (∂L/∂C).
        a: First input from forward pass (A).
        b: Second input from forward pass (B).

    Returns:
        GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs.

    Examples:
        ```
        var a = ones(List[Int](), DType.float32)
        var b = ones(List[Int](), DType.float32)
        var c = multiply(a, b)
        var grad_c = ones(List[Int](), DType.float32)
        var grads = multiply_backward(grad_c, a, b)
        var grad_a = grads.grad_a
        var grad_b = grads.grad_b
        ```
    """
    # grad_a = grad_output * b (then reduce for broadcasting)
    var grad_a_unreduced = multiply(grad_output, b)
    var grad_a = _reduce_broadcast_dims(grad_a_unreduced, a.shape())

    # grad_b = grad_output * a (then reduce for broadcasting)
    var grad_b_unreduced = multiply(grad_output, a)
    var grad_b = _reduce_broadcast_dims(grad_b_unreduced, b.shape())

    return GradientPair(grad_a^, grad_b^)


fn divide_backward(
    grad_output: AnyTensor, a: AnyTensor, b: AnyTensor
) raises -> GradientPair:
    """Compute gradients for element-wise division.

    For C = A / B, given ∂L/∂C, computes:
        ∂L/∂A = ∂L/∂C / B  (quotient rule numerator)
        ∂L/∂B = -∂L/∂C * A / B²  (quotient rule denominator)

    Includes numerical stability: adds small epsilon to prevent division by zero.

    Args:
        grad_output: Gradient from upstream (∂L/∂C).
        a: First input from forward pass (A).
        b: Second input from forward pass (B).

    Returns:
        GradientPair containing (grad_a, grad_b) - gradients w.r.t. inputs.

    Examples:
        ```
        var a = ones(List[Int](), DType.float32)
        var b = full(List[Int](), 2.0, DType.float32)
        var c = divide(a, b)
        var grad_c = ones(List[Int](), DType.float32)
        var grads = divide_backward(grad_c, a, b)
        var grad_a = grads.grad_a
        var grad_b = grads.grad_b
        ```

    Note:
        Uses epsilon = 1e-10 to prevent division by zero in b².
    """
    comptime EPSILON = 1e-10

    # grad_a = grad_output / b (then reduce for broadcasting)
    var grad_a_unreduced = divide(grad_output, b)
    var grad_a = _reduce_broadcast_dims(grad_a_unreduced, a.shape())

    # grad_b = -grad_output * a / b²
    # Add epsilon to b² for numerical stability
    var b_squared = multiply(b, b)

    # Add epsilon to prevent division by zero using tensor operations
    var epsilon_tensor = full(b_squared.shape(), EPSILON, b_squared.dtype())
    var b_squared_safe = add(b_squared, epsilon_tensor)

    # Compute -grad_output * a / b²
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


