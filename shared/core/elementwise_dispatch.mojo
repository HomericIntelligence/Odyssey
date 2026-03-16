"""Trait-based elementwise operation dispatcher for ExTensor.

Provides high-level trait-based abstraction for elementwise operations,
eliminating the need to write operation functions for each operation type.
Uses traits to define operation semantics and dispatches them through
the dtype dispatch layer.

Benefits:
- Trait-based polymorphism for operation definitions
- Reusable operation implementations
- Single source of truth for each operation
- Easy to add new operations
- Zero runtime overhead (compile-time specialization)

Architecture:
- ElementwiseUnaryOp trait: Operations with one input (e.g., exp, sin)
- ElementwiseBinaryOp trait: Operations with two inputs (e.g., add, mul)
- apply_unary[Op](): Dispatcher for unary operations
- apply_binary[Op](): Dispatcher for binary operations
- Predefined operation structs: ExpOp, LogOp, SqrtOp, AddOp, MulOp, etc.

Example usage:
    # Apply exp operation to tensor
    var result = apply_unary[ExpOp](input_tensor)

    # Apply addition to two tensors
    var sum_result = apply_binary[AddOp](tensor_a, tensor_b)

    # Define custom operation
    struct CustomOp(ElementwiseUnaryOp):
        fn apply(self, value: Float64) -> Float64:
            return value * value + 1.0

    var custom_result = apply_unary[CustomOp](tensor)

See notes/issues/elementwise-dispatch-design.md for complete design.
"""

from collections import List
from .extensor import ExTensor
from math import sqrt as math_sqrt, exp as math_exp, log as math_log
from math import sin as math_sin, cos as math_cos, tanh as math_tanh


# ============================================================================
# Trait Definitions
# ============================================================================


trait ElementwiseUnaryOp:
    """Trait for single-input elementwise operations.

    Implement this trait to define custom unary operations that can be
    dispatched through apply_unary[]. The operation receives a Float64
    value and returns the result as Float64.

    Required Methods:
        `__init__`: Default constructor
        `apply`: Apply the operation to a single value

    Contract:
        - Must have a default constructor (no arguments)
        - apply() must be deterministic
        - apply() must handle all Float64 values (including edge cases)
        - apply() should not have side effects

    Example:
        struct SquareOp(ElementwiseUnaryOp):
            '''Square each element.'''
            fn __init__(out self):
                pass

            fn apply(self, value: Float64) -> Float64:
                return value * value

        var squared = apply_unary[SquareOp](tensor)
    """

    fn __init__(out self):
        """Default constructor required for generic instantiation."""
        ...

    fn apply(self, value: Float64) -> Float64:
        """Apply operation to a single value.

        Args:
            value: Input value (Float64).

        Returns:
            Result value (Float64).

        Note:
            Must handle all Float64 values, including:
            - Positive/negative numbers.
            - Zero.
            - Infinity.
            - NaN.
        """
        ...


trait ElementwiseBinaryOp:
    """Trait for two-input elementwise operations.

    Implement this trait to define custom binary operations that can be
    dispatched through apply_binary[]. The operation receives two Float64
    values and returns the result as Float64.

    Required Methods:
        `__init__`: Default constructor
        `apply`: Apply the operation to two values

    Contract:
        - Must have a default constructor (no arguments)
        - apply() must be deterministic
        - apply() must handle all Float64 value pairs
        - apply() should not have side effects
        - apply(a, b) should not require a.shape == b.shape
          (broadcasting is handled by apply_binary)

    Example:
        struct MaxOp(ElementwiseBinaryOp):
            '''Element-wise maximum.'''
            fn __init__(out self):
                pass

            fn apply(self, a: Float64, b: Float64) -> Float64:
                if a > b:
                    return a
                else:
                    return b

        var max_result = apply_binary[MaxOp](tensor_a, tensor_b)
    """

    fn __init__(out self):
        """Default constructor required for generic instantiation."""
        ...

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Apply operation to two values.

        Args:
            a: First input value (Float64).
            b: Second input value (Float64).

        Returns:
            Result value (Float64).

        Note:
            Must handle all Float64 value pairs.
        """
        ...


# ============================================================================
# Unary Operation Dispatcher
# ============================================================================


fn apply_unary[Op: ElementwiseUnaryOp](input: ExTensor) raises -> ExTensor:
    """Apply unary operation to all elements in tensor.

    Applies the operation defined by Op to each element of the input tensor,
    returning a new tensor with the results. Uses dtype preservation.

    Args:
        input: Input tensor.

    Returns:
        New tensor with operation applied to each element.

    Raises:
        Error: If operation fails (e.g., invalid value for operation).

    Type Parameters:
        Op: Must implement ElementwiseUnaryOp trait.

    Example:
    ```
        # Apply exponential
        var exp_tensor = apply_unary[ExpOp](tensor)

        # Apply custom operation
        struct MyOp(ElementwiseUnaryOp):
            fn apply(self, value: Float64) -> Float64:
                return value + 1.0

        var result = apply_unary[MyOp](tensor)
    ```

    Note:
        - Preserves input dtype in output.
        - Single pass through data (linear time complexity).
        - Zero-copy for operation implementation.
    """
    var result = ExTensor(input.shape(), input.dtype())
    var op = Op()

    var numel = input.numel()
    for i in range(numel):
        var val = input._get_float64(i)
        var result_val = op.apply(val)
        result._set_float64(i, result_val)

    return result^


# ============================================================================
# Binary Operation Dispatcher (Element-wise)
# ============================================================================


fn apply_binary[
    Op: ElementwiseBinaryOp
](a: ExTensor, b: ExTensor) raises -> ExTensor:
    """Apply binary operation to all element pairs in two tensors.

    Applies the operation defined by Op to each pair of elements from
    the input tensors, returning a new tensor with the results.
    Both tensors must have the same shape and dtype.

    Args:
        a: First input tensor.
        b: Second input tensor.

    Returns:
        New tensor with operation applied to each element pair.

    Raises:
        Error: If shapes don't match, dtypes don't match, or operation fails.

    Type Parameters:
        Op: Must implement ElementwiseBinaryOp trait.

    Example:
    ```
        # Add two tensors
        var sum_tensor = apply_binary[AddOp](tensor_a, tensor_b)

        # Element-wise multiplication
        var prod_tensor = apply_binary[MulOp](tensor_a, tensor_b)

        # Custom operation
        struct DiffOp(ElementwiseBinaryOp):
            fn apply(self, a: Float64, b: Float64) -> Float64:
                return a - b

        var difference = apply_binary[DiffOp](tensor_a, tensor_b)
    ```

    Note:
        - Both tensors must have exact same shape and dtype.
        - Output dtype matches input dtype.
        - Single pass through data (linear time complexity).
    """
    # Validate shapes match
    if len(a.shape()) != len(b.shape()):
        raise Error("apply_binary: tensors must have same number of dimensions")

    for i in range(len(a.shape())):
        if a.shape()[i] != b.shape()[i]:
            raise Error("apply_binary: tensors must have same shape")

    # Validate dtypes match
    if a.dtype() != b.dtype():
        raise Error("apply_binary: tensors must have same dtype")

    var result = ExTensor(a.shape(), a.dtype())
    var op = Op()

    var numel = a.numel()
    for i in range(numel):
        var a_val = a._get_float64(i)
        var b_val = b._get_float64(i)
        var result_val = op.apply(a_val, b_val)
        result._set_float64(i, result_val)

    return result^


# ============================================================================
# Predefined Unary Operations
# ============================================================================


struct ExpOp(ElementwiseUnaryOp):
    """Exponential operation: e^x."""

    fn __init__(out self):
        """Initialize ExpOp."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute e^x.

        Args:
            value: Input value.

        Returns:
            Returns e^value.
        """
        return math_exp(value)


struct LogOp(ElementwiseUnaryOp):
    """Natural logarithm operation: ln(x).

    Note: Returns -inf for zero, NaN for negative values.
    """

    fn __init__(out self):
        """Initialize LogOp."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute ln(x). Returns -inf for 0, NaN for negative.

        Args:
            value: Input value.

        Returns:
            Returns ln(value), -inf for 0, NaN for negative.
        """
        # Return -inf for 0, NaN for negative (matches IEEE 754 behavior)
        return math_log(value)


struct SqrtOp(ElementwiseUnaryOp):
    """Square root operation: sqrt(x).

    Note: Returns NaN for negative values (IEEE 754 behavior).
    """

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute sqrt(x). Returns NaN for negative values.

        Args:
            value: Input value.

        Returns:
            Return sqrt(value), NaN for negative values.
        """
        # math_sqrt returns NaN for negative inputs (IEEE 754 behavior)
        return math_sqrt(value)


struct SinOp(ElementwiseUnaryOp):
    """Sine operation: sin(x)."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute sin(x).

        Args:
            value: Input value in radians.

        Returns:
            Return sin(value).
        """
        return math_sin(value)


struct CosOp(ElementwiseUnaryOp):
    """Cosine operation: cos(x)."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute cos(x).

        Args:
            value: Input value in radians.

        Returns:
            Return cos(value).
        """
        return math_cos(value)


struct TanhOp(ElementwiseUnaryOp):
    """Hyperbolic tangent operation: tanh(x)."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute tanh(x).

        Args:
            value: Input value.

        Returns:
            Return tanh(value).
        """
        return math_tanh(value)


struct AbsOp(ElementwiseUnaryOp):
    """Absolute value operation: |x|."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute |x|.

        Args:
            value: Input value.

        Returns:
            Absolute value of x.
        """
        if value >= 0.0:
            return value
        else:
            return -value


struct NegateOp(ElementwiseUnaryOp):
    """Negation operation: -x."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute -x.

        Args:
            value: Input value.

        Returns:
            Negation of x.
        """
        return -value


struct ReciprocalOp(ElementwiseUnaryOp):
    """Reciprocal operation: 1/x.

    Note: Returns inf for zero (IEEE 754 behavior).
    """

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute 1/x. Returns inf for zero.

        Args:
            value: Input value.

        Returns:
            Return 1/value, inf for zero.
        """
        # Division by zero returns inf (IEEE 754 behavior)
        return 1.0 / value


struct SquareOp(ElementwiseUnaryOp):
    """Square operation: x^2."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute x^2.

        Args:
            value: Input value.

        Returns:
            Return x squared.
        """
        return value * value


struct SignOp(ElementwiseUnaryOp):
    """Sign operation: -1, 0, or 1."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, value: Float64) -> Float64:
        """Compute sign(x).

        Args:
            value: Input value.

        Returns:
            Return 1.0 if positive, -1.0 if negative, 0.0 otherwise.
        """
        if value > 0.0:
            return 1.0
        elif value < 0.0:
            return -1.0
        else:
            return 0.0


# ============================================================================
# Predefined Binary Operations
# ============================================================================


struct AddOp(ElementwiseBinaryOp):
    """Addition operation: a + b."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a + b.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return a + b.
        """
        return a + b


struct SubtractOp(ElementwiseBinaryOp):
    """Subtraction operation: a - b."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a - b.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return a - b.
        """
        return a - b


struct MultiplyOp(ElementwiseBinaryOp):
    """Multiplication operation: a * b."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a * b.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return a * b.
        """
        return a * b


struct DivideOp(ElementwiseBinaryOp):
    """Division operation: a / b.

    Note: Returns inf for division by zero (IEEE 754 behavior).
    """

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a / b. Returns inf for division by zero.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return a / b, inf for division by zero.
        """
        # Division by zero returns inf (IEEE 754 behavior)
        return a / b


struct PowerOp(ElementwiseBinaryOp):
    """Power operation: a ^ b.

    Note: Returns inf for 0^negative, NaN for negative base with non-integer exponent.
    """

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute a ^ b (a to the power of b).

        Args:
            a: Base value.
            b: Exponent value.

        Returns:
            Return a^b using IEEE 754 semantics.

        Uses IEEE 754 semantics:
        ```
        - 0^negative = inf
        - negative^non-integer = NaN
        - 0^0 = 1 (by convention)
        ```
        """
        # Use exp and log for general case: a^b = exp(b * ln(a))
        # This naturally handles edge cases with IEEE 754 semantics:
        # - 0^negative: log(0) = -inf, -inf * negative = +inf, exp(+inf) = inf
        # - negative base: log(negative) = NaN, NaN propagates
        # - 0^0: handled specially
        if a == 0.0 and b == 0.0:
            return 1.0  # 0^0 = 1 by convention
        elif a == 0.0 and b > 0.0:
            return 0.0  # 0^positive = 0
        else:
            # For all other cases, use exp(b * log(a))
            # This gives inf for 0^negative, NaN for negative^non-integer
            return math_exp(b * math_log(a))


struct MaxOp(ElementwiseBinaryOp):
    """Maximum operation: max(a, b)."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute max(a, b).

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Maximum of a and b.
        """
        if a >= b:
            return a
        else:
            return b


struct MinOp(ElementwiseBinaryOp):
    """Minimum operation: min(a, b)."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Compute min(a, b).

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Minimum of a and b.
        """
        if a <= b:
            return a
        else:
            return b


struct EqualOp(ElementwiseBinaryOp):
    """Equality comparison: (a == b) as 1.0 or 0.0."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a == b, else 0.0.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return 1.0 if a == b, else 0.0.
        """
        if a == b:
            return 1.0
        else:
            return 0.0


struct GreaterOp(ElementwiseBinaryOp):
    """Greater than comparison: (a > b) as 1.0 or 0.0."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a > b, else 0.0.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return 1.0 if a > b, else 0.0.
        """
        if a > b:
            return 1.0
        else:
            return 0.0


struct GreaterEqualOp(ElementwiseBinaryOp):
    """Greater than or equal comparison: (a >= b) as 1.0 or 0.0."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a >= b, else 0.0.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return 1.0 if a >= b, else 0.0.
        """
        if a >= b:
            return 1.0
        else:
            return 0.0


struct LessOp(ElementwiseBinaryOp):
    """Less than comparison: (a < b) as 1.0 or 0.0."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a < b, else 0.0.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return 1.0 if a < b, else 0.0.
        """
        if a < b:
            return 1.0
        else:
            return 0.0


struct LessEqualOp(ElementwiseBinaryOp):
    """Less than or equal comparison: (a <= b) as 1.0 or 0.0."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if a <= b, else 0.0.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return 1.0 if a <= b, else 0.0.
        """
        if a <= b:
            return 1.0
        else:
            return 0.0


struct LogicalAndOp(ElementwiseBinaryOp):
    """Logical AND: (a != 0 && b != 0) as 1.0 or 0.0."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if both non-zero, else 0.0.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return 1.0 if both non-zero, else 0.0.
        """
        if (a != 0.0) and (b != 0.0):
            return 1.0
        else:
            return 0.0


struct LogicalOrOp(ElementwiseBinaryOp):
    """Logical OR: (a != 0 || b != 0) as 1.0 or 0.0."""

    fn __init__(out self):
        """Initialize operation."""
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        """Return 1.0 if either non-zero, else 0.0.

        Args:
            a: First input value.
            b: Second input value.

        Returns:
            Return 1.0 if either non-zero, else 0.0.
        """
        if (a != 0.0) or (b != 0.0):
            return 1.0
        else:
            return 0.0
