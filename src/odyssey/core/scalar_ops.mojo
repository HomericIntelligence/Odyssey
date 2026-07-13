"""Scalar mathematical operations.

This module provides pure functional scalar operations for floating-point values.
These are used throughout the library for numerical computations with single values.

Functions:
    sqrt_scalar_f32: Compute square root of a float32 scalar
    sqrt_scalar_f64: Compute square root of a float64 scalar
    pow_scalar_f32: Compute power (x^y) for float32 scalars
    pow_scalar_f64: Compute power (x^y) for float64 scalars

Example:
   ```mojo
   from .scalar_ops import sqrt_scalar_f32, pow_scalar_f32

   var x = Float32(4.0)
   var sqrt_x = sqrt_scalar_f32(x)  # 2.0

   var base = Float32(2.0)
   var exp = Float32(3.0)
   var result = pow_scalar_f32(base, exp)  # 8.0
   ```
"""


# ============================================================================
# Scalar Helper Functions
# ============================================================================


@always_inline
def sqrt_scalar_f32(x: Float32) -> Float32:
    """Compute square root of a scalar float32.

    Args:
        x: Input float32 value.

    Returns:
        Square root of x.

    Example:
       ```mojo
       from .scalar_ops import sqrt_scalar_f32

       var result = sqrt_scalar_f32(Float32(4.0))  # 2.0
       ```
    """
    return x**0.5


@always_inline
def sqrt_scalar_f64(x: Float64) -> Float64:
    """Compute square root of a scalar float64.

    Args:
        x: Input float64 value.

    Returns:
        Square root of x.

    Example:
       ```mojo
       from .scalar_ops import sqrt_scalar_f64

       var result = sqrt_scalar_f64(4.0)  # 2.0
       ```
    """
    return x**0.5


@always_inline
def pow_scalar_f32(x: Float32, y: Float32) -> Float32:
    """Compute x^y for scalar float32 values.

    Args:
        x: Base value (float32).
        y: Exponent value (float32).

    Returns:
        Result of x^y

    Example:
       ```mojo
       from .scalar_ops import pow_scalar_f32

       var result = pow_scalar_f32(Float32(2.0), Float32(3.0))  # 8.0
       ```
    """
    return x**y


@always_inline
def pow_scalar_f64(x: Float64, y: Float64) -> Float64:
    """Compute x^y for scalar float64 values.

    Args:
        x: Base value (float64).
        y: Exponent value (float64).

    Returns:
        Result of x^y

    Example:
       ```mojo
       from .scalar_ops import pow_scalar_f64

       var result = pow_scalar_f64(2.0, 3.0)  # 8.0
       ```
    """
    return x**y
