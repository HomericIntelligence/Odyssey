"""Activation operation utilities for numerical stability and computation.

This module provides pure functional utilities for activation functions
Unlike activation.mojo which contains high-level activation functions
(ReLU, sigmoid, tanh, GELU, etc.), this module contains lower-level
helper functions for computation and numerical stability.

Functions:
    exp_scalar_f32: Compute exponential of a float32 scalar
    exp_scalar_f64: Compute exponential of a float64 scalar

Example:
   ```mojo
    from .activation_ops import exp_scalar_f32, exp_scalar_f64

    # Compute exponentials
    var result_f32 = exp_scalar_f32(-2.0)
    var result_f64 = exp_scalar_f64(-2.0)
    ```
"""


# ============================================================================
# Scalar Exponential Functions
# ============================================================================


@always_inline
fn exp_scalar_f32(x: Float32) -> Float32:
    """Compute exp of a scalar float32.

    Uses the power operator (2.718281828459045 ** x) to compute exponential.
    This is a stable implementation suitable for activation function computation.

    Args:
        x: Input float32 value.

    Returns:
        Exponential of x: e^x.

    Note:
        For very negative values (x << -20), the result approaches zero.
        For very positive values (x >> 20), the result may overflow to infinity.
        Callers should handle clipping for numerical stability when needed.

    Example:
        ```mojo
        from .activation_ops import exp_scalar_f32

        var result = exp_scalar_f32(1.0)  # ≈ 2.71828
        var small = exp_scalar_f32(-10.0)  # ≈ 4.5e-5
        ```
    """
    # Using power operator for exponential
    return Float32(2.718281828459045) ** x


@always_inline
fn exp_scalar_f64(x: Float64) -> Float64:
    """Compute exp of a scalar float64.

    Uses the power operator (2.718281828459045 ** x) to compute exponential.
    This is a stable implementation suitable for activation function computation
    with higher precision than float32.

    Args:
        x: Input float64 value.

    Returns:
        Exponential of x: e^x.

    Note:
        For very negative values (x << -20), the result approaches zero.
        For very positive values (x >> 20), the result may overflow to infinity.
        Callers should handle clipping for numerical stability when needed.

    Example:
        ```mojo
        from .activation_ops import exp_scalar_f64

        var result = exp_scalar_f64(1.0)  # ≈ 2.718281828459045
        var small = exp_scalar_f64(-10.0)  # ≈ 4.54e-5
        ```
    """
    return 2.718281828459045**x
