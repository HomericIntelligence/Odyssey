"""Mathematical constants for ML operations.

This module provides centralized mathematical constants used across
the codebase for activation functions, initializers, and elementwise ops.
"""

# Pi and related constants
comptime PI: Float64 = 3.14159265358979323846  # Mathematical constant pi

# Square roots
comptime SQRT_2: Float64 = 1.4142135623730951  # Square root of 2
comptime SQRT_2_OVER_PI: Float64 = 0.7978845608028654  # sqrt(2/pi) for GELU
comptime INV_SQRT_2PI: Float64 = 0.3989422804014327  # 1/sqrt(2*pi) for normal distribution

# GELU activation constants
comptime GELU_COEFF: Float64 = 0.044715  # Coefficient in GELU approximation

# Logarithms
comptime LN2: Float64 = 0.6931471805599453  # Natural log of 2
comptime LN10: Float64 = 2.302585092994046  # Natural log of 10
