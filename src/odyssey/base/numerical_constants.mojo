"""Numerical stability constants for ML operations.

This module provides centralized epsilon and threshold values used
across the codebase for numerical stability in various operations.
"""

# Division safety - prevents division by zero
comptime EPSILON_DIV: Float64 = 1e-10  # Small constant to prevent division by zero

# Loss function stability - for log operations in BCE, cross-entropy, etc.
comptime EPSILON_LOSS: Float64 = 1e-7  # Small constant for numerical stability in log operations

# Normalization stability - for BatchNorm, LayerNorm, GroupNorm, InstanceNorm
comptime EPSILON_NORM: Float64 = 1e-5  # Small constant for numerical stability in normalization

# Gradient safety thresholds
comptime GRADIENT_MAX_NORM: Float64 = 1000.0  # Threshold for gradient explosion detection
comptime GRADIENT_MIN_NORM: Float64 = 1e-7  # Threshold for gradient vanishing detection

# Optimizer epsilon values for numerical stability
comptime EPSILON_OPTIMIZER_ADAM: Float64 = 1e-8  # Default epsilon for Adam optimizer
comptime EPSILON_OPTIMIZER_ADAGRAD: Float64 = 1e-10  # Default epsilon for AdaGrad optimizer
comptime EPSILON_OPTIMIZER_RMSPROP: Float64 = 1e-8  # Default epsilon for RMSprop optimizer

# Numerical gradient computation epsilon
comptime EPSILON_NUMERICAL_GRAD: Float64 = 1e-5  # Epsilon for finite difference gradient checking
comptime EPSILON_RELATIVE_ERROR: Float64 = 1e-8  # Epsilon for relative error checking
