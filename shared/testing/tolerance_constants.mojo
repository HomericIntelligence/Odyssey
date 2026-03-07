"""Testing tolerance constants for numerical comparisons.

This module provides centralized tolerance values for numerical comparisons
in unit tests across different data types and operations.

Constants:
    TOLERANCE_DEFAULT: Default tolerance for float32 comparisons (1e-6)
    TOLERANCE_FLOAT16: Tolerance for float16 operations (1e-3)
    TOLERANCE_FLOAT32: Tolerance for float32 operations (1e-5)
    TOLERANCE_FLOAT64: Tolerance for float64 operations (1e-10)
    TOLERANCE_GRADIENT_RTOL: Relative tolerance for gradient checks (1e-2)
    TOLERANCE_GRADIENT_ATOL: Absolute tolerance for gradient checks (1e-2)
    GRADIENT_CHECK_EPSILON: Epsilon for numerical gradient computation (1e-5)
    GRADIENT_CHECK_EPSILON_FLOAT32: Epsilon for float32 gradient checking (3e-4)
    GRADIENT_CHECK_EPSILON_OTHER: Epsilon for non-float32 gradient checking (1e-3)
    TOLERANCE_CONV: Tolerance for convolutional layer comparisons (1e-3)
    TOLERANCE_SOFTMAX: Tolerance for softmax operations (5e-4)
    TOLERANCE_CROSS_ENTROPY: Tolerance for cross-entropy loss (1e-3)
"""

# Default tolerance for general float32 comparisons
comptime TOLERANCE_DEFAULT: Float64 = 1e-6

# Dtype-specific tolerances
comptime TOLERANCE_FLOAT16: Float64 = 1e-3
comptime TOLERANCE_FLOAT32: Float64 = 1e-5
comptime TOLERANCE_FLOAT64: Float64 = 1e-10

# Gradient checking tolerances
comptime TOLERANCE_GRADIENT_RTOL: Float64 = 1e-2
comptime TOLERANCE_GRADIENT_ATOL: Float64 = 1e-2

# Epsilon for numerical gradient computation
comptime GRADIENT_CHECK_EPSILON: Float64 = 1e-5

# Float32-specific epsilon (3e-4): avoids precision loss in matmul (see issue #2704).
# Using 1e-5 causes ~56% precision loss; 3e-4 gives <1.5% gradient error.
comptime GRADIENT_CHECK_EPSILON_FLOAT32: Float64 = 3e-4

# Epsilon for non-float32 dtypes (BF16, FP16) in gradient checking.
comptime GRADIENT_CHECK_EPSILON_OTHER: Float64 = 1e-3

# Operation-specific tolerances
comptime TOLERANCE_CONV: Float64 = 1e-3
comptime TOLERANCE_SOFTMAX: Float64 = 5e-4
comptime TOLERANCE_CROSS_ENTROPY: Float64 = 1e-3
