"""Unit tests for constants in shared.core and related modules.

Tests verify that all constants are properly defined with correct values
and that they match their documented usage throughout the codebase.
"""

from shared.base.numerical_constants import (
    EPSILON_DIV,
    EPSILON_LOSS,
    EPSILON_NORM,
    GRADIENT_MAX_NORM,
    GRADIENT_MIN_NORM,
    EPSILON_OPTIMIZER_ADAM,
    EPSILON_OPTIMIZER_ADAGRAD,
    EPSILON_OPTIMIZER_RMSPROP,
    EPSILON_NUMERICAL_GRAD,
    EPSILON_RELATIVE_ERROR,
)
from shared.core.activation_constants import (
    RELU6_UPPER_BOUND,
    SIGMOID_CLIP_THRESHOLD,
    HARD_SIGMOID_OFFSET,
    HARD_SIGMOID_SCALE,
    HARD_TANH_LOWER_BOUND,
    HARD_TANH_UPPER_BOUND,
)
from shared.core.optimizer_constants import (
    DEFAULT_LEARNING_RATE_SGD,
    DEFAULT_LEARNING_RATE_ADAM,
    DEFAULT_MOMENTUM,
    DEFAULT_ADAM_BETA1,
    DEFAULT_ADAM_BETA2,
    DEFAULT_ADAM_EPSILON,
    DEFAULT_RMSPROP_ALPHA,
    DEFAULT_RMSPROP_EPSILON,
    DEFAULT_ADAGRAD_EPSILON,
)
from shared.core.types.fp_constants import (
    FP8_E4M3_MIN_NORMAL,
    FP8_E4M3_MAX_NORMAL,
    FP4_E2M1_MAX_NORMAL,
    FP4_E2M1_MIN_SUBNORMAL,
    FP4_E2M1_MANTISSA_SCALE,
    BF8_E5M2_SATURATION,
    BF8_E5M2_MANTISSA_SCALE,
    STOCHASTIC_ROUNDING_SCALE,
)
from shared.testing.tolerance_constants import (
    TOLERANCE_DEFAULT,
    TOLERANCE_FLOAT16,
    TOLERANCE_FLOAT32,
    TOLERANCE_FLOAT64,
    TOLERANCE_GRADIENT_RTOL,
    TOLERANCE_GRADIENT_ATOL,
    GRADIENT_CHECK_EPSILON,
    TOLERANCE_CONV,
    TOLERANCE_SOFTMAX,
    TOLERANCE_CROSS_ENTROPY,
)


fn test_epsilon_values() raises:
    """Verify epsilon constants are positive and appropriately small."""
    if not (EPSILON_DIV > 0.0 and EPSILON_DIV < 1e-5):
        raise Error("EPSILON_DIV validation failed")

    if not (EPSILON_LOSS > 0.0 and EPSILON_LOSS < 1e-3):
        raise Error("EPSILON_LOSS validation failed")

    if not (EPSILON_NORM > 0.0 and EPSILON_NORM < 1e-3):
        raise Error("EPSILON_NORM validation failed")

    if EPSILON_OPTIMIZER_ADAM != 1e-8:
        raise Error("EPSILON_OPTIMIZER_ADAM validation failed")

    if EPSILON_OPTIMIZER_ADAGRAD != 1e-10:
        raise Error("EPSILON_OPTIMIZER_ADAGRAD validation failed")

    if EPSILON_OPTIMIZER_RMSPROP != 1e-8:
        raise Error("EPSILON_OPTIMIZER_RMSPROP validation failed")

    if EPSILON_NUMERICAL_GRAD != 1e-5:
        raise Error("EPSILON_NUMERICAL_GRAD validation failed")

    if EPSILON_RELATIVE_ERROR != 1e-8:
        raise Error("EPSILON_RELATIVE_ERROR validation failed")


fn test_gradient_thresholds() raises:
    """Verify gradient safety thresholds are in reasonable ranges."""
    if GRADIENT_MAX_NORM != 1000.0:
        raise Error("GRADIENT_MAX_NORM validation failed")

    if not (GRADIENT_MIN_NORM > 0.0 and GRADIENT_MIN_NORM < 1e-3):
        raise Error("GRADIENT_MIN_NORM validation failed")


fn test_activation_bounds() raises:
    """Verify activation function constants match documented values."""
    if RELU6_UPPER_BOUND != 6.0:
        raise Error("RELU6_UPPER_BOUND should be 6.0")

    if SIGMOID_CLIP_THRESHOLD != 20.0:
        raise Error("SIGMOID_CLIP_THRESHOLD should be 20.0")

    if HARD_SIGMOID_OFFSET != 3.0:
        raise Error("HARD_SIGMOID_OFFSET should be 3.0")

    if HARD_SIGMOID_SCALE != 6.0:
        raise Error("HARD_SIGMOID_SCALE should be 6.0")

    if HARD_TANH_LOWER_BOUND != -1.0:
        raise Error("HARD_TANH_LOWER_BOUND should be -1.0")

    if HARD_TANH_UPPER_BOUND != 1.0:
        raise Error("HARD_TANH_UPPER_BOUND should be 1.0")


fn test_optimizer_defaults() raises:
    """Verify optimizer defaults are within reasonable ranges."""
    if not (0.0 < DEFAULT_LEARNING_RATE_SGD < 1.0):
        raise Error("DEFAULT_LEARNING_RATE_SGD out of range")

    if not (0.0 < DEFAULT_LEARNING_RATE_ADAM < 1.0):
        raise Error("DEFAULT_LEARNING_RATE_ADAM out of range")

    if not (0.0 <= DEFAULT_MOMENTUM < 1.0):
        raise Error("DEFAULT_MOMENTUM out of range")

    if not (0.0 < DEFAULT_ADAM_BETA1 < 1.0):
        raise Error("DEFAULT_ADAM_BETA1 out of range")

    if not (0.0 < DEFAULT_ADAM_BETA2 < 1.0):
        raise Error("DEFAULT_ADAM_BETA2 out of range")

    if not (DEFAULT_ADAM_BETA1 < DEFAULT_ADAM_BETA2):
        raise Error("ADAM_BETA1 should be < ADAM_BETA2")

    if DEFAULT_ADAM_EPSILON != 1e-8:
        raise Error("DEFAULT_ADAM_EPSILON should be 1e-8")

    if DEFAULT_ADAGRAD_EPSILON != 1e-10:
        raise Error("DEFAULT_ADAGRAD_EPSILON should be 1e-10")

    if DEFAULT_RMSPROP_EPSILON != 1e-8:
        raise Error("DEFAULT_RMSPROP_EPSILON should be 1e-8")

    if not (0.0 < DEFAULT_RMSPROP_ALPHA < 1.0):
        raise Error("DEFAULT_RMSPROP_ALPHA out of range")


fn test_fp_type_constants() raises:
    """Verify FP type constants match IEEE/vendor specifications."""
    if FP8_E4M3_MAX_NORMAL != 448.0:
        raise Error("FP8_E4M3_MAX_NORMAL should be 448.0")

    if FP4_E2M1_MAX_NORMAL != 6.0:
        raise Error("FP4_E2M1_MAX_NORMAL should be 6.0")

    if FP4_E2M1_MIN_SUBNORMAL != 0.5:
        raise Error("FP4_E2M1_MIN_SUBNORMAL should be 0.5")

    if FP4_E2M1_MANTISSA_SCALE != 0.5:
        raise Error("FP4_E2M1_MANTISSA_SCALE should be 0.5")

    if BF8_E5M2_SATURATION != 57344.0:
        raise Error("BF8_E5M2_SATURATION should be 57344.0")

    if BF8_E5M2_MANTISSA_SCALE != 16384.0:
        raise Error("BF8_E5M2_MANTISSA_SCALE should be 16384.0")

    if STOCHASTIC_ROUNDING_SCALE != 16777216.0:
        raise Error("STOCHASTIC_ROUNDING_SCALE should be 2^24")


fn test_tolerance_constants() raises:
    """Verify testing tolerance constants are positive and appropriately small.
    """
    if not (TOLERANCE_DEFAULT > 0.0):
        raise Error("TOLERANCE_DEFAULT must be positive")

    if not (TOLERANCE_FLOAT16 > 0.0):
        raise Error("TOLERANCE_FLOAT16 must be positive")

    if not (TOLERANCE_FLOAT32 > 0.0):
        raise Error("TOLERANCE_FLOAT32 must be positive")

    if not (TOLERANCE_FLOAT64 > 0.0):
        raise Error("TOLERANCE_FLOAT64 must be positive")

    if not (TOLERANCE_FLOAT16 > TOLERANCE_FLOAT32):
        raise Error("Float16 tolerance should be larger than float32")

    if not (TOLERANCE_FLOAT32 > TOLERANCE_FLOAT64):
        raise Error("Float32 tolerance should be larger than float64")

    if TOLERANCE_GRADIENT_RTOL != 1e-2:
        raise Error("TOLERANCE_GRADIENT_RTOL should be 1e-2")

    if TOLERANCE_GRADIENT_ATOL != 1e-2:
        raise Error("TOLERANCE_GRADIENT_ATOL should be 1e-2")

    if GRADIENT_CHECK_EPSILON != 1e-5:
        raise Error("GRADIENT_CHECK_EPSILON should be 1e-5")

    if not (TOLERANCE_CONV > 0.0):
        raise Error("TOLERANCE_CONV must be positive")

    if not (TOLERANCE_SOFTMAX > 0.0):
        raise Error("TOLERANCE_SOFTMAX must be positive")

    if not (TOLERANCE_CROSS_ENTROPY > 0.0):
        raise Error("TOLERANCE_CROSS_ENTROPY must be positive")


fn main():
    """Main test runner."""
    try:
        test_epsilon_values()
        print("✓ test_epsilon_values passed")
    except e:
        print("✗ test_epsilon_values failed")

    try:
        test_gradient_thresholds()
        print("✓ test_gradient_thresholds passed")
    except e:
        print("✗ test_gradient_thresholds failed")

    try:
        test_activation_bounds()
        print("✓ test_activation_bounds passed")
    except e:
        print("✗ test_activation_bounds failed")

    try:
        test_optimizer_defaults()
        print("✓ test_optimizer_defaults passed")
    except e:
        print("✗ test_optimizer_defaults failed")

    try:
        test_fp_type_constants()
        print("✓ test_fp_type_constants passed")
    except e:
        print("✗ test_fp_type_constants failed")

    try:
        test_tolerance_constants()
        print("✓ test_tolerance_constants passed")
    except e:
        print("✗ test_tolerance_constants failed")

    print("All tests completed!")
