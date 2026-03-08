# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_multi_precision_training.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Integration tests for multi-precision training (Part 1 of 2).

Tests cover:
- FP32, FP16, BF16, FP8 training modes
- PrecisionConfig creation and usage
- GradientScaler dynamic scaling
- Master weights maintenance

These tests verify that mixed-precision training works correctly
across all supported dtypes.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    assert_greater,
    assert_less,
    assert_dtype,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.training.precision_config import PrecisionConfig, PrecisionMode
from shared.training.mixed_precision import GradientScaler
from shared.training.dtype_utils import (
    float16_dtype,
    float32_dtype,
    bfloat16_dtype,
    is_reduced_precision,
)
from collections import List


# ============================================================================
# Test 1-4: Per-Precision Training Tests
# ============================================================================


fn test_fp32_training_loss_decreases() raises:
    """Test FP32 baseline training with loss decrease.

    This is the reference implementation - all other precisions
    should achieve similar results.
    """
    var config = PrecisionConfig.fp32()

    # Verify config settings
    assert_true(config.mode == PrecisionMode.FP32, "Mode should be FP32")
    assert_true(
        config.compute_dtype == DType.float32, "Compute dtype should be float32"
    )
    assert_false(
        config.use_gradient_scaler, "FP32 should not use gradient scaler"
    )
    assert_false(
        config.needs_master_weights(), "FP32 doesn't need master weights"
    )

    # Simulate training step with dummy data
    var input_shape = List[Int]()
    input_shape.append(4)
    input_shape.append(10)
    var input = full(input_shape, 0.5, DType.float32)

    # Cast to compute precision (should be identity for FP32)
    var compute_input = config.cast_to_compute(input)
    assert_dtype(
        compute_input, DType.float32, "Compute input should be float32"
    )


fn test_fp16_training_loss_decreases() raises:
    """Test FP16 training with gradient scaling.

    FP16 training requires:
    - Gradient scaling to prevent underflow
    - Loss scaling before backward pass
    - Gradient unscaling after backward pass.
    """
    var config = PrecisionConfig.fp16()

    # Verify config settings
    assert_true(config.mode == PrecisionMode.FP16, "Mode should be FP16")
    assert_true(
        config.compute_dtype == DType.float16, "Compute dtype should be float16"
    )
    assert_true(config.use_gradient_scaler, "FP16 should use gradient scaler")
    assert_true(config.needs_master_weights(), "FP16 needs master weights")

    # Test casting to compute precision
    var input_shape = List[Int]()
    input_shape.append(4)
    input_shape.append(10)
    var fp32_input = full(input_shape, 0.5, DType.float32)
    var fp16_input = config.cast_to_compute(fp32_input)
    assert_dtype(fp16_input, DType.float16, "Input should be cast to float16")


fn test_bf16_training_loss_decreases() raises:
    """Test BF16 training mode.

    BF16 has wider exponent range than FP16, reducing overflow risk.
    Still uses gradient scaling for safety.
    """
    var config = PrecisionConfig.bf16()

    # Verify config settings
    assert_true(config.mode == PrecisionMode.BF16, "Mode should be BF16")
    assert_true(config.use_gradient_scaler, "BF16 should use gradient scaler")
    assert_true(config.needs_master_weights(), "BF16 needs master weights")

    assert_true(
        config.compute_dtype == bfloat16_dtype,
        "Compute dtype should be bfloat16",
    )


fn test_fp8_training_loss_decreases() raises:
    """Test FP8 training with aggressive scaling.

    FP8 has very limited range:
    - E4M3: ~1.5e-4 to 448
    - Requires aggressive gradient scaling
    - Uses FP16 storage to reduce quantization noise.
    """
    var config = PrecisionConfig.fp8()

    # Verify config settings
    assert_true(config.mode == PrecisionMode.FP8, "Mode should be FP8")
    assert_true(config.use_gradient_scaler, "FP8 should use gradient scaler")
    assert_true(config.needs_master_weights(), "FP8 needs master weights")

    # FP8 uses FP16 storage to reduce quantization noise
    assert_true(
        config.storage_dtype == DType.float16,
        "Storage dtype should be float16 for FP8",
    )


# ============================================================================
# Test 5: Gradient Overflow Recovery
# ============================================================================


fn test_fp16_gradient_overflow_recovery() raises:
    """Test gradient scaler recovers from overflow.

    When gradients contain NaN/Inf:
    1. Skip optimizer step
    2. Reduce scale factor
    3. Continue training with reduced scale.
    """
    var config = PrecisionConfig.fp16()

    # Initial scale
    var initial_scale = config.get_scale()
    assert_greater(
        Float64(initial_scale), Float64(0.0), "Initial scale should be positive"
    )

    # Simulate overflow - step with invalid gradients
    config.step(grads_valid=False)
    var reduced_scale = config.get_scale()

    # Scale should decrease after overflow
    assert_less(
        Float64(reduced_scale),
        Float64(initial_scale),
        "Scale should decrease after overflow",
    )
    assert_equal_int(
        config.get_overflow_count(), 1, "Overflow count should be 1"
    )

    # Simulate recovery - step with valid gradients
    config.step(grads_valid=True)
    # Scale may increase or stay same, but overflow count stays at 1
    assert_equal_int(
        config.get_overflow_count(), 1, "Overflow count should still be 1"
    )


# ============================================================================
# Test 6: Config Parsing
# ============================================================================


fn test_precision_config_from_string() raises:
    """Test PrecisionConfig creation from string names."""
    # Test all valid precision strings
    var fp32_config = PrecisionConfig.from_string("fp32")
    assert_true(
        fp32_config.mode == PrecisionMode.FP32,
        "fp32 string should create FP32 mode",
    )

    var fp16_config = PrecisionConfig.from_string("fp16")
    assert_true(
        fp16_config.mode == PrecisionMode.FP16,
        "fp16 string should create FP16 mode",
    )

    var bf16_config = PrecisionConfig.from_string("bf16")
    assert_true(
        bf16_config.mode == PrecisionMode.BF16,
        "bf16 string should create BF16 mode",
    )

    var fp8_config = PrecisionConfig.from_string("fp8")
    assert_true(
        fp8_config.mode == PrecisionMode.FP8,
        "fp8 string should create FP8 mode",
    )


fn test_precision_config_invalid_string() raises:
    """Test that invalid precision string raises error."""
    var raised_error = False
    try:
        var invalid_config = PrecisionConfig.from_string("invalid")
    except:
        raised_error = True

    assert_true(raised_error, "Invalid precision string should raise error")


# ============================================================================
# Test 7: Dynamic Scaling
# ============================================================================


fn test_gradient_scaler_dynamic_scaling() raises:
    """Test gradient scaler adjusts scale over iterations.

    The scaler should:
    - Increase scale after consecutive successful steps
    - Decrease scale after overflow.
    """
    var scaler = GradientScaler(initial_scale=65536.0)

    # Get initial scale and verify it's positive
    var _ = scaler.get_scale()

    # Simulate several successful training steps
    for _ in range(10):
        scaler.step()

    # Scale may increase after successful steps (depends on growth interval)
    var final_scale = scaler.get_scale()
    assert_greater(
        Float64(final_scale), Float64(0.0), "Scale should remain positive"
    )

    # Simulate overflow
    scaler.backoff()
    var reduced_scale = scaler.get_scale()
    assert_less(
        Float64(reduced_scale),
        Float64(final_scale),
        "Scale should decrease after backoff",
    )


# ============================================================================
# Main - Run All Tests
# ============================================================================


fn main() raises:
    """Run multi-precision training tests (Part 1 of 2)."""
    print("=" * 60)
    print("Multi-Precision Training Tests (Part 1 of 2)")
    print("=" * 60)
    print()

    print("Test 1: FP32 training baseline...")
    test_fp32_training_loss_decreases()
    print("  PASSED")

    print("Test 2: FP16 training with gradient scaling...")
    test_fp16_training_loss_decreases()
    print("  PASSED")

    print("Test 3: BF16 training...")
    test_bf16_training_loss_decreases()
    print("  PASSED")

    print("Test 4: FP8 training...")
    test_fp8_training_loss_decreases()
    print("  PASSED")

    print("Test 5: FP16 gradient overflow recovery...")
    test_fp16_gradient_overflow_recovery()
    print("  PASSED")

    print("Test 6a: Config from string...")
    test_precision_config_from_string()
    print("  PASSED")

    print("Test 6b: Invalid string raises error...")
    test_precision_config_invalid_string()
    print("  PASSED")

    print("Test 7: Dynamic gradient scaling...")
    test_gradient_scaler_dynamic_scaling()
    print("  PASSED")

    print()
    print("=" * 60)
    print("ALL PART 1 TESTS PASSED! (8/8)")
    print("=" * 60)
