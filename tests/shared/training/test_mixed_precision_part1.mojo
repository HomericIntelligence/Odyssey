# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_mixed_precision.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for mixed precision training infrastructure (part 1).

Tests gradient scaler initialization, loss scaling, gradient unscaling,
scaler step updates, backoff, min/max limits, and FP32 master weight conversion.
"""

from shared.tensor.any_tensor import AnyTensor, full
from shared.training.mixed_precision import (
    GradientScaler,
    convert_to_fp32_master,
    update_model_from_master,
)
from testing import assert_equal, assert_true


fn test_gradient_scaler_initialization() raises:
    """Test GradientScaler initializes with correct default values."""
    print("Testing GradientScaler initialization...")

    var scaler = GradientScaler()

    assert_equal(scaler.scale, 65536.0, "Default scale should be 65536.0")
    assert_equal(
        scaler.growth_factor, 2.0, "Default growth factor should be 2.0"
    )
    assert_equal(
        scaler.backoff_factor, 0.5, "Default backoff factor should be 0.5"
    )
    assert_equal(
        scaler.growth_interval, 2000, "Default growth interval should be 2000"
    )
    assert_equal(scaler.get_num_steps(), 0, "Initial steps should be 0")

    print("✓ GradientScaler initialization test passed")


fn test_loss_scaling() raises:
    """Test loss scaling multiplies by scale factor."""
    print("Testing loss scaling...")

    var scaler = GradientScaler(initial_scale=1024.0)

    # Create a simple loss tensor (scalar)
    var loss = full([1], 0.5, DType.float32)

    # Scale the loss
    var scaled_loss = scaler.scale_loss(loss)

    # Check scaled value (0.5 * 1024 = 512)
    var scaled_val = scaled_loss._get_float64(0)
    assert_true(abs(scaled_val - 512.0) < 1e-5, "Scaled loss should be 512.0")

    print("✓ Loss scaling test passed")


fn test_gradient_unscaling() raises:
    """Test gradient unscaling divides by scale factor."""
    print("Testing gradient unscaling...")

    var scaler = GradientScaler(initial_scale=1024.0)

    # Create scaled gradients
    var scaled_grads = full([1], 2048.0, DType.float32)

    # Unscale the gradients (2048 / 1024 = 2.0)
    var unscaled_grads = scaler.unscale_gradients(scaled_grads)

    # Check unscaled value
    var val = unscaled_grads._get_float64(0)
    assert_true(abs(val - 2.0) < 1e-5, "Unscaled gradient should be 2.0")

    print("✓ Gradient unscaling test passed")


fn test_scaler_step_updates() raises:
    """Test scaler step increases scale after growth interval."""
    print("Testing scaler step updates...")

    var scaler = GradientScaler(
        initial_scale=1024.0, growth_factor=2.0, growth_interval=100
    )

    # Take 99 steps (not enough to trigger growth)
    for i in range(99):
        scaler.step()

    assert_equal(
        scaler.get_scale(),
        1024.0,
        "Scale should not increase before growth interval",
    )

    # Take one more step (100 total - should trigger growth)
    scaler.step()

    assert_equal(
        scaler.get_scale(), 2048.0, "Scale should double after growth interval"
    )

    print("✓ Scaler step updates test passed")


fn test_scaler_backoff() raises:
    """Test scaler backoff reduces scale factor."""
    print("Testing scaler backoff...")

    var scaler = GradientScaler(initial_scale=1024.0, backoff_factor=0.5)

    # Trigger backoff
    scaler.backoff()

    assert_equal(
        scaler.get_scale(), 512.0, "Scale should be halved after backoff"
    )

    print("✓ Scaler backoff test passed")


fn test_scaler_min_max_limits() raises:
    """Test scaler respects min and max scale limits."""
    print("Testing scaler min/max limits...")

    var scaler = GradientScaler(
        initial_scale=1024.0,
        min_scale=512.0,
        max_scale=2048.0,
        backoff_factor=0.5,
    )

    # Try to backoff below min_scale
    scaler.backoff()  # 1024 -> 512
    assert_equal(scaler.get_scale(), 512.0, "Scale should be at min")

    scaler.backoff()  # Try to go to 256, but should stay at 512
    assert_equal(scaler.get_scale(), 512.0, "Scale should not go below min")

    # Reset to just below max
    scaler.scale = 1536.0

    # Try to grow beyond max_scale (with very small growth interval for testing)
    scaler = GradientScaler(
        initial_scale=1536.0,
        min_scale=512.0,
        max_scale=2048.0,
        growth_factor=2.0,
        growth_interval=1,
    )

    scaler.step()  # Should grow to 3072, but capped at 2048
    assert_equal(scaler.get_scale(), 2048.0, "Scale should be capped at max")

    print("✓ Scaler min/max limits test passed")


fn test_fp32_master_conversion() raises:
    """Test converting FP16 parameters to FP32 master weights."""
    print("Testing FP32 master conversion...")

    # Create FP16 parameters (100 elements)
    var fp16_params = full([100], 0.5, DType.float16)

    # Convert to FP32 master weights
    var master_params = convert_to_fp32_master(fp16_params)

    assert_true(
        master_params.dtype() == DType.float32, "Master params should be FP32"
    )
    assert_equal(
        master_params._numel, 100, "Master params should have same size"
    )

    var val = master_params._get_float64(0)
    assert_true(abs(val - 0.5) < 1e-5, "Master params should have same values")

    print("✓ FP32 master conversion test passed")


fn test_update_model_from_master() raises:
    """Test updating FP16 model params from FP32 master weights."""
    print("Testing model update from master...")

    # Create FP16 model params and FP32 master weights (100 elements)
    var fp16_params = full([100], 1.0, DType.float16)
    var master_params = full([100], 2.0, DType.float32)

    # Update model from master
    update_model_from_master(fp16_params, master_params)

    # Check that FP16 params now have value 2.0
    var val = fp16_params._get_float64(0)
    assert_true(abs(val - 2.0) < 1e-3, "FP16 params should be updated to 2.0")

    print("✓ Update model from master test passed")


fn main() raises:
    print("\n" + "=" * 70)
    print("MIXED PRECISION TRAINING TESTS - PART 1")
    print("=" * 70)
    print()

    test_gradient_scaler_initialization()
    test_loss_scaling()
    test_gradient_unscaling()
    test_scaler_step_updates()
    test_scaler_backoff()
    test_scaler_min_max_limits()
    test_fp32_master_conversion()
    test_update_model_from_master()

    print()
    print("=" * 70)
    print("ALL MIXED PRECISION TESTS PART 1 PASSED! ✓")
    print("=" * 70)
