# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_precision_config.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for PrecisionConfig module (part 2 of 2): scaling, gradients, and clipping."""

from shared.training.precision_config import PrecisionConfig, PrecisionMode
from shared.core.any_tensor import AnyTensor, zeros, ones


fn test_scale_unscale() raises:
    """Test loss scaling and gradient unscaling."""
    print("Testing scale/unscale operations...")

    var config = PrecisionConfig.fp16(initial_scale=1000.0)

    # Create loss tensor
    var loss_shape: List[Int] = [1]
    var loss = ones(loss_shape, DType.float32)

    # Scale loss
    var scaled_loss = config.scale_loss(loss)
    var scaled_value = scaled_loss._get_float64(0)
    if scaled_value < 999.0 or scaled_value > 1001.0:
        raise Error(
            "Scaled loss should be ~1000.0, got: " + String(scaled_value)
        )

    # Create gradient tensor
    var grad_shape: List[Int] = [10]
    var grads = ones(grad_shape, DType.float32)
    for i in range(10):
        grads._set_float64(i, 1000.0)

    # Unscale gradients
    var unscaled_grads = config.unscale_gradients(grads)
    var unscaled_value = unscaled_grads._get_float64(0)
    if unscaled_value < 0.9 or unscaled_value > 1.1:
        raise Error(
            "Unscaled gradient should be ~1.0, got: " + String(unscaled_value)
        )

    print("✓ scale/unscale test passed")


fn test_gradient_checking() raises:
    """Test gradient validity checking."""
    print("Testing gradient checking...")

    var config = PrecisionConfig.fp16()

    # Create valid gradients
    var shape: List[Int] = [5]
    var valid_grads = ones(shape, DType.float32)

    if not config.check_gradients(valid_grads):
        raise Error("Valid gradients should pass check")

    print("✓ gradient checking test passed")


fn test_step_tracking() raises:
    """Test step and overflow tracking."""
    print("Testing step tracking...")

    var config = PrecisionConfig.fp16()

    if config.get_step_count() != 0:
        raise Error("Initial step count should be 0")
    if config.get_overflow_count() != 0:
        raise Error("Initial overflow count should be 0")

    # Simulate successful steps
    config.step(grads_valid=True)
    config.step(grads_valid=True)

    if config.get_step_count() != 2:
        raise Error("Step count should be 2")

    # Simulate overflow
    config.step(grads_valid=False)

    if config.get_overflow_count() != 1:
        raise Error("Overflow count should be 1")
    if config.get_step_count() != 3:
        raise Error("Step count should be 3 (includes overflow)")

    print("✓ step tracking test passed")


fn test_needs_master_weights() raises:
    """Test needs_master_weights check."""
    print("Testing needs_master_weights...")

    var fp32 = PrecisionConfig.fp32()
    if fp32.needs_master_weights():
        raise Error("FP32 should not need master weights")

    var fp16 = PrecisionConfig.fp16()
    if not fp16.needs_master_weights():
        raise Error("FP16 should need master weights")

    var bf16 = PrecisionConfig.bf16()
    if not bf16.needs_master_weights():
        raise Error("BF16 should need master weights")

    print("✓ needs_master_weights test passed")


fn test_gradient_clipping() raises:
    """Test gradient clipping."""
    print("Testing gradient clipping...")

    var config = PrecisionConfig.fp16()

    # Create gradients with large values
    var shape: List[Int] = [4]
    var grads = zeros(shape, DType.float32)
    grads._set_float64(0, 10.0)
    grads._set_float64(1, 20.0)
    grads._set_float64(2, 30.0)
    grads._set_float64(3, 40.0)

    # Clip with max_norm=2.0
    var clipped = config.clip_gradients(grads, max_norm=2.0)

    # After clipping, L2 norm should be <= 2.0
    var sum_squared = Float64(0.0)
    for i in range(4):
        var val = clipped._get_float64(i)
        sum_squared += val * val
    var norm = sum_squared**0.5

    if norm > 2.1:
        raise Error(
            "Clipped gradient norm should be <= 2.0, got: " + String(norm)
        )

    print("✓ gradient clipping test passed")


fn main() raises:
    """Run PrecisionConfig tests (part 2 of 2)."""
    print("=" * 60)
    print("PRECISION CONFIG TESTS (PART 2)")
    print("=" * 60)
    print()

    test_scale_unscale()
    test_gradient_checking()
    test_step_tracking()
    test_needs_master_weights()
    test_gradient_clipping()

    print()
    print("=" * 60)
    print("ALL PRECISION CONFIG TESTS (PART 2) PASSED! ✓")
    print("=" * 60)
