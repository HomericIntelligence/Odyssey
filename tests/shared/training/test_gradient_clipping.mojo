"""Tests for gradient clipping utilities.

Verifies correctness of gradient clipping functions for training stability.

Test Coverage:
- compute_gradient_norm_list: Compute global gradient norm
- clip_gradients_by_global_norm: Clip by global norm across all gradients
- clip_gradients_per_param: Clip each parameter independently
- clip_gradients_by_value_list: Clip by value range
- compute_gradient_statistics: Gradient monitoring and health checks
"""

from shared.core.extensor import AnyTensor, zeros, ones, full
from shared.training.gradient_clipping import (
    compute_gradient_norm_list,
    clip_gradients_by_global_norm,
    clip_gradients_per_param,
    clip_gradients_by_value_list,
    compute_gradient_statistics,
)
from shared.testing.assertions import (
    assert_true,
    assert_close_float,
    assert_equal_int,
)
from collections import List
from math import sqrt


fn create_test_gradients() raises -> List[AnyTensor]:
    """Create test gradients with known norms."""
    var grads = List[AnyTensor]()

    # Gradient 1: all ones (norm = sqrt(100) = 10)
    grads.append(ones([100], DType.float32))

    # Gradient 2: all twos (norm = sqrt(50*4) = sqrt(200))
    grads.append(full([50], 2.0, DType.float32))

    return grads^


fn test_compute_gradient_norm() raises:
    """Test global gradient norm computation."""
    var grads = create_test_gradients()

    # Grad1: 100 ones -> norm_sq = 100
    # Grad2: 50 twos -> norm_sq = 50*4 = 200
    # Total norm = sqrt(100 + 200) = sqrt(300) = 17.32...

    var norm = compute_gradient_norm_list(grads)
    var expected = Float32(sqrt(Float64(300.0)))

    assert_close_float(
        Float64(norm),
        Float64(expected),
        atol=1e-5,
        message="Gradient norm should match expected",
    )


fn test_clip_by_global_norm_no_clipping() raises:
    """Test gradient clipping when norm is below threshold."""
    var grads = create_test_gradients()

    # Norm is ~17.32, clipping at 20.0 should not change anything
    var orig_norm = compute_gradient_norm_list(grads)
    var clipped_norm = clip_gradients_by_global_norm(grads, max_norm=20.0)

    # Should return original norm
    assert_close_float(
        Float64(clipped_norm),
        Float64(orig_norm),
        atol=1e-5,
        message="Should return original norm",
    )

    # Gradients should be unchanged
    for i in range(grads[0].numel()):
        assert_close_float(
            grads[0]._get_float64(i),
            1.0,
            atol=1e-6,
            message="Grad1 should still be 1.0",
        )


fn test_clip_by_global_norm_with_clipping() raises:
    """Test gradient clipping when norm exceeds threshold."""
    var grads = create_test_gradients()

    # Norm is ~17.32, clip to 10.0
    var orig_norm = compute_gradient_norm_list(grads)
    var clipped_norm = clip_gradients_by_global_norm(grads, max_norm=10.0)

    # Should return original norm
    assert_close_float(
        Float64(clipped_norm),
        Float64(orig_norm),
        atol=1e-5,
        message="Should return original norm",
    )

    # After clipping, new norm should be close to 10.0
    var new_norm = compute_gradient_norm_list(grads)
    assert_close_float(
        Float64(new_norm),
        10.0,
        atol=1e-4,
        message="Clipped norm should be 10.0",
    )

    # Gradients should be scaled by clip_coef = 10.0 / 17.32...
    var clip_coef = 10.0 / Float64(orig_norm)
    for i in range(10):  # Check first 10 elements
        var expected = 1.0 * clip_coef
        assert_close_float(
            grads[0]._get_float64(i),
            expected,
            atol=1e-5,
            message="Grad1 should be scaled",
        )


fn test_clip_per_param() raises:
    """Test per-parameter gradient clipping."""
    var grads = List[AnyTensor]()

    # Gradient 1: all 10.0 (norm = sqrt(100*100) = 100)
    grads.append(full([100], 10.0, DType.float32))

    # Gradient 2: all 0.1 (norm = sqrt(50*0.01) = sqrt(0.5) = 0.707...)
    grads.append(full([50], 0.1, DType.float32))

    # Clip each parameter to max_norm=5.0
    clip_gradients_per_param(grads, max_norm=5.0)

    # Grad1 should be clipped (norm 100 -> 5.0)
    # Grad2 should be unchanged (norm 0.707 < 5.0)

    # Check grad1 norm
    var grad1_norm_sq = Float64(0.0)
    for i in range(grads[0].numel()):
        var val = grads[0]._get_float64(i)
        grad1_norm_sq += val * val

    var grad1_norm = Float32(sqrt(grad1_norm_sq))
    assert_close_float(
        Float64(grad1_norm), 5.0, atol=1e-4, message="Grad1 norm should be 5.0"
    )

    # Check grad2 is unchanged
    for i in range(10):  # Check first 10 elements
        assert_close_float(
            grads[1]._get_float64(i),
            0.1,
            atol=1e-6,
            message="Grad2 should be unchanged",
        )


fn test_clip_by_value() raises:
    """Test gradient clipping by value range."""
    var grads = List[AnyTensor]()

    # Create gradient with values outside [-1.0, 1.0]
    var grad = full([100], 5.0, DType.float32)
    grads.append(grad^)

    # Clip to [-1.0, 1.0]
    clip_gradients_by_value_list(grads, min_value=-1.0, max_value=1.0)

    # All values should be clipped to 1.0
    for i in range(grads[0].numel()):
        assert_close_float(
            grads[0]._get_float64(i),
            1.0,
            atol=1e-6,
            message="Values should be clipped to 1.0",
        )


fn test_clip_by_value_negative() raises:
    """Test gradient clipping with negative values."""
    var grads = List[AnyTensor]()

    # Create gradient with negative values
    var grad = full([50], -10.0, DType.float32)
    grads.append(grad^)

    # Clip to [-2.0, 2.0]
    clip_gradients_by_value_list(grads, min_value=-2.0, max_value=2.0)

    # All values should be clipped to -2.0
    for i in range(grads[0].numel()):
        assert_close_float(
            grads[0]._get_float64(i),
            -2.0,
            atol=1e-6,
            message="Negative values should be clipped to -2.0",
        )


fn test_gradient_statistics() raises:
    """Test gradient statistics computation."""
    var grads = List[AnyTensor]()

    # Create gradients with known statistics
    grads.append(ones([100], DType.float32))  # 100 ones
    grads.append(full([50], 2.0, DType.float32))  # 50 twos

    var stats = compute_gradient_statistics(grads)

    # Check statistics
    assert_equal_int(stats.num_params, 150, "Should have 150 total parameters")
    assert_equal_int(stats.num_nan, 0, "Should have no NaN values")
    assert_equal_int(stats.num_inf, 0, "Should have no Inf values")

    # Check norm (same as earlier test)
    var expected_norm = Float32(sqrt(Float64(300.0)))
    assert_close_float(
        Float64(stats.global_norm),
        Float64(expected_norm),
        atol=1e-5,
        message="Global norm should match",
    )

    # Check max/min values
    assert_close_float(
        Float64(stats.max_value),
        2.0,
        atol=1e-6,
        message="Max value should be 2.0",
    )
    assert_close_float(
        Float64(stats.min_value),
        1.0,
        atol=1e-6,
        message="Min value should be 1.0",
    )

    # Check health
    assert_true(stats.is_healthy(), "Gradients should be healthy")


fn test_gradient_statistics_empty() raises:
    """Test gradient statistics with empty list."""
    var grads = List[AnyTensor]()

    var stats = compute_gradient_statistics(grads)

    assert_equal_int(stats.num_params, 0, "Should have 0 parameters")
    assert_close_float(
        Float64(stats.global_norm),
        0.0,
        atol=1e-6,
        message="Global norm should be 0",
    )


fn test_clip_zero_gradients() raises:
    """Test clipping with zero gradients."""
    var grads = List[AnyTensor]()
    grads.append(zeros([100], DType.float32))

    # Should not crash with zero gradients
    var norm = clip_gradients_by_global_norm(grads, max_norm=1.0)

    assert_close_float(
        Float64(norm), 0.0, atol=1e-6, message="Norm of zeros should be 0"
    )


fn main() raises:
    """Run all gradient clipping tests."""
    print("Testing Gradient Clipping...")
    print("=" * 70)

    print("\n[1/9] Testing gradient norm computation...")
    test_compute_gradient_norm()
    print("✓ PASSED")

    print("[2/9] Testing global norm clipping (no clipping)...")
    test_clip_by_global_norm_no_clipping()
    print("✓ PASSED")

    print("[3/9] Testing global norm clipping (with clipping)...")
    test_clip_by_global_norm_with_clipping()
    print("✓ PASSED")

    print("[4/9] Testing per-parameter clipping...")
    test_clip_per_param()
    print("✓ PASSED")

    print("[5/9] Testing value clipping (positive)...")
    test_clip_by_value()
    print("✓ PASSED")

    print("[6/9] Testing value clipping (negative)...")
    test_clip_by_value_negative()
    print("✓ PASSED")

    print("[7/9] Testing gradient statistics...")
    test_gradient_statistics()
    print("✓ PASSED")

    print("[8/9] Testing gradient statistics (empty)...")
    test_gradient_statistics_empty()
    print("✓ PASSED")

    print("[9/9] Testing clipping with zero gradients...")
    test_clip_zero_gradients()
    print("✓ PASSED")

    print("\n" + "=" * 70)
    print("All 9 gradient clipping tests PASSED! ✓")
    print("Gradient clipping utilities are working correctly.")
