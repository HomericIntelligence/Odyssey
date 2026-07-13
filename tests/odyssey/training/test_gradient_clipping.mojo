"""Tests for gradient clipping utilities.

Verifies correctness of gradient clipping functions for training stability.

Test Coverage:
- compute_gradient_norm_list: Compute global gradient norm
- clip_gradients_by_global_norm: Clip by global norm across all gradients
- clip_gradients_per_param: Clip each parameter independently
- clip_gradients_by_value_list: Clip by value range
- compute_gradient_statistics: Gradient monitoring and health checks
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, ones, full
from odyssey.training.gradient_clipping import (
    compute_gradient_norm_list,
    clip_gradients_by_global_norm,
    clip_gradients_per_param,
    clip_gradients_by_value_list,
    compute_gradient_statistics,
)
from odyssey.testing.assertions import (
    assert_true,
    assert_close_float,
    assert_equal_int,
)
from std.collections import List
from std.math import sqrt


def create_test_gradients() raises -> List[AnyTensor]:
    """Create test gradients with known norms."""
    var grads = List[AnyTensor]()

    # Gradient 1: all ones (norm = sqrt(100) = 10)
    grads.append(ones([100], DType.float32))

    # Gradient 2: all twos (norm = sqrt(50*4) = sqrt(200))
    grads.append(full([50], 2.0, DType.float32))

    return grads^


def test_compute_gradient_norm() raises:
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


def test_clip_by_global_norm_no_clipping() raises:
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


def test_clip_by_global_norm_with_clipping() raises:
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


def test_clip_per_param() raises:
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


def test_clip_by_value() raises:
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


def test_clip_by_value_negative() raises:
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


def test_gradient_statistics() raises:
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


def test_gradient_statistics_empty() raises:
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


def test_clip_zero_gradients() raises:
    """Test clipping with zero gradients."""
    var grads = List[AnyTensor]()
    grads.append(zeros([100], DType.float32))

    # Should not crash with zero gradients
    var norm = clip_gradients_by_global_norm(grads, max_norm=1.0)

    assert_close_float(
        Float64(norm), 0.0, atol=1e-6, message="Norm of zeros should be 0"
    )


# ============================================================================
# SIMD Edge-Case Tests
# ============================================================================


def test_norm_simd_non_aligned_sizes() raises:
    """Test norm computation with tensor sizes that are NOT multiples of SIMD width.

    SIMD width is typically 8 for float32 (AVX2) or 16 (AVX-512).
    Sizes 7, 13, 33 exercise the vectorize tail handling.
    """
    # Size 7 (not a multiple of any common SIMD width)
    var grads7 = List[AnyTensor]()
    grads7.append(ones([7], DType.float32))
    var norm7 = compute_gradient_norm_list(grads7)
    var expected7 = Float32(sqrt(Float64(7.0)))
    assert_close_float(
        Float64(norm7),
        Float64(expected7),
        atol=1e-5,
        message="Norm of 7 ones should be sqrt(7)",
    )

    # Size 13
    var grads13 = List[AnyTensor]()
    grads13.append(full([13], 3.0, DType.float32))
    var norm13 = compute_gradient_norm_list(grads13)
    var expected13 = Float32(sqrt(Float64(13.0) * 9.0))
    assert_close_float(
        Float64(norm13),
        Float64(expected13),
        atol=1e-5,
        message="Norm of 13 threes should be sqrt(117)",
    )

    # Size 33
    var grads33 = List[AnyTensor]()
    grads33.append(ones([33], DType.float32))
    var norm33 = compute_gradient_norm_list(grads33)
    var expected33 = Float32(sqrt(Float64(33.0)))
    assert_close_float(
        Float64(norm33),
        Float64(expected33),
        atol=1e-5,
        message="Norm of 33 ones should be sqrt(33)",
    )

    # Size 1 (single element - pure tail)
    var grads1 = List[AnyTensor]()
    grads1.append(full([1], 5.0, DType.float32))
    var norm1 = compute_gradient_norm_list(grads1)
    assert_close_float(
        Float64(norm1),
        5.0,
        atol=1e-5,
        message="Norm of single 5.0 should be 5.0",
    )


def test_clip_large_tensor() raises:
    """Test SIMD clipping with a large tensor (10000+ elements)."""
    var grads = List[AnyTensor]()
    grads.append(full([10000], 2.0, DType.float32))

    # Norm = sqrt(10000 * 4) = sqrt(40000) = 200.0
    var expected_norm = Float32(sqrt(Float64(40000.0)))
    var norm = compute_gradient_norm_list(grads)
    assert_close_float(
        Float64(norm),
        Float64(expected_norm),
        atol=1e-3,
        message="Large tensor norm should be 200.0",
    )

    # Clip to 10.0
    var clipped_norm = clip_gradients_by_global_norm(grads, max_norm=10.0)
    assert_close_float(
        Float64(clipped_norm),
        Float64(expected_norm),
        atol=1e-3,
        message="Should return original norm",
    )

    # After clipping, new norm should be ~10.0
    var new_norm = compute_gradient_norm_list(grads)
    assert_close_float(
        Float64(new_norm),
        10.0,
        atol=1e-3,
        message="Clipped large tensor norm should be 10.0",
    )


def test_value_clip_mixed_signs() raises:
    """Test SIMD value clipping with mixed positive/negative values."""
    var grads = List[AnyTensor]()

    # Create tensor with alternating positive and negative values
    var grad = zeros([20], DType.float32)
    for i in range(20):
        if i % 2 == 0:
            grad._set_float64(i, 5.0)
        else:
            grad._set_float64(i, -5.0)
    grads.append(grad^)

    # Clip to [-2.0, 2.0]
    clip_gradients_by_value_list(grads, min_value=-2.0, max_value=2.0)

    # Verify all values are clamped correctly
    for i in range(20):
        var val = grads[0]._get_float64(i)
        if i % 2 == 0:
            assert_close_float(
                val,
                2.0,
                atol=1e-6,
                message="Positive values should be clipped to 2.0",
            )
        else:
            assert_close_float(
                val,
                -2.0,
                atol=1e-6,
                message="Negative values should be clipped to -2.0",
            )


def test_clip_per_param_non_aligned() raises:
    """Test per-param clipping with non-SIMD-aligned tensor sizes."""
    var grads = List[AnyTensor]()

    # 7 elements, all 10.0 -> norm = sqrt(7 * 100) = sqrt(700) ~ 26.46
    grads.append(full([7], 10.0, DType.float32))

    # 3 elements, all 0.1 -> norm = sqrt(3 * 0.01) = sqrt(0.03) ~ 0.173
    grads.append(full([3], 0.1, DType.float32))

    # Clip each param to max_norm=5.0
    clip_gradients_per_param(grads, max_norm=5.0)

    # Grad1 should be clipped (norm ~26.46 -> 5.0)
    var grad1_norm_sq = Float64(0.0)
    for i in range(grads[0].numel()):
        var val = grads[0]._get_float64(i)
        grad1_norm_sq += val * val
    var grad1_norm = Float32(sqrt(grad1_norm_sq))
    assert_close_float(
        Float64(grad1_norm),
        5.0,
        atol=1e-4,
        message="Grad1 (7 elements) norm should be 5.0 after clipping",
    )

    # Grad2 should be unchanged (norm ~0.173 < 5.0)
    for i in range(grads[1].numel()):
        assert_close_float(
            grads[1]._get_float64(i),
            0.1,
            atol=1e-6,
            message="Grad2 (3 elements) should be unchanged",
        )


# ============================================================================
# Float64 SIMD coverage tests (Issue #5143)
# Exercises _norm_sq_simd_f64, _scale_simd_f64, _clamp_simd_f64 code paths.
# ============================================================================


def create_test_gradients_f64() raises -> List[AnyTensor]:
    """Create float64 test gradients with known norms."""
    var grads = List[AnyTensor]()

    # Gradient 1: all ones (norm_sq = 100)
    grads.append(ones([100], DType.float64))

    # Gradient 2: all twos (norm_sq = 50 * 4 = 200)
    grads.append(full([50], 2.0, DType.float64))

    return grads^


def test_compute_gradient_norm_f64() raises:
    """Test global gradient norm computation with float64 tensors.

    Exercises _norm_sq_simd_f64 code path.
    """
    var grads = create_test_gradients_f64()

    # Total norm = sqrt(100 + 200) = sqrt(300) ~ 17.32
    var norm = compute_gradient_norm_list(grads)
    var expected = Float32(sqrt(Float64(300.0)))

    assert_close_float(
        Float64(norm),
        Float64(expected),
        atol=1e-10,
        message="Float64 gradient norm should match expected",
    )


def test_clip_by_global_norm_f64_with_clipping() raises:
    """Test global norm clipping with float64 tensors when norm exceeds threshold.

    Exercises _norm_sq_simd_f64 and _scale_simd_f64 code paths.
    """
    var grads = create_test_gradients_f64()

    var orig_norm = compute_gradient_norm_list(grads)
    # Norm ~17.32, clip to 10.0
    var clipped_norm = clip_gradients_by_global_norm(grads, max_norm=10.0)

    # Returns original norm
    assert_close_float(
        Float64(clipped_norm),
        Float64(orig_norm),
        atol=1e-10,
        message="Float64 clip should return original norm",
    )

    # After clipping, norm should be ~10.0
    var new_norm = compute_gradient_norm_list(grads)
    assert_close_float(
        Float64(new_norm),
        10.0,
        atol=1e-8,
        message="Float64 clipped norm should be 10.0",
    )

    # Verify scale factor applied correctly (higher precision than f32)
    var clip_coef = 10.0 / Float64(orig_norm)
    for i in range(10):
        var expected = 1.0 * clip_coef
        assert_close_float(
            grads[0]._get_float64(i),
            expected,
            atol=1e-10,
            message="Float64 grad1 element should be scaled",
        )


def test_clip_by_value_f64() raises:
    """Test value clamping with float64 tensors.

    Exercises _clamp_simd_f64 code path.
    """
    var grads = List[AnyTensor]()

    # Values at 5.0, clip to [-1.0, 1.0]
    grads.append(full([100], 5.0, DType.float64))
    clip_gradients_by_value_list(grads, min_value=-1.0, max_value=1.0)

    for i in range(grads[0].numel()):
        assert_close_float(
            grads[0]._get_float64(i),
            1.0,
            atol=1e-15,
            message="Float64 values should be clamped to 1.0",
        )

    # Negative side: values at -5.0, clip to [-2.0, 2.0]
    var grads2 = List[AnyTensor]()
    grads2.append(full([50], -5.0, DType.float64))
    clip_gradients_by_value_list(grads2, min_value=-2.0, max_value=2.0)

    for i in range(grads2[0].numel()):
        assert_close_float(
            grads2[0]._get_float64(i),
            -2.0,
            atol=1e-15,
            message="Float64 negative values should be clamped to -2.0",
        )


def test_clip_per_param_f64() raises:
    """Test per-parameter clipping with float64 tensors.

    Exercises _norm_sq_simd_f64 and _scale_simd_f64 code paths.
    """
    var grads = List[AnyTensor]()

    # 100 elements, all 10.0 -> norm = sqrt(100*100) = 100
    grads.append(full([100], 10.0, DType.float64))

    # 50 elements, all 0.1 -> norm = sqrt(50*0.01) = sqrt(0.5) ~ 0.707
    grads.append(full([50], 0.1, DType.float64))

    clip_gradients_per_param(grads, max_norm=5.0)

    # Grad1 clipped: norm 100 -> 5.0
    var grad1_norm_sq = Float64(0.0)
    for i in range(grads[0].numel()):
        var val = grads[0]._get_float64(i)
        grad1_norm_sq += val * val
    var grad1_norm = sqrt(grad1_norm_sq)
    assert_close_float(
        grad1_norm,
        5.0,
        atol=1e-8,
        message="Float64 grad1 norm should be 5.0 after per-param clipping",
    )

    # Grad2 unchanged: norm ~0.707 < 5.0
    for i in range(10):
        assert_close_float(
            grads[1]._get_float64(i),
            0.1,
            atol=1e-15,
            message="Float64 grad2 should be unchanged",
        )


# ============================================================================
# Scalar fallback path tests (non-float dtypes — issue #5145)
#
# compute_gradient_norm_list / clip_gradients_by_global_norm /
# clip_gradients_per_param / clip_gradients_by_value_list take a SIMD path for
# float32/float64 and a scalar `_get_float64`/`_set_float64` fallback for every
# other dtype. These tests exercise that fallback with int32 gradients.
# ============================================================================


def test_norm_scalar_fallback_int32() raises:
    """Use the scalar path of compute_gradient_norm_list for int32 tensors."""
    var grads = List[AnyTensor]()
    # 9 elements of value 2 → norm = sqrt(9 * 4) = 6.0
    grads.append(full([9], 2.0, DType.int32))
    var norm = compute_gradient_norm_list(grads)
    assert_close_float(
        Float64(norm),
        6.0,
        atol=1e-4,
        message="int32 scalar-fallback norm should be 6.0",
    )


def test_clip_by_global_norm_scalar_fallback_int32() raises:
    """Report the int32 scalar-path norm from clip_gradients_by_global_norm."""
    var grads = List[AnyTensor]()
    grads.append(full([16], 3.0, DType.int32))  # norm = sqrt(16*9) = 12.0
    var pre_norm = clip_gradients_by_global_norm(grads, max_norm=100.0)
    assert_close_float(
        Float64(pre_norm),
        12.0,
        atol=1e-4,
        message="int32 scalar-fallback pre-clip norm should be 12.0",
    )


def test_clip_per_param_scalar_fallback_int32() raises:
    """Run clip_gradients_per_param on int32 via its scalar fallback path."""
    var grads = List[AnyTensor]()
    grads.append(full([25], 2.0, DType.int32))  # local norm = sqrt(25*4) = 10
    # max_norm above the local norm: a no-op clip, but exercises the path.
    clip_gradients_per_param(grads, max_norm=100.0)
    var v = grads[0]._get_float64(0)
    assert_close_float(
        v,
        2.0,
        atol=1e-4,
        message="int32 scalar-fallback per-param value unchanged",
    )


def test_clip_by_value_scalar_fallback_int32() raises:
    """Clamp int32 gradients via the value-clip scalar fallback path."""
    var grads = List[AnyTensor]()
    grads.append(full([8], 9.0, DType.int32))  # all 9 — above max
    clip_gradients_by_value_list(grads, min_value=-5.0, max_value=5.0)
    for j in range(8):
        assert_close_float(
            grads[0]._get_float64(j),
            5.0,
            atol=1e-4,
            message="int32 value above max should clamp to 5.0",
        )


def main() raises:
    """Run all gradient clipping tests."""
    print("Testing Gradient Clipping...")
    print("=" * 70)

    print("\n[1/17] Testing gradient norm computation...")
    test_compute_gradient_norm()
    print("✓ PASSED")

    print("[2/17] Testing global norm clipping (no clipping)...")
    test_clip_by_global_norm_no_clipping()
    print("✓ PASSED")

    print("[3/17] Testing global norm clipping (with clipping)...")
    test_clip_by_global_norm_with_clipping()
    print("✓ PASSED")

    print("[4/17] Testing per-parameter clipping...")
    test_clip_per_param()
    print("✓ PASSED")

    print("[5/17] Testing value clipping (positive)...")
    test_clip_by_value()
    print("✓ PASSED")

    print("[6/17] Testing value clipping (negative)...")
    test_clip_by_value_negative()
    print("✓ PASSED")

    print("[7/17] Testing gradient statistics...")
    test_gradient_statistics()
    print("✓ PASSED")

    print("[8/17] Testing gradient statistics (empty)...")
    test_gradient_statistics_empty()
    print("✓ PASSED")

    print("[9/17] Testing clipping with zero gradients...")
    test_clip_zero_gradients()
    print("✓ PASSED")

    print("[10/17] Testing SIMD norm with non-aligned sizes...")
    test_norm_simd_non_aligned_sizes()
    print("✓ PASSED")

    print("[11/17] Testing SIMD clipping with large tensor...")
    test_clip_large_tensor()
    print("✓ PASSED")

    print("[12/17] Testing SIMD value clipping with mixed signs...")
    test_value_clip_mixed_signs()
    print("✓ PASSED")

    print("[13/17] Testing SIMD per-param clipping non-aligned...")
    test_clip_per_param_non_aligned()
    print("✓ PASSED")

    print("[14/21] Testing float64 gradient norm computation...")
    test_compute_gradient_norm_f64()
    print("✓ PASSED")

    print("[15/21] Testing float64 global norm clipping (with clipping)...")
    test_clip_by_global_norm_f64_with_clipping()
    print("✓ PASSED")

    print("[16/21] Testing float64 value clipping...")
    test_clip_by_value_f64()
    print("✓ PASSED")

    print("[17/21] Testing float64 per-param clipping...")
    test_clip_per_param_f64()
    print("✓ PASSED")

    print("[18/21] Testing int32 scalar-fallback norm...")
    test_norm_scalar_fallback_int32()
    print("✓ PASSED")

    print("[19/21] Testing int32 scalar-fallback global-norm clip...")
    test_clip_by_global_norm_scalar_fallback_int32()
    print("✓ PASSED")

    print("[20/21] Testing int32 scalar-fallback per-param clip...")
    test_clip_per_param_scalar_fallback_int32()
    print("✓ PASSED")

    print("[21/21] Testing int32 scalar-fallback value clip...")
    test_clip_by_value_scalar_fallback_int32()
    print("✓ PASSED")

    print("\n" + "=" * 70)
    print("All 21 gradient clipping tests PASSED! ✓")
    print("Gradient clipping utilities are working correctly.")
