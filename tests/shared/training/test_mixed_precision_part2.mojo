# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_mixed_precision.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for mixed precision training infrastructure (part 2).

Tests gradient finite checking, gradient clipping by value and norm,
and basic FP16 tensor operations.
"""

from shared.core.extensor import AnyTensor, full
from shared.training.mixed_precision import (
    check_gradients_finite,
    clip_gradients_by_norm,
    clip_gradients_by_value,
)
from shared.testing.special_values import create_nan_tensor, create_inf_tensor
from testing import assert_equal, assert_true, assert_false


fn test_check_gradients_finite() raises:
    """Test checking for finite gradients."""
    print("Testing gradient finite check...")

    # Create finite gradients (10 elements)
    var finite_grads = full([10], 1.0, DType.float32)
    assert_true(
        check_gradients_finite(finite_grads),
        "Finite gradients should return True",
    )

    # Test with NaN gradients
    var nan_grads = create_nan_tensor([10], DType.float32)
    assert_false(
        check_gradients_finite(nan_grads),
        "NaN gradients should return False",
    )

    # Test with +Inf gradients
    var pos_inf_grads = create_inf_tensor([10], DType.float32, positive=True)
    assert_false(
        check_gradients_finite(pos_inf_grads),
        "+Inf gradients should return False",
    )

    # Test with -Inf gradients
    var neg_inf_grads = create_inf_tensor([10], DType.float32, positive=False)
    assert_false(
        check_gradients_finite(neg_inf_grads),
        "-Inf gradients should return False",
    )

    print("✓ Gradient finite check test passed")


fn test_check_gradients_finite_mixed_precision() raises:
    """Test checking for finite gradients in mixed precision (FP16).

    Validates that NaN/Inf detection works correctly for lower precision types.
    This is critical for mixed precision training where gradient overflow is more
    likely due to reduced precision.
    """
    print("Testing gradient finite check with mixed precision (FP16)...")

    # Create finite FP16 gradients
    var finite_fp16_grads = full([10], 1.0, DType.float16)
    assert_true(
        check_gradients_finite(finite_fp16_grads),
        "Finite FP16 gradients should return True",
    )

    # Test NaN detection in FP16
    var nan_fp16_grads = create_nan_tensor([10], DType.float16)
    assert_false(
        check_gradients_finite(nan_fp16_grads),
        "NaN FP16 gradients should return False",
    )

    # Test +Inf detection in FP16
    var pos_inf_fp16_grads = create_inf_tensor(
        [10], DType.float16, positive=True
    )
    assert_false(
        check_gradients_finite(pos_inf_fp16_grads),
        "+Inf FP16 gradients should return False",
    )

    # Test -Inf detection in FP16
    var neg_inf_fp16_grads = create_inf_tensor(
        [10], DType.float16, positive=False
    )
    assert_false(
        check_gradients_finite(neg_inf_fp16_grads),
        "-Inf FP16 gradients should return False",
    )

    print("✓ Mixed precision gradient finite check test passed")


fn test_clip_gradients_by_value() raises:
    """Test clipping gradients by value range."""
    print("Testing gradient clipping by value...")

    # Create gradients with various values (5 elements)
    var shape: List[Int] = [5]
    var grads = AnyTensor(shape, DType.float32)

    # Set some values manually
    grads._set_float64(0, -2.0)
    grads._set_float64(1, -0.5)
    grads._set_float64(2, 0.0)
    grads._set_float64(3, 0.5)
    grads._set_float64(4, 2.0)

    # Clip to [-1.0, 1.0]
    var clipped = clip_gradients_by_value(grads, -1.0, 1.0)

    # Check clipped values
    assert_equal(clipped._get_float64(0), -1.0, "Should clip to -1.0")
    assert_equal(clipped._get_float64(1), -0.5, "Should not clip -0.5")
    assert_equal(clipped._get_float64(2), 0.0, "Should not clip 0.0")
    assert_equal(clipped._get_float64(3), 0.5, "Should not clip 0.5")
    assert_equal(clipped._get_float64(4), 1.0, "Should clip to 1.0")

    print("✓ Gradient clipping by value test passed")


fn test_clip_gradients_by_norm() raises:
    """Test clipping gradients by global norm."""
    print("Testing gradient clipping by norm...")

    # Create gradients with known norm (3 elements)
    var shape: List[Int] = [3]
    var grads = AnyTensor(shape, DType.float32)

    # Set values: [3.0, 4.0, 0.0] -> norm = sqrt(9 + 16) = 5.0
    grads._set_float64(0, 3.0)
    grads._set_float64(1, 4.0)
    grads._set_float64(2, 0.0)

    # Clip to max_norm = 1.0 (should scale by 1.0/5.0 = 0.2)
    var clipped = clip_gradients_by_norm(grads, 1.0)

    # Check clipped values
    var val0 = clipped._get_float64(0)
    var val1 = clipped._get_float64(1)
    var val2 = clipped._get_float64(2)

    assert_true(abs(val0 - 0.6) < 1e-5, "Should be 3.0 * 0.2 = 0.6")
    assert_true(abs(val1 - 0.8) < 1e-5, "Should be 4.0 * 0.2 = 0.8")
    assert_true(abs(val2 - 0.0) < 1e-5, "Should be 0.0")

    print("✓ Gradient clipping by norm test passed")


fn test_fp16_operations() raises:
    """Test basic FP16 tensor operations."""
    print("Testing FP16 operations...")

    # Create FP16 tensors (10 elements each)
    var a_shape = List[Int]()
    a_shape.append(10)
    var a = full(a_shape, 2.0, DType.float16)
    var b_shape = List[Int]()
    b_shape.append(10)
    var b = full(b_shape, 3.0, DType.float16)

    # Test addition
    var c = a + b
    var val_add = c._get_float64(0)
    assert_true(abs(val_add - 5.0) < 1e-2, "FP16 addition: 2 + 3 = 5")

    # Test multiplication
    var d = a * b
    var val_mul = d._get_float64(0)
    assert_true(abs(val_mul - 6.0) < 1e-2, "FP16 multiplication: 2 * 3 = 6")

    # Test division
    var e = b / a
    var val_div = e._get_float64(0)
    assert_true(abs(val_div - 1.5) < 1e-2, "FP16 division: 3 / 2 = 1.5")

    print("✓ FP16 operations test passed")


fn main() raises:
    print("\n" + "=" * 70)
    print("MIXED PRECISION TRAINING TESTS - PART 2")
    print("=" * 70)
    print()

    test_check_gradients_finite()
    test_check_gradients_finite_mixed_precision()
    test_clip_gradients_by_value()
    test_clip_gradients_by_norm()
    test_fp16_operations()

    print()
    print("=" * 70)
    print("ALL MIXED PRECISION TESTS PART 2 PASSED! ✓")
    print("=" * 70)
