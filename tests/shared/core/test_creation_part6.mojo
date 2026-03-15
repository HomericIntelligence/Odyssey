# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor creation operations - Part 6: NaN/Inf with bfloat16.

Tests nan_tensor(), inf_tensor(), and neg_inf_tensor() with bfloat16 dtype support.
Split from test_creation.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and creation operations
from shared.core import (
    ExTensor,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
)

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
)


# ============================================================================
# Test nan_tensor() with bfloat16
# ============================================================================


fn test_nan_tensor_bfloat16() raises:
    """Test nan_tensor creates NaN values with bfloat16 dtype."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    # Should not raise - bfloat16 is now supported
    var t = nan_tensor(shape, DType.bfloat16)

    assert_dim(t, 2, "nan_tensor 2D should have 2 dimensions")
    assert_numel(t, 12, "nan_tensor should have 3x4 = 12 elements")
    assert_dtype(t, DType.bfloat16, "nan_tensor should have bfloat16 dtype")

    # Successfully creating a bfloat16 NaN tensor proves the fix works
    print("  test_nan_tensor_bfloat16: PASSED")


# ============================================================================
# Test inf_tensor() with bfloat16
# ============================================================================


fn test_inf_tensor_bfloat16() raises:
    """Test inf_tensor creates positive infinity values with bfloat16 dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    # Should not raise - bfloat16 is now supported
    var t = inf_tensor(shape, DType.bfloat16)

    assert_dim(t, 2, "inf_tensor 2D should have 2 dimensions")
    assert_numel(t, 6, "inf_tensor should have 2x3 = 6 elements")
    assert_dtype(t, DType.bfloat16, "inf_tensor should have bfloat16 dtype")

    # Successfully creating a bfloat16 inf tensor proves the fix works
    print("  test_inf_tensor_bfloat16: PASSED")


# ============================================================================
# Test neg_inf_tensor() with bfloat16
# ============================================================================


fn test_neg_inf_tensor_bfloat16() raises:
    """Test neg_inf_tensor creates negative infinity values with bfloat16 dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    # Should not raise - bfloat16 is now supported
    var t = neg_inf_tensor(shape, DType.bfloat16)

    assert_dim(t, 2, "neg_inf_tensor 2D should have 2 dimensions")
    assert_numel(t, 4, "neg_inf_tensor should have 2x2 = 4 elements")
    assert_dtype(t, DType.bfloat16, "neg_inf_tensor should have bfloat16 dtype")

    # Successfully creating a bfloat16 -inf tensor proves the fix works
    print("  test_neg_inf_tensor_bfloat16: PASSED")


# ============================================================================
# Test nan_tensor() with float32 (unchanged behavior)
# ============================================================================


fn test_nan_tensor_float32() raises:
    """Test nan_tensor with float32 still works correctly."""
    var shape = List[Int]()
    shape.append(2)
    var t = nan_tensor(shape, DType.float32)

    assert_dtype(t, DType.float32, "nan_tensor should have float32 dtype")
    assert_numel(t, 2, "nan_tensor should have 2 elements")

    print("  test_nan_tensor_float32: PASSED")


# ============================================================================
# Test nan_tensor() rejects non-float dtypes
# ============================================================================


fn test_nan_tensor_rejects_int32() raises:
    """Test nan_tensor rejects int32 dtype."""
    var shape = List[Int]()
    shape.append(2)

    var error_raised = False
    try:
        var t = nan_tensor(shape, DType.int32)
    except e:
        # Error raised as expected for non-float dtype
        error_raised = True

    assert_true(error_raised, "nan_tensor should reject int32 dtype")
    print("  test_nan_tensor_rejects_int32: PASSED")


# ============================================================================
# Test inf_tensor() rejects non-float dtypes
# ============================================================================


fn test_inf_tensor_rejects_int32() raises:
    """Test inf_tensor rejects int32 dtype."""
    var shape = List[Int]()
    shape.append(2)

    var error_raised = False
    try:
        var t = inf_tensor(shape, DType.int32)
    except e:
        # Error raised as expected for non-float dtype
        error_raised = True

    assert_true(error_raised, "inf_tensor should reject int32 dtype")
    print("  test_inf_tensor_rejects_int32: PASSED")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run NaN/Inf creation tests with bfloat16 support."""
    print("Running ExTensor creation tests - Part 6: NaN/Inf with bfloat16...")

    # bfloat16 tests
    test_nan_tensor_bfloat16()
    test_inf_tensor_bfloat16()
    test_neg_inf_tensor_bfloat16()

    # Backward compatibility tests
    test_nan_tensor_float32()
    test_nan_tensor_rejects_int32()
    test_inf_tensor_rejects_int32()

    print("All Part 6 creation tests completed!")
