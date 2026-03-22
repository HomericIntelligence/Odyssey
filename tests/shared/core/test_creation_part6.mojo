# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor creation operations - Part 6: NaN/Inf dtype validation.

Tests nan_tensor(), inf_tensor(), and neg_inf_tensor() with supported float dtypes
and verifies rejection of unsupported dtypes (int32, bfloat16).
Split from test_creation.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import AnyTensor and creation operations
from shared.core.extensor import AnyTensor, nan_tensor, inf_tensor, neg_inf_tensor

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
# Test nan_tensor() rejects bfloat16 (not in supported dtype list)
# ============================================================================


fn test_nan_tensor_rejects_bfloat16() raises:
    """Test nan_tensor rejects bfloat16 dtype (not in supported float list)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var error_raised = False
    try:
        var t = nan_tensor(shape, DType.bfloat16)
    except e:
        # Error raised as expected - bfloat16 not in supported dtype list
        error_raised = True

    assert_true(error_raised, "nan_tensor should reject bfloat16 dtype")
    print("  test_nan_tensor_rejects_bfloat16: PASSED")


# ============================================================================
# Test inf_tensor() rejects bfloat16 (not in supported dtype list)
# ============================================================================


fn test_inf_tensor_rejects_bfloat16() raises:
    """Test inf_tensor rejects bfloat16 dtype (not in supported float list)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var error_raised = False
    try:
        var t = inf_tensor(shape, DType.bfloat16)
    except e:
        # Error raised as expected - bfloat16 not in supported dtype list
        error_raised = True

    assert_true(error_raised, "inf_tensor should reject bfloat16 dtype")
    print("  test_inf_tensor_rejects_bfloat16: PASSED")


# ============================================================================
# Test neg_inf_tensor() rejects bfloat16 (not in supported dtype list)
# ============================================================================


fn test_neg_inf_tensor_rejects_bfloat16() raises:
    """Test neg_inf_tensor rejects bfloat16 dtype (not in supported float list)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var error_raised = False
    try:
        var t = neg_inf_tensor(shape, DType.bfloat16)
    except e:
        # Error raised as expected - bfloat16 not in supported dtype list
        error_raised = True

    assert_true(error_raised, "neg_inf_tensor should reject bfloat16 dtype")
    print("  test_neg_inf_tensor_rejects_bfloat16: PASSED")


# ============================================================================
# Test nan_tensor() with float32 (supported behavior)
# ============================================================================


fn test_nan_tensor_float32() raises:
    """Test nan_tensor with float32 works correctly."""
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
    """Run NaN/Inf creation tests with dtype validation."""
    print("Running AnyTensor creation tests - Part 6: NaN/Inf dtype validation...")

    # bfloat16 rejection tests (not in nan/inf supported dtype list)
    test_nan_tensor_rejects_bfloat16()
    test_inf_tensor_rejects_bfloat16()
    test_neg_inf_tensor_rejects_bfloat16()

    # Supported dtype tests
    test_nan_tensor_float32()

    # Non-float rejection tests
    test_nan_tensor_rejects_int32()
    test_inf_tensor_rejects_int32()

    print("All Part 6 creation tests completed!")
