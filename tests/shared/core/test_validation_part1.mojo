# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_validation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for tensor validation utilities - Part 1: validate_tensor_shape.

Tests cover:
- validate_tensor_shape: Shape validation
"""

from tests.shared.conftest import (
    assert_equal,
    assert_true,
)
from shared.core.any_tensor import AnyTensor, zeros, ones
from shared.core.validation import (
    validate_tensor_shape,
)


# ============================================================================
# validate_tensor_shape Tests
# ============================================================================


fn test_validate_tensor_shape_1d_correct() raises:
    """Test validate_tensor_shape with correct 1D shape."""
    var x = zeros([10], DType.float32)
    var expected_shape: List[Int] = [10]
    validate_tensor_shape(x, expected_shape, "x")


fn test_validate_tensor_shape_2d_correct() raises:
    """Test validate_tensor_shape with correct 2D shape."""
    var x = zeros([3, 4], DType.float32)
    var expected_shape: List[Int] = [3, 4]
    validate_tensor_shape(x, expected_shape, "x")


fn test_validate_tensor_shape_3d_correct() raises:
    """Test validate_tensor_shape with correct 3D shape."""
    var x = zeros([2, 3, 4], DType.float32)
    var expected_shape: List[Int] = [2, 3, 4]
    validate_tensor_shape(x, expected_shape, "x")


fn test_validate_tensor_shape_wrong_dimension_count() raises:
    """Test validate_tensor_shape with wrong number of dimensions."""
    var x = zeros([3, 4], DType.float32)
    var expected_shape: List[Int] = [3, 4, 5]
    var error_raised = False

    try:
        validate_tensor_shape(x, expected_shape, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected 3D tensor, got 2D" in error_msg,
            "Error should mention expected 3D but got 2D",
        )

    assert_true(error_raised, "Error should be raised for dimension mismatch")


fn test_validate_tensor_shape_wrong_dimension_value() raises:
    """Test validate_tensor_shape with wrong dimension value."""
    var x = zeros([3, 4], DType.float32)
    var expected_shape: List[Int] = [3, 5]
    var error_raised = False

    try:
        validate_tensor_shape(x, expected_shape, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected shape" in error_msg and "[3, 5]" in error_msg,
            "Error should mention expected and actual shapes",
        )

    assert_true(error_raised, "Error should be raised for shape mismatch")


fn main() raises:
    """Run validate_tensor_shape tests."""
    print("Running validate_tensor_shape tests...")

    test_validate_tensor_shape_1d_correct()
    test_validate_tensor_shape_2d_correct()
    test_validate_tensor_shape_3d_correct()
    test_validate_tensor_shape_wrong_dimension_count()
    test_validate_tensor_shape_wrong_dimension_value()

    print("All validate_tensor_shape tests passed!")
