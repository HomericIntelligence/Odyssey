# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_validation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for tensor validation utilities - Part 2: validate_tensor_dtype and validate_matching_tensors.

Tests cover:
- validate_tensor_dtype: Dtype validation
- validate_matching_tensors: Matching shape and dtype validation
"""

from tests.shared.conftest import (
    assert_equal,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.validation import (
    validate_tensor_dtype,
    validate_matching_tensors,
)


# ============================================================================
# validate_tensor_dtype Tests
# ============================================================================


fn test_validate_tensor_dtype_float32_correct() raises:
    """Test validate_tensor_dtype with correct float32 dtype."""
    var x = zeros([3, 4], DType.float32)
    validate_tensor_dtype(x, DType.float32, "x")


fn test_validate_tensor_dtype_float64_correct() raises:
    """Test validate_tensor_dtype with correct float64 dtype."""
    var x = zeros([3, 4], DType.float64)
    validate_tensor_dtype(x, DType.float64, "x")


fn test_validate_tensor_dtype_int32_correct() raises:
    """Test validate_tensor_dtype with correct int32 dtype."""
    var x = zeros([3, 4], DType.int32)
    validate_tensor_dtype(x, DType.int32, "x")


fn test_validate_tensor_dtype_mismatch() raises:
    """Test validate_tensor_dtype with mismatched dtype."""
    var x = zeros([3, 4], DType.float32)
    var error_raised = False

    try:
        validate_tensor_dtype(x, DType.float64, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected dtype float64, got float32" in error_msg,
            "Error should mention dtype mismatch",
        )

    assert_true(error_raised, "Error should be raised for dtype mismatch")


# ============================================================================
# validate_matching_tensors Tests
# ============================================================================


fn test_validate_matching_tensors_same_shape_dtype() raises:
    """Test validate_matching_tensors with matching tensors."""
    var x = zeros([3, 4], DType.float32)
    var y = ones([3, 4], DType.float32)
    validate_matching_tensors(x, y, "x", "y")


fn test_validate_matching_tensors_different_dtype() raises:
    """Test validate_matching_tensors with different dtypes."""
    var x = zeros([3, 4], DType.float32)
    var y = ones([3, 4], DType.float64)
    var error_raised = False

    try:
        validate_matching_tensors(x, y, "x", "y")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "mismatched dtypes" in error_msg,
            "Error should mention dtype mismatch",
        )

    assert_true(error_raised, "Error should be raised for dtype mismatch")


fn test_validate_matching_tensors_different_shape() raises:
    """Test validate_matching_tensors with different shapes."""
    var x = zeros([3, 4], DType.float32)
    var y = ones([4, 5], DType.float32)
    var error_raised = False

    try:
        validate_matching_tensors(x, y, "x", "y")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "mismatched shapes" in error_msg,
            "Error should mention shape mismatch",
        )

    assert_true(error_raised, "Error should be raised for shape mismatch")


fn test_validate_matching_tensors_different_ndim() raises:
    """Test validate_matching_tensors with different number of dimensions."""
    var x = zeros([3, 4], DType.float32)
    var y = ones([3, 4, 5], DType.float32)
    var error_raised = False

    try:
        validate_matching_tensors(x, y, "x", "y")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "mismatched number of dimensions" in error_msg,
            "Error should mention dimension count mismatch",
        )

    assert_true(error_raised, "Error should be raised for ndim mismatch")


fn main() raises:
    """Run validate_tensor_dtype and validate_matching_tensors tests."""
    print(
        "Running validate_tensor_dtype and validate_matching_tensors tests..."
    )

    # validate_tensor_dtype tests
    test_validate_tensor_dtype_float32_correct()
    test_validate_tensor_dtype_float64_correct()
    test_validate_tensor_dtype_int32_correct()
    test_validate_tensor_dtype_mismatch()

    # validate_matching_tensors tests
    test_validate_matching_tensors_same_shape_dtype()
    test_validate_matching_tensors_different_dtype()
    test_validate_matching_tensors_different_shape()
    test_validate_matching_tensors_different_ndim()

    print(
        "All validate_tensor_dtype and validate_matching_tensors tests passed!"
    )
