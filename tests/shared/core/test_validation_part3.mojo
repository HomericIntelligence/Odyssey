# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_validation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for tensor validation utilities - Part 3: validate_2d_input and validate_4d_input.

Tests cover:
- validate_2d_input: 2D tensor validation
- validate_4d_input: 4D tensor validation
"""

from tests.shared.conftest import (
    assert_equal,
    assert_true,
)
from shared.core.extensor import AnyTensor, zeros, ones
from shared.core.validation import (
    validate_2d_input,
    validate_4d_input,
)


# ============================================================================
# validate_2d_input Tests
# ============================================================================


fn test_validate_2d_input_correct() raises:
    """Test validate_2d_input with correct 2D tensor."""
    var x = zeros([3, 4], DType.float32)
    validate_2d_input(x, "x")


fn test_validate_2d_input_1d() raises:
    """Test validate_2d_input with 1D tensor."""
    var x = zeros([10], DType.float32)
    var error_raised = False

    try:
        validate_2d_input(x, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected 2D tensor, got 1D" in error_msg,
            "Error should mention expected 2D but got 1D",
        )

    assert_true(error_raised, "Error should be raised for non-2D tensor")


fn test_validate_2d_input_3d() raises:
    """Test validate_2d_input with 3D tensor."""
    var x = zeros([2, 3, 4], DType.float32)
    var error_raised = False

    try:
        validate_2d_input(x, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected 2D tensor, got 3D" in error_msg,
            "Error should mention expected 2D but got 3D",
        )

    assert_true(error_raised, "Error should be raised for non-2D tensor")


fn test_validate_2d_input_4d() raises:
    """Test validate_2d_input with 4D tensor."""
    var x = zeros([2, 3, 4, 5], DType.float32)
    var error_raised = False

    try:
        validate_2d_input(x, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected 2D tensor, got 4D" in error_msg,
            "Error should mention expected 2D but got 4D",
        )

    assert_true(error_raised, "Error should be raised for non-2D tensor")


# ============================================================================
# validate_4d_input Tests
# ============================================================================


fn test_validate_4d_input_correct() raises:
    """Test validate_4d_input with correct 4D tensor."""
    var x = zeros([2, 3, 4, 5], DType.float32)
    validate_4d_input(x, "x")


fn test_validate_4d_input_2d() raises:
    """Test validate_4d_input with 2D tensor."""
    var x = zeros([3, 4], DType.float32)
    var error_raised = False

    try:
        validate_4d_input(x, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected 4D tensor, got 2D" in error_msg,
            "Error should mention expected 4D but got 2D",
        )

    assert_true(error_raised, "Error should be raised for non-4D tensor")


fn test_validate_4d_input_3d() raises:
    """Test validate_4d_input with 3D tensor."""
    var x = zeros([2, 3, 4], DType.float32)
    var error_raised = False

    try:
        validate_4d_input(x, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected 4D tensor, got 3D" in error_msg,
            "Error should mention expected 4D but got 3D",
        )

    assert_true(error_raised, "Error should be raised for non-4D tensor")


fn test_validate_4d_input_5d() raises:
    """Test validate_4d_input with 5D tensor."""
    var x = zeros([2, 3, 4, 5, 6], DType.float32)
    var error_raised = False

    try:
        validate_4d_input(x, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected 4D tensor, got 5D" in error_msg,
            "Error should mention expected 4D but got 5D",
        )

    assert_true(error_raised, "Error should be raised for non-4D tensor")


fn main() raises:
    """Run validate_2d_input and validate_4d_input tests."""
    print("Running validate_2d_input and validate_4d_input tests...")

    # validate_2d_input tests
    test_validate_2d_input_correct()
    test_validate_2d_input_1d()
    test_validate_2d_input_3d()
    test_validate_2d_input_4d()

    # validate_4d_input tests
    test_validate_4d_input_correct()
    test_validate_4d_input_2d()
    test_validate_4d_input_3d()
    test_validate_4d_input_5d()

    print("All validate_2d_input and validate_4d_input tests passed!")
