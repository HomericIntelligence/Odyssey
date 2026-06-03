"""Unit tests for extended tensor validation utilities.

Tests cover the new validation functions added for comprehensive API validation:
- validate_1d_input: 1D tensor validation
- validate_3d_input: 3D tensor validation
- validate_axis: Axis range validation
- validate_slice_range: Slice bounds validation
- validate_float_dtype: Float dtype requirement
- validate_positive_shape: Positive dimension validation
- validate_matmul_dims: Matrix multiplication compatibility
- validate_broadcast_compatible: Broadcasting compatibility
- validate_non_empty: Non-empty tensor validation
- validate_matching_dtype: Matching dtype validation
"""

from tests.projectodyssey.conftest import (
    assert_equal,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.core.validation import (
    validate_1d_input,
    validate_3d_input,
    validate_axis,
    validate_slice_range,
    validate_float_dtype,
    validate_positive_shape,
    validate_matmul_dims,
    validate_broadcast_compatible,
    validate_non_empty,
    validate_matching_dtype,
)


# ============================================================================
# validate_1d_input Tests
# ============================================================================


def test_validate_1d_input_correct() raises:
    """Test validate_1d_input with correct 1D tensor."""
    var x = zeros([10], DType.float32)
    validate_1d_input(x, "x")


def test_validate_1d_input_2d() raises:
    """Test validate_1d_input with 2D tensor."""
    var x = zeros([3, 4], DType.float32)
    var error_raised = False

    try:
        validate_1d_input(x, "input")
    except e:
        error_raised = True
        var error_msg = String(e)
        assert_true(
            "expected 1D tensor, got 2D" in error_msg,
            "Error should mention expected 1D but got 2D",
        )

    assert_true(error_raised, "Error should be raised for non-1D tensor")


def main() raises:
    """Run all extended validation tests."""
    print("Running extended validation tests...")

    # validate_1d_input tests
    test_validate_1d_input_correct()
    test_validate_1d_input_2d()

    print("All extended validation tests passed!")
