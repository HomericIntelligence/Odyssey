"""Unit tests for tensor validation utilities.

Tests cover:
- validate_tensor_shape: Shape validation
"""


from tests.projectodyssey.conftest import (
    assert_equal,
    assert_true,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.core.validation import (
    validate_2d_input,
    validate_4d_input,
    validate_matching_tensors,
    validate_tensor_dtype,
    validate_tensor_shape,
)


def test_validate_tensor_shape_1d_correct() raises:
    """Test validate_tensor_shape with correct 1D shape."""
    var x = zeros([10], DType.float32)
    var expected_shape: List[Int] = [10]
    validate_tensor_shape(x, expected_shape, "x")


def test_validate_tensor_shape_2d_correct() raises:
    """Test validate_tensor_shape with correct 2D shape."""
    var x = zeros([3, 4], DType.float32)
    var expected_shape: List[Int] = [3, 4]
    validate_tensor_shape(x, expected_shape, "x")


def test_validate_tensor_shape_3d_correct() raises:
    """Test validate_tensor_shape with correct 3D shape."""
    var x = zeros([2, 3, 4], DType.float32)
    var expected_shape: List[Int] = [2, 3, 4]
    validate_tensor_shape(x, expected_shape, "x")


def test_validate_tensor_shape_wrong_dimension_count() raises:
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


def test_validate_tensor_shape_wrong_dimension_value() raises:
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


def test_validate_tensor_dtype_float32_correct() raises:
    """Test validate_tensor_dtype with correct float32 dtype."""
    var x = zeros([3, 4], DType.float32)
    validate_tensor_dtype(x, DType.float32, "x")


def test_validate_tensor_dtype_float64_correct() raises:
    """Test validate_tensor_dtype with correct float64 dtype."""
    var x = zeros([3, 4], DType.float64)
    validate_tensor_dtype(x, DType.float64, "x")


def test_validate_tensor_dtype_int32_correct() raises:
    """Test validate_tensor_dtype with correct int32 dtype."""
    var x = zeros([3, 4], DType.int32)
    validate_tensor_dtype(x, DType.int32, "x")


def test_validate_tensor_dtype_mismatch() raises:
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


def test_validate_matching_tensors_same_shape_dtype() raises:
    """Test validate_matching_tensors with matching tensors."""
    var x = zeros([3, 4], DType.float32)
    var y = ones([3, 4], DType.float32)
    validate_matching_tensors(x, y, "x", "y")


def test_validate_matching_tensors_different_dtype() raises:
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


def test_validate_matching_tensors_different_shape() raises:
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


def test_validate_matching_tensors_different_ndim() raises:
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


def test_validate_2d_input_correct() raises:
    """Test validate_2d_input with correct 2D tensor."""
    var x = zeros([3, 4], DType.float32)
    validate_2d_input(x, "x")


def test_validate_2d_input_1d() raises:
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


def test_validate_2d_input_3d() raises:
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


def test_validate_2d_input_4d() raises:
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


def test_validate_4d_input_correct() raises:
    """Test validate_4d_input with correct 4D tensor."""
    var x = zeros([2, 3, 4, 5], DType.float32)
    validate_4d_input(x, "x")


def test_validate_4d_input_2d() raises:
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


def test_validate_4d_input_3d() raises:
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


def test_validate_4d_input_5d() raises:
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


def main() raises:
    """Run all test_validation tests."""
    print("Running test_validation tests...")

    test_validate_tensor_shape_1d_correct()
    print("✓ test_validate_tensor_shape_1d_correct")

    test_validate_tensor_shape_2d_correct()
    print("✓ test_validate_tensor_shape_2d_correct")

    test_validate_tensor_shape_3d_correct()
    print("✓ test_validate_tensor_shape_3d_correct")

    test_validate_tensor_shape_wrong_dimension_count()
    print("✓ test_validate_tensor_shape_wrong_dimension_count")

    test_validate_tensor_shape_wrong_dimension_value()
    print("✓ test_validate_tensor_shape_wrong_dimension_value")

    test_validate_tensor_dtype_float32_correct()
    print("✓ test_validate_tensor_dtype_float32_correct")

    test_validate_tensor_dtype_float64_correct()
    print("✓ test_validate_tensor_dtype_float64_correct")

    test_validate_tensor_dtype_int32_correct()
    print("✓ test_validate_tensor_dtype_int32_correct")

    test_validate_tensor_dtype_mismatch()
    print("✓ test_validate_tensor_dtype_mismatch")

    test_validate_matching_tensors_same_shape_dtype()
    print("✓ test_validate_matching_tensors_same_shape_dtype")

    test_validate_matching_tensors_different_dtype()
    print("✓ test_validate_matching_tensors_different_dtype")

    test_validate_matching_tensors_different_shape()
    print("✓ test_validate_matching_tensors_different_shape")

    test_validate_matching_tensors_different_ndim()
    print("✓ test_validate_matching_tensors_different_ndim")

    test_validate_2d_input_correct()
    print("✓ test_validate_2d_input_correct")

    test_validate_2d_input_1d()
    print("✓ test_validate_2d_input_1d")

    test_validate_2d_input_3d()
    print("✓ test_validate_2d_input_3d")

    test_validate_2d_input_4d()
    print("✓ test_validate_2d_input_4d")

    test_validate_4d_input_correct()
    print("✓ test_validate_4d_input_correct")

    test_validate_4d_input_2d()
    print("✓ test_validate_4d_input_2d")

    test_validate_4d_input_3d()
    print("✓ test_validate_4d_input_3d")

    test_validate_4d_input_5d()
    print("✓ test_validate_4d_input_5d")

    print("\nAll test_validation tests passed!")
