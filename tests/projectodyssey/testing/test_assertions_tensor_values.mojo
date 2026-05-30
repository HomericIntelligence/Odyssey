"""Tests for tensor value comparison assertion functions.

"""

from std.testing import assert_true
from projectodyssey.testing.assertions import (
    assert_all_values,
    assert_all_close,
    assert_tensor_equal,
    assert_not_equal_tensor,
)
from projectodyssey.tensor.tensor_creation import full, ones, zeros


def test_assert_all_values_passes() raises:
    """Test assert_all_values with all matching values."""
    var tensor = ones([3, 4], DType.float32)
    assert_all_values(tensor, 1.0, tolerance=1e-6)


def test_assert_all_values_fails() raises:
    """Test assert_all_values with non-matching values."""
    var tensor = ones([3, 4], DType.float32)
    var failed = False
    try:
        assert_all_values(tensor, 2.0, tolerance=1e-6)
    except:
        failed = True
    assert_true(
        failed, "assert_all_values should raise error on value mismatch"
    )


def test_assert_all_close_passes() raises:
    """Test assert_all_close with close tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = full([3, 4], 1.0000001, DType.float32)
    assert_all_close(tensor1, tensor2, tolerance=1e-5)


def test_assert_all_close_fails() raises:
    """Test assert_all_close with distant tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = full([3, 4], 2.0, DType.float32)
    var failed = False
    try:
        assert_all_close(tensor1, tensor2, tolerance=1e-5)
    except:
        failed = True
    assert_true(
        failed, "assert_all_close should raise error on tensor mismatch"
    )


def test_assert_tensor_equal_passes() raises:
    """Test assert_tensor_equal with equal tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = ones([3, 4], DType.float32)
    assert_tensor_equal(tensor1, tensor2)


def test_assert_tensor_equal_fails_shape() raises:
    """Test assert_tensor_equal with different shapes."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = ones([4, 5], DType.float32)
    var failed = False
    try:
        assert_tensor_equal(tensor1, tensor2)
    except:
        failed = True
    assert_true(
        failed, "assert_tensor_equal should raise error on shape mismatch"
    )


def test_assert_tensor_equal_fails_values() raises:
    """Test assert_tensor_equal with different values."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = full([3, 4], 2.0, DType.float32)
    var failed = False
    try:
        assert_tensor_equal(tensor1, tensor2)
    except:
        failed = True
    assert_true(
        failed, "assert_tensor_equal should raise error on value mismatch"
    )


def test_assert_not_equal_tensor_passes() raises:
    """Test assert_not_equal_tensor with different tensors."""
    var tensor1 = ones([3, 4], DType.float32)
    var tensor2 = full([3, 4], 2.0, DType.float32)
    assert_not_equal_tensor(tensor1, tensor2)


def main() raises:
    """Run tensor value assertion tests."""
    test_assert_all_values_passes()
    test_assert_all_values_fails()
    test_assert_all_close_passes()
    test_assert_all_close_fails()
    test_assert_tensor_equal_passes()
    test_assert_tensor_equal_fails_shape()
    test_assert_tensor_equal_fails_values()
    test_assert_not_equal_tensor_passes()
    print("All tensor value assertion tests passed!")
