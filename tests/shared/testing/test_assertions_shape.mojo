"""Tests for shape and tensor property assertion functions.

Note: Split from test_assertions.mojo due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from testing import assert_true
from shared.testing.assertions import (
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_shape_equal,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
)
from shared.core import ones


fn test_assert_greater_or_equal_float32_passes() raises:
    """Test assert_greater_or_equal with a >= b (Float32)."""
    assert_greater_or_equal(Float32(2.0), Float32(1.0))
    assert_greater_or_equal(Float32(1.0), Float32(1.0))


fn test_assert_greater_or_equal_float32_fails() raises:
    """Test assert_greater_or_equal with a < b (Float32)."""
    var failed = False
    try:
        assert_greater_or_equal(Float32(1.0), Float32(2.0))
    except:
        failed = True
    assert_true(failed, "assert_greater_or_equal should raise error when a < b")


fn test_assert_less_or_equal_float32_passes() raises:
    """Test assert_less_or_equal with a <= b (Float32)."""
    assert_less_or_equal(Float32(1.0), Float32(2.0))
    assert_less_or_equal(Float32(1.0), Float32(1.0))


fn test_assert_less_or_equal_float32_fails() raises:
    """Test assert_less_or_equal with a > b (Float32)."""
    var failed = False
    try:
        assert_less_or_equal(Float32(2.0), Float32(1.0))
    except:
        failed = True
    assert_true(failed, "assert_less_or_equal should raise error when a > b")


fn test_assert_shape_equal_passes() raises:
    """Test assert_shape_equal with matching shapes."""
    var shape1: List[Int] = [2, 3, 4]
    var shape2: List[Int] = [2, 3, 4]
    assert_shape_equal(shape1, shape2)


fn test_assert_shape_equal_fails_dimension() raises:
    """Test assert_shape_equal with different dimension count."""
    var shape1: List[Int] = [2, 3, 4]
    var shape2: List[Int] = [2, 3]
    var failed = False
    try:
        assert_shape_equal(shape1, shape2)
    except:
        failed = True
    assert_true(
        failed, "assert_shape_equal should raise error on dimension mismatch"
    )


fn test_assert_shape_equal_fails_size() raises:
    """Test assert_shape_equal with different dimension sizes."""
    var shape1: List[Int] = [2, 3, 4]
    var shape2: List[Int] = [2, 5, 4]
    var failed = False
    try:
        assert_shape_equal(shape1, shape2)
    except:
        failed = True
    assert_true(
        failed, "assert_shape_equal should raise error on size mismatch"
    )


fn test_assert_shape_tensor_passes() raises:
    """Test assert_shape with matching tensor shape."""
    var tensor = ones([3, 4], DType.float32)
    var expected: List[Int] = [3, 4]
    assert_shape(tensor, expected)


fn test_assert_shape_tensor_fails() raises:
    """Test assert_shape with mismatched tensor shape."""
    var tensor = ones([3, 4], DType.float32)
    var expected: List[Int] = [4, 5]
    var failed = False
    try:
        assert_shape(tensor, expected)
    except:
        failed = True
    assert_true(failed, "assert_shape should raise error on shape mismatch")


fn main() raises:
    """Run shape and tensor property assertion tests."""
    test_assert_greater_or_equal_float32_passes()
    test_assert_greater_or_equal_float32_fails()
    test_assert_less_or_equal_float32_passes()
    test_assert_less_or_equal_float32_fails()
    test_assert_shape_equal_passes()
    test_assert_shape_equal_fails_dimension()
    test_assert_shape_equal_fails_size()
    test_assert_shape_tensor_passes()
    test_assert_shape_tensor_fails()
    print("All shape assertion tests passed!")
