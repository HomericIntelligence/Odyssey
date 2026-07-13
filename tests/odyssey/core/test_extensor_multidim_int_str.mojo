"""Tests for AnyTensor __str__ on multi-dimensional int32 and bool tensors.

Verifies that __str__ correctly renders shape information for 2D+ tensors with
non-float dtypes, alongside integer formatting via _format_element().

Related: issue #4048, issue #3376
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from tests.odyssey.conftest import assert_true


def test_str_int32_2d_tensor() raises:
    """Test __str__ for 2D int32 tensor includes correct shape."""
    var t = full([2, 3], Float64(42), DType.int32)
    var s = String(t)
    assert_true(s.startswith("AnyTensor([["))
    assert_true("42" in s)
    assert_true("[2, 3]" in s)
    assert_true("dtype=int32" in s)


def test_str_int32_3d_tensor() raises:
    """Test __str__ for 3D int32 tensor includes correct shape."""
    var t = full([2, 2, 2], Float64(99), DType.int32)
    var s = String(t)
    assert_true("[2, 2, 2]" in s)
    assert_true("99" in s)
    assert_true("dtype=int32" in s)


def test_str_int64_2d_tensor() raises:
    """Test __str__ for 2D int64 tensor includes correct shape."""
    var t = full([3, 4], Float64(123456), DType.int64)
    var s = String(t)
    assert_true("[3, 4]" in s)
    assert_true("123456" in s)
    assert_true("dtype=int64" in s)


def test_str_uint32_2d_tensor() raises:
    """Test __str__ for 2D uint32 tensor includes correct shape."""
    var t = full([2, 5], Float64(1000000), DType.uint32)
    var s = String(t)
    assert_true("[2, 5]" in s)
    assert_true("1000000" in s)
    assert_true("dtype=uint32" in s)


def test_str_bool_2d_true() raises:
    """Test __str__ for 2D bool tensor with true values."""
    var t = full([2, 3], Float64(1), DType.bool)
    var s = String(t)
    assert_true("[2, 3]" in s)
    assert_true("True" in s)
    assert_true("dtype=bool" in s)


def test_str_bool_2d_false() raises:
    """Test __str__ for 2D bool tensor with false values."""
    var t = full([2, 3], Float64(0), DType.bool)
    var s = String(t)
    assert_true("[2, 3]" in s)
    assert_true("False" in s)
    assert_true("dtype=bool" in s)


def test_str_bool_3d_mixed() raises:
    """Test __str__ for 3D bool tensor."""
    # Create a 3D bool tensor - shape [2, 2, 2]
    var t = full([2, 2, 2], Float64(1), DType.bool)
    var s = String(t)
    assert_true("[2, 2, 2]" in s)
    assert_true("True" in s)
    assert_true("dtype=bool" in s)


def test_str_int16_2d_tensor() raises:
    """Test __str__ for 2D int16 tensor includes correct shape."""
    var t = full([4, 5], Float64(500), DType.int16)
    var s = String(t)
    assert_true("[4, 5]" in s)
    assert_true("500" in s)
    assert_true("dtype=int16" in s)


def test_str_uint16_2d_tensor() raises:
    """Test __str__ for 2D uint16 tensor includes correct shape."""
    var t = full([3, 3], Float64(30000), DType.uint16)
    var s = String(t)
    assert_true("[3, 3]" in s)
    assert_true("30000" in s)
    assert_true("dtype=uint16" in s)


def main() raises:
    """Run multi-dimensional int/bool __str__ tests."""
    print("Running AnyTensor __str__ multi-dimensional integer tests...")

    test_str_int32_2d_tensor()
    print("  [OK] test_str_int32_2d_tensor")

    test_str_int32_3d_tensor()
    print("  [OK] test_str_int32_3d_tensor")

    test_str_int64_2d_tensor()
    print("  [OK] test_str_int64_2d_tensor")

    test_str_uint32_2d_tensor()
    print("  [OK] test_str_uint32_2d_tensor")

    test_str_bool_2d_true()
    print("  [OK] test_str_bool_2d_true")

    test_str_bool_2d_false()
    print("  [OK] test_str_bool_2d_false")

    test_str_bool_3d_mixed()
    print("  [OK] test_str_bool_3d_mixed")

    test_str_int16_2d_tensor()
    print("  [OK] test_str_int16_2d_tensor")

    test_str_uint16_2d_tensor()
    print("  [OK] test_str_uint16_2d_tensor")

    print("All AnyTensor __str__ multi-dimensional integer tests passed!")
