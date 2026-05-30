"""Tests for AnyTensor __str__ on integer types (int8, int16, int32, int64, uint8, uint16, uint32, uint64).

Verifies that _format_element() correctly handles all integer dtype paths for string representation.

Related: issue #4047, issue #3376
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import arange, full, zeros
from tests.projectodyssey.conftest import assert_true, assert_equal


def test_str_int8() raises:
    """Test __str__ for int8 tensor."""
    var t = full([3], Float64(5), DType.int8)
    var s = String(t)
    assert_true(s.startswith("AnyTensor(["))
    assert_true("5" in s)
    assert_true("dtype=int8" in s)


def test_str_int16() raises:
    """Test __str__ for int16 tensor."""
    var t = full([3], Float64(1000), DType.int16)
    var s = String(t)
    assert_true(s.startswith("AnyTensor(["))
    assert_true("1000" in s)
    assert_true("dtype=int16" in s)


def test_str_int32() raises:
    """Test __str__ for int32 tensor."""
    var t = full([3], Float64(100000), DType.int32)
    var s = String(t)
    assert_true(s.startswith("AnyTensor(["))
    assert_true("100000" in s)
    assert_true("dtype=int32" in s)


def test_str_int64() raises:
    """Test __str__ for int64 tensor."""
    var t = full([3], Float64(9999999999), DType.int64)
    var s = String(t)
    assert_true(s.startswith("AnyTensor(["))
    assert_true("9999999999" in s)
    assert_true("dtype=int64" in s)


def test_str_uint8() raises:
    """Test __str__ for uint8 tensor."""
    var t = full([3], Float64(255), DType.uint8)
    var s = String(t)
    assert_true(s.startswith("AnyTensor(["))
    assert_true("255" in s)
    assert_true("dtype=uint8" in s)


def test_str_uint16() raises:
    """Test __str__ for uint16 tensor."""
    var t = full([3], Float64(65535), DType.uint16)
    var s = String(t)
    assert_true(s.startswith("AnyTensor(["))
    assert_true("65535" in s)
    assert_true("dtype=uint16" in s)


def test_str_uint32() raises:
    """Test __str__ for uint32 tensor."""
    var t = full([3], Float64(4294967295), DType.uint32)
    var s = String(t)
    assert_true(s.startswith("AnyTensor(["))
    assert_true("4294967295" in s)
    assert_true("dtype=uint32" in s)


def test_str_uint64() raises:
    """Test __str__ for uint64 tensor."""
    var t = full([3], Float64(9007199254740992), DType.uint64)
    var s = String(t)
    assert_true(s.startswith("AnyTensor(["))
    assert_true("9007199254740992" in s)
    assert_true("dtype=uint64" in s)


def test_str_int64_large_tensor_truncation() raises:
    """Test __str__ for large int64 tensor shows truncation."""
    # Create a large int64 tensor and verify truncation works with integer format
    var t = zeros([1001], DType.int64)
    # Manually set a few values for verification (using _set_int64 which is correct for int64)
    t._set_int64(0, 100)
    t._set_int64(1, 101)
    t._set_int64(1000, 1100)
    var s = String(t)
    assert_true("..." in s)
    assert_true("dtype=int64" in s)


def test_str_uint8_negative_not_in_output() raises:
    """Test __str__ for uint8 shows unsigned values (no negative signs)."""
    var t = full([3], Float64(200), DType.uint8)
    var s = String(t)
    # uint8(200) should never appear as negative
    assert_true("200" in s)
    assert_true("-" not in s)


def main() raises:
    """Run integer dtype __str__ tests."""
    print("Running AnyTensor __str__ integer dtype tests...")

    test_str_int8()
    print("  [OK] test_str_int8")

    test_str_int16()
    print("  [OK] test_str_int16")

    test_str_int32()
    print("  [OK] test_str_int32")

    test_str_int64()
    print("  [OK] test_str_int64")

    test_str_uint8()
    print("  [OK] test_str_uint8")

    test_str_uint16()
    print("  [OK] test_str_uint16")

    test_str_uint32()
    print("  [OK] test_str_uint32")

    test_str_uint64()
    print("  [OK] test_str_uint64")

    test_str_int64_large_tensor_truncation()
    print("  [OK] test_str_int64_large_tensor_truncation")

    test_str_uint8_negative_not_in_output()
    print("  [OK] test_str_uint8_negative_not_in_output")

    print("All AnyTensor __str__ integer dtype tests passed!")
