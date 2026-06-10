"""Tests for AnyTensor __str__ truncation behavior.

Verifies NumPy-style truncation: tensors with more than 1000 elements
show first 3 and last 3 elements with '...' in between.

Related: issue #3375
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones, full, arange
from tests.projectodyssey.conftest import assert_true, assert_equal


def test_str_empty_tensor() raises:
    """Test __str__ for empty tensor (numel=0)."""
    var t = zeros([0], DType.float32)
    var s = String(t)
    assert_equal(s, "AnyTensor([], dtype=float32)")


def test_str_single_element() raises:
    """Test __str__ for scalar / 1-element tensor."""
    var t = full([1], 3.0, DType.float32)
    var s = String(t)
    assert_equal(s, "AnyTensor([3.0], dtype=float32)")


def test_str_small_tensor_no_truncation() raises:
    """Test __str__ for tensor with numel <= 1000 shows all elements."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var s = String(t)
    # Should contain all values
    assert_true(s.startswith("AnyTensor(["))
    assert_true("0.0" in s)
    assert_true("1.0" in s)
    assert_true("4.0" in s)
    assert_true("..." not in s)


def test_str_exactly_threshold_no_truncation() raises:
    """Test __str__ for tensor with exactly 1000 elements shows all (no truncation).
    """
    var t = arange(0.0, 1000.0, 1.0, DType.float32)
    var s = String(t)
    # At exactly 1000, no truncation — all values shown
    assert_true("..." not in s)
    assert_true("0.0" in s)
    assert_true("999.0" in s)


def test_str_large_tensor_truncation() raises:
    """Test __str__ for tensor with numel > 1000 shows truncated form."""
    var t = arange(0.0, 1001.0, 1.0, DType.float32)
    var s = String(t)
    assert_true("..." in s)
    assert_true("0.0" in s)
    assert_true("1.0" in s)
    assert_true("2.0" in s)
    assert_true("1000.0" in s)
    assert_true("999.0" in s)
    assert_true("998.0" in s)


def test_str_large_tensor_format() raises:
    """Test __str__ produces correct format for large tensor."""
    var t = arange(0.0, 2000.0, 1.0, DType.float32)
    var s = String(t)
    # Must start and end correctly
    assert_true(s.startswith("AnyTensor([0.0, 1.0, 2.0, ..."))
    assert_true(s.endswith(", dtype=float32)"))
    assert_true("1999.0" in s)
    assert_true("1998.0" in s)
    assert_true("1997.0" in s)


def test_str_dtype_preserved() raises:
    """Test __str__ correctly reports dtype for large tensor."""
    var tf16 = arange(0.0, 1001.0, 1.0, DType.float16)
    var sf16 = String(tf16)
    assert_true("dtype=float16" in sf16)
    assert_true("..." in sf16)

    var tf64 = arange(0.0, 1001.0, 1.0, DType.float64)
    var sf64 = String(tf64)
    assert_true("dtype=float64" in sf64)
    assert_true("..." in sf64)


def test_str_no_truncation_for_6_elements() raises:
    """Test that a 6-element tensor is shown in full (edge case near SHOW_ELEMENTS*2).
    """
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var s = String(t)
    assert_true("..." not in s)
    assert_true("5.0" in s)


def test_str_empty_tensor_int32() raises:
    """Test __str__ for empty tensor with int32 dtype (non-float).

    This is an edge case: _format_element should never be called on an empty
    tensor, so the function should return without errors even for dtypes that
    might have format issues.
    """
    var t = zeros([0], DType.int32)
    var s = String(t)
    assert_equal(s, "AnyTensor([], dtype=int32)")


def test_str_empty_tensor_int64() raises:
    """Test __str__ for empty tensor with int64 dtype."""
    var t = zeros([0], DType.int64)
    var s = String(t)
    assert_equal(s, "AnyTensor([], dtype=int64)")


def test_str_empty_tensor_uint32() raises:
    """Test __str__ for empty tensor with uint32 dtype."""
    var t = zeros([0], DType.uint32)
    var s = String(t)
    assert_equal(s, "AnyTensor([], dtype=uint32)")


def test_str_empty_tensor_float16() raises:
    """Test __str__ for empty tensor with float16 dtype (floating-point variant).
    """
    var t = zeros([0], DType.float16)
    var s = String(t)
    assert_equal(s, "AnyTensor([], dtype=float16)")


def test_str_empty_multidim_tensor_int32() raises:
    """Test __str__ for empty multidimensional tensor (e.g., shape=[2, 0])."""
    var t = zeros([2, 0], DType.int32)
    var s = String(t)
    # Empty tensor should still show as empty list
    assert_equal(s, "AnyTensor([], dtype=int32)")


def main() raises:
    """Run __str__ truncation tests."""
    print("Running AnyTensor __str__ truncation tests...")

    test_str_empty_tensor()
    print("  [OK] test_str_empty_tensor")

    test_str_single_element()
    print("  [OK] test_str_single_element")

    test_str_small_tensor_no_truncation()
    print("  [OK] test_str_small_tensor_no_truncation")

    test_str_exactly_threshold_no_truncation()
    print("  [OK] test_str_exactly_threshold_no_truncation")

    test_str_large_tensor_truncation()
    print("  [OK] test_str_large_tensor_truncation")

    test_str_large_tensor_format()
    print("  [OK] test_str_large_tensor_format")

    test_str_dtype_preserved()
    print("  [OK] test_str_dtype_preserved")

    test_str_no_truncation_for_6_elements()
    print("  [OK] test_str_no_truncation_for_6_elements")

    test_str_empty_tensor_int32()
    print("  [OK] test_str_empty_tensor_int32")

    test_str_empty_tensor_int64()
    print("  [OK] test_str_empty_tensor_int64")

    test_str_empty_tensor_uint32()
    print("  [OK] test_str_empty_tensor_uint32")

    test_str_empty_tensor_float16()
    print("  [OK] test_str_empty_tensor_float16")

    test_str_empty_multidim_tensor_int32()
    print("  [OK] test_str_empty_multidim_tensor_int32")

    print("All AnyTensor __str__ truncation tests passed!")
