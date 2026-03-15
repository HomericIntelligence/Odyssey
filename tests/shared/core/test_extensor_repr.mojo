"""Tests for ExTensor __repr__ truncation behavior.

Verifies NumPy-style truncation: tensors with more than 1000 elements
show first 3 and last 3 elements with '...' in between.

Related: issue #4038, follow-up from #3375
"""

from shared.core.extensor import ExTensor, zeros, ones, full, arange
from tests.shared.conftest import assert_true, assert_equal


fn test_repr_empty_tensor() raises:
    """Test __repr__ for empty tensor (numel=0)."""
    var t = zeros([0], DType.float32)
    var s = repr(t)
    assert_equal(s, "ExTensor(shape=[0], dtype=float32, numel=0, data=[])")


fn test_repr_single_element() raises:
    """Test __repr__ for scalar / 1-element tensor."""
    var t = full([1], 3.0, DType.float32)
    var s = repr(t)
    assert_equal(s, "ExTensor(shape=[1], dtype=float32, numel=1, data=[3.0])")


fn test_repr_small_tensor_no_truncation() raises:
    """Test __repr__ for tensor with numel <= 1000 shows all elements."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var s = repr(t)
    # Should contain all values
    assert_true(s.startswith("ExTensor(shape="))
    assert_true("0.0" in s)
    assert_true("1.0" in s)
    assert_true("4.0" in s)
    assert_true("..." not in s)


fn test_repr_exactly_threshold_no_truncation() raises:
    """Test __repr__ for tensor with exactly 1000 elements shows all (no truncation).
    """
    var t = arange(0.0, 1000.0, 1.0, DType.float32)
    var s = repr(t)
    # At exactly 1000, no truncation — all values shown
    assert_true("..." not in s)
    assert_true("0.0" in s)
    assert_true("999.0" in s)


fn test_repr_large_tensor_truncation() raises:
    """Test __repr__ for tensor with numel > 1000 shows truncated form."""
    var t = arange(0.0, 1001.0, 1.0, DType.float32)
    var s = repr(t)
    assert_true("..." in s)
    assert_true("0.0" in s)
    assert_true("1.0" in s)
    assert_true("2.0" in s)
    assert_true("1000.0" in s)
    assert_true("999.0" in s)
    assert_true("998.0" in s)


fn test_repr_large_tensor_format() raises:
    """Test __repr__ produces correct format for large tensor."""
    var t = arange(0.0, 2000.0, 1.0, DType.float32)
    var s = repr(t)
    # Must start and end correctly
    assert_true(s.startswith("ExTensor(shape=[2000], dtype=float32, numel=2000, data=[0.0, 1.0, 2.0, ..."))
    assert_true(s.endswith("])"))
    assert_true("1999.0" in s)
    assert_true("1998.0" in s)
    assert_true("1997.0" in s)


fn test_repr_dtype_preserved() raises:
    """Test __repr__ correctly reports dtype for large tensor."""
    var tf16 = arange(0.0, 1001.0, 1.0, DType.float16)
    var sf16 = repr(tf16)
    assert_true("dtype=float16" in sf16)
    assert_true("..." in sf16)

    var tf64 = arange(0.0, 1001.0, 1.0, DType.float64)
    var sf64 = repr(tf64)
    assert_true("dtype=float64" in sf64)
    assert_true("..." in sf64)


fn test_repr_shape_preserved() raises:
    """Test __repr__ correctly shows shape for large 2D tensor."""
    var t = zeros([50, 30], DType.float32)  # numel=1500 > threshold
    var s = repr(t)
    assert_true("shape=[50, 30]" in s)
    assert_true("numel=1500" in s)
    assert_true("..." in s)


fn test_repr_no_truncation_for_6_elements() raises:
    """Test that a 6-element tensor is shown in full (edge case near SHOW_ELEMENTS*2).
    """
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    var s = repr(t)
    assert_true("..." not in s)
    assert_true("5.0" in s)


fn test_repr_empty_tensor_int32() raises:
    """Test __repr__ for empty tensor with int32 dtype."""
    var t = zeros([0], DType.int32)
    var s = repr(t)
    assert_equal(s, "ExTensor(shape=[0], dtype=int32, numel=0, data=[])")


fn test_repr_empty_tensor_float16() raises:
    """Test __repr__ for empty tensor with float16 dtype."""
    var t = zeros([0], DType.float16)
    var s = repr(t)
    assert_equal(s, "ExTensor(shape=[0], dtype=float16, numel=0, data=[])")


fn main() raises:
    """Run __repr__ truncation tests."""
    print("Running ExTensor __repr__ truncation tests...")

    test_repr_empty_tensor()
    print("  [OK] test_repr_empty_tensor")

    test_repr_single_element()
    print("  [OK] test_repr_single_element")

    test_repr_small_tensor_no_truncation()
    print("  [OK] test_repr_small_tensor_no_truncation")

    test_repr_exactly_threshold_no_truncation()
    print("  [OK] test_repr_exactly_threshold_no_truncation")

    test_repr_large_tensor_truncation()
    print("  [OK] test_repr_large_tensor_truncation")

    test_repr_large_tensor_format()
    print("  [OK] test_repr_large_tensor_format")

    test_repr_dtype_preserved()
    print("  [OK] test_repr_dtype_preserved")

    test_repr_shape_preserved()
    print("  [OK] test_repr_shape_preserved")

    test_repr_no_truncation_for_6_elements()
    print("  [OK] test_repr_no_truncation_for_6_elements")

    test_repr_empty_tensor_int32()
    print("  [OK] test_repr_empty_tensor_int32")

    test_repr_empty_tensor_float16()
    print("  [OK] test_repr_empty_tensor_float16")

    print("All ExTensor __repr__ truncation tests passed!")
