"""Tests for typed Tensor[dtype] factory functions (part 1).

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- zeros creates zero-filled tensor
- ones creates one-filled tensor
- full with typed fill value
- empty allocates tensor
- zeros_like preserves shape
- ones_like preserves shape
- full_like preserves shape with value
- arange produces correct sequence
- eye produces identity matrix
- factory float64 precision
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.factories import (
    zeros,
    ones,
    full,
    empty,
    zeros_like,
    ones_like,
    full_like,
    arange,
    eye,
)


fn test_zeros_creates_zero_tensor() raises:
    """Verify zeros[dtype] creates a tensor filled with zeros."""
    var t = zeros[DType.float32]([3, 4])
    assert_true(t.numel() == 12, "numel should be 12")
    assert_true(t.ndim() == 2, "ndim should be 2")
    for i in range(t.numel()):
        assert_almost_equal(t[i], Scalar[DType.float32](0.0), msg="element should be 0")
    print("PASS: test_zeros_creates_zero_tensor")


fn test_ones_creates_one_tensor() raises:
    """Verify ones[dtype] creates a tensor filled with ones."""
    var t = ones[DType.float32]([2, 3])
    assert_true(t.numel() == 6, "numel should be 6")
    for i in range(t.numel()):
        assert_almost_equal(t[i], Scalar[DType.float32](1.0), msg="element should be 1")
    print("PASS: test_ones_creates_one_tensor")


fn test_full_with_value() raises:
    """Verify full[dtype] fills tensor with a specific typed value."""
    var t = full[DType.float32]([3], Scalar[DType.float32](0.5))
    assert_true(t.numel() == 3, "numel should be 3")
    for i in range(t.numel()):
        assert_almost_equal(t[i], Scalar[DType.float32](0.5), msg="element should be 0.5")
    print("PASS: test_full_with_value")


fn test_empty_allocates_tensor() raises:
    """Verify empty[dtype] allocates a tensor with correct shape."""
    var t = empty[DType.float64]([4, 5])
    assert_true(t.numel() == 20, "numel should be 20")
    assert_true(t.ndim() == 2, "ndim should be 2")
    var s = t.shape()
    assert_true(s[0] == 4, "dim 0 should be 4")
    assert_true(s[1] == 5, "dim 1 should be 5")
    print("PASS: test_empty_allocates_tensor")


fn test_zeros_like_preserves_shape() raises:
    """Verify zeros_like creates zero tensor with same shape as input."""
    var original = ones[DType.float32]([2, 3, 4])
    var t = zeros_like(original)
    assert_true(t.numel() == 24, "numel should be 24")
    assert_true(t.ndim() == 3, "ndim should be 3")
    var s = t.shape()
    assert_true(s[0] == 2, "dim 0")
    assert_true(s[1] == 3, "dim 1")
    assert_true(s[2] == 4, "dim 2")
    for i in range(t.numel()):
        assert_almost_equal(t[i], Scalar[DType.float32](0.0), msg="element should be 0")
    print("PASS: test_zeros_like_preserves_shape")


fn test_ones_like_preserves_shape() raises:
    """Verify ones_like creates one tensor with same shape as input."""
    var original = zeros[DType.float64]([5, 2])
    var t = ones_like(original)
    assert_true(t.numel() == 10, "numel should be 10")
    var s = t.shape()
    assert_true(s[0] == 5, "dim 0")
    assert_true(s[1] == 2, "dim 1")
    for i in range(t.numel()):
        assert_almost_equal(t[i], Scalar[DType.float64](1.0), msg="element should be 1")
    print("PASS: test_ones_like_preserves_shape")


fn test_full_like_preserves_shape() raises:
    """Verify full_like creates tensor with same shape filled with value."""
    var original = zeros[DType.float32]([3, 3])
    var t = full_like(original, Scalar[DType.float32](0.25))
    assert_true(t.numel() == 9, "numel should be 9")
    for i in range(t.numel()):
        assert_almost_equal(t[i], Scalar[DType.float32](0.25), msg="element should be 0.25")
    print("PASS: test_full_like_preserves_shape")


fn test_arange_values() raises:
    """Verify arange produces correct sequence values."""
    var t = arange[DType.float32](
        Scalar[DType.float32](0.0),
        Scalar[DType.float32](5.0),
        Scalar[DType.float32](1.0),
    )
    assert_true(t.numel() == 5, "should have 5 elements")
    assert_almost_equal(t[0], Scalar[DType.float32](0.0), msg="element 0")
    assert_almost_equal(t[1], Scalar[DType.float32](1.0), msg="element 1")
    assert_almost_equal(t[2], Scalar[DType.float32](2.0), msg="element 2")
    assert_almost_equal(t[3], Scalar[DType.float32](3.0), msg="element 3")
    assert_almost_equal(t[4], Scalar[DType.float32](4.0), msg="element 4")
    print("PASS: test_arange_values")


fn test_eye_identity() raises:
    """Verify eye creates correct identity matrix."""
    var t = eye[DType.float32](3, 3, 0)
    assert_true(t.numel() == 9, "should have 9 elements")
    # Check diagonal is 1, off-diagonal is 0
    for i in range(3):
        for j in range(3):
            var idx = i * 3 + j
            if i == j:
                assert_almost_equal(
                    t[idx], Scalar[DType.float32](1.0), msg="diagonal should be 1"
                )
            else:
                assert_almost_equal(
                    t[idx], Scalar[DType.float32](0.0), msg="off-diagonal should be 0"
                )
    print("PASS: test_eye_identity")


fn test_factory_float64_precision() raises:
    """Verify float64 factory functions preserve full precision."""
    # Use a value that would lose precision if round-tripped through float32
    var precise_val = Scalar[DType.float64](1.0000000000000002)
    var t = full[DType.float64]([1], precise_val)
    # Direct comparison — should be exact for float64
    assert_true(
        t[0] == precise_val,
        "float64 should preserve full precision",
    )
    print("PASS: test_factory_float64_precision")


fn main() raises:
    test_zeros_creates_zero_tensor()
    test_ones_creates_one_tensor()
    test_full_with_value()
    test_empty_allocates_tensor()
    test_zeros_like_preserves_shape()
    test_ones_like_preserves_shape()
    test_full_like_preserves_shape()
    test_arange_values()
    test_eye_identity()
    test_factory_float64_precision()
