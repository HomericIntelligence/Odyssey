"""Tests for Tensor[dtype] typed shape operation overloads.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- reshape preserves dtype and values
- squeeze removes size-1 dims
- unsqueeze adds dimension
- flatten to 1D
- expand_dims adds dimension at position
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.core.shape import (
    reshape,
    squeeze,
    unsqueeze,
    expand_dims,
    flatten,
)


fn test_reshape_typed() raises:
    """Reshape preserves dtype and values."""
    var t = Tensor[DType.float32]([2, 3])
    # Set some values
    t[0] = 1.0
    t[1] = 2.0
    t[2] = 3.0
    t[3] = 4.0
    t[4] = 5.0
    t[5] = 6.0

    var r = reshape(t, [3, 2])
    assert_true(r.numel() == 6, "numel should be 6")
    assert_true(r.ndim() == 2, "ndim should be 2")
    var s = r.shape()
    assert_true(s[0] == 3, "dim 0 should be 3")
    assert_true(s[1] == 2, "dim 1 should be 2")
    assert_true(r.dtype() == DType.float32, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float32](1.0), msg="val 0")
    assert_almost_equal(r[5], Scalar[DType.float32](6.0), msg="val 5")
    print("PASS: test_reshape_typed")


fn test_squeeze_typed() raises:
    """Squeeze removes size-1 dims."""
    var t = Tensor[DType.float32]([1, 3, 1, 4])
    for i in range(12):
        t[i] = Scalar[DType.float32](i)

    var r = squeeze(t)
    assert_true(r.ndim() == 2, "ndim should be 2 after squeeze")
    var s = r.shape()
    assert_true(s[0] == 3, "dim 0 should be 3")
    assert_true(s[1] == 4, "dim 1 should be 4")
    assert_true(r.dtype() == DType.float32, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float32](0.0), msg="val 0")
    assert_almost_equal(r[11], Scalar[DType.float32](11.0), msg="val 11")
    print("PASS: test_squeeze_typed")


fn test_unsqueeze_typed() raises:
    """Unsqueeze adds dimension."""
    var t = Tensor[DType.float32]([3, 4])
    for i in range(12):
        t[i] = Scalar[DType.float32](i)

    var r = unsqueeze(t, axis=0)
    assert_true(r.ndim() == 3, "ndim should be 3")
    var s = r.shape()
    assert_true(s[0] == 1, "dim 0 should be 1")
    assert_true(s[1] == 3, "dim 1 should be 3")
    assert_true(s[2] == 4, "dim 2 should be 4")
    assert_true(r.dtype() == DType.float32, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float32](0.0), msg="val 0")
    print("PASS: test_unsqueeze_typed")


fn test_flatten_typed() raises:
    """Flatten to 1D."""
    var t = Tensor[DType.float64]([2, 3])
    for i in range(6):
        t[i] = Scalar[DType.float64](i + 1)

    var r = flatten(t)
    assert_true(r.ndim() == 1, "ndim should be 1")
    var s = r.shape()
    assert_true(s[0] == 6, "dim 0 should be 6")
    assert_true(r.dtype() == DType.float64, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float64](1.0), msg="val 0")
    assert_almost_equal(r[5], Scalar[DType.float64](6.0), msg="val 5")
    print("PASS: test_flatten_typed")


fn test_expand_dims_typed() raises:
    """Expand_dims adds dimension at position."""
    var t = Tensor[DType.float32]([4])
    for i in range(4):
        t[i] = Scalar[DType.float32](i)

    var r = expand_dims(t, axis=0)
    assert_true(r.ndim() == 2, "ndim should be 2")
    var s = r.shape()
    assert_true(s[0] == 1, "dim 0 should be 1")
    assert_true(s[1] == 4, "dim 1 should be 4")
    assert_true(r.dtype() == DType.float32, "dtype preserved")
    assert_almost_equal(r[0], Scalar[DType.float32](0.0), msg="val 0")
    assert_almost_equal(r[3], Scalar[DType.float32](3.0), msg="val 3")
    print("PASS: test_expand_dims_typed")


fn main() raises:
    test_reshape_typed()
    test_squeeze_typed()
    test_unsqueeze_typed()
    test_flatten_typed()
    test_expand_dims_typed()
