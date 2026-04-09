"""Tests for AnyTensor shape operations.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- reshape: Reshape via AnyTensor
- squeeze: Remove size-1 dimensions via AnyTensor
- unsqueeze: Insert size-1 dimension via AnyTensor
- flatten: Flatten to 1D via AnyTensor
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.any_tensor import AnyTensor, ones as any_ones, full as any_full
from shared.core.shape import (
    reshape,
    squeeze,
    unsqueeze,
    flatten,
)


def test_reshape() raises:
    """Reshape preserves dtype and values through round-trip."""
    var t = any_full([2, 3], 1.5, DType.float32)
    var r = reshape(t, [3, 2])
    var s = r.shape()
    assert_true(s[0] == 3, "dim 0 should be 3")
    assert_true(s[1] == 2, "dim 1 should be 2")
    assert_true(r.numel() == 6, "numel should be preserved")
    assert_true(r.dtype() == DType.float32, "dtype should be preserved")
    for i in range(6):
        assert_almost_equal(
            Float64(r[i]), 1.5, atol=1e-6, msg="value should be preserved"
        )
    print("PASS: test_reshape")


def test_reshape_to_1d() raises:
    """Reshape to 1D works correctly."""
    var t = any_ones([2, 3], DType.float32)
    var r = reshape(t, [6])
    var s = r.shape()
    assert_true(len(s) == 1, "should be 1D")
    assert_true(s[0] == 6, "should have 6 elements")
    print("PASS: test_reshape_to_1d")


def test_squeeze() raises:
    """Squeeze removes size-1 dimensions."""
    var t = any_ones([1, 3, 1], DType.float32)
    var s = squeeze(t)
    var shape = s.shape()
    assert_true(len(shape) == 1, "should be 1D after squeeze")
    assert_true(shape[0] == 3, "remaining dim should be 3")
    assert_true(s.dtype() == DType.float32, "dtype should be preserved")
    print("PASS: test_squeeze")


def test_squeeze_axis() raises:
    """Squeeze with specific axis removes only that dimension."""
    var t = any_ones([1, 3, 1], DType.float32)
    var s = squeeze(t, axis=0)
    var shape = s.shape()
    assert_true(len(shape) == 2, "should be 2D")
    assert_true(shape[0] == 3, "first dim should be 3")
    assert_true(shape[1] == 1, "second dim should remain 1")
    print("PASS: test_squeeze_axis")


def test_unsqueeze() raises:
    """Unsqueeze inserts size-1 dimension."""
    var t = any_ones([3, 4], DType.float32)
    var u = unsqueeze(t, axis=0)
    var shape = u.shape()
    assert_true(len(shape) == 3, "should be 3D")
    assert_true(shape[0] == 1, "new dim should be 1")
    assert_true(shape[1] == 3, "original dim 0")
    assert_true(shape[2] == 4, "original dim 1")
    assert_true(u.dtype() == DType.float32, "dtype should be preserved")
    print("PASS: test_unsqueeze")


def test_unsqueeze_last() raises:
    """Unsqueeze at last axis works correctly."""
    var t = any_ones([3, 4], DType.float32)
    var u = unsqueeze(t, axis=2)
    var shape = u.shape()
    assert_true(len(shape) == 3, "should be 3D")
    assert_true(shape[0] == 3, "dim 0 preserved")
    assert_true(shape[1] == 4, "dim 1 preserved")
    assert_true(shape[2] == 1, "new dim at end")
    print("PASS: test_unsqueeze_last")


def test_flatten() raises:
    """Flatten converts to 1D preserving values."""
    var t = any_full([2, 3], 0.25, DType.float32)
    var f = flatten(t)
    var shape = f.shape()
    assert_true(len(shape) == 1, "should be 1D")
    assert_true(shape[0] == 6, "should have 6 elements")
    assert_true(f.dtype() == DType.float32, "dtype should be preserved")
    for i in range(6):
        assert_almost_equal(
            Float64(f[i]), 0.25, atol=1e-6, msg="value should be preserved"
        )
    print("PASS: test_flatten")


def test_flatten_already_1d() raises:
    """Flatten of 1D tensor returns same shape."""
    var t = any_ones([5], DType.float32)
    var f = flatten(t)
    var shape = f.shape()
    assert_true(len(shape) == 1, "should remain 1D")
    assert_true(shape[0] == 5, "should have 5 elements")
    print("PASS: test_flatten_already_1d")


def main() raises:
    test_reshape()
    test_reshape_to_1d()
    test_squeeze()
    test_squeeze_axis()
    test_unsqueeze()
    test_unsqueeze_last()
    test_flatten()
    test_flatten_already_1d()
    print("All test_tensor_shape_ops tests passed!")
