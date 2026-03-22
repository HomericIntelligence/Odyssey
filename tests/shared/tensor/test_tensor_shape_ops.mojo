"""Tests for typed Tensor[dtype] shape operations.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- reshape[dt]: Reshape typed tensor
- squeeze[dt]: Remove size-1 dimensions
- unsqueeze[dt]: Insert size-1 dimension
- flatten[dt]: Flatten to 1D
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.factories import ones, full
from shared.core.shape import (
    reshape,
    squeeze,
    unsqueeze,
    flatten,
)


fn test_reshape_typed() raises:
    """reshape preserves dtype and values through round-trip."""
    var t = full[DType.float32]([2, 3], 1.5)
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
    print("PASS: test_reshape_typed")


fn test_reshape_to_1d() raises:
    """reshape to 1D works correctly."""
    var t = ones[DType.float32]([2, 3])
    var r = reshape(t, [6])
    var s = r.shape()
    assert_true(len(s) == 1, "should be 1D")
    assert_true(s[0] == 6, "should have 6 elements")
    print("PASS: test_reshape_to_1d")


fn test_squeeze_typed() raises:
    """squeeze removes size-1 dimensions."""
    var t = ones[DType.float32]([1, 3, 1])
    var s = squeeze(t)
    var shape = s.shape()
    assert_true(len(shape) == 1, "should be 1D after squeeze")
    assert_true(shape[0] == 3, "remaining dim should be 3")
    assert_true(s.dtype() == DType.float32, "dtype should be preserved")
    print("PASS: test_squeeze_typed")


fn test_squeeze_axis() raises:
    """squeeze with specific axis removes only that dimension."""
    var t = ones[DType.float32]([1, 3, 1])
    var s = squeeze(t, axis=0)
    var shape = s.shape()
    assert_true(len(shape) == 2, "should be 2D")
    assert_true(shape[0] == 3, "first dim should be 3")
    assert_true(shape[1] == 1, "second dim should remain 1")
    print("PASS: test_squeeze_axis")


fn test_unsqueeze_typed() raises:
    """unsqueeze inserts size-1 dimension."""
    var t = ones[DType.float32]([3, 4])
    var u = unsqueeze(t, axis=0)
    var shape = u.shape()
    assert_true(len(shape) == 3, "should be 3D")
    assert_true(shape[0] == 1, "new dim should be 1")
    assert_true(shape[1] == 3, "original dim 0")
    assert_true(shape[2] == 4, "original dim 1")
    assert_true(u.dtype() == DType.float32, "dtype should be preserved")
    print("PASS: test_unsqueeze_typed")


fn test_unsqueeze_last() raises:
    """unsqueeze at last axis works correctly."""
    var t = ones[DType.float32]([3, 4])
    var u = unsqueeze(t, axis=2)
    var shape = u.shape()
    assert_true(len(shape) == 3, "should be 3D")
    assert_true(shape[0] == 3, "dim 0 preserved")
    assert_true(shape[1] == 4, "dim 1 preserved")
    assert_true(shape[2] == 1, "new dim at end")
    print("PASS: test_unsqueeze_last")


fn test_flatten_typed() raises:
    """flatten converts to 1D preserving values."""
    var t = full[DType.float32]([2, 3], 0.25)
    var f = flatten(t)
    var shape = f.shape()
    assert_true(len(shape) == 1, "should be 1D")
    assert_true(shape[0] == 6, "should have 6 elements")
    assert_true(f.dtype() == DType.float32, "dtype should be preserved")
    for i in range(6):
        assert_almost_equal(
            Float64(f[i]), 0.25, atol=1e-6, msg="value should be preserved"
        )
    print("PASS: test_flatten_typed")


fn test_flatten_already_1d() raises:
    """flatten of 1D tensor returns same shape."""
    var t = ones[DType.float32]([5])
    var f = flatten(t)
    var shape = f.shape()
    assert_true(len(shape) == 1, "should remain 1D")
    assert_true(shape[0] == 5, "should have 5 elements")
    print("PASS: test_flatten_already_1d")


fn main() raises:
    test_reshape_typed()
    test_reshape_to_1d()
    test_squeeze_typed()
    test_squeeze_axis()
    test_unsqueeze_typed()
    test_unsqueeze_last()
    test_flatten_typed()
    test_flatten_already_1d()
    print("All test_tensor_shape_ops tests passed!")
