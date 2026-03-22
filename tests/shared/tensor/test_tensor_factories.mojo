"""Tests for typed Tensor[dtype] factory functions.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- zeros[dtype]: Zero-filled tensor creation
- ones[dtype]: One-filled tensor creation
- full[dtype]: Constant-filled tensor creation
- zeros_like[dtype]: Zero-filled clone by shape
- arange[dtype]: Evenly spaced 1D tensor
- eye[dtype]: Identity-like 2D tensor
- linspace[dtype]: Linearly spaced 1D tensor
- Factory with float64 dtype
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.factories import (
    zeros,
    ones,
    full,
    zeros_like,
    arange,
    eye,
    linspace,
)


fn test_zeros() raises:
    """zeros[DType.float32] creates a zero-filled tensor."""
    var t = zeros[DType.float32]([3, 4])
    assert_true(t.numel() == 12, "numel should be 12")
    assert_true(t.dtype() == DType.float32, "dtype should be float32")
    for i in range(12):
        assert_almost_equal(
            Float64(t[i]), 0.0, atol=1e-6, msg="element should be 0"
        )
    print("PASS: test_zeros")


fn test_ones() raises:
    """ones[DType.float32] creates a one-filled tensor."""
    var t = ones[DType.float32]([2, 3])
    assert_true(t.numel() == 6, "numel should be 6")
    for i in range(6):
        assert_almost_equal(
            Float64(t[i]), 1.0, atol=1e-6, msg="element should be 1"
        )
    print("PASS: test_ones")


fn test_full() raises:
    """full[DType.float32] creates a constant-filled tensor."""
    var t = full[DType.float32]([2, 2], 0.5)
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_almost_equal(
            Float64(t[i]), 0.5, atol=1e-6, msg="element should be 0.5"
        )
    print("PASS: test_full")


fn test_zeros_like() raises:
    """zeros_like creates a zero tensor with same shape."""
    var original = ones[DType.float32]([3, 2])
    var z = zeros_like(original)
    assert_true(z.numel() == 6, "numel should match original")
    assert_true(z.dtype() == DType.float32, "dtype should match")
    for i in range(6):
        assert_almost_equal(
            Float64(z[i]), 0.0, atol=1e-6, msg="element should be 0"
        )
    print("PASS: test_zeros_like")


fn test_arange() raises:
    """arange[DType.float32] creates a 1D sequence tensor."""
    var t = arange[DType.float32](0.0, 4.0, 1.0)
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_almost_equal(
            Float64(t[i]),
            Float64(i),
            atol=1e-6,
            msg="element should match index",
        )
    print("PASS: test_arange")


fn test_eye() raises:
    """eye[DType.float32] creates a 2D identity-like tensor."""
    var t = eye[DType.float32](3, 3)
    assert_true(t.numel() == 9, "numel should be 9")
    for i in range(3):
        for j in range(3):
            var expected = 1.0 if i == j else 0.0
            assert_almost_equal(
                Float64(t[i * 3 + j]),
                expected,
                atol=1e-6,
                msg="diagonal element check",
            )
    print("PASS: test_eye")


fn test_linspace() raises:
    """linspace[DType.float32] creates evenly spaced values."""
    var t = linspace[DType.float32](0.0, 1.0, 5)
    assert_true(t.numel() == 5, "numel should be 5")
    assert_almost_equal(Float64(t[0]), 0.0, atol=1e-6, msg="first element")
    assert_almost_equal(
        Float64(t[2]), 0.5, atol=1e-6, msg="middle element"
    )
    assert_almost_equal(Float64(t[4]), 1.0, atol=1e-6, msg="last element")
    print("PASS: test_linspace")


fn test_factory_float64() raises:
    """Factory functions work with float64 dtype."""
    var t = zeros[DType.float64]([2, 2])
    assert_true(t.dtype() == DType.float64, "dtype should be float64")
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_almost_equal(
            Float64(t[i]), 0.0, atol=1e-10, msg="element should be 0"
        )
    print("PASS: test_factory_float64")


fn main() raises:
    test_zeros()
    test_ones()
    test_full()
    test_zeros_like()
    test_arange()
    test_eye()
    test_linspace()
    test_factory_float64()
    print("All test_tensor_factories tests passed!")
