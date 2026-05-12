"""Tests for typed Tensor[dtype] factory functions (part 2).

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

Tests cover:
- randn[dtype]: Random normal tensor
- nan_tensor[dtype]: NaN-filled tensor
- inf_tensor[dtype]: Positive infinity tensor
- neg_inf_tensor[dtype]: Negative infinity tensor
- empty[dtype]: Uninitialized tensor
- ones_like[dtype]: One-filled clone by shape
- full_like[dtype]: Constant-filled clone by shape
"""


from std.testing import assert_true, assert_almost_equal
from shared.tensor.factories import (
    arange,
    empty,
    eye,
    full,
    full_like,
    inf_tensor,
    linspace,
    nan_tensor,
    neg_inf_tensor,
    ones,
    ones_like,
    randn,
    zeros,
    zeros_like,
)
from std.math import isnan, isinf

def test_zeros() raises:
    """Zeros[DType.float32] creates a zero-filled tensor."""
    var t = zeros[DType.float32]([3, 4])
    assert_true(t.numel() == 12, "numel should be 12")
    assert_true(t.get_dtype() == DType.float32, "dtype should be float32")
    for i in range(12):
        assert_almost_equal(
            Float64(t[i]), 0.0, atol=1e-6, msg="element should be 0"
        )
    print("PASS: test_zeros")


def test_ones() raises:
    """Ones[DType.float32] creates a one-filled tensor."""
    var t = ones[DType.float32]([2, 3])
    assert_true(t.numel() == 6, "numel should be 6")
    for i in range(6):
        assert_almost_equal(
            Float64(t[i]), 1.0, atol=1e-6, msg="element should be 1"
        )
    print("PASS: test_ones")


def test_full() raises:
    """Full[DType.float32] creates a constant-filled tensor."""
    var t = full[DType.float32]([2, 2], 0.5)
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_almost_equal(
            Float64(t[i]), 0.5, atol=1e-6, msg="element should be 0.5"
        )
    print("PASS: test_full")


def test_zeros_like() raises:
    """Zeros_like creates a zero tensor with same shape."""
    var original = ones[DType.float32]([3, 2])
    var z = zeros_like(original)
    assert_true(z.numel() == 6, "numel should match original")
    assert_true(z.get_dtype() == DType.float32, "dtype should match")
    for i in range(6):
        assert_almost_equal(
            Float64(z[i]), 0.0, atol=1e-6, msg="element should be 0"
        )
    print("PASS: test_zeros_like")


def test_arange() raises:
    """Arange[DType.float32] creates a 1D sequence tensor."""
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


def test_eye() raises:
    """Eye[DType.float32] creates a 2D identity-like tensor."""
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


def test_linspace() raises:
    """Linspace[DType.float32] creates evenly spaced values."""
    var t = linspace[DType.float32](0.0, 1.0, 5)
    assert_true(t.numel() == 5, "numel should be 5")
    assert_almost_equal(Float64(t[0]), 0.0, atol=1e-6, msg="first element")
    assert_almost_equal(Float64(t[2]), 0.5, atol=1e-6, msg="middle element")
    assert_almost_equal(Float64(t[4]), 1.0, atol=1e-6, msg="last element")
    print("PASS: test_linspace")


def test_factory_float64() raises:
    """Factory functions work with float64 dtype."""
    var t = zeros[DType.float64]([2, 2])
    assert_true(t.get_dtype() == DType.float64, "dtype should be float64")
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_almost_equal(
            Float64(t[i]), 0.0, atol=1e-10, msg="element should be 0"
        )
    print("PASS: test_factory_float64")


def test_randn() raises:
    """Randn[DType.float32] creates a tensor with random values."""
    var t = randn[DType.float32]([10], seed=42)
    assert_true(t.numel() == 10, "numel should be 10")
    assert_true(t.get_dtype() == DType.float32, "dtype should be float32")
    # Just verify shape and dtype; values are random
    print("PASS: test_randn")


def test_nan_tensor() raises:
    """Nan_tensor[DType.float32] creates a NaN-filled tensor."""
    var t = nan_tensor[DType.float32]([2, 2])
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_true(isnan(Float32(t[i])), "element should be NaN")
    print("PASS: test_nan_tensor")


def test_inf_tensor() raises:
    """Inf_tensor[DType.float32] creates a +inf tensor."""
    var t = inf_tensor[DType.float32]([2, 2])
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_true(isinf(Float32(t[i])), "element should be inf")
        assert_true(Float32(t[i]) > Float32(0.0), "element should be positive")
    print("PASS: test_inf_tensor")


def test_neg_inf_tensor() raises:
    """Neg_inf_tensor[DType.float32] creates a -inf tensor."""
    var t = neg_inf_tensor[DType.float32]([2, 2])
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_true(isinf(Float32(t[i])), "element should be inf")
        assert_true(Float32(t[i]) < Float32(0.0), "element should be negative")
    print("PASS: test_neg_inf_tensor")


def test_empty() raises:
    """Empty[DType.float32] creates a tensor (values uninitialized)."""
    var t = empty[DType.float32]([3, 4])
    assert_true(t.numel() == 12, "numel should be 12")
    assert_true(t.get_dtype() == DType.float32, "dtype should be float32")
    var s = t.shape()
    assert_true(s[0] == 3, "dim 0 should be 3")
    assert_true(s[1] == 4, "dim 1 should be 4")
    print("PASS: test_empty")


def test_ones_like() raises:
    """Ones_like creates a one-filled tensor with same shape."""
    var original = ones[DType.float32]([2, 3])
    var o = ones_like(original)
    assert_true(o.numel() == 6, "numel should match original")
    for i in range(6):
        assert_almost_equal(
            Float64(o[i]), 1.0, atol=1e-6, msg="element should be 1"
        )
    print("PASS: test_ones_like")


def test_full_like() raises:
    """Full_like creates a constant-filled tensor with same shape."""
    var original = ones[DType.float32]([2, 2])
    var f = full_like(original, 0.25)
    assert_true(f.numel() == 4, "numel should match original")
    for i in range(4):
        assert_almost_equal(
            Float64(f[i]), 0.25, atol=1e-6, msg="element should be 0.25"
        )
    print("PASS: test_full_like")


def main() raises:
    """Run all test_tensor_factories tests."""
    print("Running test_tensor_factories tests...")

    test_zeros()
    print("✓ test_zeros")

    test_ones()
    print("✓ test_ones")

    test_full()
    print("✓ test_full")

    test_zeros_like()
    print("✓ test_zeros_like")

    test_arange()
    print("✓ test_arange")

    test_eye()
    print("✓ test_eye")

    test_linspace()
    print("✓ test_linspace")

    test_factory_float64()
    print("✓ test_factory_float64")

    test_randn()
    print("✓ test_randn")

    test_nan_tensor()
    print("✓ test_nan_tensor")

    test_inf_tensor()
    print("✓ test_inf_tensor")

    test_neg_inf_tensor()
    print("✓ test_neg_inf_tensor")

    test_empty()
    print("✓ test_empty")

    test_ones_like()
    print("✓ test_ones_like")

    test_full_like()
    print("✓ test_full_like")

    print("\nAll test_tensor_factories tests passed!")
