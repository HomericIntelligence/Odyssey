"""Tests for typed Tensor[dtype] factory functions (part 2).

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- randn[dtype]: Random normal tensor
- nan_tensor[dtype]: NaN-filled tensor
- inf_tensor[dtype]: Positive infinity tensor
- neg_inf_tensor[dtype]: Negative infinity tensor
- empty[dtype]: Uninitialized tensor
- ones_like[dtype]: One-filled clone by shape
- full_like[dtype]: Constant-filled clone by shape
"""

from testing import assert_true, assert_almost_equal
from math import isnan, isinf
from shared.tensor.factories import (
    randn,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
    empty,
    ones,
    ones_like,
    full_like,
)


fn test_randn() raises:
    """randn[DType.float32] creates a tensor with random values."""
    var t = randn[DType.float32]([10], seed=42)
    assert_true(t.numel() == 10, "numel should be 10")
    assert_true(t.dtype() == DType.float32, "dtype should be float32")
    # Just verify shape and dtype; values are random
    print("PASS: test_randn")


fn test_nan_tensor() raises:
    """nan_tensor[DType.float32] creates a NaN-filled tensor."""
    var t = nan_tensor[DType.float32]([2, 2])
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_true(
            isnan(Float32(t[i])), "element should be NaN"
        )
    print("PASS: test_nan_tensor")


fn test_inf_tensor() raises:
    """inf_tensor[DType.float32] creates a +inf tensor."""
    var t = inf_tensor[DType.float32]([2, 2])
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_true(
            isinf(Float32(t[i])), "element should be inf"
        )
        assert_true(
            Float32(t[i]) > Float32(0.0), "element should be positive"
        )
    print("PASS: test_inf_tensor")


fn test_neg_inf_tensor() raises:
    """neg_inf_tensor[DType.float32] creates a -inf tensor."""
    var t = neg_inf_tensor[DType.float32]([2, 2])
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(4):
        assert_true(
            isinf(Float32(t[i])), "element should be inf"
        )
        assert_true(
            Float32(t[i]) < Float32(0.0), "element should be negative"
        )
    print("PASS: test_neg_inf_tensor")


fn test_empty() raises:
    """empty[DType.float32] creates a tensor (values uninitialized)."""
    var t = empty[DType.float32]([3, 4])
    assert_true(t.numel() == 12, "numel should be 12")
    assert_true(t.dtype() == DType.float32, "dtype should be float32")
    var s = t.shape()
    assert_true(s[0] == 3, "dim 0 should be 3")
    assert_true(s[1] == 4, "dim 1 should be 4")
    print("PASS: test_empty")


fn test_ones_like() raises:
    """ones_like creates a one-filled tensor with same shape."""
    var original = ones[DType.float32]([2, 3])
    var o = ones_like(original)
    assert_true(o.numel() == 6, "numel should match original")
    for i in range(6):
        assert_almost_equal(
            Float64(o[i]), 1.0, atol=1e-6, msg="element should be 1"
        )
    print("PASS: test_ones_like")


fn test_full_like() raises:
    """full_like creates a constant-filled tensor with same shape."""
    var original = ones[DType.float32]([2, 2])
    var f = full_like(original, 0.25)
    assert_true(f.numel() == 4, "numel should match original")
    for i in range(4):
        assert_almost_equal(
            Float64(f[i]), 0.25, atol=1e-6, msg="element should be 0.25"
        )
    print("PASS: test_full_like")


fn main() raises:
    test_randn()
    test_nan_tensor()
    test_inf_tensor()
    test_neg_inf_tensor()
    test_empty()
    test_ones_like()
    test_full_like()
    print("All test_tensor_factories_part2 tests passed!")
