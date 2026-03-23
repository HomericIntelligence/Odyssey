"""Tests for contiguous tensor fast path optimizations - Part 1: Helper functions.

Tests the contiguous tensor fast path helper functions:
- shapes_match helper with various shape combinations
- can_use_fast_path with contiguous and non-contiguous tensors

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_arithmetic_contiguous.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal_int,
    assert_false,
    assert_shape,
    assert_true,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.arithmetic import add, subtract, multiply, divide


# ============================================================================
# Helper Tests
# ============================================================================


fn test_shapes_match_identical_1d() raises:
    """Test shapes_match helper with identical 1D shapes."""
    from shared.core.arithmetic_contiguous import shapes_match

    var a = ones([5], DType.float32)
    var b = ones([5], DType.float32)

    assert_true(shapes_match(a, b), "Identical 1D shapes should match")


fn test_shapes_match_identical_2d() raises:
    """Test shapes_match helper with identical 2D shapes."""
    from shared.core.arithmetic_contiguous import shapes_match

    var a = ones([3, 4], DType.float32)
    var b = ones([3, 4], DType.float32)

    assert_true(shapes_match(a, b), "Identical 2D shapes should match")


fn test_shapes_match_different_shapes() raises:
    """Test shapes_match helper with different shapes."""
    from shared.core.arithmetic_contiguous import shapes_match

    var a = ones([3, 4], DType.float32)
    var b = ones([3, 5], DType.float32)

    assert_false(shapes_match(a, b), "Different shapes should not match")


fn test_shapes_match_different_dims() raises:
    """Test shapes_match helper with different number of dimensions."""
    from shared.core.arithmetic_contiguous import shapes_match

    var a = ones([3, 4], DType.float32)
    var b = ones([3, 4, 1], DType.float32)

    assert_false(
        shapes_match(a, b), "Different dimension counts should not match"
    )


fn test_can_use_fast_path_contiguous_same_shape() raises:
    """Test can_use_fast_path with contiguous same-shape tensors."""
    from shared.core.arithmetic_contiguous import can_use_fast_path

    var a = ones([3, 4], DType.float32)
    var b = ones([3, 4], DType.float32)

    # Both should be contiguous by default (newly created)
    assert_true(a.is_contiguous(), "a should be contiguous")
    assert_true(b.is_contiguous(), "b should be contiguous")
    assert_true(
        can_use_fast_path(a, b),
        "Contiguous same-shape tensors should use fast path",
    )


fn test_can_use_fast_path_different_shapes() raises:
    """Test can_use_fast_path rejects different shapes."""
    from shared.core.arithmetic_contiguous import can_use_fast_path

    var a = ones([3, 4], DType.float32)
    var b = ones([3, 5], DType.float32)

    assert_false(
        can_use_fast_path(a, b),
        "Different shapes should not use fast path",
    )


fn test_can_use_fast_path_different_dtypes() raises:
    """Test can_use_fast_path rejects different dtypes."""
    from shared.core.arithmetic_contiguous import can_use_fast_path

    var a = ones([3, 4], DType.float32)
    var b = ones([3, 4], DType.float64)

    assert_false(
        can_use_fast_path(a, b),
        "Different dtypes should not use fast path",
    )


fn main() raises:
    """Run all helper function tests."""
    print("Running contiguous fast path helper tests (part 1)...")

    print("  Testing helper functions...")
    test_shapes_match_identical_1d()
    test_shapes_match_identical_2d()
    test_shapes_match_different_shapes()
    test_shapes_match_different_dims()
    test_can_use_fast_path_contiguous_same_shape()
    test_can_use_fast_path_different_shapes()
    test_can_use_fast_path_different_dtypes()

    print("All contiguous fast path helper tests passed!")
