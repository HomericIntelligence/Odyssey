"""Tests for ExTensor utility operations - Part 4: type conversions, __str__/__repr__, __hash__, and diff().

Tests type conversions, string representations, hash consistency,
and consecutive difference computations.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_utility.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    clone,
    item,
    diff,
    as_contiguous,
    transpose_view,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_equal,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_equal_int,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
)


# ============================================================================
# Test type conversions
# ============================================================================


fn test_int_conversion() raises:
    """Test int conversion via item()."""
    var shape = List[Int]()
    var t = full(shape, 42.5, DType.float32)

    # Use item() to extract value, then convert to Int
    var val = Int(item(t))
    assert_equal_int(val, 42, "item() + Int should convert to int")


fn test_float_conversion() raises:
    """Test float conversion via item()."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.int32)

    # Use item() to extract value as Float64
    var val = item(t)
    assert_almost_equal(val, 42.0, 1e-6, "item() should return Float64 value")


# ============================================================================
# Test __str__ and __repr__
# ============================================================================


fn test_str_readable() raises:
    """Test __str__ produces readable output."""
    var t = arange(0.0, 3.0, 1.0, DType.float32)
    var s = String(t)
    assert_equal(
        s, "ExTensor([0.0, 1.0, 2.0], dtype=float32)", "__str__ format"
    )


fn test_repr_complete() raises:
    """Test __repr__ produces complete representation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var r = repr(t)
    assert_equal(
        r,
        (
            "ExTensor(shape=[2, 2], dtype=float32, numel=4, data=[1.0, 1.0,"
            " 1.0, 1.0])"
        ),
        "__repr__ format",
    )


# ============================================================================
# Test __hash__
# ============================================================================


fn test_hash_immutable() raises:
    """Test __hash__ for immutable tensors."""
    var a = arange(0.0, 3.0, 1.0, DType.float32)
    var b = arange(0.0, 3.0, 1.0, DType.float32)

    var hash_a = hash(a)
    var hash_b = hash(b)
    assert_equal_int(
        Int(hash_a), Int(hash_b), "Equal tensors should have same hash"
    )


fn test_hash_different_values_differ() raises:
    """Test that tensors with different values produce different hashes."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 1.0, DType.float64)
    var b = full(shape, 2.0, DType.float64)

    var hash_a = hash(a)
    var hash_b = hash(b)
    if hash_a == hash_b:
        raise Error(
            "Tensors with different values should have different hashes"
        )


# ============================================================================
# Test diff() - consecutive differences
# ============================================================================


fn test_diff_1d() raises:
    """Test computing consecutive differences."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    var d = diff(t)

    # Result: [1, 1, 1, 1] (4 elements)
    assert_numel(d, 4, "diff should have n-1 elements")
    for i in range(4):
        assert_value_at(d, i, 1.0, 1e-6, "Consecutive differences should be 1")


fn test_diff_higher_order() raises:
    """Test higher-order differences."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var d = diff(t, 2)

    # Second differences of [0,1,2,3,4] -> [0,0,0]
    assert_numel(d, 3, "Second diff should have n-2 elements")
    for i in range(3):
        assert_value_at(d, i, 0.0, 1e-6, "Second-order differences should be 0")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run utility operation tests - Part 4: conversions, str/repr, hash, and diff.
    """
    print("Running ExTensor utility operation tests (Part 4)...")

    # Type conversions
    print("  Testing type conversions...")
    test_int_conversion()
    test_float_conversion()

    # __str__ and __repr__
    print("  Testing __str__ and __repr__...")
    test_str_readable()
    test_repr_complete()

    # __hash__
    print("  Testing __hash__...")
    test_hash_immutable()
    test_hash_different_values_differ()

    # diff()
    print("  Testing diff()...")
    test_diff_1d()
    test_diff_higher_order()

    print("All utility operation tests (Part 4) completed!")
