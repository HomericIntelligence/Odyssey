"""Tests for AnyTensor broadcasting operations - Part 5: are_shapes_broadcastable ndim guard.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_broadcasting.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests the ndim guard added to are_shapes_broadcastable() in Issue #3859.
Broadcasting cannot reduce the number of dimensions: if shape2 has fewer
dimensions than shape1, are_shapes_broadcastable must return False.
"""

from shared.base.broadcasting import are_shapes_broadcastable
from testing import assert_true


# ============================================================================
# Tests for ndim guard: shape2 fewer dims than shape1 -> False
# ============================================================================


fn test_are_shapes_broadcastable_ndim_reduction_returns_false() raises:
    """Verify are_shapes_broadcastable([3,4,5], [4,5]) returns False (ndim reduction)."""
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    shape1.append(5)
    var shape2 = List[Int]()
    shape2.append(4)
    shape2.append(5)
    assert_true(
        not are_shapes_broadcastable(shape1, shape2),
        "3D->2D ndim reduction must return False",
    )


fn test_are_shapes_broadcastable_1d_vs_2d_reduction() raises:
    """Verify are_shapes_broadcastable([3,4], [4]) returns False (ndim reduction)."""
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(4)
    assert_true(
        not are_shapes_broadcastable(shape1, shape2),
        "2D->1D ndim reduction must return False",
    )


fn test_are_shapes_broadcastable_empty_target_returns_false() raises:
    """Verify are_shapes_broadcastable([3], []) returns False (empty target)."""
    var shape1 = List[Int]()
    shape1.append(3)
    var shape2 = List[Int]()
    assert_true(
        not are_shapes_broadcastable(shape1, shape2),
        "Non-empty source vs empty target must return False",
    )


# ============================================================================
# Tests for valid broadcasting (expanding dims or same ndim)
# ============================================================================


fn test_are_shapes_broadcastable_expanding_ndim_ok() raises:
    """Verify are_shapes_broadcastable([4,5], [3,4,5]) returns True (expanding dims)."""
    var shape1 = List[Int]()
    shape1.append(4)
    shape1.append(5)
    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    shape2.append(5)
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "2D->3D expansion must return True",
    )


fn test_are_shapes_broadcastable_same_ndim_compatible() raises:
    """Verify are_shapes_broadcastable([3,4], [3,4]) returns True (same shape)."""
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "Identical shapes must return True",
    )


fn test_are_shapes_broadcastable_broadcast_1_dim_ok() raises:
    """Verify are_shapes_broadcastable([1,4], [3,4]) returns True (dim-1 broadcast)."""
    var shape1 = List[Int]()
    shape1.append(1)
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "Dim-1 broadcast [1,4]->[3,4] must return True",
    )


fn test_are_shapes_broadcastable_incompatible_dims_unchanged() raises:
    """Verify are_shapes_broadcastable([3,4], [5,4]) returns False (incompatible dims)."""
    var shape1 = List[Int]()
    shape1.append(3)
    shape1.append(4)
    var shape2 = List[Int]()
    shape2.append(5)
    shape2.append(4)
    assert_true(
        not are_shapes_broadcastable(shape1, shape2),
        "Incompatible dims [3,4] vs [5,4] must return False",
    )


fn test_are_shapes_broadcastable_scalar_source_empty_target() raises:
    """Verify are_shapes_broadcastable([], []) returns True (both empty/scalar)."""
    var shape1 = List[Int]()
    var shape2 = List[Int]()
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "Both empty shapes (scalar) must return True",
    )


fn test_are_shapes_broadcastable_scalar_source_to_1d() raises:
    """Verify are_shapes_broadcastable([], [3]) returns True (scalar to 1D)."""
    var shape1 = List[Int]()
    var shape2 = List[Int]()
    shape2.append(3)
    assert_true(
        are_shapes_broadcastable(shape1, shape2),
        "Scalar source to 1D target must return True",
    )


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run broadcasting part 5 tests (ndim guard in are_shapes_broadcastable)."""
    print("Running AnyTensor broadcasting tests - Part 5...")

    # ndim reduction cases (must return False)
    print("  Testing ndim reduction returns False...")
    test_are_shapes_broadcastable_ndim_reduction_returns_false()
    test_are_shapes_broadcastable_1d_vs_2d_reduction()
    test_are_shapes_broadcastable_empty_target_returns_false()

    # Valid broadcasting cases
    print("  Testing valid broadcasting cases...")
    test_are_shapes_broadcastable_expanding_ndim_ok()
    test_are_shapes_broadcastable_same_ndim_compatible()
    test_are_shapes_broadcastable_broadcast_1_dim_ok()
    test_are_shapes_broadcastable_incompatible_dims_unchanged()
    test_are_shapes_broadcastable_scalar_source_empty_target()
    test_are_shapes_broadcastable_scalar_source_to_1d()

    print("All broadcasting part 5 tests completed!")
