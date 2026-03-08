# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_extensor_slicing.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor slice edge cases and copy semantics."""

from shared.core.extensor import ExTensor, zeros, ones, full, arange
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


# ============================================================================
# Edge Cases
# ============================================================================


fn test_slice_empty() raises:
    """Test empty slice [5:5]."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    var sliced = t[5:5]

    assert_equal(sliced.numel(), 0)


fn test_slice_single_element() raises:
    """Test single element slice [3:4]."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    var sliced = t[3:4]

    assert_equal(sliced.numel(), 1)
    assert_almost_equal(Float64(sliced[0]), 3.0, tolerance=1e-6)


fn test_slice_out_of_bounds_clamped() raises:
    """Test slice with out-of-bounds indices (should be clamped)."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [8:20] should be clamped to [8:10]
    var sliced = t[8:20]

    assert_equal(sliced.numel(), 2)
    assert_almost_equal(Float64(sliced[0]), 8.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 9.0, tolerance=1e-6)


# ============================================================================
# View Semantics Tests
# ============================================================================


fn test_slice_creates_copy() raises:
    """Test that __getitem__(Slice) creates a copy, not a view.

    This is the designed behavior: `t[start:end:step]` always allocates a new
    buffer and copies the (possibly strided) data. Use `tensor.slice()` when
    a memory-sharing view over the first axis is required.
    """
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    var sliced = t[2:7]

    # By design: __getitem__(Slice) creates a copy
    assert_true(not sliced._is_view)


fn test_slice_modification_doesnt_affect_original() raises:
    """Test that modifying a slice doesn't affect the original tensor.

    Because `__getitem__(Slice)` returns a copy, mutations to the result
    must not propagate back to the source tensor. This is the expected,
    designed behavior — not a limitation.
    """
    var t = zeros([10], DType.float32)

    var sliced = t[2:7]

    # Modify slice
    sliced._set_float32(0, Float32(99.0))

    # Check original is NOT affected (copy semantics)
    assert_almost_equal(Float64(t[2]), 0.0, tolerance=1e-6)


fn main() raises:
    """Run all slice edge case and copy semantics tests."""
    test_slice_empty()
    test_slice_single_element()
    test_slice_out_of_bounds_clamped()
    test_slice_creates_copy()
    test_slice_modification_doesnt_affect_original()
    print("All slice edge case and copy semantics tests passed!")
