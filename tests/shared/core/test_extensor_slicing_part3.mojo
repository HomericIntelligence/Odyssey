# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_extensor_slicing.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for ExTensor edge cases and view semantics (#3013).

Tests cover:
- Edge cases (out-of-bounds clamping)
- View semantics (copy vs view behavior)

Following TDD principles - tests written before implementation.
"""

from shared.core.extensor import ExTensor, zeros, ones, full, arange
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


# ============================================================================
# Edge Cases (continued)
# ============================================================================


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


fn test_slice_2d_value_correctness() raises:
    """Test 2D slice returns correct element values. Closes #3693."""
    # Create a 5x4 tensor with arange values: [[0,1,2,3],[4,5,6,7],...]
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var shape = List[Int]()
    shape.append(5)
    shape.append(4)
    var t2d = t.reshape(shape)

    # Slice rows [1:4] (rows 1,2,3)
    var sliced = t2d[1:4]

    # Row 1 of original = [4,5,6,7], should be row 0 of slice
    assert_almost_equal(
        Float64(sliced._get_float32(0)), 4.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(sliced._get_float32(1)), 5.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(sliced._get_float32(2)), 6.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(sliced._get_float32(3)), 7.0, tolerance=1e-6
    )

    # Row 2 of original = [8,9,10,11], should be row 1 of slice
    assert_almost_equal(
        Float64(sliced._get_float32(4)), 8.0, tolerance=1e-6
    )

    print("PASS: test_slice_2d_value_correctness")


fn test_negative_step_empty_result() raises:
    """Test negative step with invalid range produces empty result. Closes #3699."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [1:5:-1] - start < end with negative step should be empty
    var sliced = t[1:5:-1]

    assert_equal(sliced.numel(), 0)
    print("PASS: test_negative_step_empty_result")


fn test_slice_step_value_correctness() raises:
    """Test slice with step returns correct values. Closes #3693."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [1:8:2] -> elements at indices 1,3,5,7 -> values 1,3,5,7
    var sliced = t[1:8:2]

    assert_equal(sliced.numel(), 4)
    assert_almost_equal(Float64(sliced[0]), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 5.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[3]), 7.0, tolerance=1e-6)

    print("PASS: test_slice_step_value_correctness")


fn test_slice_method_returns_zero_copy_view() raises:
    """Test that slice() method returns a zero-copy view that shares memory.

    Mutating the slice should affect the original tensor.
    This verifies the documented semantics of the slice() method.
    """
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Call slice() method explicitly (not [start:end] syntax which uses __getitem__)
    var sliced = t.slice(2, 7)

    # By design: slice() returns a view
    assert_true(sliced._is_view)

    # Verify initial values match
    assert_almost_equal(Float64(sliced[0]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[4]), 6.0, tolerance=1e-6)

    # Mutate the slice
    sliced._set_float32(0, Float32(99.0))

    # Check that original IS affected (view semantics - zero-copy)
    assert_almost_equal(Float64(t[2]), 99.0, tolerance=1e-6)

    print("PASS: test_slice_method_returns_zero_copy_view")


fn test_getitem_slice_is_copy_not_view() raises:
    """Test that __getitem__(Slice) creates a copy, not a zero-copy view.

    Mutating a slice obtained via [start:end] syntax should NOT affect original.
    This contrasts with slice() method which returns a view.
    """
    var t = zeros([10], DType.float32)

    # Use __getitem__(Slice) syntax - returns a copy
    var sliced = t[2:7]

    # By design: __getitem__(Slice) creates a copy, not a view
    assert_true(not sliced._is_view)

    # Mutate the slice
    sliced._set_float32(0, Float32(99.0))

    # Check original is NOT affected (copy semantics)
    assert_almost_equal(Float64(t[2]), 0.0, tolerance=1e-6)

    print("PASS: test_getitem_slice_is_copy_not_view")


fn main() raises:
    """Run all tests."""
    # Edge cases (continued)
    print("Testing edge cases (continued)...")
    test_slice_out_of_bounds_clamped()
    print("Edge cases: PASSED")

    # Copy semantics (current implementation)
    print("Testing copy semantics...")
    test_slice_creates_copy()
    test_slice_modification_doesnt_affect_original()
    print("Copy semantics: PASSED")

    # Value correctness tests
    print("Testing value correctness...")
    test_slice_2d_value_correctness()
    test_negative_step_empty_result()
    test_slice_step_value_correctness()
    print("Value correctness: PASSED")

    # Zero-copy view semantics (slice() method)
    print("Testing zero-copy view semantics...")
    test_slice_method_returns_zero_copy_view()
    test_getitem_slice_is_copy_not_view()
    print("Zero-copy view semantics: PASSED")

    print("\nAll tests PASSED!")
