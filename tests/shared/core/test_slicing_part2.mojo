# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_slicing.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tensor slicing tests part 2: edge cases and batch extraction.

Test Categories:
1. Edge Cases (4 tests)
2. Batch Extraction (2 tests)

Total: 6 tests.
"""

from shared.core import ExTensor, zeros, ones
from shared.data import extract_batch, extract_batch_pair
from tests.shared.conftest import assert_equal, assert_almost_equal


fn test_slice_single_element() raises:
    """Test slicing a single element."""
    var tensor = zeros([5], DType.float32)
    for i in range(5):
        tensor._set_float32(i, Float32(i))

    # Slice single element [2:3]
    var single = tensor.slice(2, 3, axis=0)

    # Verify shape
    assert_equal(single.shape()[0], 1, "Single element slice shape")

    # Verify value
    assert_almost_equal(Float64(single._get_float32(0)), 2.0, tolerance=1e-6)

    print("PASS: test_slice_single_element")


fn test_slice_out_of_bounds_start() raises:
    """Test that out-of-bounds start index raises error."""
    var tensor = zeros([5], DType.float32)

    try:
        var bad_slice = tensor.slice(10, 15, axis=0)
        raise Error("Expected error for out-of-bounds start")
    except:
        # Expected
        print("PASS: test_slice_out_of_bounds_start")


fn test_slice_out_of_bounds_end() raises:
    """Test that out-of-bounds end index raises error."""
    var tensor = zeros([5], DType.float32)

    try:
        var bad_slice = tensor.slice(0, 100, axis=0)
        raise Error("Expected error for out-of-bounds end")
    except:
        # Expected
        print("PASS: test_slice_out_of_bounds_end")


fn test_slice_invalid_axis() raises:
    """Test that invalid axis raises error."""
    var tensor = zeros([5, 3], DType.float32)

    try:
        var bad_slice = tensor.slice(0, 2, axis=5)
        raise Error("Expected error for invalid axis")
    except:
        # Expected
        print("PASS: test_slice_invalid_axis")


fn test_batch_extraction_uses_view() raises:
    """Verify that batch extraction creates views, not copies."""
    # Create dataset with 10 samples of shape (3, 2)
    var dataset = zeros([10, 3, 2], DType.float32)
    var idx = 0.0
    for i in range(10):
        for j in range(3):
            for k in range(2):
                var linear_idx = i * 6 + j * 2 + k
                dataset._set_float32(linear_idx, Float32(idx))
                idx += 1.0

    # Extract a batch
    var batch = extract_batch(dataset, 2, 3)

    # Verify shape
    assert_equal(batch.shape()[0], 3, "Batch shape[0]")
    assert_equal(batch.shape()[1], 3, "Batch shape[1]")
    assert_equal(batch.shape()[2], 2, "Batch shape[2]")

    # Verify batch is a view (marked as such)
    assert_equal(batch._is_view, True, "Batch is marked as view")

    # Verify batch sees original data
    var expected_start = 2.0 * 3.0 * 2.0
    assert_almost_equal(
        Float64(batch._get_float32(0)), expected_start, tolerance=1e-6
    )

    print("PASS: test_batch_extraction_uses_view")


fn test_batch_extraction_pair() raises:
    """Verify that batch pair extraction creates views for both data and labels.
    """
    # Create paired dataset and labels
    var images = zeros([5, 2], DType.float32)
    var labels = zeros([5], DType.float32)

    for i in range(5):
        labels._set_float32(i, Float32(i * 10))
        for j in range(2):
            var idx = i * 2 + j
            images._set_float32(idx, Float32(i * 10 + j))

    # Extract batch pair
    var (batch_imgs, batch_lbls) = extract_batch_pair(images, labels, 1, 3)

    # Verify shapes
    assert_equal(batch_imgs.shape()[0], 3, "Batch images shape[0]")
    assert_equal(batch_lbls.shape()[0], 3, "Batch labels shape[0]")

    # Verify both are views
    assert_equal(batch_imgs._is_view, True, "Batch images are view")
    assert_equal(batch_lbls._is_view, True, "Batch labels are view")

    # Verify values
    assert_almost_equal(
        Float64(batch_lbls._get_float32(0)), 10.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(batch_lbls._get_float32(1)), 20.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(batch_lbls._get_float32(2)), 30.0, tolerance=1e-6
    )

    print("PASS: test_batch_extraction_pair")


fn test_slice_view_mutation_visible() raises:
    """Test that mutating a view slice is visible in original. Closes #3900."""
    var t = zeros([10], DType.float32)
    for i in range(10):
        t._set_float32(i, Float32(i))

    # .slice() creates a view
    var view = t.slice(2, 5, axis=0)

    # Mutate through the view
    view._set_float32(0, Float32(99.0))

    # Original should reflect the change
    assert_almost_equal(
        Float64(t._get_float32(2)), 99.0, tolerance=1e-6
    )

    print("PASS: test_slice_view_mutation_visible")


fn test_chained_slice_shares_memory() raises:
    """Test slice-of-slice shares memory with root tensor. Closes #3901."""
    var t = zeros([10], DType.float32)
    for i in range(10):
        t._set_float32(i, Float32(i))

    # First slice: view of elements [2:8]
    var view1 = t.slice(2, 8, axis=0)

    # Second slice of view: elements [1:4] of view1 = [3:6] of original
    var view2 = view1.slice(1, 4, axis=0)

    # Mutate through second slice
    view2._set_float32(0, Float32(77.0))

    # Original should see the change at index 3
    assert_almost_equal(
        Float64(t._get_float32(3)), 77.0, tolerance=1e-6
    )

    print("PASS: test_chained_slice_shares_memory")


fn main() raises:
    """Run slicing tests part 2."""
    print("Running tensor slicing tests part 2...")

    # Edge case tests
    test_slice_single_element()
    test_slice_out_of_bounds_start()
    test_slice_out_of_bounds_end()
    test_slice_invalid_axis()

    # Batch extraction tests
    test_batch_extraction_uses_view()
    test_batch_extraction_pair()

    # View mutation tests
    test_slice_view_mutation_visible()
    test_chained_slice_shares_memory()

    print("\nAll tests passed!")
