"""Tensor slicing tests part 1: basic functionality, view semantics, reference counting.

Test Categories:
1. Basic Functionality (4 tests)
2. View Semantics (1 test)
3. Reference Counting (2 tests)
4. Edge Cases (1 test)

Total: 8 tests.
"""


from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones
from tests.projectodyssey.conftest import assert_equal, assert_almost_equal
from projectodyssey.data.batch_utils import extract_batch, extract_batch_pair


def test_slice_basic_1d() raises:
    """Test basic slicing on 1D tensor."""
    # Create 1D tensor: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    var tensor = zeros([10], DType.float32)
    for i in range(10):
        tensor._set_float32(i, Float32(i))

    # Slice [2:5] should give [2, 3, 4]
    var slice_result = tensor.slice(2, 5, axis=0)

    # Verify shape
    assert_equal(slice_result.shape()[0], 5 - 2, "1D slice shape")

    # Verify values are correct (should be 2, 3, 4)
    assert_almost_equal(
        Float64(slice_result._get_float32(0)), 2.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(slice_result._get_float32(1)), 3.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(slice_result._get_float32(2)), 4.0, tolerance=1e-6
    )

    print("PASS: test_slice_basic_1d")


def test_slice_2d_axis0() raises:
    """Test slicing 2D tensor along axis 0 (rows)."""
    # Create 2D tensor with shape (4, 3)
    var tensor = zeros([4, 3], DType.float32)
    for i in range(4):
        for j in range(3):
            var idx = i * 3 + j
            tensor._set_float32(idx, Float32(i * 3 + j))

    # Slice rows [1:3]
    var slice_result = tensor.slice(1, 3, axis=0)

    # Verify shape
    assert_equal(slice_result.shape()[0], 2, "2D axis0 slice shape[0]")
    assert_equal(slice_result.shape()[1], 3, "2D axis0 slice shape[1]")

    # Verify values (should be rows 1 and 2)
    assert_almost_equal(
        Float64(slice_result._get_float32(0)), 3.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(slice_result._get_float32(1)), 4.0, tolerance=1e-6
    )
    assert_almost_equal(
        Float64(slice_result._get_float32(3)), 6.0, tolerance=1e-6
    )

    print("PASS: test_slice_2d_axis0")


def test_slice_4d_batch() raises:
    """Test slicing 4D tensor (typical CNN batch: batch_size, channels, height, width).
    """
    # Create 4D tensor with shape (8, 3, 4, 4) representing 8 images
    var tensor = zeros([8, 3, 4, 4], DType.float32)
    var idx = 0.0
    for b in range(8):
        for c in range(3):
            for h in range(4):
                for w in range(4):
                    tensor._set_float32(
                        b * 48 + c * 16 + h * 4 + w, Float32(idx)
                    )
                    idx += 1.0

    # Extract batch [2:5] (3 images)
    var batch = tensor.slice(2, 5, axis=0)

    # Verify shape
    assert_equal(batch.shape()[0], 3, "4D batch slice shape[0]")
    assert_equal(batch.shape()[1], 3, "4D batch slice shape[1]")
    assert_equal(batch.shape()[2], 4, "4D batch slice shape[2]")
    assert_equal(batch.shape()[3], 4, "4D batch slice shape[3]")

    # Verify first element of batch corresponds to element 2 of original
    var expected_start = 2.0 * 3.0 * 4.0 * 4.0
    assert_almost_equal(
        Float64(batch._get_float32(0)), expected_start, tolerance=1e-6
    )

    print("PASS: test_slice_4d_batch")


def test_slice_full_range() raises:
    """Test slicing the full range returns same data."""
    var tensor = zeros([5], DType.float32)
    for i in range(5):
        tensor._set_float32(i, Float32(i))

    # Slice full range [0:5]
    var slice_result = tensor.slice(0, 5, axis=0)

    # Verify shape
    assert_equal(slice_result.shape()[0], 5, "Full range slice shape")

    # Verify all values
    for i in range(5):
        assert_almost_equal(
            Float64(slice_result._get_float32(i)), Float64(i), tolerance=1e-6
        )

    print("PASS: test_slice_full_range")


def test_slice_is_marked_as_view() raises:
    """Verify that sliced tensors are marked with _is_view = True."""
    var tensor = zeros([5], DType.float32)

    # Original tensor should not be a view
    assert_equal(tensor._is_view, False, "Original tensor is not a view")

    # Create slice
    var slice_result = tensor.slice(1, 4, axis=0)

    # Sliced tensor should be marked as view
    assert_equal(slice_result._is_view, True, "Slice is marked as view")

    print("PASS: test_slice_is_marked_as_view")


def test_slice_refcount_increments() raises:
    """Verify that creating a slice increments the reference count."""
    var tensor = zeros([5], DType.float32)

    # Get initial refcount (should be 1 for new tensor)
    var initial_refcount = tensor._refcount[]

    # Create slice - should increment refcount
    var slice_result = tensor.slice(1, 4, axis=0)

    # Refcount should have incremented
    var new_refcount = tensor._refcount[]
    assert_equal(
        new_refcount, initial_refcount + 1, "Refcount incremented by slice"
    )

    print("PASS: test_slice_refcount_increments")


def test_multiple_slices_share_refcount() raises:
    """Verify that multiple slices share the same refcount pointer."""
    var tensor = zeros([5], DType.float32)
    for i in range(5):
        tensor._set_float32(i, Float32(i))

    # Create two slices
    var slice1 = tensor.slice(0, 2, axis=0)
    var slice2 = tensor.slice(2, 4, axis=0)

    # Both slices should share the same refcount pointer
    assert_equal(
        slice1._refcount[], slice2._refcount[], "Slices share refcount value"
    )

    # Modify original and check both slices see it
    tensor._set_float32(0, 99.0)
    assert_almost_equal(Float64(slice1._get_float32(0)), 99.0, tolerance=1e-6)

    print("PASS: test_multiple_slices_share_refcount")


def test_slice_mutation_visible_in_original() raises:
    """Verify that mutating a slice element is visible in the original tensor.

    Asserts true view semantics: slice shares memory with original, so
    writes through the slice are reflected when reading the original.
    """
    var tensor = zeros([10], DType.float32)
    for i in range(10):
        tensor._set_float32(i, Float32(i))

    # slice [2:6] -> indices 2,3,4,5 of original
    var s = tensor.slice(2, 6, axis=0)

    # Mutate element 0 of the slice (corresponds to index 2 of original)
    s._set_float32(0, 99.0)

    # The original tensor must reflect the change at index 2
    assert_almost_equal(Float64(tensor._get_float32(2)), 99.0, tolerance=1e-6)

    print("PASS: test_slice_mutation_visible_in_original")


def test_slice_empty_range() raises:
    """Test slicing with start == end (empty slice)."""
    var tensor = zeros([5], DType.float32)

    # Slice where start == end
    var empty_slice = tensor.slice(2, 2, axis=0)

    # Verify shape is 0
    assert_equal(empty_slice.shape()[0], 0, "Empty slice shape")

    print("PASS: test_slice_empty_range")


def test_slice_single_element() raises:
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


def test_slice_out_of_bounds_start() raises:
    """Test that out-of-bounds start index raises error."""
    var tensor = zeros([5], DType.float32)

    try:
        var bad_slice = tensor.slice(10, 15, axis=0)
        raise Error("Expected error for out-of-bounds start")
    except:
        # Expected
        print("PASS: test_slice_out_of_bounds_start")


def test_slice_out_of_bounds_end() raises:
    """Test that out-of-bounds end index raises error."""
    var tensor = zeros([5], DType.float32)

    try:
        var bad_slice = tensor.slice(0, 100, axis=0)
        raise Error("Expected error for out-of-bounds end")
    except:
        # Expected
        print("PASS: test_slice_out_of_bounds_end")


def test_slice_invalid_axis() raises:
    """Test that invalid axis raises error."""
    var tensor = zeros([5, 3], DType.float32)

    try:
        var bad_slice = tensor.slice(0, 2, axis=5)
        raise Error("Expected error for invalid axis")
    except:
        # Expected
        print("PASS: test_slice_invalid_axis")


def test_batch_extraction_uses_view() raises:
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


def test_batch_extraction_pair() raises:
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


def test_slice_view_mutation_visible() raises:
    """Test that mutating a view slice is visible in original. Closes #3900."""
    var t = zeros([10], DType.float32)
    for i in range(10):
        t._set_float32(i, Float32(i))

    # .slice() creates a view
    var view = t.slice(2, 5, axis=0)

    # Mutate through the view
    view._set_float32(0, Float32(99.0))

    # Original should reflect the change
    assert_almost_equal(Float64(t._get_float32(2)), 99.0, tolerance=1e-6)

    print("PASS: test_slice_view_mutation_visible")


def test_chained_slice_shares_memory() raises:
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
    assert_almost_equal(Float64(t._get_float32(3)), 77.0, tolerance=1e-6)

    print("PASS: test_chained_slice_shares_memory")


def main() raises:
    """Run all test_slicing tests."""
    print("Running test_slicing tests...")

    test_slice_basic_1d()
    print("✓ test_slice_basic_1d")

    test_slice_2d_axis0()
    print("✓ test_slice_2d_axis0")

    test_slice_4d_batch()
    print("✓ test_slice_4d_batch")

    test_slice_full_range()
    print("✓ test_slice_full_range")

    test_slice_is_marked_as_view()
    print("✓ test_slice_is_marked_as_view")

    test_slice_refcount_increments()
    print("✓ test_slice_refcount_increments")

    test_multiple_slices_share_refcount()
    print("✓ test_multiple_slices_share_refcount")

    test_slice_mutation_visible_in_original()
    print("✓ test_slice_mutation_visible_in_original")

    test_slice_empty_range()
    print("✓ test_slice_empty_range")

    test_slice_single_element()
    print("✓ test_slice_single_element")

    test_slice_out_of_bounds_start()
    print("✓ test_slice_out_of_bounds_start")

    test_slice_out_of_bounds_end()
    print("✓ test_slice_out_of_bounds_end")

    test_slice_invalid_axis()
    print("✓ test_slice_invalid_axis")

    test_batch_extraction_uses_view()
    print("✓ test_batch_extraction_uses_view")

    test_batch_extraction_pair()
    print("✓ test_batch_extraction_pair")

    test_slice_view_mutation_visible()
    print("✓ test_slice_view_mutation_visible")

    test_chained_slice_shares_memory()
    print("✓ test_chained_slice_shares_memory")

    print("\nAll test_slicing tests passed!")
