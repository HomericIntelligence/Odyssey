# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_slicing.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tensor slicing tests part 1: basic functionality, view semantics, reference counting.

Test Categories:
1. Basic Functionality (4 tests)
2. View Semantics (1 test)
3. Reference Counting (2 tests)
4. Edge Cases (1 test)

Total: 8 tests.
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones
from tests.shared.conftest import assert_equal, assert_almost_equal


fn test_slice_basic_1d() raises:
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


fn test_slice_2d_axis0() raises:
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


fn test_slice_4d_batch() raises:
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


fn test_slice_full_range() raises:
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


fn test_slice_is_marked_as_view() raises:
    """Verify that sliced tensors are marked with _is_view = True."""
    var tensor = zeros([5], DType.float32)

    # Original tensor should not be a view
    assert_equal(tensor._is_view, False, "Original tensor is not a view")

    # Create slice
    var slice_result = tensor.slice(1, 4, axis=0)

    # Sliced tensor should be marked as view
    assert_equal(slice_result._is_view, True, "Slice is marked as view")

    print("PASS: test_slice_is_marked_as_view")


fn test_slice_refcount_increments() raises:
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


fn test_multiple_slices_share_refcount() raises:
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


fn test_slice_mutation_visible_in_original() raises:
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


fn test_slice_empty_range() raises:
    """Test slicing with start == end (empty slice)."""
    var tensor = zeros([5], DType.float32)

    # Slice where start == end
    var empty_slice = tensor.slice(2, 2, axis=0)

    # Verify shape is 0
    assert_equal(empty_slice.shape()[0], 0, "Empty slice shape")

    print("PASS: test_slice_empty_range")


fn main() raises:
    """Run slicing tests part 1."""
    print("Running tensor slicing tests part 1...")

    # Basic functionality tests
    test_slice_basic_1d()
    test_slice_2d_axis0()
    test_slice_4d_batch()
    test_slice_full_range()

    # View semantics tests
    test_slice_is_marked_as_view()
    test_slice_refcount_increments()
    test_multiple_slices_share_refcount()
    test_slice_mutation_visible_in_original()

    # Edge case tests
    test_slice_empty_range()

    print("\nAll tests passed!")
