"""Unit tests for AnyTensor basic 1D and strided slicing operations (#3013).

Tests cover:
- Basic 1D slicing (tensor[start:end])
- Strided slicing (tensor[start:end:step])
- Negative indices

Following TDD principles - tests written before implementation.
"""


from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


fn test_slice_1d_basic() raises:
    """Test basic 1D slicing [start:end]."""
    # Create tensor [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [2:7] should give [2, 3, 4, 5, 6]
    var sliced = t[2:7]

    assert_equal(sliced.numel(), 5)
    assert_almost_equal(Float64(sliced[0]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 4.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[3]), 5.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[4]), 6.0, tolerance=1e-6)


fn test_slice_1d_from_start() raises:
    """Test slicing from start [:end]."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [:5] should give [0, 1, 2, 3, 4]
    var sliced = t[:5]

    assert_equal(sliced.numel(), 5)
    for i in range(5):
        assert_almost_equal(Float64(sliced[i]), Float64(i), tolerance=1e-6)


fn test_slice_1d_to_end() raises:
    """Test slicing to end [start:]."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [7:] should give [7, 8, 9]
    var sliced = t[7:]

    assert_equal(sliced.numel(), 3)
    assert_almost_equal(Float64(sliced[0]), 7.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 8.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 9.0, tolerance=1e-6)


fn test_slice_1d_full() raises:
    """Test full slice [:]."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)

    # Slice [:] should give entire tensor
    var sliced = t[:]

    assert_equal(sliced.numel(), 5)
    for i in range(5):
        assert_almost_equal(Float64(sliced[i]), Float64(i), tolerance=1e-6)


fn test_slice_1d_negative_indices() raises:
    """Test slicing with negative indices."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [-3:] should give [7, 8, 9]
    var sliced = t[-3:]

    assert_equal(sliced.numel(), 3)
    assert_almost_equal(Float64(sliced[0]), 7.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 8.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 9.0, tolerance=1e-6)


fn test_slice_1d_strided() raises:
    """Test strided slicing [start:end:step]."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [0:10:2] should give [0, 2, 4, 6, 8]
    var sliced = t[0:10:2]

    assert_equal(sliced.numel(), 5)
    assert_almost_equal(Float64(sliced[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 4.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[3]), 6.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[4]), 8.0, tolerance=1e-6)


fn test_slice_1d_strided_step3() raises:
    """Test strided slicing with step=3."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [0:10:3] should give [0, 3, 6, 9]
    var sliced = t[0:10:3]

    assert_equal(sliced.numel(), 4)
    assert_almost_equal(Float64(sliced[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 6.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[3]), 9.0, tolerance=1e-6)


fn test_slice_1d_reverse() raises:
    """Test reverse slicing with negative step [::-1]."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)

    # Slice [::-1] should give [4, 3, 2, 1, 0]
    var sliced = t[::-1]

    assert_equal(sliced.numel(), 5)
    assert_almost_equal(Float64(sliced[0]), 4.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[3]), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[4]), 0.0, tolerance=1e-6)


fn test_slice_2d_single_dim() raises:
    """Test slicing along single dimension in 2D tensor."""
    # Create 5x4 tensor with sequential values
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # Slice rows [1:4, :] should give 3x4 tensor
    var sliced = t2d[1:4, :]

    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 4)


fn test_slice_2d_both_dims() raises:
    """Test slicing along both dimensions in 2D tensor."""
    # Create 5x4 tensor
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # Slice [1:4, 1:3] should give 3x2 tensor
    var sliced = t2d[1:4, 1:3]

    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 2)


fn test_slice_3d_partial() raises:
    """Test slicing in 3D tensor."""
    # Create 4x3x2 tensor
    var t = arange(0.0, 24.0, 1.0, DType.float32)
    var t3d = t.reshape([4, 3, 2])

    # Slice [1:3, :, :] should give 2x3x2 tensor
    var sliced = t3d[1:3, :, :]

    var shape = sliced.shape()
    assert_equal(len(shape), 3)
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 2)


fn test_batch_extraction_basic() raises:
    """Test extracting a batch from dataset (critical for training)."""
    # Simulate dataset: 100 samples, each 3x32x32 (like CIFAR-10)
    var batch_size = 16
    var num_samples = 100

    # Create mock dataset [100, 3, 32, 32]
    var data = zeros([num_samples, 3, 32, 32], DType.float32)

    # Extract first batch [0:16, :, :, :]
    var batch = data[0:batch_size, :, :, :]

    var shape = batch.shape()
    assert_equal(len(shape), 4)
    assert_equal(shape[0], batch_size)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 32)
    assert_equal(shape[3], 32)


fn test_batch_extraction_offset() raises:
    """Test extracting batch at offset (second batch)."""
    var batch_size = 16
    var num_samples = 100

    var data = zeros([num_samples, 3, 32, 32], DType.float32)

    # Extract second batch [16:32, :, :, :]
    var batch = data[batch_size : 2 * batch_size, :, :, :]

    var shape = batch.shape()
    assert_equal(shape[0], batch_size)


fn test_batch_extraction_last_partial() raises:
    """Test extracting last partial batch."""
    var batch_size = 16
    var num_samples = 50  # Not evenly divisible

    var data = zeros([num_samples, 3, 32, 32], DType.float32)

    # Extract last batch [48:50, :, :, :] (only 2 samples)
    var last_start = (num_samples // batch_size) * batch_size
    var batch = data[last_start:num_samples, :, :, :]

    var shape = batch.shape()
    assert_equal(shape[0], 2)  # Only 2 samples in last batch


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


fn test_slice_empty_negative_step() raises:
    """Test empty-result slices with negative step.

    Edge cases where negative-step slicing yields 0 elements:
    - [1:3:-1]: start < end with negative step -> empty
    - [2:5:-1]: start < end with negative step -> empty

    These verify the max(0, ...) logic correctly handles invalid ranges
    with negative steps.
    """
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Test case 1: [1:3:-1] should give empty result
    var sliced1 = t[1:3:-1]
    assert_equal(sliced1.numel(), 0)

    # Test case 2: [2:5:-1] should give empty result
    var sliced2 = t[2:5:-1]
    assert_equal(sliced2.numel(), 0)


fn test_slice_out_of_bounds_clamped() raises:
    """Test slice with out-of-bounds indices (should be clamped)."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [8:20] should be clamped to [8:10]
    var sliced = t[8:20]

    assert_equal(sliced.numel(), 2)
    assert_almost_equal(Float64(sliced[0]), 8.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 9.0, tolerance=1e-6)


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

    # Slice rows [1:4] (rows 1,2,3) using multi-dimensional slice syntax
    var sliced = t2d[1:4, :]

    # Row 1 of original = [4,5,6,7], should be row 0 of slice
    assert_almost_equal(Float64(sliced._get_float32(0)), 4.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced._get_float32(1)), 5.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced._get_float32(2)), 6.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced._get_float32(3)), 7.0, tolerance=1e-6)

    # Row 2 of original = [8,9,10,11], should be row 1 of slice
    assert_almost_equal(Float64(sliced._get_float32(4)), 8.0, tolerance=1e-6)

    print("PASS: test_slice_2d_value_correctness")


fn test_negative_step_empty_result() raises:
    """Test negative step with invalid range produces empty result. Closes #3699.
    """
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
    """Run all test_extensor_slicing tests."""
    print("Running test_extensor_slicing tests...")

    test_slice_1d_basic()
    print("✓ test_slice_1d_basic")

    test_slice_1d_from_start()
    print("✓ test_slice_1d_from_start")

    test_slice_1d_to_end()
    print("✓ test_slice_1d_to_end")

    test_slice_1d_full()
    print("✓ test_slice_1d_full")

    test_slice_1d_negative_indices()
    print("✓ test_slice_1d_negative_indices")

    test_slice_1d_strided()
    print("✓ test_slice_1d_strided")

    test_slice_1d_strided_step3()
    print("✓ test_slice_1d_strided_step3")

    test_slice_1d_reverse()
    print("✓ test_slice_1d_reverse")

    test_slice_2d_single_dim()
    print("✓ test_slice_2d_single_dim")

    test_slice_2d_both_dims()
    print("✓ test_slice_2d_both_dims")

    test_slice_3d_partial()
    print("✓ test_slice_3d_partial")

    test_batch_extraction_basic()
    print("✓ test_batch_extraction_basic")

    test_batch_extraction_offset()
    print("✓ test_batch_extraction_offset")

    test_batch_extraction_last_partial()
    print("✓ test_batch_extraction_last_partial")

    test_slice_empty()
    print("✓ test_slice_empty")

    test_slice_single_element()
    print("✓ test_slice_single_element")

    test_slice_empty_negative_step()
    print("✓ test_slice_empty_negative_step")

    test_slice_out_of_bounds_clamped()
    print("✓ test_slice_out_of_bounds_clamped")

    test_slice_creates_copy()
    print("✓ test_slice_creates_copy")

    test_slice_modification_doesnt_affect_original()
    print("✓ test_slice_modification_doesnt_affect_original")

    test_slice_2d_value_correctness()
    print("✓ test_slice_2d_value_correctness")

    test_negative_step_empty_result()
    print("✓ test_negative_step_empty_result")

    test_slice_step_value_correctness()
    print("✓ test_slice_step_value_correctness")

    test_slice_method_returns_zero_copy_view()
    print("✓ test_slice_method_returns_zero_copy_view")

    test_getitem_slice_is_copy_not_view()
    print("✓ test_getitem_slice_is_copy_not_view")

    print("\nAll test_extensor_slicing tests passed!")
