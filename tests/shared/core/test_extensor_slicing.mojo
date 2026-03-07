"""Unit tests for ExTensor slicing and indexing operations (#3013).

Tests cover:
- Basic 1D slicing (tensor[start:end])
- Strided slicing (tensor[start:end:step])
- Multi-dimensional slicing (tensor[a:b, c:d])
- Negative indices and steps
- Edge cases (empty slices, single elements)
- Batch extraction for training loops

Following TDD principles - tests written before implementation.
"""

from shared.core.extensor import ExTensor, zeros, ones, full, arange
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


# ============================================================================
# Basic 1D Slicing Tests
# ============================================================================


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


# ============================================================================
# Strided Slicing Tests
# ============================================================================


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


# ============================================================================
# Multi-dimensional Slicing Tests
# ============================================================================


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


# ============================================================================
# Batch Extraction Tests (Critical for Training Loops)
# ============================================================================


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
    """Run all tests."""
    # Basic 1D slicing
    print("Testing basic 1D slicing...")
    test_slice_1d_basic()
    test_slice_1d_from_start()
    test_slice_1d_to_end()
    test_slice_1d_full()
    test_slice_1d_negative_indices()
    print("Basic 1D slicing: PASSED")

    # Strided slicing
    print("Testing strided slicing...")
    test_slice_1d_strided()
    test_slice_1d_strided_step3()
    test_slice_1d_reverse()
    print("Strided slicing: PASSED")

    # Multi-dimensional slicing
    print("Testing multi-dimensional slicing...")
    test_slice_2d_single_dim()
    test_slice_2d_both_dims()
    test_slice_3d_partial()
    print("Multi-dimensional slicing: PASSED")

    # Batch extraction (critical path)
    print("Testing batch extraction...")
    test_batch_extraction_basic()
    test_batch_extraction_offset()
    test_batch_extraction_last_partial()
    print("Batch extraction: PASSED")

    # Edge cases
    print("Testing edge cases...")
    test_slice_empty()
    test_slice_single_element()
    test_slice_out_of_bounds_clamped()
    print("Edge cases: PASSED")

    # Copy semantics (current implementation)
    print("Testing copy semantics...")
    test_slice_creates_copy()
    test_slice_modification_doesnt_affect_original()
    print("Copy semantics: PASSED")

    print("\nAll tests PASSED!")
