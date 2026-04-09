# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_anytensor_slicing.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor multi-dimensional slicing and batch extraction."""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


# ============================================================================
# Multi-dimensional Slicing Tests
# ============================================================================


def test_slice_2d_single_dim() raises:
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

    # Verify values: sliced should start at row 1, elements 4-7
    # Original 5x4 tensor (row-major): [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19]
    # sliced[0,:] = original[1,:] = [4,5,6,7]
    var data_ptr = sliced._data.bitcast[Float32]()
    assert_almost_equal(Float64(data_ptr[0]), Float64(4.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[1]), Float64(5.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[2]), Float64(6.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[3]), Float64(7.0), Float64(1e-5))
    # sliced[1,:] = original[2,:] = [8,9,10,11]
    assert_almost_equal(Float64(data_ptr[4]), Float64(8.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[5]), Float64(9.0), Float64(1e-5))


def test_slice_2d_both_dims() raises:
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

    # Verify values: sliced[i,j] = original[i+1, j+1]
    # Original 5x4 tensor: [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19]
    # sliced[0,:] = original[1,1:3] = [5,6]
    # sliced[1,:] = original[2,1:3] = [9,10]
    # sliced[2,:] = original[3,1:3] = [13,14]
    var data_ptr = sliced._data.bitcast[Float32]()
    assert_almost_equal(Float64(data_ptr[0]), Float64(5.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[1]), Float64(6.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[2]), Float64(9.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[3]), Float64(10.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[4]), Float64(13.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[5]), Float64(14.0), Float64(1e-5))


def test_slice_3d_partial() raises:
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

    # Verify values: sliced[i,:,:] = original[i+1,:,:]
    # Original 4x3x2 tensor (row-major): [0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15, 16,17, 18,19, 20,21, 22,23]
    # sliced[0,:,:] = original[1,:,:] starts at index 6: [6,7, 8,9, 10,11]
    # sliced[1,:,:] = original[2,:,:] starts at index 12: [12,13, 14,15, 16,17]
    var data_ptr = sliced._data.bitcast[Float32]()
    assert_almost_equal(Float64(data_ptr[0]), Float64(6.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[1]), Float64(7.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[2]), Float64(8.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[3]), Float64(9.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[6]), Float64(12.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[7]), Float64(13.0), Float64(1e-5))


# ============================================================================
# Batch Extraction Tests (Critical for Training Loops)
# ============================================================================


def test_batch_extraction_basic() raises:
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


def test_batch_extraction_offset() raises:
    """Test extracting batch at offset (second batch)."""
    var batch_size = 16
    var num_samples = 100

    var data = zeros([num_samples, 3, 32, 32], DType.float32)

    # Extract second batch [16:32, :, :, :]
    var batch = data[batch_size : 2 * batch_size, :, :, :]

    var shape = batch.shape()
    assert_equal(shape[0], batch_size)


def test_batch_extraction_last_partial() raises:
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
# Negative Index Tests
# ============================================================================


def test_slice_2d_negative_start() raises:
    """Test slicing with negative start index in 2D tensor."""
    # Create 5x4 tensor with sequential values
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # Slice [-2:, :] should give last 2 rows (rows 3-4)
    var sliced = t2d[-2:, :]

    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 4)

    # Verify values: sliced should be rows [3:5,:] of original
    # Original 5x4 tensor: [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19]
    # sliced[0,:] = original[3,:] = [12,13,14,15]
    var data_ptr = sliced._data.bitcast[Float32]()
    assert_almost_equal(Float64(data_ptr[0]), Float64(12.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[1]), Float64(13.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[2]), Float64(14.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[3]), Float64(15.0), Float64(1e-5))


def test_slice_2d_negative_end() raises:
    """Test slicing with negative end index in 2D tensor."""
    # Create 5x4 tensor with sequential values
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # Slice [:, -2:] should give last 2 columns
    var sliced = t2d[:, -2:]

    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 5)
    assert_equal(shape[1], 2)

    # Verify values: sliced should be columns [2:4,:] of original
    # sliced[0,:] = original[0,2:4] = [2,3]
    var data_ptr = sliced._data.bitcast[Float32]()
    assert_almost_equal(Float64(data_ptr[0]), Float64(2.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[1]), Float64(3.0), Float64(1e-5))


def test_slice_2d_negative_both_dims() raises:
    """Test slicing with negative indices in both dimensions."""
    # Create 5x4 tensor with sequential values
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # Slice [-2:, -2:] should give last 2 rows and last 2 columns
    var sliced = t2d[-2:, -2:]

    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 2)

    # Verify values: sliced should be rows [3:5,2:4] of original
    # Original 5x4 tensor: [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19]
    # sliced[0,:] = original[3,2:4] = [14,15]
    # sliced[1,:] = original[4,2:4] = [18,19]
    var data_ptr = sliced._data.bitcast[Float32]()
    assert_almost_equal(Float64(data_ptr[0]), Float64(14.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[1]), Float64(15.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[2]), Float64(18.0), Float64(1e-5))
    assert_almost_equal(Float64(data_ptr[3]), Float64(19.0), Float64(1e-5))


def main() raises:
    """Run all multi-dimensional slicing and batch extraction tests."""
    test_slice_2d_single_dim()
    test_slice_2d_both_dims()
    test_slice_3d_partial()
    test_batch_extraction_basic()
    test_batch_extraction_offset()
    test_batch_extraction_last_partial()
    test_slice_2d_negative_start()
    test_slice_2d_negative_end()
    test_slice_2d_negative_both_dims()
    print("All multi-dimensional slicing and batch extraction tests passed!")
