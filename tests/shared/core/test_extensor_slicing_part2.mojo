# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_anytensor_slicing.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for AnyTensor multi-dimensional slicing and batch extraction (#3013).

Tests cover:
- Multi-dimensional slicing (tensor[a:b, c:d])
- Batch extraction for training loops
- Edge cases (empty slices, single elements)

Following TDD principles - tests written before implementation.
"""

from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


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


fn main() raises:
    """Run all tests."""
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
    test_slice_empty_negative_step()
    print("Edge cases: PASSED")

    print("\nAll tests PASSED!")
