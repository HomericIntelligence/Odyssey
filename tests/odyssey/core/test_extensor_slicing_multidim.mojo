"""Tests for AnyTensor multi-dimensional slicing.

Covers:
- __getitem__(*slices: Slice) for N-D tensors (one Slice per dimension)
- slice(start, end, axis) for axis-0 extraction on N-D tensors

Note: __getitem__(Slice) (single Slice) is 1D-only by design. Multi-dim
tensors must use __getitem__(*slices) with one Slice per dimension, or
the slice(start, end, axis) method. The *slices overload does not support
step; only start:end per dimension.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, ones, full, arange
from tests.odyssey.conftest import (
    assert_true,
    assert_almost_equal,
    assert_equal,
)


def test_multislice_2d_axis0_basic() raises:
    """Test t[1:3, :] on a 2D tensor slices along axis 0."""
    # 4x5 tensor with sequential values 0..19
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([4, 5])

    var sliced = t2d[1:3, :]

    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 5)

    # sliced[0,:] = row 1 of original = [5,6,7,8,9]
    # sliced[1,:] = row 2 of original = [10,11,12,13,14]
    assert_almost_equal(Float64(sliced[0]), 5.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[4]), 9.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[5]), 10.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[9]), 14.0, tolerance=1e-6)


def test_multislice_2d_axis0_full() raises:
    """Test t[:, :] on a 2D tensor returns a full copy."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var t2d = t.reshape([3, 4])

    var sliced = t2d[:, :]

    var shape = sliced.shape()
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 4)
    assert_equal(sliced.numel(), 12)

    # All values should match original
    for i in range(12):
        assert_almost_equal(Float64(sliced[i]), Float64(i), tolerance=1e-6)


def test_multislice_3d_axis0_basic() raises:
    """Test t[1:3, :, :] on a 3D tensor slices along axis 0."""
    # 4x3x2 tensor with sequential values 0..23
    var t = arange(0.0, 24.0, 1.0, DType.float32)
    var t3d = t.reshape([4, 3, 2])

    var sliced = t3d[1:3, :, :]

    var shape = sliced.shape()
    assert_equal(len(shape), 3)
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 2)

    # sliced[0,:,:] = original[1,:,:] = elements 6..11
    assert_almost_equal(Float64(sliced[0]), 6.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[5]), 11.0, tolerance=1e-6)
    # sliced[1,:,:] = original[2,:,:] = elements 12..17
    assert_almost_equal(Float64(sliced[6]), 12.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[11]), 17.0, tolerance=1e-6)


def test_multislice_2d_is_copy_not_view() raises:
    """Test that __getitem__(*slices) result is a copy -- mutating does not affect original.
    """
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([4, 5])

    var sliced = t2d[1:3, :]

    # Mutate sliced
    sliced[0] = 999.0

    # Original must be unchanged at the corresponding flat index (row 1, col 0 = index 5)
    assert_almost_equal(Float64(t2d[5]), 5.0, tolerance=1e-6)


def test_multislice_2d_negative_start() raises:
    """Test t[-2:, :] on a 2D tensor selects the last two rows."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([4, 5])

    var sliced = t2d[-2:, :]

    var shape = sliced.shape()
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 5)

    # sliced[0,:] = row 2 of original = [10,11,12,13,14]
    assert_almost_equal(Float64(sliced[0]), 10.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[4]), 14.0, tolerance=1e-6)
    # sliced[1,:] = row 3 of original = [15,16,17,18,19]
    assert_almost_equal(Float64(sliced[5]), 15.0, tolerance=1e-6)


def test_multislice_2d_empty_result() raises:
    """Test t[3:1, :] on a 2D tensor returns an empty tensor with shape (0, cols).
    """
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([4, 5])

    var sliced = t2d[3:1, :]

    var shape = sliced.shape()
    assert_equal(shape[0], 0)
    assert_equal(shape[1], 5)
    assert_equal(sliced.numel(), 0)


def test_slice_method_2d_axis0() raises:
    """Test slice(start, end, axis=0) on a 2D tensor for axis-0 extraction."""
    # 6x3 tensor with sequential values 0..17
    var t = arange(0.0, 18.0, 1.0, DType.float32)
    var t2d = t.reshape([6, 3])

    # Extract rows 1..3 (view)
    var sliced = t2d.slice(1, 4, axis=0)

    var shape = sliced.shape()
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 3)

    # Row 0 of sliced = row 1 of original = [3,4,5]
    assert_almost_equal(Float64(sliced[0]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 5.0, tolerance=1e-6)
    # Row 1 of sliced = row 2 of original = [6,7,8]
    assert_almost_equal(Float64(sliced[3]), 6.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[5]), 8.0, tolerance=1e-6)
    # Row 2 of sliced = row 3 of original = [9,10,11]
    assert_almost_equal(Float64(sliced[6]), 9.0, tolerance=1e-6)


def test_multislice_2d_inner_dim() raises:
    """Test t[:, 1:3] on a 2D tensor slices along axis 1 (columns)."""
    # 3x5 tensor with sequential values 0..14
    var t = arange(0.0, 15.0, 1.0, DType.float32)
    var t2d = t.reshape([3, 5])

    var sliced = t2d[:, 1:3]

    var shape = sliced.shape()
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 2)

    # Row 0, cols 1-2 = [1, 2]
    assert_almost_equal(Float64(sliced[0]), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 2.0, tolerance=1e-6)
    # Row 1, cols 1-2 = [6, 7]
    assert_almost_equal(Float64(sliced[2]), 6.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[3]), 7.0, tolerance=1e-6)
    # Row 2, cols 1-2 = [11, 12]
    assert_almost_equal(Float64(sliced[4]), 11.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[5]), 12.0, tolerance=1e-6)


def test_slice_1d_regression() raises:
    """Regression: __getitem__(Slice) on 1D tensors still works correctly."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    var fwd = t[2:7]
    assert_equal(fwd.numel(), 5)
    assert_almost_equal(Float64(fwd[0]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(fwd[4]), 6.0, tolerance=1e-6)

    var rev = t[::-1]
    assert_equal(rev.numel(), 10)
    assert_almost_equal(Float64(rev[0]), 9.0, tolerance=1e-6)
    assert_almost_equal(Float64(rev[9]), 0.0, tolerance=1e-6)

    var strided = t[0:10:2]
    assert_equal(strided.numel(), 5)
    assert_almost_equal(Float64(strided[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(strided[4]), 8.0, tolerance=1e-6)


def main() raises:
    """Run all multi-dimensional slicing tests."""
    test_multislice_2d_axis0_basic()
    test_multislice_2d_axis0_full()
    test_multislice_3d_axis0_basic()
    test_multislice_2d_is_copy_not_view()
    test_multislice_2d_negative_start()
    test_multislice_2d_empty_result()
    test_slice_method_2d_axis0()
    test_multislice_2d_inner_dim()
    test_slice_1d_regression()
    print("All multi-dimensional slicing tests passed!")
