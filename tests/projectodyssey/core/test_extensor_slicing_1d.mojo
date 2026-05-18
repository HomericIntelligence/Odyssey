# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""Tests for AnyTensor 1D slicing operations (basic and strided)."""

from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from tests.projectodyssey.conftest import assert_true, assert_almost_equal, assert_equal


# ============================================================================
# Basic 1D Slicing Tests
# ============================================================================


def test_slice_1d_basic() raises:
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


def test_slice_1d_from_start() raises:
    """Test slicing from start [:end]."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [:5] should give [0, 1, 2, 3, 4]
    var sliced = t[:5]

    assert_equal(sliced.numel(), 5)
    for i in range(5):
        assert_almost_equal(Float64(sliced[i]), Float64(i), tolerance=1e-6)


def test_slice_1d_to_end() raises:
    """Test slicing to end [start:]."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [7:] should give [7, 8, 9]
    var sliced = t[7:]

    assert_equal(sliced.numel(), 3)
    assert_almost_equal(Float64(sliced[0]), 7.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 8.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 9.0, tolerance=1e-6)


def test_slice_1d_full() raises:
    """Test full slice [:]."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)

    # Slice [:] should give entire tensor
    var sliced = t[:]

    assert_equal(sliced.numel(), 5)
    for i in range(5):
        assert_almost_equal(Float64(sliced[i]), Float64(i), tolerance=1e-6)


def test_slice_1d_negative_indices() raises:
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


def test_slice_1d_strided() raises:
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


def test_slice_1d_strided_step3() raises:
    """Test strided slicing with step=3."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    # Slice [0:10:3] should give [0, 3, 6, 9]
    var sliced = t[0:10:3]

    assert_equal(sliced.numel(), 4)
    assert_almost_equal(Float64(sliced[0]), 0.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[1]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[2]), 6.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced[3]), 9.0, tolerance=1e-6)


def test_slice_1d_reverse() raises:
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


def test_slice_1d_reverse_parametric() raises:
    """Test reverse slicing with explicit bounds and steps.

    Parametric test covering multiple reverse slicing patterns:
    - [3:1:-1] -> [3, 2]: explicit start/end with step=-1
    - [4::-1] -> [4, 3, 2, 1, 0]: end explicit, start default with step=-1
    - [::-2] -> [4, 2, 0]: full reverse with step=-2
    """
    var t = arange(0.0, 5.0, 1.0, DType.float32)

    # Test case 1: [3:1:-1] should give [3, 2]
    var sliced1 = t[3:1:-1]
    assert_equal(sliced1.numel(), 2)
    assert_almost_equal(Float64(sliced1[0]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced1[1]), 2.0, tolerance=1e-6)

    # Test case 2: [4::-1] should give [4, 3, 2, 1, 0]
    var sliced2 = t[4::-1]
    assert_equal(sliced2.numel(), 5)
    assert_almost_equal(Float64(sliced2[0]), 4.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced2[1]), 3.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced2[2]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced2[3]), 1.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced2[4]), 0.0, tolerance=1e-6)

    # Test case 3: [::-2] should give [4, 2, 0]
    var sliced3 = t[::-2]
    assert_equal(sliced3.numel(), 3)
    assert_almost_equal(Float64(sliced3[0]), 4.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced3[1]), 2.0, tolerance=1e-6)
    assert_almost_equal(Float64(sliced3[2]), 0.0, tolerance=1e-6)


def main() raises:
    """Run all 1D slicing tests."""
    test_slice_1d_basic()
    test_slice_1d_from_start()
    test_slice_1d_to_end()
    test_slice_1d_full()
    test_slice_1d_negative_indices()
    test_slice_1d_strided()
    test_slice_1d_strided_step3()
    test_slice_1d_reverse()
    test_slice_1d_reverse_parametric()
    print("All 1D slicing tests passed!")
