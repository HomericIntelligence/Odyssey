"""Tests for reduction operations on non-contiguous tensors.

Verifies that sum() and mean() produce correct results when given
non-contiguous inputs. Without the as_contiguous() guard these operations
read from wrong memory positions, silently producing wrong values.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

Follow-up from #3236.
"""


from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_false,
    assert_true,
)
from shared.tensor.any_tensor import (
    AnyTensor,
    arange,
    full,
    ones,
    zeros,
)
from shared.core.reduction import (
    max_reduce,
    mean,
    min_reduce,
    sum,
)
from shared.core.matrix import transpose_view


def _make_nc_2x3() raises -> AnyTensor:
    """Non-contiguous 3×2 tensor (transpose of 2×3 sequential).

    Logical layout (3×2 row-major): 0,3,1,4,2,5
    Sum of all logical elements = 0+3+1+4+2+5 = 15
    Mean = 15/6 = 2.5
    """
    var base = arange(0.0, 6.0, 1.0, DType.float32)
    var shaped = base.reshape([2, 3])
    var nc = transpose_view(shaped)
    assert_false(nc.is_contiguous(), "fixture must be non-contiguous")
    return nc^


def test_sum_all_noncontiguous() raises:
    """Reduction sum() over all elements of a non-contiguous tensor should be correct.
    """
    var nc = _make_nc_2x3()  # logical values: 0,3,1,4,2,5; sum=15

    var result = sum(nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(15.0), tolerance=1e-4)


def test_sum_axis0_noncontiguous() raises:
    """Reduction sum(axis=0) on non-contiguous 3×2 should sum along rows correctly.
    """
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # sum axis 0: [0+1+2, 3+4+5] = [3, 12]
    var nc = _make_nc_2x3()

    var result = sum(nc, axis=0)

    assert_equal_int(result.shape()[0], 2)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(3.0), tolerance=1e-4)
    assert_almost_equal(ptr[1], Float32(12.0), tolerance=1e-4)


def test_sum_axis1_noncontiguous() raises:
    """Reduction sum(axis=1) on non-contiguous 3×2 should sum along columns correctly.
    """
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # sum axis 1: [0+3, 1+4, 2+5] = [3, 5, 7]
    var nc = _make_nc_2x3()

    var result = sum(nc, axis=1)

    assert_equal_int(result.shape()[0], 3)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(3.0), tolerance=1e-4)
    assert_almost_equal(ptr[1], Float32(5.0), tolerance=1e-4)
    assert_almost_equal(ptr[2], Float32(7.0), tolerance=1e-4)


def test_mean_all_noncontiguous() raises:
    """Reduction mean() over all elements of a non-contiguous tensor should be correct.
    """
    var nc = _make_nc_2x3()  # logical values: 0,3,1,4,2,5; mean=2.5

    var result = mean(nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(2.5), tolerance=1e-4)


def test_mean_axis0_noncontiguous() raises:
    """Reduction mean(axis=0) on non-contiguous 3×2 should average along rows correctly.
    """
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # mean axis 0: [0+1+2/3, 3+4+5/3] = [1.0, 4.0]
    var nc = _make_nc_2x3()

    var result = mean(nc, axis=0)

    assert_equal_int(result.shape()[0], 2)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(1.0), tolerance=1e-4)
    assert_almost_equal(ptr[1], Float32(4.0), tolerance=1e-4)


def test_mean_axis1_noncontiguous() raises:
    """Reduction mean(axis=1) on non-contiguous 3×2 should average along columns correctly.
    """
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # mean axis 1: [0+3/2, 1+4/2, 2+5/2] = [1.5, 2.5, 3.5]
    var nc = _make_nc_2x3()

    var result = mean(nc, axis=1)

    assert_equal_int(result.shape()[0], 3)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(1.5), tolerance=1e-4)
    assert_almost_equal(ptr[1], Float32(2.5), tolerance=1e-4)
    assert_almost_equal(ptr[2], Float32(3.5), tolerance=1e-4)


def test_sum_noncontiguous_matches_contiguous_baseline() raises:
    """Non-contiguous sum must match contiguous baseline."""
    # Build contiguous tensor with same logical values as nc
    var logical = zeros([3, 2], DType.float32)
    var lp = logical._data.bitcast[Float32]()
    lp[0] = 0.0
    lp[1] = 3.0
    lp[2] = 1.0
    lp[3] = 4.0
    lp[4] = 2.0
    lp[5] = 5.0

    var baseline = sum(logical, axis=0)
    var nc = _make_nc_2x3()
    var result = sum(nc, axis=0)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(2):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-4)


def test_mean_noncontiguous_matches_contiguous_baseline() raises:
    """Non-contiguous mean must match contiguous baseline."""
    var logical = zeros([3, 2], DType.float32)
    var lp = logical._data.bitcast[Float32]()
    lp[0] = 0.0
    lp[1] = 3.0
    lp[2] = 1.0
    lp[3] = 4.0
    lp[4] = 2.0
    lp[5] = 5.0

    var baseline = mean(logical)
    var nc = _make_nc_2x3()
    var result = mean(nc)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    assert_almost_equal(rp[0], bp[0], tolerance=1e-4)


def test_sum_keepdims_noncontiguous() raises:
    """Reduction sum(keepdims=True) on non-contiguous tensor should work correctly.
    """
    var nc = _make_nc_2x3()  # shape (3,2), sum=15

    var result = sum(nc, axis=-1, keepdims=True)

    # keepdims=True with axis=-1 should give shape [1, 1] scalar-like
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(15.0), tolerance=1e-4)


def test_max_all_noncontiguous() raises:
    """Max_reduce() over all elements of a non-contiguous tensor should be correct.
    """
    var nc = _make_nc_2x3()  # logical values: 0,3,1,4,2,5; max=5

    var result = max_reduce(nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(5.0), tolerance=1e-5)


def test_max_axis0_noncontiguous() raises:
    """Max_reduce(axis=0) on non-contiguous 3×2 should find column maxima."""
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # max axis 0: [max(0,1,2), max(3,4,5)] = [2, 5]
    var nc = _make_nc_2x3()

    var result = max_reduce(nc, axis=0)

    assert_equal_int(result.shape()[0], 2)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(2.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(5.0), tolerance=1e-5)


def test_max_axis1_noncontiguous() raises:
    """Max_reduce(axis=1) on non-contiguous 3×2 should find row maxima."""
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # max axis 1: [max(0,3), max(1,4), max(2,5)] = [3, 4, 5]
    var nc = _make_nc_2x3()

    var result = max_reduce(nc, axis=1)

    assert_equal_int(result.shape()[0], 3)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(3.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(4.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(5.0), tolerance=1e-5)


def test_min_all_noncontiguous() raises:
    """Min_reduce() over all elements of a non-contiguous tensor should be correct.
    """
    var nc = _make_nc_2x3()  # logical values: 0,3,1,4,2,5; min=0

    var result = min_reduce(nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)


def test_min_axis0_noncontiguous() raises:
    """Min_reduce(axis=0) on non-contiguous 3×2 should find column minima."""
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # min axis 0: [min(0,1,2), min(3,4,5)] = [0, 3]
    var nc = _make_nc_2x3()

    var result = min_reduce(nc, axis=0)

    assert_equal_int(result.shape()[0], 2)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(3.0), tolerance=1e-5)


def test_min_axis1_noncontiguous() raises:
    """Min_reduce(axis=1) on non-contiguous 3×2 should find row minima."""
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # min axis 1: [min(0,3), min(1,4), min(2,5)] = [0, 1, 2]
    var nc = _make_nc_2x3()

    var result = min_reduce(nc, axis=1)

    assert_equal_int(result.shape()[0], 3)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(0.0), tolerance=1e-5)
    assert_almost_equal(ptr[1], Float32(1.0), tolerance=1e-5)
    assert_almost_equal(ptr[2], Float32(2.0), tolerance=1e-5)


def test_max_noncontiguous_matches_contiguous_baseline() raises:
    """Non-contiguous max_reduce must match contiguous baseline."""
    var logical = zeros([3, 2], DType.float32)
    var lp = logical._data.bitcast[Float32]()
    lp[0] = 0.0
    lp[1] = 3.0
    lp[2] = 1.0
    lp[3] = 4.0
    lp[4] = 2.0
    lp[5] = 5.0

    var baseline = max_reduce(logical, axis=1)
    var nc = _make_nc_2x3()
    var result = max_reduce(nc, axis=1)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(3):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-5)


def test_min_noncontiguous_matches_contiguous_baseline() raises:
    """Non-contiguous min_reduce must match contiguous baseline."""
    var logical = zeros([3, 2], DType.float32)
    var lp = logical._data.bitcast[Float32]()
    lp[0] = 0.0
    lp[1] = 3.0
    lp[2] = 1.0
    lp[3] = 4.0
    lp[4] = 2.0
    lp[5] = 5.0

    var baseline = min_reduce(logical, axis=0)
    var nc = _make_nc_2x3()
    var result = min_reduce(nc, axis=0)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(2):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-5)


def test_max_larger_noncontiguous() raises:
    """Max_reduce on a larger non-contiguous tensor should be correct."""
    # Create a 3×4 contiguous tensor and transpose to (4,3) non-contiguous
    var base = arange(0.0, 12.0, 1.0, DType.float32)
    var shaped = base.reshape([3, 4])
    var nc = transpose_view(shaped)  # shape (4,3), non-contiguous
    assert_false(nc.is_contiguous(), "fixture must be non-contiguous")

    var result = max_reduce(nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(11.0), tolerance=1e-5)


def test_min_larger_noncontiguous() raises:
    """Min_reduce on a larger non-contiguous tensor should be correct."""
    var base = arange(1.0, 13.0, 1.0, DType.float32)
    var shaped = base.reshape([3, 4])
    var nc = transpose_view(shaped)  # shape (4,3), non-contiguous
    assert_false(nc.is_contiguous(), "fixture must be non-contiguous")

    var result = min_reduce(nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(1.0), tolerance=1e-5)


def main() raises:
    """Run all test_reduction_noncontiguous tests."""
    print("Running test_reduction_noncontiguous tests...")

    test_sum_all_noncontiguous()
    print("✓ test_sum_all_noncontiguous")

    test_sum_axis0_noncontiguous()
    print("✓ test_sum_axis0_noncontiguous")

    test_sum_axis1_noncontiguous()
    print("✓ test_sum_axis1_noncontiguous")

    test_mean_all_noncontiguous()
    print("✓ test_mean_all_noncontiguous")

    test_mean_axis0_noncontiguous()
    print("✓ test_mean_axis0_noncontiguous")

    test_mean_axis1_noncontiguous()
    print("✓ test_mean_axis1_noncontiguous")

    test_sum_noncontiguous_matches_contiguous_baseline()
    print("✓ test_sum_noncontiguous_matches_contiguous_baseline")

    test_mean_noncontiguous_matches_contiguous_baseline()
    print("✓ test_mean_noncontiguous_matches_contiguous_baseline")

    test_sum_keepdims_noncontiguous()
    print("✓ test_sum_keepdims_noncontiguous")

    test_max_all_noncontiguous()
    print("✓ test_max_all_noncontiguous")

    test_max_axis0_noncontiguous()
    print("✓ test_max_axis0_noncontiguous")

    test_max_axis1_noncontiguous()
    print("✓ test_max_axis1_noncontiguous")

    test_min_all_noncontiguous()
    print("✓ test_min_all_noncontiguous")

    test_min_axis0_noncontiguous()
    print("✓ test_min_axis0_noncontiguous")

    test_min_axis1_noncontiguous()
    print("✓ test_min_axis1_noncontiguous")

    test_max_noncontiguous_matches_contiguous_baseline()
    print("✓ test_max_noncontiguous_matches_contiguous_baseline")

    test_min_noncontiguous_matches_contiguous_baseline()
    print("✓ test_min_noncontiguous_matches_contiguous_baseline")

    test_max_larger_noncontiguous()
    print("✓ test_max_larger_noncontiguous")

    test_min_larger_noncontiguous()
    print("✓ test_min_larger_noncontiguous")

    print("\nAll test_reduction_noncontiguous tests passed!")
