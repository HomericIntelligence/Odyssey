"""Tests for reduction operations on non-contiguous tensors - Part 1.

Verifies that sum() and mean() produce correct results when given
non-contiguous inputs. Without the as_contiguous() guard these operations
read from wrong memory positions, silently producing wrong values.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Follow-up from #3236.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_false,
    assert_true,
)
from shared.core.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core.reduction import sum, mean
from shared.core.matrix import transpose_view


fn _make_nc_2x3() raises -> AnyTensor:
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


fn test_sum_all_noncontiguous() raises:
    """Reduction sum() over all elements of a non-contiguous tensor should be correct."""
    var nc = _make_nc_2x3()  # logical values: 0,3,1,4,2,5; sum=15

    var result = sum(nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(15.0), tolerance=1e-4)


fn test_sum_axis0_noncontiguous() raises:
    """Reduction sum(axis=0) on non-contiguous 3×2 should sum along rows correctly."""
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # sum axis 0: [0+1+2, 3+4+5] = [3, 12]
    var nc = _make_nc_2x3()

    var result = sum(nc, axis=0)

    assert_equal_int(result.shape()[0], 2)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(3.0), tolerance=1e-4)
    assert_almost_equal(ptr[1], Float32(12.0), tolerance=1e-4)


fn test_sum_axis1_noncontiguous() raises:
    """Reduction sum(axis=1) on non-contiguous 3×2 should sum along columns correctly."""
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # sum axis 1: [0+3, 1+4, 2+5] = [3, 5, 7]
    var nc = _make_nc_2x3()

    var result = sum(nc, axis=1)

    assert_equal_int(result.shape()[0], 3)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(3.0), tolerance=1e-4)
    assert_almost_equal(ptr[1], Float32(5.0), tolerance=1e-4)
    assert_almost_equal(ptr[2], Float32(7.0), tolerance=1e-4)


fn test_mean_all_noncontiguous() raises:
    """Reduction mean() over all elements of a non-contiguous tensor should be correct."""
    var nc = _make_nc_2x3()  # logical values: 0,3,1,4,2,5; mean=2.5

    var result = mean(nc)

    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(2.5), tolerance=1e-4)


fn test_mean_axis0_noncontiguous() raises:
    """Reduction mean(axis=0) on non-contiguous 3×2 should average along rows correctly."""
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # mean axis 0: [0+1+2/3, 3+4+5/3] = [1.0, 4.0]
    var nc = _make_nc_2x3()

    var result = mean(nc, axis=0)

    assert_equal_int(result.shape()[0], 2)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(1.0), tolerance=1e-4)
    assert_almost_equal(ptr[1], Float32(4.0), tolerance=1e-4)


fn test_mean_axis1_noncontiguous() raises:
    """Reduction mean(axis=1) on non-contiguous 3×2 should average along columns correctly."""
    # nc shape (3,2), logical rows: [0,3], [1,4], [2,5]
    # mean axis 1: [0+3/2, 1+4/2, 2+5/2] = [1.5, 2.5, 3.5]
    var nc = _make_nc_2x3()

    var result = mean(nc, axis=1)

    assert_equal_int(result.shape()[0], 3)
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(1.5), tolerance=1e-4)
    assert_almost_equal(ptr[1], Float32(2.5), tolerance=1e-4)
    assert_almost_equal(ptr[2], Float32(3.5), tolerance=1e-4)


fn test_sum_noncontiguous_matches_contiguous_baseline() raises:
    """Non-contiguous sum must match contiguous baseline."""
    # Build contiguous tensor with same logical values as nc
    var logical = zeros([3, 2], DType.float32)
    var lp = logical._data.bitcast[Float32]()
    lp[0] = 0.0; lp[1] = 3.0
    lp[2] = 1.0; lp[3] = 4.0
    lp[4] = 2.0; lp[5] = 5.0

    var baseline = sum(logical, axis=0)
    var nc = _make_nc_2x3()
    var result = sum(nc, axis=0)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    for i in range(2):
        assert_almost_equal(rp[i], bp[i], tolerance=1e-4)


fn test_mean_noncontiguous_matches_contiguous_baseline() raises:
    """Non-contiguous mean must match contiguous baseline."""
    var logical = zeros([3, 2], DType.float32)
    var lp = logical._data.bitcast[Float32]()
    lp[0] = 0.0; lp[1] = 3.0
    lp[2] = 1.0; lp[3] = 4.0
    lp[4] = 2.0; lp[5] = 5.0

    var baseline = mean(logical)
    var nc = _make_nc_2x3()
    var result = mean(nc)

    var bp = baseline._data.bitcast[Float32]()
    var rp = result._data.bitcast[Float32]()
    assert_almost_equal(rp[0], bp[0], tolerance=1e-4)


fn test_sum_keepdims_noncontiguous() raises:
    """Reduction sum(keepdims=True) on non-contiguous tensor should work correctly."""
    var nc = _make_nc_2x3()  # shape (3,2), sum=15

    var result = sum(nc, axis=-1, keepdims=True)

    # keepdims=True with axis=-1 should give shape [1, 1] scalar-like
    var ptr = result._data.bitcast[Float32]()
    assert_almost_equal(ptr[0], Float32(15.0), tolerance=1e-4)


fn main() raises:
    """Run all non-contiguous reduction tests (part 1)."""
    print("Running reduction non-contiguous tests (part 1)...")

    test_sum_all_noncontiguous()
    test_sum_axis0_noncontiguous()
    test_sum_axis1_noncontiguous()
    test_mean_all_noncontiguous()
    test_mean_axis0_noncontiguous()
    test_mean_axis1_noncontiguous()
    test_sum_noncontiguous_matches_contiguous_baseline()
    test_mean_noncontiguous_matches_contiguous_baseline()
    test_sum_keepdims_noncontiguous()

    print("All reduction non-contiguous tests (part 1) passed!")
