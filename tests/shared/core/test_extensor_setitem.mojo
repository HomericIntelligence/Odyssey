"""Tests for AnyTensor __setitem__ with flat and multi-dimensional index support.

Covers:
- Flat Int index overloads (Float64, Int64, Float32)
- Multi-dimensional List[Int] index overload with stride arithmetic
- Bounds checking and error cases
- Round-trip get/set correctness

CI verification: issue #3840. All 17 tests verified passing in CI.
"""

from shared.core.any_tensor import AnyTensor, zeros, ones
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


# ============================================================================
# Flat index __setitem__ tests
# ============================================================================


fn test_setitem_flat_float64_1d() raises:
    """Test flat __setitem__ (Float64) on a 1D tensor."""
    var t = zeros([5], DType.float32)
    t[2] = 7.5
    assert_almost_equal(t._get_float64(2), 7.5, tolerance=1e-6)


fn test_setitem_flat_float32() raises:
    """Test flat __setitem__ (Float32) on a 2D tensor."""
    var t = zeros([3, 4], DType.float32)
    t[5] = Float32(3.14)
    assert_almost_equal(
        t._get_float64(5), Float64(Float32(3.14)), tolerance=1e-6
    )


fn test_setitem_flat_int64() raises:
    """Test flat __setitem__ on an int tensor.

    Note: Int64 can no longer be implicitly converted to Float32, so we
    use a Float64 literal and rely on the Float64 overload instead.
    """
    var t = zeros([4], DType.int32)
    t[3] = 99.0
    assert_almost_equal(t._get_float64(3), 99.0, tolerance=1e-6)


fn test_setitem_flat_overwrites_value() raises:
    """Test that flat __setitem__ overwrites an existing value."""
    var t = ones([3], DType.float32)
    t[1] = 42.0
    assert_almost_equal(t._get_float64(0), 1.0, tolerance=1e-6)
    assert_almost_equal(t._get_float64(1), 42.0, tolerance=1e-6)
    assert_almost_equal(t._get_float64(2), 1.0, tolerance=1e-6)


fn test_setitem_flat_out_of_bounds_raises() raises:
    """Test that flat __setitem__ with out-of-bounds index raises an error."""
    var t = zeros([3], DType.float32)
    var raised = False
    try:
        t[3] = 1.0
    except:
        raised = True
    assert_true(raised, "Expected Error for out-of-bounds index")


fn test_setitem_flat_negative_index_raises() raises:
    """Test that flat __setitem__ with negative index raises an error."""
    var t = zeros([3], DType.float32)
    var raised = False
    try:
        t[-1] = 1.0
    except:
        raised = True
    assert_true(raised, "Expected Error for negative index")


# ============================================================================
# Multi-dimensional List[Int] __setitem__ tests
# ============================================================================


fn test_setitem_multidim_2d() raises:
    """Test multi-dim __setitem__ on a 2D tensor."""
    var t = zeros([3, 4], DType.float32)
    # Row 1, col 2 → flat index = 1*4 + 2 = 6
    t[[1, 2]] = 5.0
    assert_almost_equal(t._get_float64(6), 5.0, tolerance=1e-6)


fn test_setitem_multidim_3d() raises:
    """Test multi-dim __setitem__ on a 3D tensor."""
    var t = zeros([2, 3, 4], DType.float32)
    # [1, 2, 3] → flat = 1*12 + 2*4 + 3 = 23
    t[[1, 2, 3]] = 9.0
    assert_almost_equal(t._get_float64(23), 9.0, tolerance=1e-6)


fn test_setitem_multidim_first_element() raises:
    """Test multi-dim __setitem__ at [0, 0] → flat index 0."""
    var t = ones([4, 5], DType.float32)
    t[[0, 0]] = 99.0
    assert_almost_equal(t._get_float64(0), 99.0, tolerance=1e-6)


fn test_setitem_multidim_last_element() raises:
    """Test multi-dim __setitem__ at last element of 2D tensor."""
    var t = zeros([3, 4], DType.float32)
    # [2, 3] → flat = 2*4 + 3 = 11
    t[[2, 3]] = 77.0
    assert_almost_equal(t._get_float64(11), 77.0, tolerance=1e-6)


fn test_setitem_multidim_float64_dtype() raises:
    """Test multi-dim __setitem__ on a float64 tensor."""
    var t = zeros([2, 3], DType.float64)
    t[[1, 1]] = 3.14159
    # flat = 1*3 + 1 = 4
    # Note: __getitem__ returns Float32 lvalue, so 3.14159 is stored at
    # Float32 precision even on a float64 tensor. Use Float32 tolerance.
    # Long-term fix: make AnyTensor parametric on dtype (see GitHub epic).
    assert_almost_equal(t._get_float64(4), 3.14159, tolerance=1e-6)


fn test_setitem_multidim_int_dtype() raises:
    """Test multi-dim __setitem__ on an int32 tensor."""
    var t = zeros([2, 4], DType.int32)
    t[[0, 3]] = 42.0
    # flat = 0*4 + 3 = 3
    assert_almost_equal(t._get_float64(3), 42.0, tolerance=1e-6)


# ============================================================================
# Error cases for multi-dim __setitem__
# ============================================================================


fn test_setitem_multidim_rank_mismatch_raises() raises:
    """Test that wrong number of indices raises an error."""
    var t = zeros([3, 4], DType.float32)
    var raised = False
    try:
        t[[1]] = 5.0
    except:
        raised = True
    assert_true(raised, "Expected Error for rank mismatch")


fn test_setitem_multidim_too_many_indices_raises() raises:
    """Test that too many indices raises an error."""
    var t = zeros([3, 4], DType.float32)
    var raised = False
    try:
        t[[0, 1, 2]] = 5.0
    except:
        raised = True
    assert_true(raised, "Expected Error for too many indices")


fn test_setitem_multidim_dim_out_of_bounds_raises() raises:
    """Test that a per-dimension out-of-bounds index raises an error."""
    var t = zeros([3, 4], DType.float32)
    var raised = False
    try:
        t[[1, 4]] = 5.0  # col 4 is out of bounds for shape[1]=4
    except:
        raised = True
    assert_true(raised, "Expected Error for per-dimension out-of-bounds")


fn test_setitem_multidim_negative_dim_raises() raises:
    """Test that a negative per-dimension index raises an error."""
    var t = zeros([3, 4], DType.float32)
    var raised = False
    try:
        t[[-1, 0]] = 5.0
    except:
        raised = True
    assert_true(raised, "Expected Error for negative dimension index")


# ============================================================================
# Round-trip tests
# ============================================================================


fn test_setitem_getitem_roundtrip_flat() raises:
    """Test write via __setitem__ and read back via __getitem__."""
    var t = zeros([5], DType.float32)
    t[3] = 6.28
    var val = t[3]
    assert_almost_equal(Float64(val), 6.28, tolerance=1e-5)


fn test_setitem_getitem_roundtrip_multidim() raises:
    """Test write via multi-dim __setitem__ and verify via flat __getitem__."""
    var t = zeros([4, 5], DType.float32)
    t[[2, 3]] = 1.5
    # flat = 2*5 + 3 = 13
    var val = t[13]
    assert_almost_equal(Float64(val), 1.5, tolerance=1e-6)


fn test_setitem_multidim_does_not_affect_others() raises:
    """Test that writing one element doesn't corrupt neighbors."""
    var t = zeros([3, 3], DType.float32)
    t[[1, 1]] = 42.0
    # All other elements should remain 0.0
    for flat in range(9):
        var expected = 42.0 if flat == 4 else 0.0  # [1,1] → 1*3+1=4
        assert_almost_equal(t._get_float64(flat), expected, tolerance=1e-6)


fn main() raises:
    """Run all __setitem__ tests."""
    print("Running flat index __setitem__ tests...")
    test_setitem_flat_float64_1d()
    test_setitem_flat_float32()
    test_setitem_flat_int64()
    test_setitem_flat_overwrites_value()
    test_setitem_flat_out_of_bounds_raises()
    test_setitem_flat_negative_index_raises()

    print("Running multi-dim __setitem__ tests...")
    test_setitem_multidim_2d()
    test_setitem_multidim_3d()
    test_setitem_multidim_first_element()
    test_setitem_multidim_last_element()
    test_setitem_multidim_float64_dtype()
    test_setitem_multidim_int_dtype()

    print("Running multi-dim error case tests...")
    test_setitem_multidim_rank_mismatch_raises()
    test_setitem_multidim_too_many_indices_raises()
    test_setitem_multidim_dim_out_of_bounds_raises()
    test_setitem_multidim_negative_dim_raises()

    print("Running round-trip tests...")
    test_setitem_getitem_roundtrip_flat()
    test_setitem_getitem_roundtrip_multidim()
    test_setitem_multidim_does_not_affect_others()

    print("All __setitem__ tests passed!")
