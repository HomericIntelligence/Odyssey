"""Tests for typed Tensor[dtype] factory functions (part 2).

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- linspace produces evenly spaced values
- linspace without endpoint
- randn produces values (deterministic with seed)
- nan_tensor fills with NaN
- inf_tensor fills with positive infinity
- neg_inf_tensor fills with negative infinity
- eye with diagonal offset
- arange with step 0.5
"""

from testing import assert_true, assert_almost_equal
from math import isnan
from shared.tensor.tensor import Tensor
from shared.tensor.factories import (
    linspace,
    randn,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
    eye,
    arange,
)


fn test_linspace_values() raises:
    """Verify linspace produces evenly spaced values including endpoint."""
    var t = linspace[DType.float64](
        Scalar[DType.float64](0.0),
        Scalar[DType.float64](1.0),
        5,
    )
    assert_true(t.numel() == 5, "should have 5 elements")
    assert_almost_equal(t[0], Scalar[DType.float64](0.0), msg="element 0")
    assert_almost_equal(t[1], Scalar[DType.float64](0.25), msg="element 1")
    assert_almost_equal(t[2], Scalar[DType.float64](0.5), msg="element 2")
    assert_almost_equal(t[3], Scalar[DType.float64](0.75), msg="element 3")
    assert_almost_equal(t[4], Scalar[DType.float64](1.0), msg="element 4")
    print("PASS: test_linspace_values")


fn test_linspace_no_endpoint() raises:
    """Verify linspace without endpoint excludes the stop value."""
    var t = linspace[DType.float64](
        Scalar[DType.float64](0.0),
        Scalar[DType.float64](1.0),
        4,
        endpoint=False,
    )
    assert_true(t.numel() == 4, "should have 4 elements")
    assert_almost_equal(t[0], Scalar[DType.float64](0.0), msg="element 0")
    assert_almost_equal(t[1], Scalar[DType.float64](0.25), msg="element 1")
    assert_almost_equal(t[2], Scalar[DType.float64](0.5), msg="element 2")
    assert_almost_equal(t[3], Scalar[DType.float64](0.75), msg="element 3")
    print("PASS: test_linspace_no_endpoint")


fn test_randn_produces_values() raises:
    """Verify randn produces values with deterministic seed."""
    var t = randn[DType.float32]([10], seed=42)
    assert_true(t.numel() == 10, "should have 10 elements")
    # Verify values are not all zero (would indicate broken RNG)
    var has_nonzero = False
    for i in range(t.numel()):
        if t[i] != Scalar[DType.float32](0.0):
            has_nonzero = True
            break
    assert_true(has_nonzero, "randn should produce non-zero values")
    print("PASS: test_randn_produces_values")


fn test_nan_tensor_fills_nan() raises:
    """Verify nan_tensor fills all elements with NaN."""
    var t = nan_tensor[DType.float32]([2, 3])
    assert_true(t.numel() == 6, "numel should be 6")
    for i in range(t.numel()):
        assert_true(isnan(t[i]), "element should be NaN")
    print("PASS: test_nan_tensor_fills_nan")


fn test_inf_tensor_fills_inf() raises:
    """Verify inf_tensor fills all elements with positive infinity."""
    var t = inf_tensor[DType.float32]([4])
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(t.numel()):
        assert_true(t[i] > Scalar[DType.float32](1e30), "element should be very large (inf)")
    print("PASS: test_inf_tensor_fills_inf")


fn test_neg_inf_tensor_fills_neg_inf() raises:
    """Verify neg_inf_tensor fills all elements with negative infinity."""
    var t = neg_inf_tensor[DType.float32]([4])
    assert_true(t.numel() == 4, "numel should be 4")
    for i in range(t.numel()):
        assert_true(t[i] < Scalar[DType.float32](-1e30), "element should be very negative (neg inf)")
    print("PASS: test_neg_inf_tensor_fills_neg_inf")


fn test_eye_with_offset() raises:
    """Verify eye with k=1 places ones on upper diagonal."""
    var t = eye[DType.float32](3, 4, 1)
    assert_true(t.numel() == 12, "should have 12 elements")
    # Row 0: [0, 1, 0, 0]
    assert_almost_equal(t[0], Scalar[DType.float32](0.0), msg="[0,0]")
    assert_almost_equal(t[1], Scalar[DType.float32](1.0), msg="[0,1]")
    assert_almost_equal(t[2], Scalar[DType.float32](0.0), msg="[0,2]")
    assert_almost_equal(t[3], Scalar[DType.float32](0.0), msg="[0,3]")
    # Row 1: [0, 0, 1, 0]
    assert_almost_equal(t[4], Scalar[DType.float32](0.0), msg="[1,0]")
    assert_almost_equal(t[5], Scalar[DType.float32](0.0), msg="[1,1]")
    assert_almost_equal(t[6], Scalar[DType.float32](1.0), msg="[1,2]")
    assert_almost_equal(t[7], Scalar[DType.float32](0.0), msg="[1,3]")
    # Row 2: [0, 0, 0, 1]
    assert_almost_equal(t[8], Scalar[DType.float32](0.0), msg="[2,0]")
    assert_almost_equal(t[9], Scalar[DType.float32](0.0), msg="[2,1]")
    assert_almost_equal(t[10], Scalar[DType.float32](0.0), msg="[2,2]")
    assert_almost_equal(t[11], Scalar[DType.float32](1.0), msg="[2,3]")
    print("PASS: test_eye_with_offset")


fn test_arange_fractional_step() raises:
    """Verify arange with step=0.5 produces correct fractional sequence."""
    var t = arange[DType.float64](
        Scalar[DType.float64](0.0),
        Scalar[DType.float64](2.0),
        Scalar[DType.float64](0.5),
    )
    assert_true(t.numel() == 4, "should have 4 elements")
    assert_almost_equal(t[0], Scalar[DType.float64](0.0), msg="element 0")
    assert_almost_equal(t[1], Scalar[DType.float64](0.5), msg="element 1")
    assert_almost_equal(t[2], Scalar[DType.float64](1.0), msg="element 2")
    assert_almost_equal(t[3], Scalar[DType.float64](1.5), msg="element 3")
    print("PASS: test_arange_fractional_step")


fn main() raises:
    test_linspace_values()
    test_linspace_no_endpoint()
    test_randn_produces_values()
    test_nan_tensor_fills_nan()
    test_inf_tensor_fills_inf()
    test_neg_inf_tensor_fills_neg_inf()
    test_eye_with_offset()
    test_arange_fractional_step()
