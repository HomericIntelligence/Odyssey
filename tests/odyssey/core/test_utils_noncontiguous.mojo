"""Tests for core.utils index ops on non-contiguous (strided) tensors.

Verifies argmax (flat + axis), top_k_indices, top_k, and argsort produce
correct results on a strided AnyTensor view (e.g. from slice(..., axis=1)).
Without the as_contiguous() guard these ops synthesize row-major strides and
read the raw buffer via _get_float64, silently returning wrong indices.

Regression test for #5572.
"""


from tests.odyssey.conftest import (
    assert_equal_int,
    assert_almost_equal,
    assert_false,
    assert_true,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.utils import argmax, top_k_indices, top_k, argsort
from odyssey.training.metrics.accuracy import argmax as metrics_argmax


def _make_strided_2x4() raises -> AnyTensor:
    """Columns 0..3 of a (2, 8) buffer — a non-contiguous inner-axis view.

    (2, 8) row-major buffer:
        row 0: [0, 5, 0, 0, 9, 0, 0, 0]
        row 1: [0, 0, 7, 0, 0, 0, 0, 0]
    The slice [:, 0:4] is logically [[0, 5, 0, 0], [0, 0, 7, 0]] but keeps the
    parent's stride of 8, so row 1 starts at flat offset 8 (not 4). A buggy
    reader would read row 1 at flat offset 4 (the 9.0, outside the slice).
    """
    var t = zeros([2, 8], DType.float32)
    t.set(1, Float32(5.0))  # row 0, col 1  (inside slice)
    t.set(4, Float32(9.0))  # row 0, col 4  (OUTSIDE slice — the trap)
    t.set(10, Float32(7.0))  # row 1, col 2  (inside slice)
    var view = t.slice(0, 4, axis=1)
    assert_false(view.is_contiguous(), "fixture must be non-contiguous")
    return view^


def test_argmax_axis_noncontiguous() raises:
    """Axis argmax reads the view's real rows, not flat offsets."""
    var view = _make_strided_2x4()
    var pred = argmax(view, axis=1)
    assert_equal_int(pred.shape()[0], 2)
    # Row 0 max is 5.0 at col 1; row 1 max is 7.0 at col 2.
    assert_equal_int(Int(pred.load[DType.int64](0)), 1)
    assert_equal_int(Int(pred.load[DType.int64](1)), 2)


def test_argmax_flat_noncontiguous() raises:
    """Flat argmax over a strided view scans only the view's real elements.

    The flat view [0, 5, 0, 0, 0, 0, 7, 0] (logical order) has its max 7.0 at
    logical index 6 — NOT the 9.0 at raw offset 4 that lives outside the slice.
    """
    var view = _make_strided_2x4()
    var idx = argmax(view)
    assert_equal_int(idx, 6)


def test_top_k_indices_noncontiguous() raises:
    """Top-k indices are the logical indices of the true top values."""
    var view = _make_strided_2x4()
    # Logical flat order: [0,5,0,0, 0,0,7,0]; top-2 are 7.0 (idx 6), 5.0 (idx 1).
    var idx = top_k_indices(view, 2)
    assert_equal_int(len(idx), 2)
    assert_equal_int(idx[0], 6)
    assert_equal_int(idx[1], 1)


def test_top_k_values_noncontiguous() raises:
    """Top-k values are the true top values (read from the contiguous view)."""
    var view = _make_strided_2x4()
    var result = top_k(view, 2)
    var values = result[0]
    assert_almost_equal(
        Float32(values.load[DType.float32](0)), Float32(7.0), tolerance=1e-4
    )
    assert_almost_equal(
        Float32(values.load[DType.float32](1)), Float32(5.0), tolerance=1e-4
    )


def test_argsort_noncontiguous() raises:
    """Argsort orders the view's real elements, not raw-buffer garbage."""
    var view = _make_strided_2x4()
    # Logical: [0,5,0,0, 0,0,7,0]. Descending order by value -> idx 6 (7.0),
    # then idx 1 (5.0), then the zeros in ascending-index order.
    var idx = argsort(view, descending=True)
    assert_equal_int(len(idx), 8)
    assert_equal_int(idx[0], 6)
    assert_equal_int(idx[1], 1)


def test_argsort_ascending_noncontiguous() raises:
    """Ascending argsort on a strided view puts the true max last."""
    var view = _make_strided_2x4()
    # Logical: [0,5,0,0, 0,0,7,0]. Ascending -> zeros first, then 5.0 (idx 1),
    # then the max 7.0 (idx 6) last.
    var idx = argsort(view, descending=False)
    assert_equal_int(len(idx), 8)
    assert_equal_int(idx[len(idx) - 1], 6)
    assert_equal_int(idx[len(idx) - 2], 1)


def test_metrics_argmax_noncontiguous() raises:
    """Metrics argmax also guards strided views (#5572, same class)."""
    var view = _make_strided_2x4()
    var pred = metrics_argmax(view, axis=1)
    assert_equal_int(pred.shape()[0], 2)
    # Row 0 max is 5.0 at col 1; row 1 max is 7.0 at col 2 (result dtype int32).
    assert_equal_int(Int(pred.load[DType.int32](0)), 1)
    assert_equal_int(Int(pred.load[DType.int32](1)), 2)


def test_contiguous_baseline_unchanged() raises:
    """Contiguous inputs still behave correctly (guard is a no-op for them)."""
    var t = zeros([2, 4], DType.float32)
    t.set(1, Float32(5.0))  # row 0, col 1
    t.set(6, Float32(7.0))  # row 1, col 2
    assert_true(t.is_contiguous(), "baseline fixture must be contiguous")
    var pred = argmax(t, axis=1)
    assert_equal_int(Int(pred.load[DType.int64](0)), 1)
    assert_equal_int(Int(pred.load[DType.int64](1)), 2)


def main() raises:
    print("test_argmax_axis_noncontiguous...", end="")
    test_argmax_axis_noncontiguous()
    print(" PASS")
    print("test_argmax_flat_noncontiguous...", end="")
    test_argmax_flat_noncontiguous()
    print(" PASS")
    print("test_top_k_indices_noncontiguous...", end="")
    test_top_k_indices_noncontiguous()
    print(" PASS")
    print("test_top_k_values_noncontiguous...", end="")
    test_top_k_values_noncontiguous()
    print(" PASS")
    print("test_argsort_noncontiguous...", end="")
    test_argsort_noncontiguous()
    print(" PASS")
    print("test_argsort_ascending_noncontiguous...", end="")
    test_argsort_ascending_noncontiguous()
    print(" PASS")
    print("test_metrics_argmax_noncontiguous...", end="")
    test_metrics_argmax_noncontiguous()
    print(" PASS")
    print("test_contiguous_baseline_unchanged...", end="")
    test_contiguous_baseline_unchanged()
    print(" PASS")
    print("ALL test_utils_noncontiguous TESTS PASSED")
