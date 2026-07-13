"""Direct unit tests for the tensor_creation module (issue #5158).

Other tests reach these creation functions through the
`odyssey.tensor.any_tensor` re-export layer. This file imports
straight from `odyssey.tensor.tensor_creation` so the module is
verified independently of that re-export.
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from std.math import isnan, isinf
from odyssey.tensor.tensor_creation import (
    zeros,
    ones,
    full,
    empty,
    arange,
    eye,
    linspace,
    ones_like,
    zeros_like,
    full_like,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
)


def test_zeros_shape_and_values() raises:
    """Zeros creates a correctly shaped all-zero tensor."""
    var t = zeros([2, 3], DType.float32)
    assert_equal(t.numel(), 6, "zeros numel")
    assert_equal(t.shape()[0], 2, "zeros dim0")
    assert_equal(t.shape()[1], 3, "zeros dim1")
    for i in range(t.numel()):
        assert_equal(t._get_float64(i), 0.0, "zeros element")


def test_ones_values() raises:
    """Ones creates an all-one tensor."""
    var t = ones([4], DType.float32)
    assert_equal(t.numel(), 4, "ones numel")
    for i in range(t.numel()):
        assert_equal(t._get_float64(i), 1.0, "ones element")


def test_full_fill_value() raises:
    """Full fills every element with the given value."""
    var t = full([5], 3.5, DType.float32)
    for i in range(t.numel()):
        assert_almost_equal(t._get_float64(i), 3.5, atol=1e-6)


def test_empty_shape() raises:
    """Empty allocates a tensor of the requested shape."""
    var t = empty([3, 2], DType.float32)
    assert_equal(t.numel(), 6, "empty numel")
    assert_equal(t.shape()[0], 3, "empty dim0")


def test_arange_sequence() raises:
    """Arange produces a 0,1,2,... sequence."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    assert_equal(t.numel(), 5, "arange length")
    for i in range(5):
        assert_almost_equal(t._get_float64(i), Float64(i), atol=1e-6)


def test_eye_identity() raises:
    """Eye places ones on the main diagonal and zeros elsewhere."""
    var t = eye(3, 3, 0, DType.float32)
    assert_equal(t.numel(), 9, "eye numel")
    for r in range(3):
        for c in range(3):
            var expected = 1.0 if r == c else 0.0
            assert_almost_equal(t._get_float64(r * 3 + c), expected, atol=1e-6)


def test_linspace_endpoints() raises:
    """Linspace spans start..stop inclusive with even spacing."""
    var t = linspace(0.0, 1.0, 5, DType.float32)
    assert_equal(t.numel(), 5, "linspace length")
    assert_almost_equal(t._get_float64(0), 0.0, atol=1e-6)
    assert_almost_equal(t._get_float64(4), 1.0, atol=1e-6)
    assert_almost_equal(t._get_float64(2), 0.5, atol=1e-6)


def test_ones_like_matches_shape() raises:
    """Ones_like copies the source shape and fills with ones."""
    var src = zeros([2, 4], DType.float32)
    var t = ones_like(src)
    assert_equal(t.numel(), 8, "ones_like numel")
    for i in range(t.numel()):
        assert_equal(t._get_float64(i), 1.0, "ones_like element")


def test_zeros_like_matches_shape() raises:
    """Zeros_like copies the source shape and fills with zeros."""
    var src = ones([6], DType.float32)
    var t = zeros_like(src)
    assert_equal(t.numel(), 6, "zeros_like numel")
    for i in range(t.numel()):
        assert_equal(t._get_float64(i), 0.0, "zeros_like element")


def test_full_like_fill() raises:
    """Full_like copies the source shape and fills with a constant."""
    var src = zeros([3], DType.float32)
    var t = full_like(src, 7.0)
    for i in range(t.numel()):
        assert_almost_equal(t._get_float64(i), 7.0, atol=1e-6)


def test_nan_tensor_is_nan() raises:
    """Nan_tensor fills every element with NaN."""
    var t = nan_tensor([4], DType.float32)
    for i in range(t.numel()):
        assert_true(isnan(t._get_float64(i)), "nan_tensor element is NaN")


def test_inf_tensor_is_positive_inf() raises:
    """Inf_tensor fills with +infinity."""
    var t = inf_tensor([4], DType.float32)
    for i in range(t.numel()):
        var v = t._get_float64(i)
        assert_true(isinf(v) and v > 0.0, "inf_tensor element is +inf")


def test_neg_inf_tensor_is_negative_inf() raises:
    """Neg_inf_tensor fills with -infinity."""
    var t = neg_inf_tensor([4], DType.float32)
    for i in range(t.numel()):
        var v = t._get_float64(i)
        assert_true(isinf(v) and v < 0.0, "neg_inf_tensor element is -inf")


def main() raises:
    """Run all tensor_creation tests."""
    print("Running test_tensor_creation tests...")

    test_zeros_shape_and_values()
    print("✓ test_zeros_shape_and_values")

    test_ones_values()
    print("✓ test_ones_values")

    test_full_fill_value()
    print("✓ test_full_fill_value")

    test_empty_shape()
    print("✓ test_empty_shape")

    test_arange_sequence()
    print("✓ test_arange_sequence")

    test_eye_identity()
    print("✓ test_eye_identity")

    test_linspace_endpoints()
    print("✓ test_linspace_endpoints")

    test_ones_like_matches_shape()
    print("✓ test_ones_like_matches_shape")

    test_zeros_like_matches_shape()
    print("✓ test_zeros_like_matches_shape")

    test_full_like_fill()
    print("✓ test_full_like_fill")

    test_nan_tensor_is_nan()
    print("✓ test_nan_tensor_is_nan")

    test_inf_tensor_is_positive_inf()
    print("✓ test_inf_tensor_is_positive_inf")

    test_neg_inf_tensor_is_negative_inf()
    print("✓ test_neg_inf_tensor_is_negative_inf")

    print("\nAll test_tensor_creation tests passed!")
