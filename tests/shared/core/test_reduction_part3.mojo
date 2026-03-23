# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for median and percentile reduction operations with gradient checking.

Tests cover:
- Median reduction along axes
- Percentile reduction along axes
- Shape validation for backward passes

All tests validate forward and backward passes produce correct values.
"""

from tests.shared.conftest import (
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.reduction import (
    median,
    percentile,
    median_backward,
    percentile_backward,
)
from shared.testing import check_gradient


# ============================================================================
# Median Tests
# ============================================================================


fn test_median_forward_odd() raises:
    """Test median with odd count."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 3.0
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 4.0
    x._data.bitcast[Float32]()[3] = 2.0
    x._data.bitcast[Float32]()[4] = 5.0

    var result = median(x, axis=-1)
    assert_close_float(result._get_float64(0), 3.0, rtol=1e-5, atol=1e-7)


fn test_median_forward_even() raises:
    """Test median with even count (average of two middle values)."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0
    x._data.bitcast[Float32]()[3] = 4.0

    var result = median(x, axis=-1)
    assert_close_float(result._get_float64(0), 2.5, rtol=1e-5, atol=1e-7)


fn test_median_backward_shapes() raises:
    """Test that median_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    for i in range(6):
        x._data.bitcast[Float32]()[i] = Float32(i) + 1.0

    var result = median(x, axis=1)
    var grad_output = ones_like(result)
    var grad_input = median_backward(grad_output, x, axis=1)

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)


# ============================================================================
# Percentile Tests
# ============================================================================


fn test_percentile_forward_p50() raises:
    """Test that 50th percentile equals median."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0
    x._data.bitcast[Float32]()[3] = 4.0
    x._data.bitcast[Float32]()[4] = 5.0

    var result = percentile(x, 50.0, axis=-1)
    assert_close_float(result._get_float64(0), 3.0, rtol=1e-5, atol=1e-7)


fn test_percentile_forward_p0_p100() raises:
    """Test that 0th and 100th percentiles equal min and max."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0
    x._data.bitcast[Float32]()[3] = 4.0
    x._data.bitcast[Float32]()[4] = 5.0

    var p0 = percentile(x, 0.0, axis=-1)
    assert_close_float(p0._get_float64(0), 1.0, rtol=1e-5, atol=1e-7)

    var p100 = percentile(x, 100.0, axis=-1)
    assert_close_float(p100._get_float64(0), 5.0, rtol=1e-5, atol=1e-7)


fn test_percentile_backward_shapes() raises:
    """Test that percentile_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    for i in range(6):
        x._data.bitcast[Float32]()[i] = Float32(i) + 1.0

    var result = percentile(x, 50.0, axis=1)
    var grad_output = ones_like(result)
    var grad_input = percentile_backward(grad_output, x, 50.0, axis=1)

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run median and percentile reduction tests."""
    print("Running reduction part 3 tests (median, percentile)...")

    # Median tests
    test_median_forward_odd()
    print("✓ test_median_forward_odd")

    test_median_forward_even()
    print("✓ test_median_forward_even")

    test_median_backward_shapes()
    print("✓ test_median_backward_shapes")

    # Percentile tests
    test_percentile_forward_p50()
    print("✓ test_percentile_forward_p50")

    test_percentile_forward_p0_p100()
    print("✓ test_percentile_forward_p0_p100")

    test_percentile_backward_shapes()
    print("✓ test_percentile_backward_shapes")

    print("\nAll reduction part 3 tests passed!")
