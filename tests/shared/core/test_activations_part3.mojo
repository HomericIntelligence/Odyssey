"""Tests for activation functions - Part 3: Sigmoid (stability/dtype), Tanh, Softmax (basic).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests: test_sigmoid_numerical_stability, test_sigmoid_float16, test_sigmoid_float64,
       test_tanh_basic, test_tanh_values, test_tanh_backward,
       test_tanh_range, test_softmax_basic_2d
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_true,
)
from shared.core.extensor import (
    AnyTensor,
    zeros,
    ones_like,
)
from shared.core.activation import (
    sigmoid,
    tanh,
    softmax,
    sigmoid_backward,
    tanh_backward,
)
from shared.testing import (
    check_gradient,
)
from math import tanh as math_tanh


# ============================================================================
# Sigmoid Tests (continued)
# ============================================================================


fn test_sigmoid_numerical_stability() raises:
    """Test sigmoid with extreme values."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -100.0
    x._data.bitcast[Float32]()[1] = -20.0
    x._data.bitcast[Float32]()[2] = 20.0
    x._data.bitcast[Float32]()[3] = 100.0

    var y = sigmoid(x)

    # Large negative values should be close to 0
    assert_true(y._data.bitcast[Float32]()[0] < 1e-6)
    assert_true(y._data.bitcast[Float32]()[1] < 1e-6)

    # Large positive values should be close to 1
    assert_true(y._data.bitcast[Float32]()[2] > 0.999999)
    assert_true(y._data.bitcast[Float32]()[3] > 0.999999)


fn test_sigmoid_float16() raises:
    """Test sigmoid with float16."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float16)
    x._data.bitcast[Float16]()[0] = Float16(-1.0)
    x._data.bitcast[Float16]()[1] = Float16(0.0)
    x._data.bitcast[Float16]()[2] = Float16(1.0)

    var y = sigmoid(x)

    # Check sigmoid(0) = 0.5
    var val_0 = Float32(y._data.bitcast[Float16]()[1])
    assert_almost_equal(val_0, Float32(0.5), tolerance=0.01)

    # Check range (0, 1)
    for i in range(3):
        var val = Float32(y._data.bitcast[Float16]()[i])
        assert_true(val > 0.0 and val < 1.0)


fn test_sigmoid_float64() raises:
    """Test sigmoid with float64 dtype."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float64)

    x._data.bitcast[Float64]()[0] = 0.0

    var y = sigmoid(x)

    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.5, tolerance=1e-10)


# ============================================================================
# Tanh Tests
# ============================================================================


fn test_tanh_basic() raises:
    """Test tanh with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -100.0  # Should be ~-1
    x._data.bitcast[Float32]()[1] = 0.0  # Should be 0
    x._data.bitcast[Float32]()[2] = 100.0  # Should be ~1

    var y = tanh(x)

    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-3
    )


fn test_tanh_values() raises:
    """Test tanh against known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float64)
    x._data.bitcast[Float64]()[0] = 0.0
    x._data.bitcast[Float64]()[1] = 1.0
    x._data.bitcast[Float64]()[2] = -1.0

    var y = tanh(x)

    # tanh(0) = 0
    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.0, tolerance=1e-10)

    # tanh(1) ≈ 0.7616
    var expected_tanh_1 = math_tanh(1.0)
    assert_almost_equal(
        y._data.bitcast[Float64]()[1], expected_tanh_1, tolerance=1e-10
    )

    # tanh(-1) ≈ -0.7616
    assert_almost_equal(
        y._data.bitcast[Float64]()[2], -expected_tanh_1, tolerance=1e-10
    )


fn test_tanh_backward() raises:
    """Test tanh gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Use multiple test points for better coverage
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    # Forward function wrapper
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return tanh(x)

    var y = tanh(x)
    var grad_out = ones_like(y)

    # Note: tanh_backward takes output y, not input x
    fn backward_fn(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        var out = tanh(x)  # Recompute output inside wrapper
        return tanh_backward(grad, out)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


fn test_tanh_range() raises:
    """Test tanh output is in (-1, 1)."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -10.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 10.0

    var y = tanh(x)

    # All values should be in [-1, 1] (inclusive due to floating point precision)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val >= -1.0, "tanh output should be >= -1.0")
        assert_true(val <= 1.0, "tanh output should be <= 1.0")


# ============================================================================
# Softmax Tests
# ============================================================================


fn test_softmax_basic_2d() raises:
    """Test softmax 2D normalization."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # All zeros should give uniform distribution
    var y = softmax(x, axis=1)

    # Sum should be 1.0
    var sum = (
        y._data.bitcast[Float32]()[0]
        + y._data.bitcast[Float32]()[1]
        + y._data.bitcast[Float32]()[2]
    )
    assert_almost_equal(sum, Float32(1.0), tolerance=1e-5)

    # Each value should be ~1/3
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.333333), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.333333), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.333333), tolerance=1e-3
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_activations_part3.mojo")
    print("=" * 70 + "\n")

    # test_sigmoid_numerical_stability
    total += 1
    try:
        test_sigmoid_numerical_stability()
        passed += 1
        print("  ✓ test_sigmoid_numerical_stability")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_numerical_stability:", e)

    # test_sigmoid_float16
    total += 1
    try:
        test_sigmoid_float16()
        passed += 1
        print("  ✓ test_sigmoid_float16")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_float16:", e)

    # test_sigmoid_float64
    total += 1
    try:
        test_sigmoid_float64()
        passed += 1
        print("  ✓ test_sigmoid_float64")
    except e:
        failed += 1
        print("  ✗ test_sigmoid_float64:", e)

    # test_tanh_basic
    total += 1
    try:
        test_tanh_basic()
        passed += 1
        print("  ✓ test_tanh_basic")
    except e:
        failed += 1
        print("  ✗ test_tanh_basic:", e)

    # test_tanh_values
    total += 1
    try:
        test_tanh_values()
        passed += 1
        print("  ✓ test_tanh_values")
    except e:
        failed += 1
        print("  ✗ test_tanh_values:", e)

    # test_tanh_backward
    total += 1
    try:
        test_tanh_backward()
        passed += 1
        print("  ✓ test_tanh_backward")
    except e:
        failed += 1
        print("  ✗ test_tanh_backward:", e)

    # test_tanh_range
    total += 1
    try:
        test_tanh_range()
        passed += 1
        print("  ✓ test_tanh_range")
    except e:
        failed += 1
        print("  ✗ test_tanh_range:", e)

    # test_softmax_basic_2d
    total += 1
    try:
        test_softmax_basic_2d()
        passed += 1
        print("  ✓ test_softmax_basic_2d")
    except e:
        failed += 1
        print("  ✗ test_softmax_basic_2d:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
