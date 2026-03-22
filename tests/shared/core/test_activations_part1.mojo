"""Tests for activation functions - Part 1: ReLU and Leaky ReLU (basic).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests: test_relu_basic, test_relu_non_negativity, test_relu_backward,
       test_relu_shape, test_relu_integer_types, test_relu_float64,
       test_leaky_relu_basic, test_leaky_relu_custom_alpha
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import (
    AnyTensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
)
from shared.core.activation import (
    relu,
    leaky_relu,
    relu_backward,
    leaky_relu_backward,
)
from shared.testing import (
    check_gradient,
)


# ============================================================================
# ReLU Tests
# ============================================================================


fn test_relu_basic() raises:
    """Test ReLU with known values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    # Set test values: [-2, -1, 0, 1, 2]
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var y = relu(x)

    # Expected: [0, 0, 0, 1, 2]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


fn test_relu_non_negativity() raises:
    """Test ReLU always produces non-negative outputs."""
    var shape = List[Int]()
    shape.append(100)
    var x = zeros(shape, DType.float32)

    # Fill with values from -50 to 50
    for i in range(100):
        x._data.bitcast[Float32]()[i] = Float32(i - 50)

    var y = relu(x)

    # All outputs should be >= 0
    for i in range(100):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val >= 0.0)


fn test_relu_backward() raises:
    """Test ReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)

    # Set test values: [-1, 1e-4, 0.5, 2]
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 1e-4
    x._data.bitcast[Float32]()[2] = 0.5
    x._data.bitcast[Float32]()[3] = 2.0

    # Forward function wrapper
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        return relu(x)

    var y = relu(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_wrapper(
        grad: AnyTensor, x: AnyTensor
    ) raises escaping -> AnyTensor:
        return relu_backward(grad, x)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(forward, backward_wrapper, x, grad_out, rtol=1e-3, atol=1e-6)


fn test_relu_shape() raises:
    """Test ReLU preserves shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var y = relu(x)

    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)
    assert_equal(y.shape()[2], 4)


fn test_relu_integer_types() raises:
    """Test ReLU with integer types."""
    # Test int32
    var shape = List[Int]()
    shape.append(5)
    var x_int32 = zeros(shape, DType.int32)
    x_int32._data.bitcast[Int32]()[0] = -2
    x_int32._data.bitcast[Int32]()[1] = -1
    x_int32._data.bitcast[Int32]()[2] = 0
    x_int32._data.bitcast[Int32]()[3] = 1
    x_int32._data.bitcast[Int32]()[4] = 2

    var y_int32 = relu(x_int32)

    # Expected: [0, 0, 0, 1, 2]
    assert_equal(y_int32._data.bitcast[Int32]()[0], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[1], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[2], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[3], 1)
    assert_equal(y_int32._data.bitcast[Int32]()[4], 2)

    # Test uint8 (already non-negative)
    var x_uint8 = zeros(shape, DType.uint8)
    x_uint8._data.bitcast[UInt8]()[0] = 0
    x_uint8._data.bitcast[UInt8]()[1] = 1
    x_uint8._data.bitcast[UInt8]()[2] = 128
    x_uint8._data.bitcast[UInt8]()[3] = 255
    x_uint8._data.bitcast[UInt8]()[4] = 100

    var y_uint8 = relu(x_uint8)

    # Should be unchanged
    assert_equal(y_uint8._data.bitcast[UInt8]()[0], 0)
    assert_equal(y_uint8._data.bitcast[UInt8]()[1], 1)
    assert_equal(y_uint8._data.bitcast[UInt8]()[2], 128)
    assert_equal(y_uint8._data.bitcast[UInt8]()[3], 255)


fn test_relu_float64() raises:
    """Test ReLU with float64 dtype."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float64)

    x._data.bitcast[Float64]()[0] = -1.0
    x._data.bitcast[Float64]()[1] = 1.0

    var y = relu(x)

    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.0, tolerance=1e-10)
    assert_almost_equal(y._data.bitcast[Float64]()[1], 1.0, tolerance=1e-10)


# ============================================================================
# Leaky ReLU Tests
# ============================================================================


fn test_leaky_relu_basic() raises:
    """Test Leaky ReLU with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 2.0

    var y = leaky_relu(x, alpha=0.1)

    # Expected: [-0.2, 0, 2.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.2), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_leaky_relu_custom_alpha() raises:
    """Test Leaky ReLU with custom alpha value."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -4.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 4.0

    var y = leaky_relu(x, alpha=0.25)

    # Expected with alpha=0.25: [-1.0, 0, 4.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(4.0), tolerance=1e-5
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
    print("Running tests from: test_activations_part1.mojo")
    print("=" * 70 + "\n")

    # test_relu_basic
    total += 1
    try:
        test_relu_basic()
        passed += 1
        print("  ✓ test_relu_basic")
    except e:
        failed += 1
        print("  ✗ test_relu_basic:", e)

    # test_relu_non_negativity
    total += 1
    try:
        test_relu_non_negativity()
        passed += 1
        print("  ✓ test_relu_non_negativity")
    except e:
        failed += 1
        print("  ✗ test_relu_non_negativity:", e)

    # test_relu_backward
    total += 1
    try:
        test_relu_backward()
        passed += 1
        print("  ✓ test_relu_backward")
    except e:
        failed += 1
        print("  ✗ test_relu_backward:", e)

    # test_relu_shape
    total += 1
    try:
        test_relu_shape()
        passed += 1
        print("  ✓ test_relu_shape")
    except e:
        failed += 1
        print("  ✗ test_relu_shape:", e)

    # test_relu_integer_types
    total += 1
    try:
        test_relu_integer_types()
        passed += 1
        print("  ✓ test_relu_integer_types")
    except e:
        failed += 1
        print("  ✗ test_relu_integer_types:", e)

    # test_relu_float64
    total += 1
    try:
        test_relu_float64()
        passed += 1
        print("  ✓ test_relu_float64")
    except e:
        failed += 1
        print("  ✗ test_relu_float64:", e)

    # test_leaky_relu_basic
    total += 1
    try:
        test_leaky_relu_basic()
        passed += 1
        print("  ✓ test_leaky_relu_basic")
    except e:
        failed += 1
        print("  ✗ test_leaky_relu_basic:", e)

    # test_leaky_relu_custom_alpha
    total += 1
    try:
        test_leaky_relu_custom_alpha()
        passed += 1
        print("  ✓ test_leaky_relu_custom_alpha")
    except e:
        failed += 1
        print("  ✗ test_leaky_relu_custom_alpha:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
