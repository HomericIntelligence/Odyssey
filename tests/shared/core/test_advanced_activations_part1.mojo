# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_advanced_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for advanced activation functions - Part 1: Swish/SiLU.

Tests cover:
- Swish/SiLU activation forward pass
- Swish/SiLU activation backward pass

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import ExTensor, zeros, ones
from shared.core.activation import (
    swish,
    swish_backward,
)
from shared.core.elementwise import exp
from shared.core.arithmetic import add, multiply
from math import sqrt


# ============================================================================
# Swish/SiLU Tests
# ============================================================================


fn test_swish_shapes() raises:
    """Test that swish returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var output = swish(x)

    # Check shape
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


fn test_swish_values() raises:
    """Test that swish computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    # Test values: [-2, -1, 0, 1, 2]
    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 2.0

    var output = swish(x)

    # swish(x) = x * sigmoid(x)
    # swish(-2) = -2 * sigmoid(-2) = -2 * 0.119 ≈ -0.238
    # swish(-1) = -1 * sigmoid(-1) = -1 * 0.269 ≈ -0.269
    # swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0.0
    # swish(1) = 1 * sigmoid(1) = 1 * 0.731 ≈ 0.731
    # swish(2) = 2 * sigmoid(2) = 2 * 0.881 ≈ 1.762

    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(-0.238), tolerance=0.01
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[1], Float32(-0.269), tolerance=0.01
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[3], Float32(0.731), tolerance=0.01
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[4], Float32(1.762), tolerance=0.01
    )


fn test_swish_backward_shapes() raises:
    """Test that swish_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(5)
    var x = ones(shape, DType.float32)

    var output = swish(x)
    var grad_output = ones(shape, DType.float32)
    var grad_input = swish_backward(grad_output, x)

    # Check shape
    assert_equal(grad_input.shape()[0], 3)
    assert_equal(grad_input.shape()[1], 5)


fn test_swish_backward_zero() raises:
    """Test swish backward at x=0."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    var grad_output = ones(shape, DType.float32)
    var grad_input = swish_backward(grad_output, x)

    # At x=0: sigmoid(0) = 0.5
    # d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    # = 0.5 + 0 * 0.5 * 0.5 = 0.5
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.5), tolerance=1e-5
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run Swish/SiLU advanced activation tests."""
    print("Running advanced activation tests (Part 1: Swish/SiLU)...")

    # Swish tests
    test_swish_shapes()
    print("✓ test_swish_shapes")

    test_swish_values()
    print("✓ test_swish_values")

    test_swish_backward_shapes()
    print("✓ test_swish_backward_shapes")

    test_swish_backward_zero()
    print("✓ test_swish_backward_zero")

    print("\nAll Swish/SiLU tests passed!")
