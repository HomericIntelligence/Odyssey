# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_advanced_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for advanced activation functions - Part 3: ELU backward pass.

Tests cover:
- ELU activation continuity at zero
- ELU backward pass correctness

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
from shared.core.any_tensor import AnyTensor, zeros, ones
from shared.core.activation import (
    elu,
    elu_backward,
)
from shared.core.elementwise import exp
from shared.core.arithmetic import add, multiply
from math import sqrt


# ============================================================================
# ELU Continuity and Backward Tests
# ============================================================================


fn test_elu_at_zero() raises:
    """Test that elu is continuous at zero."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 0.0

    var output = elu(x, alpha=1.0)

    # At x=0: elu(0) = 0
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


fn test_elu_backward_shapes() raises:
    """Test that elu_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(5)
    var x = ones(shape, DType.float32)

    var output = elu(x, alpha=1.0)
    var grad_output = ones(shape, DType.float32)
    var grad_input = elu_backward(grad_output, x, alpha=1.0)

    # Check shape
    assert_equal(grad_input.shape()[0], 3)
    assert_equal(grad_input.shape()[1], 5)


fn test_elu_backward_positive() raises:
    """Test elu backward gradient for positive inputs."""
    var shape = List[Int]()
    shape.append(1)
    var x = ones(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0

    var grad_output = ones(shape, DType.float32)
    var grad_input = elu_backward(grad_output, x, alpha=1.0)

    # For x > 0: d/dx[elu(x)] = 1
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )


fn test_elu_backward_negative() raises:
    """Test elu backward gradient for negative inputs."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0

    var grad_output = ones(shape, DType.float32)
    var grad_input = elu_backward(grad_output, x, alpha=1.0)

    # For x < 0: d/dx[elu(x)] = alpha * exp(x)
    # d/dx[elu(-1)] = 1.0 * exp(-1) = 0.368
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.368), tolerance=0.01
    )


fn test_elu_backward_at_zero() raises:
    """Test elu backward gradient at zero."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 0.0

    var grad_output = ones(shape, DType.float32)
    var grad_input = elu_backward(grad_output, x, alpha=1.0)

    # At x=0: d/dx[elu(x)] = 1
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run ELU backward advanced activation tests."""
    print("Running advanced activation tests (Part 3: ELU backward)...")

    # ELU continuity and backward tests
    test_elu_at_zero()
    print("✓ test_elu_at_zero")

    test_elu_backward_shapes()
    print("✓ test_elu_backward_shapes")

    test_elu_backward_positive()
    print("✓ test_elu_backward_positive")

    test_elu_backward_negative()
    print("✓ test_elu_backward_negative")

    test_elu_backward_at_zero()
    print("✓ test_elu_backward_at_zero")

    print("\nAll ELU backward tests passed!")
