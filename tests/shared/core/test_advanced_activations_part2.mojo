# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_advanced_activations.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for advanced activation functions - Part 2: Mish and ELU forward.

Tests cover:
- Mish activation forward pass
- Mish activation backward pass
- ELU activation forward pass (shapes and values)

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
    mish,
    mish_backward,
    elu,
)
from shared.core.elementwise import exp
from shared.core.arithmetic import add, multiply
from math import sqrt


# ============================================================================
# Mish Tests
# ============================================================================


fn test_mish_shapes() raises:
    """Test that mish returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var output = mish(x)

    # Check shape
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


fn test_mish_values() raises:
    """Test that mish computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Test values: [-1, 0, 1]
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 1.0

    var output = mish(x)

    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # mish(-1) = -1 * tanh(ln(1 + exp(-1))) = -1 * tanh(0.313) ≈ -1 * 0.303 ≈ -0.303
    # mish(0) = 0 * tanh(ln(1 + 1)) = 0 * tanh(0.693) ≈ 0
    # mish(1) = 1 * tanh(ln(1 + exp(1))) = 1 * tanh(1.313) ≈ 1 * 0.866 ≈ 0.866

    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(-0.303), tolerance=0.01
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[2], Float32(0.866), tolerance=0.01
    )


fn test_mish_backward_shapes() raises:
    """Test that mish_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(5)
    var x = ones(shape, DType.float32)

    var output = mish(x)
    var grad_output = ones(shape, DType.float32)
    var grad_input = mish_backward(grad_output, x)

    # Check shape
    assert_equal(grad_input.shape()[0], 3)
    assert_equal(grad_input.shape()[1], 5)


fn test_mish_backward_positive() raises:
    """Test that mish backward gradient is positive for positive inputs."""
    var shape = List[Int]()
    shape.append(1)
    var x = ones(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0

    var grad_output = ones(shape, DType.float32)
    var grad_input = mish_backward(grad_output, x)

    # For positive x, mish gradient should be positive
    assert_true(grad_input._data.bitcast[Float32]()[0] > 0.0)


# ============================================================================
# ELU Forward Tests
# ============================================================================


fn test_elu_shapes() raises:
    """Test that elu returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var output = elu(x, alpha=1.0)

    # Check shape
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


fn test_elu_positive_values() raises:
    """Test that elu passes through positive values unchanged."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Test positive values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 2.0

    var output = elu(x, alpha=1.0)

    # For x > 0: elu(x) = x
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_elu_negative_values() raises:
    """Test that elu applies exponential to negative values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Test negative values
    x._data.bitcast[Float32]()[0] = -1.0
    x._data.bitcast[Float32]()[1] = -0.5
    x._data.bitcast[Float32]()[2] = -2.0

    var output = elu(x, alpha=1.0)

    # For x <= 0: elu(x) = alpha * (exp(x) - 1)
    # elu(-1) = 1.0 * (exp(-1) - 1) = 0.368 - 1 = -0.632
    # elu(-0.5) = 1.0 * (exp(-0.5) - 1) = 0.607 - 1 = -0.393
    # elu(-2) = 1.0 * (exp(-2) - 1) = 0.135 - 1 = -0.865

    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(-0.632), tolerance=0.01
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[1], Float32(-0.393), tolerance=0.01
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[2], Float32(-0.865), tolerance=0.01
    )


fn test_elu_alpha_parameter() raises:
    """Test that elu alpha parameter works correctly."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = -1.0

    # Test with alpha = 2.0
    var output = elu(x, alpha=2.0)

    # elu(-1, alpha=2.0) = 2.0 * (exp(-1) - 1) = 2.0 * -0.632 = -1.264
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(-1.264), tolerance=0.01
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run Mish and ELU forward advanced activation tests."""
    print("Running advanced activation tests (Part 2: Mish and ELU forward)...")

    # Mish tests
    test_mish_shapes()
    print("✓ test_mish_shapes")

    test_mish_values()
    print("✓ test_mish_values")

    test_mish_backward_shapes()
    print("✓ test_mish_backward_shapes")

    test_mish_backward_positive()
    print("✓ test_mish_backward_positive")

    # ELU forward tests
    test_elu_shapes()
    print("✓ test_elu_shapes")

    test_elu_positive_values()
    print("✓ test_elu_positive_values")

    test_elu_negative_values()
    print("✓ test_elu_negative_values")

    test_elu_alpha_parameter()
    print("✓ test_elu_alpha_parameter")

    print("\nAll Mish and ELU forward tests passed!")
