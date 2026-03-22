"""Unit tests for activation functions - Part 2: Tanh and Softmax (basic).

Tests cover:
- tanh: Forward and backward passes
- softmax: Basic forward and backward passes

All tests use pure functional API - no internal state.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_activation_funcs.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_true,
)
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.activation import (
    tanh,
    tanh_backward,
    softmax,
    softmax_backward,
)


# ============================================================================
# Tanh Tests
# ============================================================================


fn test_tanh_range() raises:
    """Test tanh output is in [-1, 1] range."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(1)
    input_shape.append(5)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -10.0
    input_data[1] = -1.0
    input_data[2] = 0.0
    input_data[3] = 1.0
    input_data[4] = 10.0

    var output = tanh(input)

    var output_data = output._data.bitcast[Float32]()
    for i in range(5):
        assert_greater_or_equal(output_data[i], -1.0, "Tanh lower bound")
        assert_less_or_equal(output_data[i], 1.0, "Tanh upper bound")


fn test_tanh_at_zero() raises:
    """Test tanh(0) = 0."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var input = zeros(input_shape, DType.float32)

    var output = tanh(input)

    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.0, tolerance=1e-5)


fn test_tanh_antisymmetry() raises:
    """Test tanh(-x) = -tanh(x)."""
    var input_shape = List[Int]()
    input_shape.append(1)
    var x = zeros(input_shape, DType.float32)
    var x_data = x._data.bitcast[Float32]()
    x_data[0] = 2.0

    var neg_x = zeros(input_shape, DType.float32)
    var neg_x_data = neg_x._data.bitcast[Float32]()
    neg_x_data[0] = -2.0

    var tanh_x = tanh(x)
    var tanh_neg_x = tanh(neg_x)

    var tanh_x_data = tanh_x._data.bitcast[Float32]()
    var tanh_neg_x_data = tanh_neg_x._data.bitcast[Float32]()

    assert_almost_equal(tanh_x_data[0], -tanh_neg_x_data[0], tolerance=1e-4)


fn test_tanh_backward() raises:
    """Test tanh backward pass computes correct gradients."""
    var output_shape = List[Int]()
    output_shape.append(1)
    var output = zeros(output_shape, DType.float32)
    var output_data = output._data.bitcast[Float32]()
    output_data[0] = 0.5

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = tanh_backward(grad_output, output)

    # For tanh output = 0.5, gradient = 1 * (1 - 0.5^2) = 0.75
    var grad_data = grad_input._data.bitcast[Float32]()
    assert_almost_equal(grad_data[0], 0.75, tolerance=1e-4)


# ============================================================================
# Softmax Tests (Basic)
# ============================================================================


fn test_softmax_output_sum() raises:
    """Test softmax outputs sum to 1 along last axis."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = 1.0
    input_data[1] = 2.0
    input_data[2] = 3.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    var sum_val = output_data[0] + output_data[1] + output_data[2]
    assert_almost_equal(sum_val, 1.0, tolerance=1e-5)


fn test_softmax_positive_outputs() raises:
    """Test softmax outputs are all positive."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(3)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -5.0
    input_data[1] = 0.0
    input_data[2] = 5.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    for i in range(3):
        assert_greater_or_equal(output_data[i], 0.0, "Softmax positive")


fn test_softmax_uniform() raises:
    """Test softmax on uniform input gives uniform output."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(4)
    var input = zeros(input_shape, DType.float32)

    var input_data = input._data.bitcast[Float32]()
    for i in range(4):
        input_data[i] = 5.0

    var output = softmax(input, axis=-1)

    var output_data = output._data.bitcast[Float32]()
    for i in range(4):
        assert_almost_equal(output_data[i], 0.25, tolerance=1e-5)


fn test_softmax_backward() raises:
    """Test softmax backward pass."""
    var output_shape = List[Int]()
    output_shape.append(1)
    output_shape.append(3)
    var output = zeros(output_shape, DType.float32)

    var output_data = output._data.bitcast[Float32]()
    output_data[0] = 0.1
    output_data[1] = 0.6
    output_data[2] = 0.3

    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    grad_output_shape.append(3)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = softmax_backward(grad_output, output, axis=-1)

    var grad_shape = grad_input.shape()
    assert_equal(grad_shape[0], 1)
    assert_equal(grad_shape[1], 3)


fn main() raises:
    """Run activation tests - Part 2: Tanh and Softmax (basic)."""
    print("Running activation tests - Part 2: Tanh and Softmax (basic)...")

    test_tanh_range()
    print("✓ test_tanh_range")

    test_tanh_at_zero()
    print("✓ test_tanh_at_zero")

    test_tanh_antisymmetry()
    print("✓ test_tanh_antisymmetry")

    test_tanh_backward()
    print("✓ test_tanh_backward")

    test_softmax_output_sum()
    print("✓ test_softmax_output_sum")

    test_softmax_positive_outputs()
    print("✓ test_softmax_positive_outputs")

    test_softmax_uniform()
    print("✓ test_softmax_uniform")

    test_softmax_backward()
    print("✓ test_softmax_backward")

    print("\nAll Part 2 activation tests passed!")
