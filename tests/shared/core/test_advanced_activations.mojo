"""Tests for advanced activation functions.

Tests cover:
- Swish/SiLU activation forward pass
- Swish/SiLU activation backward pass

All tests use pure functional API.
"""


from tests.shared.conftest import (
    TestFixtures,
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from shared.core.activation import (
    elu,
    elu_backward,
    mish,
    mish_backward,
    swish,
    swish_backward,
)
from shared.core.elementwise import exp
from shared.core.arithmetic import add, multiply
from std.math import sqrt


def test_swish_shapes() raises:
    """Test that swish returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var output = swish(x)

    # Check shape
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


def test_swish_values() raises:
    """Test that swish computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    # Test values: [-2, -1, 0, 1, 2]
    x.set(0, Float32(-2.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(2.0))

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


def test_swish_backward_shapes() raises:
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


def test_swish_backward_zero() raises:
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


def test_mish_shapes() raises:
    """Test that mish returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var output = mish(x)

    # Check shape
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


def test_mish_values() raises:
    """Test that mish computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Test values: [-1, 0, 1]
    x.set(0, Float32(-1.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(1.0))

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


def test_mish_backward_shapes() raises:
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


def test_mish_backward_positive() raises:
    """Test that mish backward gradient is positive for positive inputs."""
    var shape = List[Int]()
    shape.append(1)
    var x = ones(shape, DType.float32)
    x.set(0, Float32(1.0))

    var grad_output = ones(shape, DType.float32)
    var grad_input = mish_backward(grad_output, x)

    # For positive x, mish gradient should be positive
    assert_true(grad_input._data.bitcast[Float32]()[0] > 0.0)


def test_elu_shapes() raises:
    """Test that elu returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var output = elu(x, alpha=1.0)

    # Check shape
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


def test_elu_positive_values() raises:
    """Test that elu passes through positive values unchanged."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Test positive values
    x.set(0, Float32(0.5))
    x.set(1, Float32(1.0))
    x.set(2, Float32(2.0))

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


def test_elu_negative_values() raises:
    """Test that elu applies exponential to negative values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Test negative values
    x.set(0, Float32(-1.0))
    x.set(1, Float32(-0.5))
    x.set(2, Float32(-2.0))

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


def test_elu_alpha_parameter() raises:
    """Test that elu alpha parameter works correctly."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-1.0))

    # Test with alpha = 2.0
    var output = elu(x, alpha=2.0)

    # elu(-1, alpha=2.0) = 2.0 * (exp(-1) - 1) = 2.0 * -0.632 = -1.264
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(-1.264), tolerance=0.01
    )


def test_elu_at_zero() raises:
    """Test that elu is continuous at zero."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(0.0))

    var output = elu(x, alpha=1.0)

    # At x=0: elu(0) = 0
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


def test_elu_backward_shapes() raises:
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


def test_elu_backward_positive() raises:
    """Test elu backward gradient for positive inputs."""
    var shape = List[Int]()
    shape.append(1)
    var x = ones(shape, DType.float32)
    x.set(0, Float32(1.0))

    var grad_output = ones(shape, DType.float32)
    var grad_input = elu_backward(grad_output, x, alpha=1.0)

    # For x > 0: d/dx[elu(x)] = 1
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )


def test_elu_backward_negative() raises:
    """Test elu backward gradient for negative inputs."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-1.0))

    var grad_output = ones(shape, DType.float32)
    var grad_input = elu_backward(grad_output, x, alpha=1.0)

    # For x < 0: d/dx[elu(x)] = alpha * exp(x)
    # d/dx[elu(-1)] = 1.0 * exp(-1) = 0.368
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.368), tolerance=0.01
    )


def test_elu_backward_at_zero() raises:
    """Test elu backward gradient at zero."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(0.0))

    var grad_output = ones(shape, DType.float32)
    var grad_input = elu_backward(grad_output, x, alpha=1.0)

    # At x=0: d/dx[elu(x)] = 1
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )


def main() raises:
    """Run all test_advanced_activations tests."""
    print("Running test_advanced_activations tests...")

    test_swish_shapes()
    print("✓ test_swish_shapes")

    test_swish_values()
    print("✓ test_swish_values")

    test_swish_backward_shapes()
    print("✓ test_swish_backward_shapes")

    test_swish_backward_zero()
    print("✓ test_swish_backward_zero")

    test_mish_shapes()
    print("✓ test_mish_shapes")

    test_mish_values()
    print("✓ test_mish_values")

    test_mish_backward_shapes()
    print("✓ test_mish_backward_shapes")

    test_mish_backward_positive()
    print("✓ test_mish_backward_positive")

    test_elu_shapes()
    print("✓ test_elu_shapes")

    test_elu_positive_values()
    print("✓ test_elu_positive_values")

    test_elu_negative_values()
    print("✓ test_elu_negative_values")

    test_elu_alpha_parameter()
    print("✓ test_elu_alpha_parameter")

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

    print("\nAll test_advanced_activations tests passed!")
