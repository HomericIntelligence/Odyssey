"""Tests for activation functions

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
from shared.tensor.any_tensor import (
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
    NumericalForward,
    NumericalBackward,
)
from tests.shared.conftest import (
    assert_almost_equal,
    assert_true,
)
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    full,
    ones_like,
)
from shared.core.activation import (
    leaky_relu,
    prelu,
    sigmoid,
    leaky_relu_backward,
    prelu_backward,
    sigmoid_backward,
)
from shared.tensor.any_tensor import (
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
from std.math import tanh as math_tanh
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    ones_like,
)
from shared.core.activation import (
    softmax,
    gelu,
    softmax_backward,
)
from shared.core.activation import (
    gelu,
    swish,
    mish,
    gelu_backward,
    swish_backward,
)
from shared.core.activation import (
    relu,
    sigmoid,
    mish,
    elu,
    relu_backward,
    sigmoid_backward,
    mish_backward,
    elu_backward,
)


def test_relu_basic() raises:
    """Test ReLU with known values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    # Set test values: [-2, -1, 0, 1, 2]
    x.set(0, Float32(-2.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(2.0))

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


def test_relu_non_negativity() raises:
    """Test ReLU always produces non-negative outputs."""
    var shape = List[Int]()
    shape.append(100)
    var x = zeros(shape, DType.float32)

    # Fill with values from -50 to 50
    for i in range(100):
        x.set(i, Float32(i - 50))

    var y = relu(x)

    # All outputs should be >= 0
    for i in range(100):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val >= 0.0)


@fieldwise_init
struct _ReluFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return relu(x)


@fieldwise_init
struct _ReluBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return relu_backward(grad, x)


def test_relu_backward() raises:
    """Test ReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)

    # Set test values: [-1, 1e-4, 0.5, 2]
    x.set(0, Float32(-1.0))
    x.set(1, Float32(1e-4))
    x.set(2, Float32(0.5))
    x.set(3, Float32(2.0))

    var y = relu(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(_ReluFwd(), _ReluBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_relu_shape() raises:
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


def test_relu_integer_types() raises:
    """Test ReLU with integer types."""
    # Test int32
    var shape = List[Int]()
    shape.append(5)
    var x_int32 = zeros(shape, DType.int32)
    x_int32.set(0, Int32(-2))
    x_int32.set(1, Int32(-1))
    x_int32.set(2, Int32(0))
    x_int32.set(3, Int32(1))
    x_int32.set(4, Int32(2))

    var y_int32 = relu(x_int32)

    # Expected: [0, 0, 0, 1, 2]
    assert_equal(y_int32._data.bitcast[Int32]()[0], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[1], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[2], 0)
    assert_equal(y_int32._data.bitcast[Int32]()[3], 1)
    assert_equal(y_int32._data.bitcast[Int32]()[4], 2)

    # Test uint8 (already non-negative)
    var x_uint8 = zeros(shape, DType.uint8)
    x_uint8.set(0, UInt8(0))
    x_uint8.set(1, UInt8(1))
    x_uint8.set(2, UInt8(128))
    x_uint8.set(3, UInt8(255))
    x_uint8.set(4, UInt8(100))

    var y_uint8 = relu(x_uint8)

    # Should be unchanged
    assert_equal(y_uint8._data.bitcast[UInt8]()[0], 0)
    assert_equal(y_uint8._data.bitcast[UInt8]()[1], 1)
    assert_equal(y_uint8._data.bitcast[UInt8]()[2], 128)
    assert_equal(y_uint8._data.bitcast[UInt8]()[3], 255)


def test_relu_float64() raises:
    """Test ReLU with float64 dtype."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float64)

    x.set(0, Float64(-1.0))
    x.set(1, Float64(1.0))

    var y = relu(x)

    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.0, tolerance=1e-10)
    assert_almost_equal(y._data.bitcast[Float64]()[1], 1.0, tolerance=1e-10)


def test_leaky_relu_basic() raises:
    """Test Leaky ReLU with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-2.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(2.0))

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


def test_leaky_relu_custom_alpha() raises:
    """Test Leaky ReLU with custom alpha value."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-4.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(4.0))

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


@fieldwise_init
struct _LeakyReluFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return leaky_relu(x, alpha=0.1)


@fieldwise_init
struct _LeakyReluBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return leaky_relu_backward(grad, x, alpha=0.1)


def test_leaky_relu_backward() raises:
    """Test Leaky ReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-1.0))
    x.set(1, Float32(1.0))

    var y = leaky_relu(x, alpha=0.1)
    var grad_out = ones_like(y)

    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(_LeakyReluFwd(), _LeakyReluBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_prelu_basic() raises:
    """Test PReLU with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    var alpha = zeros(shape, DType.float32)

    x.set(0, Float32(-2.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(2.0))

    alpha.set(0, Float32(0.25))
    alpha.set(1, Float32(0.25))
    alpha.set(2, Float32(0.25))

    var y = prelu(x, alpha)

    # Expected: [-0.5, 0, 2.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.5), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


def test_prelu_scalar_alpha() raises:
    """Test PReLU with scalar alpha parameter."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-2.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(2.0))

    # Scalar alpha = 0.2
    var alpha_shape = List[Int]()
    alpha_shape.append(1)
    var alpha = full(alpha_shape, 0.2, DType.float32)

    var y = prelu(x, alpha)

    # Expected with alpha=0.2: [-0.4, -0.2, 0, 1, 2]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.4), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(-0.2), tolerance=1e-6
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


def test_prelu_elementwise_alpha() raises:
    """Test PReLU with element-wise alpha parameters."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-2.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(2.0))

    # Element-wise alpha = [0.1, 0.2, 0.3]
    var alpha = zeros(shape, DType.float32)
    alpha.set(0, Float32(0.1))
    alpha.set(1, Float32(0.2))
    alpha.set(2, Float32(0.3))

    var y = prelu(x, alpha)

    # Expected: [-0.2, -0.2, 2.0]
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.2), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(-0.2), tolerance=1e-6
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


@fieldwise_init
struct _PreluFwd(NumericalForward):
    var alpha: AnyTensor

    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return prelu(x, self.alpha)


@fieldwise_init
struct _PreluBwd(NumericalBackward):
    var alpha: AnyTensor

    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var result = prelu_backward(grad, x, self.alpha)
        return result.grad_a


def test_prelu_backward() raises:
    """Test PReLU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var alpha = zeros(shape, DType.float32)

    x.set(0, Float32(-1.0))
    x.set(1, Float32(1.0))

    alpha.set(0, Float32(0.5))
    alpha.set(1, Float32(0.5))

    var y = prelu(x, alpha)
    var grad_out = ones_like(y)

    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(_PreluFwd(alpha), _PreluBwd(alpha), x, grad_out, rtol=1e-3, atol=1e-6)


def test_sigmoid_basic() raises:
    """Test sigmoid with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-100.0))  # Should be ~0
    x.set(1, Float32(0.0))  # Should be 0.5
    x.set(2, Float32(100.0))  # Should be ~1

    var y = sigmoid(x)

    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-3
    )


@fieldwise_init
struct _SigmoidFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return sigmoid(x)


@fieldwise_init
struct _SigmoidBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var out = sigmoid(x)
        return sigmoid_backward(grad, out)


def test_sigmoid_backward() raises:
    """Test sigmoid gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Use multiple test points for better coverage
    x.set(0, Float32(-1.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(1.0))

    var y = sigmoid(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(_SigmoidFwd(), _SigmoidBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_sigmoid_range() raises:
    """Test sigmoid output is in (0, 1)."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-10.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(10.0))

    var y = sigmoid(x)

    # All values should be in (0, 1)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val > 0.0)
        assert_true(val < 1.0)


def test_sigmoid_numerical_stability() raises:
    """Test sigmoid with extreme values."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-100.0))
    x.set(1, Float32(-20.0))
    x.set(2, Float32(20.0))
    x.set(3, Float32(100.0))

    var y = sigmoid(x)

    # Large negative values should be close to 0
    assert_true(y._data.bitcast[Float32]()[0] < 1e-6)
    assert_true(y._data.bitcast[Float32]()[1] < 1e-6)

    # Large positive values should be close to 1
    assert_true(y._data.bitcast[Float32]()[2] > 0.999999)
    assert_true(y._data.bitcast[Float32]()[3] > 0.999999)


def test_sigmoid_float16() raises:
    """Test sigmoid with float16."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float16)
    x.set(0, Float16(-1.0))
    x.set(1, Float16(0.0))
    x.set(2, Float16(1.0))

    var y = sigmoid(x)

    # Check sigmoid(0) = 0.5
    var val_0 = Float32(y._data.bitcast[Float16]()[1])
    assert_almost_equal(val_0, Float32(0.5), tolerance=0.01)

    # Check range (0, 1)
    for i in range(3):
        var val = Float32(y._data.bitcast[Float16]()[i])
        assert_true(val > 0.0 and val < 1.0)


def test_sigmoid_float64() raises:
    """Test sigmoid with float64 dtype."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float64)

    x.set(0, Float64(0.0))

    var y = sigmoid(x)

    assert_almost_equal(y._data.bitcast[Float64]()[0], 0.5, tolerance=1e-10)


def test_tanh_basic() raises:
    """Test tanh with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-100.0))  # Should be ~-1
    x.set(1, Float32(0.0))  # Should be 0
    x.set(2, Float32(100.0))  # Should be ~1

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


def test_tanh_values() raises:
    """Test tanh against known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float64)
    x.set(0, Float64(0.0))
    x.set(1, Float64(1.0))
    x.set(2, Float64(-1.0))

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


@fieldwise_init
struct _TanhFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return tanh(x)


@fieldwise_init
struct _TanhBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var out = tanh(x)
        return tanh_backward(grad, out)


def test_tanh_backward() raises:
    """Test tanh gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Use multiple test points for better coverage
    x.set(0, Float32(-1.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(1.0))

    var y = tanh(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(_TanhFwd(), _TanhBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_tanh_range() raises:
    """Test tanh output is in (-1, 1)."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-10.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(10.0))

    var y = tanh(x)

    # All values should be in [-1, 1] (inclusive due to floating point precision)
    for i in range(5):
        var val = y._data.bitcast[Float32]()[i]
        assert_true(val >= -1.0, "tanh output should be >= -1.0")
        assert_true(val <= 1.0, "tanh output should be <= 1.0")


def test_softmax_basic_2d() raises:
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


def test_softmax_one_hot() raises:
    """Test softmax with large difference (one-hot-like)."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))
    x.set(1, Float32(10.0))
    x.set(2, Float32(0.0))

    var y = softmax(x, axis=1)

    # Middle value should be ~1.0, others ~0.0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-3
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-3
    )


def test_softmax_sum_to_one() raises:
    """Test softmax probabilities sum to 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(4)
    var x = zeros(shape, DType.float32)

    # Set random values
    for i in range(8):
        x.set(i, Float32(i % 5) - 2.0)

    var y = softmax(x, axis=1)

    # Each row should sum to 1.0
    var sum_row0 = Float32(0.0)
    var sum_row1 = Float32(0.0)
    for i in range(4):
        sum_row0 += y._data.bitcast[Float32]()[i]
        sum_row1 += y._data.bitcast[Float32]()[4 + i]

    assert_almost_equal(sum_row0, Float32(1.0), tolerance=1e-5)
    assert_almost_equal(sum_row1, Float32(1.0), tolerance=1e-5)


def test_softmax_numerical_stability() raises:
    """Test softmax with large values (numerical stability)."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(1000.0))
    x.set(1, Float32(1001.0))
    x.set(2, Float32(1002.0))

    var y = softmax(x, axis=-1)

    # Should still sum to 1 (no overflow)
    var sum: Float32 = 0.0
    for i in range(3):
        sum += y._data.bitcast[Float32]()[i]

    assert_almost_equal(sum, Float32(1.0), tolerance=1e-5)

    # Largest value should have largest probability
    assert_true(y._data.bitcast[Float32]()[2] > y._data.bitcast[Float32]()[1])
    assert_true(y._data.bitcast[Float32]()[1] > y._data.bitcast[Float32]()[0])


@fieldwise_init
struct _SoftmaxFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return softmax(x, axis=1)


@fieldwise_init
struct _SoftmaxBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        var out = softmax(x, axis=1)
        return softmax_backward(grad, out, axis=1)


def test_softmax_backward() raises:
    """Test softmax gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set test values
    x.set(0, Float32(-1.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(1.0))
    x.set(3, Float32(-0.5))
    x.set(4, Float32(0.5))
    x.set(5, Float32(1.5))

    var y = softmax(x, axis=1)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3, atol=5e-4 is needed for float32 softmax gradients
    # Softmax involves exp() and division which amplify numerical errors,
    # especially at the edges of the distribution
    check_gradient(_SoftmaxFwd(), _SoftmaxBwd(), x, grad_out, rtol=1e-3, atol=5e-4)


def test_gelu_basic() raises:
    """Test GELU with known value at x=0."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))

    var y = gelu(x)

    # GELU(0) = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


def test_gelu_positive() raises:
    """Test GELU with positive values."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(1.0))
    x.set(1, Float32(2.0))

    var y = gelu(x)

    # For positive x, GELU(x) ≈ x (asymptotically)
    # GELU(1) ≈ 0.84, GELU(2) ≈ 1.96
    assert_true(y._data.bitcast[Float32]()[0] > 0.8)
    assert_true(y._data.bitcast[Float32]()[0] < 1.0)
    assert_true(y._data.bitcast[Float32]()[1] > 1.9)
    assert_true(y._data.bitcast[Float32]()[1] < 2.0)


def test_gelu_shape() raises:
    """Test GELU preserves shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var y = gelu(x)

    assert_equal(y.shape()[0], 3)
    assert_equal(y.shape()[1], 4)


def test_gelu_approximate() raises:
    """Test GELU with tanh approximation."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-2.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(2.0))

    var y = gelu(x, approximate=True)

    # GELU(0) should be 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )

    # GELU is NOT symmetric (unlike relu). For x < 0, GELU(x) is close to 0.
    # For x > 0, GELU(x) is close to x.
    var val_neg2 = y._data.bitcast[Float32]()[0]  # GELU(-2.0) ≈ -0.045
    var val_pos2 = y._data.bitcast[Float32]()[4]  # GELU(2.0) ≈ 1.954

    # For large positive x, GELU(x) ≈ x
    assert_true(val_pos2 > 1.9, "GELU(2.0) should be close to 2.0")

    # For large negative x, GELU(x) ≈ 0 (small negative value)
    assert_true(abs(val_neg2) < 0.1, "GELU(-2.0) should be close to 0")


def test_gelu_exact() raises:
    """Test GELU with exact erf implementation."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-2.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(2.0))

    var y = gelu(x, approximate=False)

    # GELU(0) = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )

    # For large positive x, GELU(x) ≈ x
    assert_true(y._data.bitcast[Float32]()[4] > 1.9)

    # For large negative x, GELU(x) ≈ 0
    assert_true(abs(y._data.bitcast[Float32]()[0]) < 0.1)


def test_gelu_comparison() raises:
    """Compare approximate and exact GELU implementations."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-2.0))
    x.set(1, Float32(-1.0))
    x.set(2, Float32(0.0))
    x.set(3, Float32(1.0))
    x.set(4, Float32(2.0))

    var y_approx = gelu(x, approximate=True)
    var y_exact = gelu(x, approximate=False)

    # Approximate and exact should be close
    for i in range(5):
        var approx_val = y_approx._data.bitcast[Float32]()[i]
        var exact_val = y_exact._data.bitcast[Float32]()[i]
        var diff = abs(approx_val - exact_val)

        # Approximation error should be small (< 1%)
        if abs(exact_val) > 0.01:
            var rel_error = diff / abs(exact_val)
            assert_true(rel_error < 0.01)


def test_gelu_float16() raises:
    """Test GELU with float16."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float16)
    x.set(0, Float16(-1.0))
    x.set(1, Float16(0.0))
    x.set(2, Float16(1.0))

    var y = gelu(x, approximate=True)

    # GELU(0) should be 0
    var val_0 = Float32(y._data.bitcast[Float16]()[1])
    assert_almost_equal(val_0, Float32(0.0), tolerance=0.01)


@fieldwise_init
struct _GeluFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return gelu(x, approximate=False)


@fieldwise_init
struct _GeluBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return gelu_backward(grad, x, approximate=False)


def test_gelu_backward_gradient() raises:
    """Test GELU backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.0))
    x.set(2, Float32(0.5))

    var y = gelu(x, approximate=False)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_GeluFwd(), _GeluBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_swish_basic() raises:
    """Test swish with known values."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))

    var y = swish(x)

    # swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


def test_swish_positive() raises:
    """Test swish with large positive value."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(10.0))

    var y = swish(x)

    # swish(10) ≈ 10 * sigmoid(10) ≈ 10 * 1 ≈ 10
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(10.0), tolerance=0.01
    )


@fieldwise_init
struct _SwishFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return swish(x)


@fieldwise_init
struct _SwishBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return swish_backward(grad, x)


def test_swish_backward_gradient() raises:
    """Test Swish backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.0))
    x.set(2, Float32(0.5))

    var y = swish(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_SwishFwd(), _SwishBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_mish_basic() raises:
    """Test mish with known values."""
    var shape = List[Int]()
    shape.append(1)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(0.0))

    var y = mish(x)

    # mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(log(2)) ≈ 0
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.01
    )


def test_mish_shape() raises:
    """Test mish preserves shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var y = mish(x)

    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)
    assert_equal(y.shape()[2], 4)


@fieldwise_init
struct _MishFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return mish(x)


@fieldwise_init
struct _MishBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return mish_backward(grad, x)


def test_mish_backward_gradient() raises:
    """Test Mish backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.0))
    x.set(2, Float32(0.5))

    var y = mish(x)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    check_gradient(_MishFwd(), _MishBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_elu_basic() raises:
    """Test ELU with known values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-1.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(1.0))

    var y = elu(x, alpha=1.0)

    # ELU(-1) = 1.0 * (exp(-1) - 1) ≈ -0.632
    # ELU(0) = 0
    # ELU(1) = 1
    assert_almost_equal(
        y._data.bitcast[Float32]()[0], Float32(-0.632), tolerance=0.01
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        y._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )


@fieldwise_init
struct _EluFwd(NumericalForward):
    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return elu(x, alpha=1.0)


@fieldwise_init
struct _EluBwd(NumericalBackward):
    def __call__(self, grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return elu_backward(grad, x, alpha=1.0)


def test_elu_backward() raises:
    """Test ELU gradient with numerical validation."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x.set(0, Float32(-1.0))
    x.set(1, Float32(0.0))
    x.set(2, Float32(1.0))

    var y = elu(x, alpha=1.0)
    var grad_out = ones_like(y)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=1e-3 is appropriate for float32 finite differences
    check_gradient(_EluFwd(), _EluBwd(), x, grad_out, rtol=1e-3, atol=1e-6)


def test_integration_forward_backward() raises:
    """Integration test: Complete forward and backward pass through activations.

    Simulates a simple neural network layer with:
    - Input -> ReLU -> Sigmoid -> Output
    - Loss gradient flows back through the network.
    """
    # Input data
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x.set(0, Float32(-1.0))
    x.set(1, Float32(0.5))
    x.set(2, Float32(2.0))

    # Forward pass: x -> ReLU -> Sigmoid
    var relu_out = relu(x)
    var sigmoid_out = sigmoid(relu_out)

    # Check forward pass values
    # After ReLU: [0, 0.5, 2.0]
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.001
    )
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[1], Float32(0.5), tolerance=0.001
    )
    assert_almost_equal(
        relu_out._data.bitcast[Float32]()[2], Float32(2.0), tolerance=0.001
    )

    # After Sigmoid: [0.5, sigmoid(0.5), sigmoid(2.0)]
    var sig_0_5 = sigmoid_out._data.bitcast[Float32]()[1]
    var sig_2_0 = sigmoid_out._data.bitcast[Float32]()[2]
    assert_true(sig_0_5 > 0.6 and sig_0_5 < 0.7)
    assert_true(sig_2_0 > 0.8 and sig_2_0 < 0.9)

    # Simulate loss gradient (all ones)
    var grad_loss = ones(shape, DType.float32)

    # Backward pass: Sigmoid <- ReLU <- x
    var grad_sigmoid = sigmoid_backward(grad_loss, sigmoid_out)
    var grad_x = relu_backward(grad_sigmoid, x)

    # Check backward pass values
    # Gradient through ReLU should be 0 at x=-1 (negative input)
    assert_almost_equal(
        grad_x._data.bitcast[Float32]()[0], Float32(0.0), tolerance=0.001
    )

    # Gradients at positive inputs should be non-zero
    assert_true(grad_x._data.bitcast[Float32]()[1] > 0.0)
    assert_true(grad_x._data.bitcast[Float32]()[2] > 0.0)


def main() raises:
    """Run all test_activations tests."""
    print("Running test_activations tests...")

    test_relu_basic()
    print("✓ test_relu_basic")

    test_relu_non_negativity()
    print("✓ test_relu_non_negativity")

    test_relu_backward()
    print("✓ test_relu_backward")

    test_relu_shape()
    print("✓ test_relu_shape")

    test_relu_integer_types()
    print("✓ test_relu_integer_types")

    test_relu_float64()
    print("✓ test_relu_float64")

    test_leaky_relu_basic()
    print("✓ test_leaky_relu_basic")

    test_leaky_relu_custom_alpha()
    print("✓ test_leaky_relu_custom_alpha")

    test_leaky_relu_backward()
    print("✓ test_leaky_relu_backward")

    test_prelu_basic()
    print("✓ test_prelu_basic")

    test_prelu_scalar_alpha()
    print("✓ test_prelu_scalar_alpha")

    test_prelu_elementwise_alpha()
    print("✓ test_prelu_elementwise_alpha")

    test_prelu_backward()
    print("✓ test_prelu_backward")

    test_sigmoid_basic()
    print("✓ test_sigmoid_basic")

    test_sigmoid_backward()
    print("✓ test_sigmoid_backward")

    test_sigmoid_range()
    print("✓ test_sigmoid_range")

    test_sigmoid_numerical_stability()
    print("✓ test_sigmoid_numerical_stability")

    test_sigmoid_float16()
    print("✓ test_sigmoid_float16")

    test_sigmoid_float64()
    print("✓ test_sigmoid_float64")

    test_tanh_basic()
    print("✓ test_tanh_basic")

    test_tanh_values()
    print("✓ test_tanh_values")

    test_tanh_backward()
    print("✓ test_tanh_backward")

    test_tanh_range()
    print("✓ test_tanh_range")

    test_softmax_basic_2d()
    print("✓ test_softmax_basic_2d")

    test_softmax_one_hot()
    print("✓ test_softmax_one_hot")

    test_softmax_sum_to_one()
    print("✓ test_softmax_sum_to_one")

    test_softmax_numerical_stability()
    print("✓ test_softmax_numerical_stability")

    test_softmax_backward()
    print("✓ test_softmax_backward")

    test_gelu_basic()
    print("✓ test_gelu_basic")

    test_gelu_positive()
    print("✓ test_gelu_positive")

    test_gelu_shape()
    print("✓ test_gelu_shape")

    test_gelu_approximate()
    print("✓ test_gelu_approximate")

    test_gelu_exact()
    print("✓ test_gelu_exact")

    test_gelu_comparison()
    print("✓ test_gelu_comparison")

    test_gelu_float16()
    print("✓ test_gelu_float16")

    test_gelu_backward_gradient()
    print("✓ test_gelu_backward_gradient")

    test_swish_basic()
    print("✓ test_swish_basic")

    test_swish_positive()
    print("✓ test_swish_positive")

    test_swish_backward_gradient()
    print("✓ test_swish_backward_gradient")

    test_mish_basic()
    print("✓ test_mish_basic")

    test_mish_shape()
    print("✓ test_mish_shape")

    test_mish_backward_gradient()
    print("✓ test_mish_backward_gradient")

    test_elu_basic()
    print("✓ test_elu_basic")

    test_elu_backward()
    print("✓ test_elu_backward")

    test_integration_forward_backward()
    print("✓ test_integration_forward_backward")

    print("\nAll test_activations tests passed!")
