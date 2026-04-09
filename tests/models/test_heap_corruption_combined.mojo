"""Heap corruption split part 1: Conv1, Conv2, and ReLU (forward) tests.

Split from test_heap_corruption_combined.mojo (ADR-009).
Contains 8 fn test_ functions (limit: 10).
"""


from shared.tensor.any_tensor import AnyTensor
from shared.core.conv import conv2d
from shared.core.activation import relu
from shared.testing.layer_params import ConvFixture
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
)
from shared.testing.layer_testers import LayerTester
from shared.core.pooling import maxpool2d
from shared.core.linear import linear
from shared.testing.layer_params import LinearFixture
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    SPECIAL_VALUE_ONE,
)
from std.math import isnan, isinf


def create_conv1_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv1 layer parameters (1→6, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=1, out_channels=6, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


def create_conv2_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv2 layer parameters (6→16, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=6, out_channels=16, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


def create_fc1_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create FC1 layer parameters (400→120)."""
    var fixture = LinearFixture(in_features=400, out_features=120, dtype=dtype)
    return fixture.weights, fixture.bias


def create_fc2_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create FC2 layer parameters (120→84)."""
    var fixture = LinearFixture(in_features=120, out_features=84, dtype=dtype)
    return fixture.weights, fixture.bias


def create_fc3_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create FC3 layer parameters (84→10)."""
    var fixture = LinearFixture(in_features=84, out_features=10, dtype=dtype)
    return fixture.weights, fixture.bias


def test_conv1_forward_float32() raises:
    """Test Conv1 forward pass float32."""
    var dtype = DType.float32
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=28,
        input_w=28,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


def test_conv1_forward_float16() raises:
    """Test Conv1 forward pass float16."""
    var dtype = DType.float16
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=28,
        input_w=28,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


def test_conv1_backward_float32() raises:
    """Test Conv1 backward pass."""
    var dtype = DType.float32
    var _result = create_conv1_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer_backward(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


def test_conv2_forward_float32() raises:
    """Test Conv2 forward pass float32."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=14,
        input_w=14,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


def test_conv2_forward_float16() raises:
    """Test Conv2 forward pass float16."""
    var dtype = DType.float16
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=14,
        input_w=14,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


def test_conv2_backward_float32() raises:
    """Test Conv2 backward pass."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)
    var kernel = _result[0]
    var bias = _result[1]
    LayerTester.test_conv_layer_backward(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=8,
        input_w=8,
        weights=kernel,
        bias=bias,
        dtype=dtype,
    )


def test_relu_forward_float32() raises:
    """Test ReLU forward pass float32."""
    var shape: List[Int] = [1, 6, 24, 24]
    LayerTester.test_activation_layer(shape, DType.float32, activation="relu")


def test_relu_forward_float16() raises:
    """Test ReLU forward pass float16."""
    var shape: List[Int] = [1, 6, 24, 24]
    LayerTester.test_activation_layer(shape, DType.float16, activation="relu")


def test_relu_backward_float32() raises:
    """Test ReLU backward pass."""
    var shape: List[Int] = [1, 6, 8, 8]
    LayerTester.test_activation_layer_backward(
        shape, DType.float32, activation="relu"
    )


def test_maxpool1_forward_float32() raises:
    """Test MaxPool1 forward pass float32."""
    LayerTester.test_pooling_layer(
        channels=6,
        input_h=24,
        input_w=24,
        pool_size=2,
        stride=2,
        dtype=DType.float32,
        pool_type="max",
        padding=0,
    )


def test_maxpool1_forward_float16() raises:
    """Test MaxPool1 forward pass float16."""
    LayerTester.test_pooling_layer(
        channels=6,
        input_h=24,
        input_w=24,
        pool_size=2,
        stride=2,
        dtype=DType.float16,
        pool_type="max",
        padding=0,
    )


def test_maxpool2_forward_float32() raises:
    """Test MaxPool2 forward pass float32."""
    LayerTester.test_pooling_layer(
        channels=16,
        input_h=10,
        input_w=10,
        pool_size=2,
        stride=2,
        dtype=DType.float32,
        pool_type="max",
        padding=0,
    )


def test_maxpool2_forward_float16() raises:
    """Test MaxPool2 forward pass float16."""
    LayerTester.test_pooling_layer(
        channels=16,
        input_h=10,
        input_w=10,
        pool_size=2,
        stride=2,
        dtype=DType.float16,
        pool_type="max",
        padding=0,
    )


def test_fc1_forward_float32() raises:
    """Test FC1 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=400,
        out_features=120,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


def test_fc1_forward_float16() raises:
    """Test FC1 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=400,
        out_features=120,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


def test_fc1_backward_float32() raises:
    """Test FC1 backward pass."""
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=400,
        out_features=120,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


def test_fc2_forward_float32() raises:
    """Test FC2 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=120,
        out_features=84,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


def test_fc2_forward_float16() raises:
    """Test FC2 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=120,
        out_features=84,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


def test_fc2_backward_float32() raises:
    """Test FC2 backward pass."""
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=120,
        out_features=84,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


def test_fc3_forward_float32() raises:
    """Test FC3 forward pass float32."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


def test_fc3_forward_float16() raises:
    """Test FC3 forward pass float16."""
    var dtype = DType.float16
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


def test_fc3_backward_float32() raises:
    """Test FC3 backward pass."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)
    var weights = _result[0]
    var bias = _result[1]
    LayerTester.test_linear_layer_backward(
        in_features=84, out_features=10, weights=weights, bias=bias, dtype=dtype
    )


def test_flatten_operation_float32() raises:
    """Test reshape/flatten operation (16, 5, 5) -> (400,)."""
    var dtype = DType.float32
    var input = create_special_value_tensor(
        [1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE
    )
    var flattened = input.reshape([1, 400])
    assert_shape(flattened, [1, 400], "Flatten shape mismatch")
    assert_dtype(flattened, dtype, "Flatten dtype mismatch")
    for i in range(flattened.numel()):
        var val = flattened._get_float64(i)
        assert_false(isnan(val), "Flatten produced NaN")
        assert_false(isinf(val), "Flatten produced Inf")


def test_flatten_operation_float16() raises:
    """Test flatten with float16."""
    var dtype = DType.float16
    var input = create_special_value_tensor(
        [1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE
    )
    var flattened = input.reshape([1, 400])
    assert_shape(flattened, [1, 400], "Flatten shape mismatch (float16)")
    assert_dtype(flattened, dtype, "Flatten dtype mismatch (float16)")


def main() raises:
    """Run all test_heap_corruption_combined tests."""
    print("Running test_heap_corruption_combined tests...")

    test_conv1_forward_float32()
    print("✓ test_conv1_forward_float32")

    test_conv1_forward_float16()
    print("✓ test_conv1_forward_float16")

    test_conv1_backward_float32()
    print("✓ test_conv1_backward_float32")

    test_conv2_forward_float32()
    print("✓ test_conv2_forward_float32")

    test_conv2_forward_float16()
    print("✓ test_conv2_forward_float16")

    test_conv2_backward_float32()
    print("✓ test_conv2_backward_float32")

    test_relu_forward_float32()
    print("✓ test_relu_forward_float32")

    test_relu_forward_float16()
    print("✓ test_relu_forward_float16")

    test_relu_backward_float32()
    print("✓ test_relu_backward_float32")

    test_maxpool1_forward_float32()
    print("✓ test_maxpool1_forward_float32")

    test_maxpool1_forward_float16()
    print("✓ test_maxpool1_forward_float16")

    test_maxpool2_forward_float32()
    print("✓ test_maxpool2_forward_float32")

    test_maxpool2_forward_float16()
    print("✓ test_maxpool2_forward_float16")

    test_fc1_forward_float32()
    print("✓ test_fc1_forward_float32")

    test_fc1_forward_float16()
    print("✓ test_fc1_forward_float16")

    test_fc1_backward_float32()
    print("✓ test_fc1_backward_float32")

    test_fc2_forward_float32()
    print("✓ test_fc2_forward_float32")

    test_fc2_forward_float16()
    print("✓ test_fc2_forward_float16")

    test_fc2_backward_float32()
    print("✓ test_fc2_backward_float32")

    test_fc3_forward_float32()
    print("✓ test_fc3_forward_float32")

    test_fc3_forward_float16()
    print("✓ test_fc3_forward_float16")

    test_fc3_backward_float32()
    print("✓ test_fc3_backward_float32")

    test_flatten_operation_float32()
    print("✓ test_flatten_operation_float32")

    test_flatten_operation_float16()
    print("✓ test_flatten_operation_float16")

    print("\nAll test_heap_corruption_combined tests passed!")
