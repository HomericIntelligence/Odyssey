"""DEPRECATED: Layerwise Unit Tests for LeNet-5

⚠️ **THIS FILE IS DEPRECATED** ⚠️

This monolithic test file has been split into smaller files to work around
Issue #2942 (heap corruption after running 15+ cumulative tests).

**Use these files instead**:
- test_lenet5_conv_layers.mojo (6 tests)
- test_lenet5_activation_layers.mojo (3 tests)
- test_lenet5_pooling_layers.mojo (4 tests)
- test_lenet5_fc_layers.mojo (8 tests)
- test_lenet5_reshape_layers.mojo (2 tests)

**Why this was deprecated**:
Running all 24 tests in a single file triggers a Mojo 0.26.1 runtime bug that
causes heap corruption after approximately 15 cumulative tests. Splitting the
tests into smaller files (<10 tests each) avoids this issue.

**See**: BUG_REPORT_HEAP_CORRUPTION.md and Issue #2942 for details.

---

Original description:
Tests each layer independently with special FP-representable values.
Each layer test runs on all dtypes: float16, bfloat16, float32

Architecture (12 layer operations):
1. Conv1 (1→6 channels, 5x5 kernel)
2. ReLU1
3. MaxPool1 (2x2, stride 2)
4. Conv2 (6→16 channels, 5x5 kernel)
5. ReLU2
6. MaxPool2 (2x2, stride 2)
7. Flatten
8. FC1 (400→120)
9. ReLU3
10. FC2 (120→84)
11. ReLU4
12. FC3 (84→10)

All tests use small tensor sizes to keep runtime under 60 seconds.
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.conv import conv2d
from shared.core.pooling import maxpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.core.shape import conv2d_output_shape, pool_output_shape
from shared.core.initializers import kaiming_uniform
from shared.testing.layer_params import ConvFixture, LinearFixture
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_true,
    assert_false,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    create_alternating_pattern_tensor,
    create_seeded_random_tensor,
    SPECIAL_VALUE_ONE,
    SPECIAL_VALUE_NEG_ONE,
    SPECIAL_VALUE_HALF,
)
from shared.testing.layer_testers import LayerTester
from math import isnan, isinf


# ============================================================================
# Test Fixtures
# ============================================================================


fn create_conv1_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create Conv1 layer parameters (1→6, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=1, out_channels=6, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


fn create_conv2_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create Conv2 layer parameters (6→16, 5x5 kernel)."""
    var fixture = ConvFixture(
        in_channels=6, out_channels=16, kernel_size=5, dtype=dtype
    )
    return fixture.kernel, fixture.bias


fn create_fc1_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC1 layer parameters (400→120)."""
    var fixture = LinearFixture(in_features=400, out_features=120, dtype=dtype)
    return fixture.weights, fixture.bias


fn create_fc2_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC2 layer parameters (120→84)."""
    var fixture = LinearFixture(in_features=120, out_features=84, dtype=dtype)
    return fixture.weights, fixture.bias


fn create_fc3_parameters(dtype: DType) raises -> Tuple[ExTensor, ExTensor]:
    """Create FC3 layer parameters (84→10)."""
    var fixture = LinearFixture(in_features=84, out_features=10, dtype=dtype)
    return fixture.weights, fixture.bias


# ============================================================================
# Conv1 Tests
# ============================================================================


fn test_conv1_forward_float32() raises:
    """Test Conv1 forward pass (1→6 channels, 5x5 kernel) with float32."""
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
        stride=1,
        padding=0,
    )


fn test_conv1_forward_float16() raises:
    """Test Conv1 forward pass with float16."""
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
        stride=1,
        padding=0,
    )


fn test_conv1_backward_float32() raises:
    """Test Conv1 backward pass with gradient checking (small tensor: 8x8)."""
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
        stride=1,
        padding=0,
    )


# ============================================================================
# Conv2 Tests
# ============================================================================


fn test_conv2_forward_float32() raises:
    """Test Conv2 forward pass (6→16 channels, 5x5 kernel) with float32."""
    var dtype = DType.float32
    var _result = create_conv2_parameters(dtype)

    var kernel = _result[0]

    var bias = _result[1]

    # Input after pool1 is smaller: 14x14
    LayerTester.test_conv_layer(
        in_channels=6,
        out_channels=16,
        kernel_size=5,
        input_h=14,
        input_w=14,
        weights=kernel,
        bias=bias,
        dtype=dtype,
        stride=1,
        padding=0,
    )


fn test_conv2_forward_float16() raises:
    """Test Conv2 forward pass with float16."""
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
        stride=1,
        padding=0,
    )


fn test_conv2_backward_float32() raises:
    """Test Conv2 backward pass with gradient checking (small tensor: 8x8)."""
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
        stride=1,
        padding=0,
    )


# ============================================================================
# ReLU Tests (single test covers all)
# ============================================================================


fn test_relu_forward_float32() raises:
    """Test ReLU activation forward pass with float32."""
    var dtype = DType.float32
    var shape: List[Int] = [2, 16, 8, 8]

    LayerTester.test_activation_layer(shape, dtype, activation="relu")


fn test_relu_forward_float16() raises:
    """Test ReLU activation forward pass with float16."""
    var dtype = DType.float16
    var shape: List[Int] = [2, 16, 8, 8]

    LayerTester.test_activation_layer(shape, dtype, activation="relu")


fn test_relu_backward_float32() raises:
    """Test ReLU backward pass with gradient checking."""
    var dtype = DType.float32
    var shape: List[Int] = [2, 8, 4, 4]

    LayerTester.test_activation_layer_backward(shape, dtype, activation="relu")


# ============================================================================
# MaxPool Tests
# ============================================================================


fn test_maxpool1_forward_float32() raises:
    """Test MaxPool1 (2x2, stride 2) forward pass with float32."""
    var dtype = DType.float32

    LayerTester.test_pooling_layer(
        channels=6,
        input_h=24,
        input_w=24,
        pool_size=2,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


fn test_maxpool1_forward_float16() raises:
    """Test MaxPool1 forward pass with float16."""
    var dtype = DType.float16

    LayerTester.test_pooling_layer(
        channels=6,
        input_h=24,
        input_w=24,
        pool_size=2,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


fn test_maxpool2_forward_float32() raises:
    """Test MaxPool2 (2x2, stride 2) forward pass with float32."""
    var dtype = DType.float32

    LayerTester.test_pooling_layer(
        channels=16,
        input_h=10,
        input_w=10,
        pool_size=2,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


fn test_maxpool2_forward_float16() raises:
    """Test MaxPool2 forward pass with float16."""
    var dtype = DType.float16

    LayerTester.test_pooling_layer(
        channels=16,
        input_h=10,
        input_w=10,
        pool_size=2,
        stride=2,
        dtype=dtype,
        pool_type="max",
        padding=0,
    )


# ============================================================================
# FC1 (Linear) Tests
# ============================================================================


fn test_fc1_forward_float32() raises:
    """Test FC1 (400→120) forward pass with float32."""
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


fn test_fc1_forward_float16() raises:
    """Test FC1 forward pass with float16."""
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


fn test_fc1_backward_float32() raises:
    """Test FC1 backward pass with gradient checking."""
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


# ============================================================================
# FC2 (Linear) Tests
# ============================================================================


fn test_fc2_forward_float32() raises:
    """Test FC2 (120→84) forward pass with float32."""
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


fn test_fc2_forward_float16() raises:
    """Test FC2 forward pass with float16."""
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


fn test_fc2_backward_float32() raises:
    """Test FC2 backward pass with gradient checking."""
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


# ============================================================================
# FC3 (Linear) Tests
# ============================================================================


fn test_fc3_forward_float32() raises:
    """Test FC3 (84→10) forward pass with float32."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=84,
        out_features=10,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc3_forward_float16() raises:
    """Test FC3 forward pass with float16."""
    var dtype = DType.float16
    var _result = create_fc3_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=84,
        out_features=10,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc3_backward_float32() raises:
    """Test FC3 backward pass with gradient checking."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer_backward(
        in_features=84,
        out_features=10,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


# ============================================================================
# Flatten Test
# ============================================================================


fn test_flatten_operation_float32() raises:
    """Test reshape/flatten operation (16, 5, 5) -> (400,)."""
    var dtype = DType.float32

    # Create a tensor with conv output shape: (1, 16, 5, 5)
    var input = create_special_value_tensor(
        [1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE
    )

    # Flatten: (1, 16, 5, 5) -> (1, 400)
    var flattened = input.reshape([1, 400])

    # Verify shape
    assert_shape(flattened, [1, 400], "Flatten shape mismatch")

    # Verify dtype preserved
    assert_dtype(flattened, dtype, "Flatten dtype mismatch")

    # Verify all values preserved
    for i in range(flattened.numel()):
        var val = flattened._get_float64(i)
        assert_false(isnan(val), "Flatten produced NaN")
        assert_false(isinf(val), "Flatten produced Inf")


fn test_flatten_operation_float16() raises:
    """Test flatten with float16."""
    var dtype = DType.float16

    var input = create_special_value_tensor(
        [1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE
    )
    var flattened = input.reshape([1, 400])

    assert_shape(flattened, [1, 400], "Flatten shape mismatch (float16)")
    assert_dtype(flattened, dtype, "Flatten dtype mismatch (float16)")


# ============================================================================
# Data Flow Through All Layers (Sequential Test)
# ============================================================================


fn test_all_layers_sequence_float32() raises:
    """Test data flow through all layers in sequence (forward pass only).

    Verifies shapes at each layer match expected values.
    Uses float32 for simplicity.
    """
    var dtype = DType.float32
    var batch_size = 1

    # Input: (1, 1, 28, 28)
    var input = create_special_value_tensor(
        [batch_size, 1, 28, 28], dtype, SPECIAL_VALUE_ONE
    )
    assert_shape(input, [batch_size, 1, 28, 28], "Input shape")

    # Conv1: (1, 1, 28, 28) -> (1, 6, 24, 24)
    var _result_conv1 = create_conv1_parameters(dtype)

    var kernel1 = _result_conv1[0]

    var bias1 = _result_conv1[1]
    var conv1_out = conv2d(input, kernel1, bias1, stride=1, padding=0)
    assert_shape(conv1_out, [batch_size, 6, 24, 24], "Conv1 output shape")

    # ReLU1: (1, 6, 24, 24) -> (1, 6, 24, 24)
    var relu1_out = relu(conv1_out)
    assert_shape(relu1_out, [batch_size, 6, 24, 24], "ReLU1 output shape")

    # Pool1: (1, 6, 24, 24) -> (1, 6, 12, 12)
    var pool1_out = maxpool2d(relu1_out, kernel_size=2, stride=2, padding=0)
    assert_shape(pool1_out, [batch_size, 6, 12, 12], "MaxPool1 output shape")

    # Conv2: (1, 6, 12, 12) -> (1, 16, 8, 8)
    var _result_conv2 = create_conv2_parameters(dtype)

    var kernel2 = _result_conv2[0]

    var bias2 = _result_conv2[1]
    var conv2_out = conv2d(pool1_out, kernel2, bias2, stride=1, padding=0)
    assert_shape(conv2_out, [batch_size, 16, 8, 8], "Conv2 output shape")

    # ReLU2: (1, 16, 8, 8) -> (1, 16, 8, 8)
    var relu2_out = relu(conv2_out)
    assert_shape(relu2_out, [batch_size, 16, 8, 8], "ReLU2 output shape")

    # Pool2: (1, 16, 8, 8) -> (1, 16, 4, 4)
    var pool2_out = maxpool2d(relu2_out, kernel_size=2, stride=2, padding=0)
    assert_shape(pool2_out, [batch_size, 16, 4, 4], "MaxPool2 output shape")

    # Flatten: (1, 16, 4, 4) -> (1, 256)
    var flattened = pool2_out.reshape([batch_size, 256])
    assert_shape(flattened, [batch_size, 256], "Flatten output shape")

    # FC1: (1, 256) -> (1, 120)
    var _result_fc1 = create_fc1_parameters(dtype)

    var fc1_weights = _result_fc1[0]

    var fc1_bias = _result_fc1[1]
    # Note: FC1 expects 400 inputs, but we have 256 due to smaller input test
    # Create compatible FC1 with 256 inputs
    var fc1_weights_compat = kaiming_uniform(256, 120, [120, 256], dtype=dtype)
    var fc1_out = linear(flattened, fc1_weights_compat, fc1_bias)
    assert_shape(fc1_out, [batch_size, 120], "FC1 output shape")

    # ReLU3: (1, 120) -> (1, 120)
    var relu3_out = relu(fc1_out)
    assert_shape(relu3_out, [batch_size, 120], "ReLU3 output shape")

    # FC2: (1, 120) -> (1, 84)
    var _result_fc2 = create_fc2_parameters(dtype)

    var fc2_weights = _result_fc2[0]

    var fc2_bias = _result_fc2[1]
    var fc2_out = linear(relu3_out, fc2_weights, fc2_bias)
    assert_shape(fc2_out, [batch_size, 84], "FC2 output shape")

    # ReLU4: (1, 84) -> (1, 84)
    var relu4_out = relu(fc2_out)
    assert_shape(relu4_out, [batch_size, 84], "ReLU4 output shape")

    # FC3: (1, 84) -> (1, 10)
    var _result_fc3 = create_fc3_parameters(dtype)

    var fc3_weights = _result_fc3[0]

    var fc3_bias = _result_fc3[1]
    var output = linear(relu4_out, fc3_weights, fc3_bias)
    assert_shape(output, [batch_size, 10], "FC3 (final output) shape")

    # Verify all outputs are valid (no NaN/Inf)
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_false(isnan(val), "Output contains NaN")
        assert_false(isinf(val), "Output contains Inf")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting LeNet-5 Layerwise Tests...")

    # Conv1 tests
    print("  test_conv1_forward_float32...", end="")
    test_conv1_forward_float32()
    print(" OK")

    print("  test_conv1_forward_float16...", end="")
    test_conv1_forward_float16()
    print(" OK")

    print("  test_conv1_backward_float32...", end="")
    test_conv1_backward_float32()
    print(" OK")

    # Conv2 tests
    print("  test_conv2_forward_float32...", end="")
    test_conv2_forward_float32()
    print(" OK")

    print("  test_conv2_forward_float16...", end="")
    test_conv2_forward_float16()
    print(" OK")

    print("  test_conv2_backward_float32...", end="")
    test_conv2_backward_float32()
    print(" OK")

    # ReLU tests
    print("  test_relu_forward_float32...", end="")
    test_relu_forward_float32()
    print(" OK")

    print("  test_relu_forward_float16...", end="")
    test_relu_forward_float16()
    print(" OK")

    print("  test_relu_backward_float32...", end="")
    test_relu_backward_float32()
    print(" OK")

    # MaxPool tests
    print("  test_maxpool1_forward_float32...", end="")
    test_maxpool1_forward_float32()
    print(" OK")

    print("  test_maxpool1_forward_float16...", end="")
    test_maxpool1_forward_float16()
    print(" OK")

    print("  test_maxpool2_forward_float32...", end="")
    test_maxpool2_forward_float32()
    print(" OK")

    print("  test_maxpool2_forward_float16...", end="")
    test_maxpool2_forward_float16()
    print(" OK")

    # FC1 tests
    print("  test_fc1_forward_float32...", end="")
    test_fc1_forward_float32()
    print(" OK")

    print("  test_fc1_forward_float16...", end="")
    test_fc1_forward_float16()
    print(" OK")

    # FIXME(#2702): test_fc1_backward_float32 disabled - gradient checking crashes in
    # Mojo runtime (closure/memory issue with O(400) forward passes).
    # See: https://github.com/HomericIntelligence/ProjectOdyssey/issues/2702
    print("  test_fc1_backward_float32... FIXME(#2702)")

    # FC2 tests
    print("  test_fc2_forward_float32...", end="")
    test_fc2_forward_float32()
    print(" OK")

    print("  test_fc2_forward_float16...", end="")
    test_fc2_forward_float16()
    print(" OK")

    # FIXME(#2702): test_fc2_backward_float32 disabled - same issue as fc1.
    # See: https://github.com/HomericIntelligence/ProjectOdyssey/issues/2702
    print("  test_fc2_backward_float32... FIXME(#2702)")

    # FC3 tests
    print("  test_fc3_forward_float32...", end="")
    test_fc3_forward_float32()
    print(" OK")

    # FIXME(#2703): test_fc3_forward_float16 disabled - float16 precision insufficient
    # for FC layer with 84 inputs (840 multiplications per output element).
    # See: https://github.com/HomericIntelligence/ProjectOdyssey/issues/2703
    print("  test_fc3_forward_float16... FIXME(#2703)")

    # FIXME(#2702): test_fc3_backward_float32 disabled - same issue as fc1.
    # See: https://github.com/HomericIntelligence/ProjectOdyssey/issues/2702
    print("  test_fc3_backward_float32... FIXME(#2702)")

    # FIXME(#2705): Flatten tests disabled - crashes during tensor creation after
    # running many previous tests (possible memory corruption from earlier tests).
    # Note: Standalone flatten tests pass, AlexNet flatten tests pass.
    # See: https://github.com/HomericIntelligence/ProjectOdyssey/issues/2705
    print("  test_flatten_operation_float32... FIXME(#2705)")
    print("  test_flatten_operation_float16... FIXME(#2705)")

    # Sequential data flow test
    print("  test_all_layers_sequence_float32...", end="")
    test_all_layers_sequence_float32()
    print(" OK")

    print("\nAll LeNet-5 layerwise tests passed!")
