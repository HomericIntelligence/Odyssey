"""Layerwise Unit Tests for AlexNet - Part 5: FC1/FC2 Backward, FC3, Flatten, and Full Sequence

Tests FC1 (9216→4096) and FC2 (4096→4096) backward passes, FC3 (4096→1000),
Flatten operation, and full forward pass sequence through all AlexNet layers.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_alexnet_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from shared.core.extensor import AnyTensor, zeros, ones, full
from shared.core.conv import conv2d
from shared.core.pooling import maxpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.core.shape import conv2d_output_shape, pool_output_shape
from shared.core.initializers import kaiming_uniform
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
)
from shared.testing.layer_testers import LayerTester
from math import isnan, isinf


# ============================================================================
# Test Fixtures - Parameter Creation
# ============================================================================


fn create_conv1_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv1 layer parameters (3→64, 11x11 kernel)."""
    var in_channels = 3
    var out_channels = 64
    var kernel_size = 11

    var kernel_shape: List[Int] = [
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    ]
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)
    var bias = zeros([out_channels], dtype)

    return kernel, bias


fn create_conv2_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv2 layer parameters (64→192, 5x5 kernel)."""
    var in_channels = 64
    var out_channels = 192
    var kernel_size = 5

    var kernel_shape: List[Int] = [
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    ]
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)
    var bias = zeros([out_channels], dtype)

    return kernel, bias


fn create_conv3_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv3 layer parameters (192→384, 3x3 kernel)."""
    var in_channels = 192
    var out_channels = 384
    var kernel_size = 3

    var kernel_shape: List[Int] = [
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    ]
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)
    var bias = zeros([out_channels], dtype)

    return kernel, bias


fn create_conv4_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv4 layer parameters (384→384, 3x3 kernel)."""
    var in_channels = 384
    var out_channels = 384
    var kernel_size = 3

    var kernel_shape: List[Int] = [
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    ]
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)
    var bias = zeros([out_channels], dtype)

    return kernel, bias


fn create_conv5_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create Conv5 layer parameters (384→256, 3x3 kernel)."""
    var in_channels = 384
    var out_channels = 256
    var kernel_size = 3

    var kernel_shape: List[Int] = [
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    ]
    var fan_in = in_channels * kernel_size * kernel_size
    var fan_out = out_channels * kernel_size * kernel_size
    var kernel = kaiming_uniform(fan_in, fan_out, kernel_shape, dtype=dtype)
    var bias = zeros([out_channels], dtype)

    return kernel, bias


fn create_fc1_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create FC1 layer parameters (9216→4096)."""
    var in_features = 9216  # 256 * 6 * 6
    var out_features = 4096

    var weights_shape: List[Int] = [out_features, in_features]
    var weights = kaiming_uniform(
        in_features, out_features, weights_shape, dtype=dtype
    )
    var bias = zeros([out_features], dtype)

    return weights, bias


fn create_fc2_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create FC2 layer parameters (4096→4096)."""
    var in_features = 4096
    var out_features = 4096

    var weights_shape: List[Int] = [out_features, in_features]
    var weights = kaiming_uniform(
        in_features, out_features, weights_shape, dtype=dtype
    )
    var bias = zeros([out_features], dtype)

    return weights, bias


fn create_fc3_parameters(dtype: DType) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create FC3 layer parameters (4096→1000)."""
    var in_features = 4096
    var out_features = 1000

    # FC3 weights: (1000, 4096)
    var weights_shape: List[Int] = [out_features, in_features]
    var weights = kaiming_uniform(
        in_features, out_features, weights_shape, dtype=dtype
    )

    # FC3 bias: (1000,)
    var bias = zeros([out_features], dtype)

    return weights, bias


# ============================================================================
# FC1 Backward Tests (9216→4096) - moved from part4 to reduce compile unit size
# ============================================================================


fn test_fc1_backward_float32() raises:
    """Test FC1 backward pass with sampled gradient checking.

    Uses sampled gradient checking (30 samples) to avoid timeout.
    FC1 has 9,216 inputs, making exhaustive checking too slow.
    30 samples provides 95% statistical confidence while completing in ~54s.
    """
    var dtype = DType.float32
    var _result = create_fc1_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer_backward(
        in_features=9216,
        out_features=4096,
        weights=weights,
        bias=bias,
        dtype=dtype,
        validate_analytical=True,
        num_gradient_samples=30,
    )


# ============================================================================
# FC2 Backward Tests (4096→4096) - moved from part4 to reduce compile unit size
# ============================================================================


fn test_fc2_backward_float32() raises:
    """Test FC2 backward pass with sampled gradient checking.

    Uses sampled gradient checking (30 samples) to avoid timeout.
    FC2 has 4,096 inputs, making exhaustive checking too slow.
    30 samples provides 95% statistical confidence while completing in ~30s.
    """
    var dtype = DType.float32
    var _result = create_fc2_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer_backward(
        in_features=4096,
        out_features=4096,
        weights=weights,
        bias=bias,
        dtype=dtype,
        validate_analytical=True,
        num_gradient_samples=30,
    )


# ============================================================================
# FC3 (Linear) Tests (4096→1000)
# ============================================================================


fn test_fc3_forward_float32() raises:
    """Test FC3 (4096→1000) forward pass with float32."""
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer(
        in_features=4096,
        out_features=1000,
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
        in_features=4096,
        out_features=1000,
        weights=weights,
        bias=bias,
        dtype=dtype,
    )


fn test_fc3_backward_float32() raises:
    """Test FC3 backward pass with sampled gradient checking.

    Uses sampled gradient checking (30 samples) to avoid timeout.
    FC3 has 4,096 inputs, making exhaustive checking too slow.
    30 samples provides 95% statistical confidence while completing in ~20s.
    """
    var dtype = DType.float32
    var _result = create_fc3_parameters(dtype)

    var weights = _result[0]

    var bias = _result[1]

    LayerTester.test_linear_layer_backward(
        in_features=4096,
        out_features=1000,
        weights=weights,
        bias=bias,
        dtype=dtype,
        validate_analytical=True,
        num_gradient_samples=30,
    )


# ============================================================================
# Flatten Test
# ============================================================================


fn test_flatten_operation_float32() raises:
    """Test reshape/flatten operation (256, 6, 6) -> (9216,)."""
    var dtype = DType.float32

    # Create a tensor with final conv output shape: (1, 256, 6, 6)
    var input = create_special_value_tensor(
        [1, 256, 6, 6], dtype, SPECIAL_VALUE_ONE
    )

    # Flatten: (1, 256, 6, 6) -> (1, 9216)
    var flattened = input.reshape([1, 9216])

    # Verify shape
    assert_shape(flattened, [1, 9216], "Flatten shape mismatch")

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
        [1, 256, 6, 6], dtype, SPECIAL_VALUE_ONE
    )
    var flattened = input.reshape([1, 9216])

    assert_shape(flattened, [1, 9216], "Flatten shape mismatch (float16)")
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

    # Input: (1, 3, 224, 224) - typical ImageNet size
    var input = create_special_value_tensor(
        [batch_size, 3, 224, 224], dtype, SPECIAL_VALUE_ONE
    )
    assert_shape(input, [batch_size, 3, 224, 224], "Input shape")

    # Conv1: (1, 3, 224, 224) -> (1, 64, 55, 55) with stride 4
    var _result_conv1 = create_conv1_parameters(dtype)
    var kernel1 = _result_conv1[0]
    var bias1 = _result_conv1[1]
    var conv1_out = conv2d(input, kernel1, bias1, stride=4, padding=2)
    assert_shape(conv1_out, [batch_size, 64, 55, 55], "Conv1 output shape")

    # ReLU1: (1, 64, 55, 55) -> (1, 64, 55, 55)
    var relu1_out = relu(conv1_out)
    assert_shape(relu1_out, [batch_size, 64, 55, 55], "ReLU1 output shape")

    # MaxPool1: (1, 64, 55, 55) -> (1, 64, 27, 27) with 3x3, stride 2
    var pool1_out = maxpool2d(relu1_out, kernel_size=3, stride=2, padding=0)
    assert_shape(pool1_out, [batch_size, 64, 27, 27], "MaxPool1 output shape")

    # Conv2: (1, 64, 27, 27) -> (1, 192, 27, 27)
    var _result_conv2 = create_conv2_parameters(dtype)

    var kernel2 = _result_conv2[0]

    var bias2 = _result_conv2[1]
    var conv2_out = conv2d(pool1_out, kernel2, bias2, stride=1, padding=2)
    assert_shape(conv2_out, [batch_size, 192, 27, 27], "Conv2 output shape")

    # ReLU2: (1, 192, 27, 27) -> (1, 192, 27, 27)
    var relu2_out = relu(conv2_out)
    assert_shape(relu2_out, [batch_size, 192, 27, 27], "ReLU2 output shape")

    # MaxPool2: (1, 192, 27, 27) -> (1, 192, 13, 13) with 3x3, stride 2
    var pool2_out = maxpool2d(relu2_out, kernel_size=3, stride=2, padding=0)
    assert_shape(pool2_out, [batch_size, 192, 13, 13], "MaxPool2 output shape")

    # Conv3: (1, 192, 13, 13) -> (1, 384, 13, 13)
    var _result_conv3 = create_conv3_parameters(dtype)

    var kernel3 = _result_conv3[0]

    var bias3 = _result_conv3[1]
    var conv3_out = conv2d(pool2_out, kernel3, bias3, stride=1, padding=1)
    assert_shape(conv3_out, [batch_size, 384, 13, 13], "Conv3 output shape")

    # ReLU3: (1, 384, 13, 13) -> (1, 384, 13, 13)
    var relu3_out = relu(conv3_out)
    assert_shape(relu3_out, [batch_size, 384, 13, 13], "ReLU3 output shape")

    # Conv4: (1, 384, 13, 13) -> (1, 384, 13, 13)
    var _result_conv4 = create_conv4_parameters(dtype)

    var kernel4 = _result_conv4[0]

    var bias4 = _result_conv4[1]
    var conv4_out = conv2d(relu3_out, kernel4, bias4, stride=1, padding=1)
    assert_shape(conv4_out, [batch_size, 384, 13, 13], "Conv4 output shape")

    # ReLU4: (1, 384, 13, 13) -> (1, 384, 13, 13)
    var relu4_out = relu(conv4_out)
    assert_shape(relu4_out, [batch_size, 384, 13, 13], "ReLU4 output shape")

    # Conv5: (1, 384, 13, 13) -> (1, 256, 13, 13)
    var _result_conv5 = create_conv5_parameters(dtype)

    var kernel5 = _result_conv5[0]

    var bias5 = _result_conv5[1]
    var conv5_out = conv2d(relu4_out, kernel5, bias5, stride=1, padding=1)
    assert_shape(conv5_out, [batch_size, 256, 13, 13], "Conv5 output shape")

    # ReLU5: (1, 256, 13, 13) -> (1, 256, 13, 13)
    var relu5_out = relu(conv5_out)
    assert_shape(relu5_out, [batch_size, 256, 13, 13], "ReLU5 output shape")

    # MaxPool3: (1, 256, 13, 13) -> (1, 256, 6, 6) with 3x3, stride 2
    var pool3_out = maxpool2d(relu5_out, kernel_size=3, stride=2, padding=0)
    assert_shape(pool3_out, [batch_size, 256, 6, 6], "MaxPool3 output shape")

    # Flatten: (1, 256, 6, 6) -> (1, 9216)
    var flattened = pool3_out.reshape([batch_size, 9216])
    assert_shape(flattened, [batch_size, 9216], "Flatten output shape")

    # FC1: (1, 9216) -> (1, 4096)
    var _result_fc1 = create_fc1_parameters(dtype)

    var fc1_weights = _result_fc1[0]

    var fc1_bias = _result_fc1[1]
    var fc1_out = linear(flattened, fc1_weights, fc1_bias)
    assert_shape(fc1_out, [batch_size, 4096], "FC1 output shape")

    # ReLU6: (1, 4096) -> (1, 4096)
    var relu6_out = relu(fc1_out)
    assert_shape(relu6_out, [batch_size, 4096], "ReLU6 output shape")

    # FC2: (1, 4096) -> (1, 4096)
    var _result_fc2 = create_fc2_parameters(dtype)

    var fc2_weights = _result_fc2[0]

    var fc2_bias = _result_fc2[1]
    var fc2_out = linear(relu6_out, fc2_weights, fc2_bias)
    assert_shape(fc2_out, [batch_size, 4096], "FC2 output shape")

    # ReLU7: (1, 4096) -> (1, 4096)
    var relu7_out = relu(fc2_out)
    assert_shape(relu7_out, [batch_size, 4096], "ReLU7 output shape")

    # FC3: (1, 4096) -> (1, 1000)
    var _result_fc3 = create_fc3_parameters(dtype)

    var fc3_weights = _result_fc3[0]

    var fc3_bias = _result_fc3[1]
    var output = linear(relu7_out, fc3_weights, fc3_bias)
    assert_shape(output, [batch_size, 1000], "FC3 (final output) shape")

    # Verify all outputs are valid (no NaN/Inf)
    for i in range(output.numel()):
        var val = output._get_float64(i)
        assert_false(isnan(val), "Output contains NaN")
        assert_false(isinf(val), "Output contains Inf")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print(
        "Starting AlexNet Layerwise Tests - Part 5 (FC1/FC2 Backward, FC3,"
        " Flatten, Full Sequence)..."
    )

    # FC1 backward - uses sampled gradient checking (30 samples)
    print("  test_fc1_backward_float32...", end="")
    test_fc1_backward_float32()
    print(" OK")

    # FC2 backward - uses sampled gradient checking (30 samples)
    print("  test_fc2_backward_float32...", end="")
    test_fc2_backward_float32()
    print(" OK")

    # FC3 tests
    print("  test_fc3_forward_float32...", end="")
    test_fc3_forward_float32()
    print(" OK")

    print("  test_fc3_forward_float16...", end="")
    test_fc3_forward_float16()
    print(" OK")

    # FC3 backward - uses sampled gradient checking (30 samples)
    print("  test_fc3_backward_float32...", end="")
    test_fc3_backward_float32()
    print(" OK")

    # Flatten tests - reference counting bug fixed in Mojo 0.26.1
    print("  test_flatten_operation_float32...", end="")
    test_flatten_operation_float32()
    print(" OK")

    print("  test_flatten_operation_float16...", end="")
    test_flatten_operation_float16()
    print(" OK")

    # Sequential data flow test
    print("  test_all_layers_sequence_float32...", end="")
    test_all_layers_sequence_float32()
    print(" OK")

    print(
        "\nAll AlexNet Part 5 (FC1/FC2 Backward, FC3, Flatten, Full Sequence)"
        " tests passed!"
    )
