# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_googlenet_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Layerwise unit tests for GoogLeNet (Inception-v1) model components - Part 2.

Tests cover:
- Multi-tensor concatenation: Value validation
- Initial convolution block: Entry layer before Inception modules
- Global average pooling: Spatial dimension reduction
- FC layer: Final classification layer

All tests use small tensors for fast execution (< 90 seconds total).
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, full, randn
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, global_avgpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.core.initializers import kaiming_normal, xavier_normal, constant


fn concatenate_depthwise(
    t1: ExTensor, t2: ExTensor, t3: ExTensor, t4: ExTensor
) raises -> ExTensor:
    """Concatenate 4 tensors along the channel dimension (axis=1).

    Args:
        t1: Tensor 1 (batch, C1, H, W).
        t2: Tensor 2 (batch, C2, H, W).
        t3: Tensor 3 (batch, C3, H, W).
        t4: Tensor 4 (batch, C4, H, W).

    Returns:
        Concatenated tensor (batch, C1+C2+C3+C4, H, W).
    """
    var batch_size = t1.shape()[0]
    var c1 = t1.shape()[1]
    var c2 = t2.shape()[1]
    var c3 = t3.shape()[1]
    var c4 = t4.shape()[1]
    var height = t1.shape()[2]
    var width = t1.shape()[3]

    var total_channels = c1 + c2 + c3 + c4
    var result = zeros([batch_size, total_channels, height, width], t1.dtype())

    # Copy data from each tensor
    var result_data = result._data.bitcast[Float32]()
    var t1_data = t1._data.bitcast[Float32]()
    var t2_data = t2._data.bitcast[Float32]()
    var t3_data = t3._data.bitcast[Float32]()
    var t4_data = t4._data.bitcast[Float32]()

    var hw = height * width

    for b in range(batch_size):
        # Copy t1 channels
        for c in range(c1):
            for i in range(hw):
                var src_idx = ((b * c1 + c) * hw) + i
                var dst_idx = ((b * total_channels + c) * hw) + i
                result_data[dst_idx] = t1_data[src_idx]

        # Copy t2 channels (offset by c1)
        for c in range(c2):
            for i in range(hw):
                var src_idx = ((b * c2 + c) * hw) + i
                var dst_idx = ((b * total_channels + (c1 + c)) * hw) + i
                result_data[dst_idx] = t2_data[src_idx]

        # Copy t3 channels (offset by c1+c2)
        for c in range(c3):
            for i in range(hw):
                var src_idx = ((b * c3 + c) * hw) + i
                var dst_idx = ((b * total_channels + (c1 + c2 + c)) * hw) + i
                result_data[dst_idx] = t3_data[src_idx]

        # Copy t4 channels (offset by c1+c2+c3)
        for c in range(c4):
            for i in range(hw):
                var src_idx = ((b * c4 + c) * hw) + i
                var dst_idx = (
                    (b * total_channels + (c1 + c2 + c3 + c)) * hw
                ) + i
                result_data[dst_idx] = t4_data[src_idx]

    return result


# ============================================================================
# Multi-Tensor Concatenation Tests
# ============================================================================


fn test_concatenate_depthwise_values() raises:
    """Test that concatenation preserves values in correct order.

    Concatenation should preserve values from each tensor in channel dimension.
    """
    var batch_size = 1
    var height = 2
    var width = 2

    # Create tensors with distinct values
    var t1 = full([batch_size, 2, height, width], 1.0, DType.float32)
    var t2 = full([batch_size, 2, height, width], 2.0, DType.float32)
    var t3 = full([batch_size, 2, height, width], 3.0, DType.float32)
    var t4 = full([batch_size, 2, height, width], 4.0, DType.float32)

    # Concatenate
    var result = concatenate_depthwise(t1, t2, t3, t4)

    # Verify shape
    assert_equal(result.shape()[0], batch_size)
    assert_equal(result.shape()[1], 8)
    assert_equal(result.shape()[2], height)
    assert_equal(result.shape()[3], width)

    # Sample values to verify correct concatenation
    var result_data = result._data.bitcast[Float32]()

    # Result structure: [t1_c0, t1_c1, t2_c0, t2_c1, t3_c0, t3_c1, t4_c0, t4_c1] along channels
    # Each channel has height*width values

    # Check first channel value (from t1) is 1.0
    var idx_t1 = 0
    assert_close_float(Float64(result_data[idx_t1]), 1.0)

    # Check third channel value (from t2) is 2.0
    var idx_t2 = 2 * height * width
    assert_close_float(Float64(result_data[idx_t2]), 2.0)

    # Check fifth channel value (from t3) is 3.0
    var idx_t3 = 4 * height * width
    assert_close_float(Float64(result_data[idx_t3]), 3.0)

    # Check seventh channel value (from t4) is 4.0
    var idx_t4 = 6 * height * width
    assert_close_float(Float64(result_data[idx_t4]), 4.0)


# ============================================================================
# Initial Convolution Block Tests
# ============================================================================


fn test_initial_conv_block() raises:
    """Test initial convolution block (before Inception modules).

    Structure: Conv2d (3×3) → ReLU

    Input: (batch=2, channels=3, height=32, width=32)
    Output: (batch=2, channels=64, height=32, width=32)
    """
    var batch_size = 2
    var in_channels = 3
    var out_channels = 64
    var height = 32
    var width = 32

    # Create input
    var input = ones([batch_size, in_channels, height, width], DType.float32)

    # Create weights and bias
    var weights = kaiming_normal(
        in_channels * 9, out_channels, [out_channels, in_channels, 3, 3]
    )
    var bias = zeros([out_channels], DType.float32)

    # Forward pass
    var output = conv2d(input, weights, bias, stride=1, padding=1)
    output = relu(output)

    # Verify shape
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], out_channels)
    assert_equal(output.shape()[2], height)
    assert_equal(output.shape()[3], width)


# ============================================================================
# Global Average Pooling Tests
# ============================================================================


fn test_global_avgpool() raises:
    """Test global average pooling layer.

    Reduces spatial dimensions to 1×1 by averaging.

    Input: (batch=2, channels=1024, height=1, width=1) (already spatial 1x1)
    Output: (batch=2, channels=1024)
    """
    var batch_size = 2
    var channels = 1024
    var height = 1
    var width = 1

    # Create input
    var input = ones([batch_size, channels, height, width], DType.float32)
    var input_data = input._data.bitcast[Float32]()
    for i in range(input.numel()):
        input_data[i] = 2.0

    # Apply global average pooling
    var output = global_avgpool2d(input)

    # Verify output shape: (batch, channels)
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], channels)

    # Verify output values (should be 2.0 since input was all 2.0)
    var output_data = output._data.bitcast[Float32]()
    for i in range(output.numel()):
        assert_close_float(Float64(output_data[i]), 2.0)


fn test_global_avgpool_larger_spatial() raises:
    """Test global average pooling with larger spatial dimensions.

    Input: (batch=2, channels=512, height=4, width=4)
    Output: (batch=2, channels=512)
    """
    var batch_size = 2
    var channels = 512
    var height = 4
    var width = 4

    # Create input with known values
    var input = full([batch_size, channels, height, width], 4.0, DType.float32)

    # Apply global average pooling
    var output = global_avgpool2d(input)

    # Verify output shape
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], channels)

    # Verify averaging: all values should be 4.0
    var output_data = output._data.bitcast[Float32]()
    for i in range(output.numel()):
        assert_close_float(Float64(output_data[i]), 4.0)


# ============================================================================
# FC Layer Tests
# ============================================================================


fn test_fc_layer() raises:
    """Test final fully connected layer.

    Linear transformation from feature vector to class logits.

    Input: (batch=2, features=1024)
    Output: (batch=2, classes=10)
    """
    var batch_size = 2
    var in_features = 1024
    var num_classes = 10

    # Create input
    var input = ones([batch_size, in_features], DType.float32)

    # Create weights and bias
    var weights = xavier_normal(
        in_features, num_classes, [num_classes, in_features]
    )
    var bias = zeros([num_classes], DType.float32)

    # Forward pass: y = xW^T + b
    var output = linear(input, weights, bias)

    # Verify shape
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], num_classes)


fn test_fc_layer_different_sizes() raises:
    """Test FC layer with different feature and class sizes.

    Input: (batch=4, features=512)
    Output: (batch=4, classes=100)
    """
    var batch_size = 4
    var in_features = 512
    var num_classes = 100

    # Create input
    var input = ones([batch_size, in_features], DType.float32)

    # Create weights and bias
    var weights = xavier_normal(
        in_features, num_classes, [num_classes, in_features]
    )
    var bias = zeros([num_classes], DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)

    # Verify shape
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], num_classes)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting GoogLeNet Layerwise Tests Part 2...")

    print("  test_concatenate_depthwise_values...", end="")
    test_concatenate_depthwise_values()
    print(" OK")

    print("  test_initial_conv_block...", end="")
    test_initial_conv_block()
    print(" OK")

    print("  test_global_avgpool...", end="")
    test_global_avgpool()
    print(" OK")

    print("  test_global_avgpool_larger_spatial...", end="")
    test_global_avgpool_larger_spatial()
    print(" OK")

    print("  test_fc_layer...", end="")
    test_fc_layer()
    print(" OK")

    print("  test_fc_layer_different_sizes...", end="")
    test_fc_layer_different_sizes()
    print(" OK")

    print("All GoogLeNet layerwise tests Part 2 passed!")
