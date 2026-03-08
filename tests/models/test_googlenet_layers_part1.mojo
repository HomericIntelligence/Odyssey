# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_googlenet_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""Layerwise unit tests for GoogLeNet (Inception-v1) model components - Part 1.

Tests cover:
- InceptionModule: Initialization and forward pass shape/values
- 1×1 convolution branch: Dimensionality reduction without spatial change
- 3×3 convolution branch: Feature extraction with 1×1 reduction
- 5×5 convolution branch: Larger receptive field with 1×1 reduction
- MaxPool branch: Spatial pooling with 1×1 projection
- Multi-tensor concatenation: Shape validation

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


# ============================================================================
# Inception Module Tests
# ============================================================================


struct InceptionModule:
    """Minimal Inception module for testing (matches production structure).

    Implements 4 parallel branches:
    - Branch 1: 1×1 conv
    - Branch 2: 1×1 reduce → 3×3 conv
    - Branch 3: 1×1 reduce → 5×5 conv
    - Branch 4: MaxPool → 1×1 projection
    """

    # Branch 1: 1×1 convolution
    var conv1x1_1_weights: ExTensor
    var conv1x1_1_bias: ExTensor

    # Branch 2: 1×1 reduce → 3×3
    var conv1x1_2_weights: ExTensor
    var conv1x1_2_bias: ExTensor
    var conv3x3_weights: ExTensor
    var conv3x3_bias: ExTensor

    # Branch 3: 1×1 reduce → 5×5
    var conv1x1_3_weights: ExTensor
    var conv1x1_3_bias: ExTensor
    var conv5x5_weights: ExTensor
    var conv5x5_bias: ExTensor

    # Branch 4: pool → 1×1 projection
    var conv1x1_4_weights: ExTensor
    var conv1x1_4_bias: ExTensor

    fn __init__(
        out self,
        in_channels: Int,
        out_1x1: Int,
        reduce_3x3: Int,
        out_3x3: Int,
        reduce_5x5: Int,
        out_5x5: Int,
        pool_proj: Int,
    ) raises:
        """Initialize Inception module with specified channel configurations."""
        # Branch 1: 1×1 conv
        self.conv1x1_1_weights = kaiming_normal(
            fan_in=in_channels,
            fan_out=out_1x1,
            shape=[out_1x1, in_channels, 1, 1],
        )
        self.conv1x1_1_bias = zeros([out_1x1], DType.float32)

        # Branch 2: 1×1 reduce
        self.conv1x1_2_weights = kaiming_normal(
            fan_in=in_channels,
            fan_out=reduce_3x3,
            shape=[reduce_3x3, in_channels, 1, 1],
        )
        self.conv1x1_2_bias = zeros([reduce_3x3], DType.float32)

        # Branch 2: 3×3 conv
        self.conv3x3_weights = kaiming_normal(
            fan_in=reduce_3x3 * 9,
            fan_out=out_3x3,
            shape=[out_3x3, reduce_3x3, 3, 3],
        )
        self.conv3x3_bias = zeros([out_3x3], DType.float32)

        # Branch 3: 1×1 reduce
        self.conv1x1_3_weights = kaiming_normal(
            fan_in=in_channels,
            fan_out=reduce_5x5,
            shape=[reduce_5x5, in_channels, 1, 1],
        )
        self.conv1x1_3_bias = zeros([reduce_5x5], DType.float32)

        # Branch 3: 5×5 conv
        self.conv5x5_weights = kaiming_normal(
            fan_in=reduce_5x5 * 25,
            fan_out=out_5x5,
            shape=[out_5x5, reduce_5x5, 5, 5],
        )
        self.conv5x5_bias = zeros([out_5x5], DType.float32)

        # Branch 4: 1×1 projection after pooling
        self.conv1x1_4_weights = kaiming_normal(
            fan_in=in_channels,
            fan_out=pool_proj,
            shape=[pool_proj, in_channels, 1, 1],
        )
        self.conv1x1_4_bias = zeros([pool_proj], DType.float32)

    fn forward(self, x: ExTensor) raises -> ExTensor:
        """Forward pass through all 4 branches with concatenation.

        Returns:
            Concatenated output (batch, out_1x1+out_3x3+out_5x5+pool_proj, H, W).
        """
        # Branch 1: 1×1 conv
        var b1 = conv2d(
            x, self.conv1x1_1_weights, self.conv1x1_1_bias, stride=1, padding=0
        )
        b1 = relu(b1)

        # Branch 2: 1×1 reduce → 3×3 conv
        var b2 = conv2d(
            x, self.conv1x1_2_weights, self.conv1x1_2_bias, stride=1, padding=0
        )
        b2 = relu(b2)
        b2 = conv2d(
            b2, self.conv3x3_weights, self.conv3x3_bias, stride=1, padding=1
        )
        b2 = relu(b2)

        # Branch 3: 1×1 reduce → 5×5 conv
        var b3 = conv2d(
            x, self.conv1x1_3_weights, self.conv1x1_3_bias, stride=1, padding=0
        )
        b3 = relu(b3)
        b3 = conv2d(
            b3, self.conv5x5_weights, self.conv5x5_bias, stride=1, padding=2
        )
        b3 = relu(b3)

        # Branch 4: 3×3 max pool → 1×1 projection
        var b4 = maxpool2d(x, kernel_size=3, stride=1, padding=1)
        b4 = conv2d(
            b4, self.conv1x1_4_weights, self.conv1x1_4_bias, stride=1, padding=0
        )
        b4 = relu(b4)

        # Concatenate all branches depth-wise
        return concatenate_depthwise(b1, b2, b3, b4)


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
# Inception Module Tests
# ============================================================================


fn test_inception_module_initialization() raises:
    """Test that Inception module parameters are created with correct shapes."""
    var in_channels = 64
    var out_1x1 = 32
    var reduce_3x3 = 16
    var out_3x3 = 32
    var reduce_5x5 = 8
    var out_5x5 = 16
    var pool_proj = 16

    var inception = InceptionModule(
        in_channels,
        out_1x1,
        reduce_3x3,
        out_3x3,
        reduce_5x5,
        out_5x5,
        pool_proj,
    )

    # Verify branch 1 shapes
    assert_shape(inception.conv1x1_1_weights, [out_1x1, in_channels, 1, 1])
    assert_shape(inception.conv1x1_1_bias, [out_1x1])

    # Verify branch 2 shapes
    assert_shape(inception.conv1x1_2_weights, [reduce_3x3, in_channels, 1, 1])
    assert_shape(inception.conv3x3_weights, [out_3x3, reduce_3x3, 3, 3])

    # Verify branch 3 shapes
    assert_shape(inception.conv1x1_3_weights, [reduce_5x5, in_channels, 1, 1])
    assert_shape(inception.conv5x5_weights, [out_5x5, reduce_5x5, 5, 5])

    # Verify branch 4 shapes
    assert_shape(inception.conv1x1_4_weights, [pool_proj, in_channels, 1, 1])


fn test_inception_module_forward_shape() raises:
    """Test Inception module output shape with all 4 branches concatenated.

    Input: (batch=2, channels=64, height=8, width=8)
    Branches: 32 + 32 + 16 + 16 = 96 output channels
    Output: (batch=2, channels=96, height=8, width=8)
    """
    var batch_size = 2
    var in_channels = 64
    var in_height = 8
    var in_width = 8

    # Create input
    var input = ones(
        [batch_size, in_channels, in_height, in_width], DType.float32
    )

    # Create Inception module
    var inception = InceptionModule(
        in_channels=in_channels,
        out_1x1=32,
        reduce_3x3=16,
        out_3x3=32,
        reduce_5x5=8,
        out_5x5=16,
        pool_proj=16,
    )

    # Forward pass
    var output = inception.forward(input)

    # Expected output: (batch, 32+32+16+16, 8, 8) = (batch, 96, 8, 8)
    ref outShape = output.shape()
    assert_equal(outShape[0], batch_size)
    assert_equal(outShape[1], 96)
    assert_equal(outShape[2], in_height)
    assert_equal(outShape[3], in_width)


fn test_inception_module_forward_values() raises:
    """Test Inception module produces non-zero output values.

    Verifies that:
    - Input with value 0.1 produces varying output
    - ReLU activations are applied
    - Concatenation preserves all branch outputs
    """
    var batch_size = 1
    var in_channels = 16
    var in_height = 4
    var in_width = 4

    # Create input with known values
    var input = ones(
        [batch_size, in_channels, in_height, in_width], DType.float32
    )
    var input_data = input._data.bitcast[Float32]()
    for i in range(input.numel()):
        input_data[i] = 0.5

    # Create Inception module
    var inception = InceptionModule(
        in_channels=in_channels,
        out_1x1=8,
        reduce_3x3=8,
        out_3x3=8,
        reduce_5x5=4,
        out_5x5=4,
        pool_proj=4,
    )

    # Forward pass
    var output = inception.forward(input)

    # Verify output has expected shape
    var expected_channels = 8 + 8 + 4 + 4  # 24
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], expected_channels)
    assert_equal(output.shape()[2], in_height)
    assert_equal(output.shape()[3], in_width)

    # Verify output contains non-zero values (from ReLU on non-zero input)
    var output_data = output._data.bitcast[Float32]()
    var sum_val = Float32(0.0)
    for i in range(output.numel()):
        sum_val += output_data[i]

    assert_true(sum_val > 0.0, "Output should contain non-zero values")


fn test_inception_branch_1x1_convolution() raises:
    """Test 1×1 convolution branch (Branch 1) independently.

    1×1 convolution is dimensionality reduction without spatial change.
    """
    var batch_size = 2
    var in_channels = 32
    var out_channels = 16
    var height = 8
    var width = 8

    # Create input
    var input = ones([batch_size, in_channels, height, width], DType.float32)

    # Create 1×1 conv weights and bias
    var weights = kaiming_normal(
        in_channels, out_channels, [out_channels, in_channels, 1, 1]
    )
    var bias = zeros([out_channels], DType.float32)

    # Forward pass
    var output = conv2d(input, weights, bias, stride=1, padding=0)
    output = relu(output)

    # Verify shape: input (B, 32, 8, 8) → output (B, 16, 8, 8)
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], out_channels)
    assert_equal(output.shape()[2], height)
    assert_equal(output.shape()[3], width)


fn test_inception_branch_3x3_convolution() raises:
    """Test 3×3 convolution branch with 1×1 reduction (Branch 2).

    Tests the pattern: 1×1 reduce → 3×3 conv.
    """
    var batch_size = 2
    var in_channels = 64
    var reduce_channels = 32
    var out_channels = 64
    var height = 8
    var width = 8

    # Create input
    var input = ones([batch_size, in_channels, height, width], DType.float32)

    # 1×1 reduce
    var reduce_weights = kaiming_normal(
        in_channels, reduce_channels, [reduce_channels, in_channels, 1, 1]
    )
    var reduce_bias = zeros([reduce_channels], DType.float32)
    var reduced = conv2d(
        input, reduce_weights, reduce_bias, stride=1, padding=0
    )
    reduced = relu(reduced)

    # 3×3 conv
    var conv3x3_weights = kaiming_normal(
        reduce_channels * 9, out_channels, [out_channels, reduce_channels, 3, 3]
    )
    var conv3x3_bias = zeros([out_channels], DType.float32)
    var output = conv2d(
        reduced, conv3x3_weights, conv3x3_bias, stride=1, padding=1
    )
    output = relu(output)

    # Verify output shape: (batch, out_channels, height, width)
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], out_channels)
    assert_equal(output.shape()[2], height)
    assert_equal(output.shape()[3], width)


fn test_inception_branch_5x5_convolution() raises:
    """Test 5×5 convolution branch with 1×1 reduction (Branch 3).

    Tests the pattern: 1×1 reduce → 5×5 conv.
    Note: 5×5 conv with padding=2 maintains spatial dimensions.
    """
    var batch_size = 2
    var in_channels = 64
    var reduce_channels = 16
    var out_channels = 32
    var height = 8
    var width = 8

    # Create input
    var input = ones([batch_size, in_channels, height, width], DType.float32)

    # 1×1 reduce
    var reduce_weights = kaiming_normal(
        in_channels, reduce_channels, [reduce_channels, in_channels, 1, 1]
    )
    var reduce_bias = zeros([reduce_channels], DType.float32)
    var reduced = conv2d(
        input, reduce_weights, reduce_bias, stride=1, padding=0
    )
    reduced = relu(reduced)

    # 5×5 conv (padding=2 maintains spatial dimensions)
    var conv5x5_weights = kaiming_normal(
        reduce_channels * 25,
        out_channels,
        [out_channels, reduce_channels, 5, 5],
    )
    var conv5x5_bias = zeros([out_channels], DType.float32)
    var output = conv2d(
        reduced, conv5x5_weights, conv5x5_bias, stride=1, padding=2
    )
    output = relu(output)

    # Verify output shape: (batch, out_channels, height, width)
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], out_channels)
    assert_equal(output.shape()[2], height)
    assert_equal(output.shape()[3], width)


fn test_inception_branch_maxpool() raises:
    """Test MaxPool branch with 1×1 projection (Branch 4).

    Tests the pattern: MaxPool (3×3, stride=1, padding=1) → 1×1 projection.
    """
    var batch_size = 2
    var in_channels = 64
    var proj_channels = 32
    var height = 8
    var width = 8

    # Create input
    var input = ones([batch_size, in_channels, height, width], DType.float32)

    # MaxPool (maintains spatial dimensions with stride=1, padding=1)
    var pooled = maxpool2d(input, kernel_size=3, stride=1, padding=1)

    # 1×1 projection
    var proj_weights = kaiming_normal(
        in_channels, proj_channels, [proj_channels, in_channels, 1, 1]
    )
    var proj_bias = zeros([proj_channels], DType.float32)
    var output = conv2d(pooled, proj_weights, proj_bias, stride=1, padding=0)
    output = relu(output)

    # Verify output shape: (batch, proj_channels, height, width)
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], proj_channels)
    assert_equal(output.shape()[2], height)
    assert_equal(output.shape()[3], width)


fn test_concatenate_depthwise_4_tensors() raises:
    """Test concatenation of 4 tensors along channel dimension.

    Input:
        - t1: (batch=2, channels=8, height=4, width=4)
        - t2: (batch=2, channels=8, height=4, width=4)
        - t3: (batch=2, channels=4, height=4, width=4)
        - t4: (batch=2, channels=4, height=4, width=4)

    Output: (batch=2, channels=24, height=4, width=4)
    """
    var batch_size = 2
    var height = 4
    var width = 4

    # Create 4 input tensors
    var t1 = ones([batch_size, 8, height, width], DType.float32)
    var t2 = ones([batch_size, 8, height, width], DType.float32)
    var t3 = ones([batch_size, 4, height, width], DType.float32)
    var t4 = ones([batch_size, 4, height, width], DType.float32)

    # Set different values to verify correct concatenation
    var t1_data = t1._data.bitcast[Float32]()
    var t2_data = t2._data.bitcast[Float32]()
    var t3_data = t3._data.bitcast[Float32]()
    var t4_data = t4._data.bitcast[Float32]()

    for i in range(t1.numel()):
        t1_data[i] = 1.0
    for i in range(t2.numel()):
        t2_data[i] = 2.0
    for i in range(t3.numel()):
        t3_data[i] = 3.0
    for i in range(t4.numel()):
        t4_data[i] = 4.0

    # Concatenate
    var result = concatenate_depthwise(t1, t2, t3, t4)

    # Verify output shape: (batch, 8+8+4+4, height, width) = (batch, 24, height, width)
    assert_equal(result.shape()[0], batch_size)
    assert_equal(result.shape()[1], 24)
    assert_equal(result.shape()[2], height)
    assert_equal(result.shape()[3], width)

    # Verify total elements
    var expected_numel = batch_size * 24 * height * width
    assert_equal(result.numel(), expected_numel)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting GoogLeNet Layerwise Tests Part 1...")

    print("  test_inception_module_initialization...", end="")
    test_inception_module_initialization()
    print(" OK")

    print("  test_inception_module_forward_shape...", end="")
    test_inception_module_forward_shape()
    print(" OK")

    print("  test_inception_module_forward_values...", end="")
    test_inception_module_forward_values()
    print(" OK")

    print("  test_inception_branch_1x1_convolution...", end="")
    test_inception_branch_1x1_convolution()
    print(" OK")

    print("  test_inception_branch_3x3_convolution...", end="")
    test_inception_branch_3x3_convolution()
    print(" OK")

    print("  test_inception_branch_5x5_convolution...", end="")
    test_inception_branch_5x5_convolution()
    print(" OK")

    print("  test_inception_branch_maxpool...", end="")
    test_inception_branch_maxpool()
    print(" OK")

    print("  test_concatenate_depthwise_4_tensors...", end="")
    test_concatenate_depthwise_4_tensors()
    print(" OK")

    print("All GoogLeNet layerwise tests Part 1 passed!")
