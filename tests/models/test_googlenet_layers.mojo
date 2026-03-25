"""Layerwise unit tests for GoogLeNet (Inception-v1) model components

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
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, randn
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import maxpool2d, global_avgpool2d
from shared.core.linear import linear
from shared.core.activation import relu
from shared.core.initializers import kaiming_normal, xavier_normal, constant


struct InceptionModule:
    """Minimal Inception module for testing (matches production structure).

    Implements 4 parallel branches:
    - Branch 1: 1×1 conv
    - Branch 2: 1×1 reduce → 3×3 conv
    - Branch 3: 1×1 reduce → 5×5 conv
    - Branch 4: MaxPool → 1×1 projection
    """

    # Branch 1: 1×1 convolution
    var conv1x1_1_weights: AnyTensor
    var conv1x1_1_bias: AnyTensor

    # Branch 2: 1×1 reduce → 3×3
    var conv1x1_2_weights: AnyTensor
    var conv1x1_2_bias: AnyTensor
    var conv3x3_weights: AnyTensor
    var conv3x3_bias: AnyTensor

    # Branch 3: 1×1 reduce → 5×5
    var conv1x1_3_weights: AnyTensor
    var conv1x1_3_bias: AnyTensor
    var conv5x5_weights: AnyTensor
    var conv5x5_bias: AnyTensor

    # Branch 4: pool → 1×1 projection
    var conv1x1_4_weights: AnyTensor
    var conv1x1_4_bias: AnyTensor

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

    fn forward(self, x: AnyTensor) raises -> AnyTensor:
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
    t1: AnyTensor, t2: AnyTensor, t3: AnyTensor, t4: AnyTensor


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


fn test_inception_branch_1x1_backward() raises:
    """Test backward pass through 1×1 convolution branch.

    Verifies gradient computation for weight updates.
    """
    var batch_size = 2
    var in_channels = 32
    var out_channels = 16
    var height = 4
    var width = 4

    # Create input and weights
    var input = ones([batch_size, in_channels, height, width], DType.float32)
    var weights = ones([out_channels, in_channels, 1, 1], DType.float32)
    var bias = zeros([out_channels], DType.float32)

    # Forward pass
    var output = conv2d(input, weights, bias, stride=1, padding=0)

    # Create gradient from upstream
    var grad_output = ones(
        [batch_size, out_channels, height, width], DType.float32
    )

    # Backward pass
    var _result = conv2d_backward(
        grad_output, input, weights, stride=1, padding=0
    )
    var grad_input = _result.grad_input
    var grad_weights = _result.grad_weights
    var grad_bias = _result.grad_bias

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())
    assert_shape(grad_weights, weights.shape())
    assert_shape(grad_bias, bias.shape())


fn test_inception_branch_3x3_backward() raises:
    """Test backward pass through 3×3 convolution branch.

    Tests gradient computation with padding=1.
    """
    var batch_size = 2
    var in_channels = 16
    var out_channels = 16
    var height = 4
    var width = 4

    # Create input and weights
    var input = ones([batch_size, in_channels, height, width], DType.float32)
    var weights = ones([out_channels, in_channels, 3, 3], DType.float32)
    var bias = zeros([out_channels], DType.float32)

    # Forward pass
    var output = conv2d(input, weights, bias, stride=1, padding=1)

    # Create gradient from upstream
    var grad_output = ones(
        [batch_size, out_channels, height, width], DType.float32
    )

    # Backward pass
    var _result = conv2d_backward(
        grad_output, input, weights, stride=1, padding=1
    )
    var grad_input = _result.grad_input
    var grad_weights = _result.grad_weights
    var grad_bias = _result.grad_bias

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())
    assert_shape(grad_weights, weights.shape())
    assert_shape(grad_bias, bias.shape())


fn test_inception_branch_5x5_backward() raises:
    """Test backward pass through 5×5 convolution branch.

    Tests gradient computation with padding=2.
    """
    var batch_size = 2
    var in_channels = 8
    var out_channels = 8
    var height = 4
    var width = 4

    # Create input and weights
    var input = ones([batch_size, in_channels, height, width], DType.float32)
    var weights = ones([out_channels, in_channels, 5, 5], DType.float32)
    var bias = zeros([out_channels], DType.float32)

    # Forward pass
    var output = conv2d(input, weights, bias, stride=1, padding=2)

    # Create gradient from upstream
    var grad_output = ones(
        [batch_size, out_channels, height, width], DType.float32
    )

    # Backward pass
    var _result = conv2d_backward(
        grad_output, input, weights, stride=1, padding=2
    )
    var grad_input = _result.grad_input
    var grad_weights = _result.grad_weights
    var grad_bias = _result.grad_bias

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())
    assert_shape(grad_weights, weights.shape())
    assert_shape(grad_bias, bias.shape())


fn test_concatenate_gradient_preservation() raises:
    """Test that concatenation preserves gradients correctly.

    Gradient flows backward through concatenation to each input tensor.
    """
    var batch_size = 1
    var height = 2
    var width = 2

    # Create input tensors
    var t1 = ones([batch_size, 2, height, width], DType.float32)
    var t2 = ones([batch_size, 2, height, width], DType.float32)
    var t3 = ones([batch_size, 2, height, width], DType.float32)
    var t4 = ones([batch_size, 2, height, width], DType.float32)

    # Forward pass
    var result = concatenate_depthwise(t1, t2, t3, t4)

    # Create gradient from upstream (all ones for simplicity)
    var grad_result = ones(result.shape(), DType.float32)

    # Verify gradient shape
    assert_equal(grad_result.shape()[0], batch_size)
    assert_equal(grad_result.shape()[1], 8)
    assert_equal(grad_result.shape()[2], height)
    assert_equal(grad_result.shape()[3], width)


fn main() raises:
    """Run all test_googlenet_layers tests."""
    print("Running test_googlenet_layers tests...")

    test_inception_module_initialization()
    print("✓ test_inception_module_initialization")

    test_inception_module_forward_shape()
    print("✓ test_inception_module_forward_shape")

    test_inception_module_forward_values()
    print("✓ test_inception_module_forward_values")

    test_inception_branch_1x1_convolution()
    print("✓ test_inception_branch_1x1_convolution")

    test_inception_branch_3x3_convolution()
    print("✓ test_inception_branch_3x3_convolution")

    test_inception_branch_5x5_convolution()
    print("✓ test_inception_branch_5x5_convolution")

    test_inception_branch_maxpool()
    print("✓ test_inception_branch_maxpool")

    test_concatenate_depthwise_4_tensors()
    print("✓ test_concatenate_depthwise_4_tensors")

    test_concatenate_depthwise_values()
    print("✓ test_concatenate_depthwise_values")

    test_initial_conv_block()
    print("✓ test_initial_conv_block")

    test_global_avgpool()
    print("✓ test_global_avgpool")

    test_global_avgpool_larger_spatial()
    print("✓ test_global_avgpool_larger_spatial")

    test_fc_layer()
    print("✓ test_fc_layer")

    test_fc_layer_different_sizes()
    print("✓ test_fc_layer_different_sizes")

    test_inception_branch_1x1_backward()
    print("✓ test_inception_branch_1x1_backward")

    test_inception_branch_3x3_backward()
    print("✓ test_inception_branch_3x3_backward")

    test_inception_branch_5x5_backward()
    print("✓ test_inception_branch_5x5_backward")

    test_concatenate_gradient_preservation()
    print("✓ test_concatenate_gradient_preservation")

    print("\nAll test_googlenet_layers tests passed!")
