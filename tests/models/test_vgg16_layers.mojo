"""Unit tests for VGG-16 convolutional layer operations.

VGG-16 Architecture Overview (~25 layer operations):
- 13 Conv2D layers (3x3 kernels, various channel depths)
- 5 MaxPool2D layers (2x2, stride 2)
- 3 Fully Connected (FC) layers
- ReLU activation after each conv and FC
- Dropout (0.5) in FC layers during training

Layer Deduplication Strategy:
VGG-16 contains 13 convolutional layers but only ~5 unique architectures
distinguished by channel count. Rather than testing all 13 identical conv
operations, we test one representative conv per unique channel configuration:

1. Conv with 64 output channels (appears 2x in VGG-16: Conv1_1, Conv1_2)
2. Conv with 128 output channels (appears 2x: Conv2_1, Conv2_2)
3. Conv with 256 output channels (appears 3x: Conv3_1, Conv3_2, Conv3_3)
4. Conv with 512 output channels (appears 3x: Conv4_1, Conv4_2, Conv4_3)
5. Conv with 512 output channels again (appears 3x: Conv5_1, Conv5_2, Conv5_3)

All use: kernel_size=3x3, padding=1, stride=1, followed by ReLU and MaxPool

This file contains Part 1: Conv layer tests (64, 128, 256, 512 channels).
See test_vgg16_layers_part2.mojo for MaxPool, ReLU, and FC layer tests.

All tests use pure functional API - no internal state or parameters.
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
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.conv import (
    conv2d,
    conv2d_no_bias,
    conv2d_backward,
    conv2d_no_bias_backward,
)
from shared.core.linear import (
    linear,
    linear_no_bias,
    linear_backward,
    linear_no_bias_backward,
)
from shared.core.activation import (
    relu,
    relu_backward,
)
from shared.core.pooling import (
    maxpool2d,
    maxpool2d_backward,
)


def test_vgg16_conv64_forward() raises:
    """Test Conv2D forward pass with 64 output channels (VGG layer 1).

    Configuration:
        Input: (4, 3, 32, 32) - 4 samples, 3 input channels, 32x32 spatial
        Conv: 3x3 kernel, 64 output channels, padding=1, stride=1
        Output: (4, 64, 32, 32) - spatial size preserved by padding

    Deduplication Note:
        This tests Conv1_1 and Conv1_2 which are identical in architecture.
        We test once and document that both use this configuration.
    """
    var batch_size = 4
    var in_channels = 3
    var out_channels = 64
    var height = 32
    var width = 32
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input: (batch_size, in_channels, height, width)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel: (out_channels, in_channels, kernel_size, kernel_size)
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias: (out_channels,)
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], height)  # Preserved by padding=1
    assert_equal(output_shape[3], width)


def test_vgg16_conv64_backward() raises:
    """Test Conv2D backward pass with 64 output channels.

    Verifies gradient computation w.r.t. input, kernel, and bias.
    """
    var batch_size = 2
    var in_channels = 3
    var out_channels = 64
    var height = 32
    var width = 32
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Create gradient w.r.t. output
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var _backward_result = conv2d_backward(
        grad_output, input, kernel, stride, padding
    )
    var grad_input = _backward_result.grad_input

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())


def test_vgg16_conv128_forward() raises:
    """Test Conv2D forward pass with 128 output channels (VGG layer 2).

    Configuration:
        Input: (4, 64, 16, 16) - from after MaxPool1
        Conv: 3x3 kernel, 128 output channels, padding=1, stride=1
        Output: (4, 128, 16, 16)

    Deduplication Note:
        Conv2_1 and Conv2_2 both have this configuration.
    """
    var batch_size = 4
    var in_channels = 64
    var out_channels = 128
    var height = 16
    var width = 16
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], height)
    assert_equal(output_shape[3], width)


def test_vgg16_conv128_backward() raises:
    """Test Conv2D backward pass with 128 output channels."""
    var batch_size = 2
    var in_channels = 64
    var out_channels = 128
    var height = 16
    var width = 16
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Create gradient
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var _backward_result = conv2d_backward(
        grad_output, input, kernel, stride, padding
    )
    var grad_input = _backward_result.grad_input

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())


def test_vgg16_conv256_forward() raises:
    """Test Conv2D forward pass with 256 output channels (VGG layer 3).

    Configuration:
        Input: (4, 128, 8, 8) - from after MaxPool2
        Conv: 3x3 kernel, 256 output channels, padding=1, stride=1
        Output: (4, 256, 8, 8)

    Deduplication Note:
        Conv3_1, Conv3_2, Conv3_3 all have this configuration.
        We test once as they are identical.
    """
    var batch_size = 4
    var in_channels = 128
    var out_channels = 256
    var height = 8
    var width = 8
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], height)
    assert_equal(output_shape[3], width)


def test_vgg16_conv256_backward() raises:
    """Test Conv2D backward pass with 256 output channels."""
    var batch_size = 2
    var in_channels = 128
    var out_channels = 256
    var height = 8
    var width = 8
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Create gradient
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var _backward_result = conv2d_backward(
        grad_output, input, kernel, stride, padding
    )
    var grad_input = _backward_result.grad_input

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())


def test_vgg16_conv512_forward() raises:
    """Test Conv2D forward pass with 512 output channels (VGG layers 4-5).

    Configuration:
        Input: (4, 256, 4, 4) - from after MaxPool3
        Conv: 3x3 kernel, 512 output channels, padding=1, stride=1
        Output: (4, 512, 4, 4)

    Deduplication Note:
        Conv4_1, Conv4_2, Conv4_3, Conv5_1, Conv5_2, Conv5_3 all use 512
        output channels. We test this once to represent all 6 layers.
    """
    var batch_size = 4
    var in_channels = 256
    var out_channels = 512
    var height = 4
    var width = 4
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_channels)
    assert_equal(output_shape[2], height)
    assert_equal(output_shape[3], width)


def test_vgg16_conv512_backward() raises:
    """Test Conv2D backward pass with 512 output channels."""
    var batch_size = 2
    var in_channels = 256
    var out_channels = 512
    var height = 4
    var width = 4
    var kernel_size = 3
    var padding = 1
    var stride = 1

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Create kernel
    var kernel_shape = List[Int]()
    kernel_shape.append(out_channels)
    kernel_shape.append(in_channels)
    kernel_shape.append(kernel_size)
    kernel_shape.append(kernel_size)
    var kernel = ones(kernel_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_channels)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = conv2d(input, kernel, bias, stride, padding)

    # Create gradient
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var _backward_result = conv2d_backward(
        grad_output, input, kernel, stride, padding
    )
    var grad_input = _backward_result.grad_input

    # Verify gradient shapes
    assert_shape(grad_input, input.shape())


def test_vgg16_maxpool_forward() raises:
    """Test MaxPool2D forward pass (2x2 kernel, stride 2).

    VGG-16 has 5 MaxPool layers between conv groups.
    All use identical: kernel_size=2, stride=2, no padding.

    Example: (batch, 64, 32, 32) -> (batch, 64, 16, 16)
    """
    var batch_size = 4
    var channels = 64
    var height = 32
    var width = 32

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Set some varying values for max pool selection
    for i in range(batch_size * channels * height * width):
        input[i] = Float32(i % 10)

    # MaxPool2D with 2x2 kernel, stride 2
    var kernel_size = 2
    var stride = 2
    var output = maxpool2d(input, kernel_size, stride)

    # Verify output shape (height and width are halved)
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], channels)
    assert_equal(output_shape[2], height // stride)
    assert_equal(output_shape[3], width // stride)


def test_vgg16_maxpool_backward() raises:
    """Test MaxPool2D backward pass."""
    var batch_size = 2
    var channels = 64
    var height = 32
    var width = 32

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Set varying values
    for i in range(batch_size * channels * height * width):
        input[i] = Float32(i % 10)

    # Forward pass
    var kernel_size = 2
    var stride = 2
    var output = maxpool2d(input, kernel_size, stride)

    # Create gradient w.r.t. output
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var grad_input = maxpool2d_backward(grad_output, input, kernel_size, stride)

    # Verify gradient shape matches input
    assert_shape(grad_input, input.shape())


def test_vgg16_relu_forward() raises:
    """Test ReLU forward pass.

    VGG-16 applies ReLU after all conv and FC layers.
    """
    var shape = List[Int]()
    shape.append(4)
    shape.append(64)
    shape.append(32)
    shape.append(32)
    var input = zeros(shape, DType.float32)

    # Create mixed positive and negative values
    for i in range(4 * 64 * 32 * 32):
        if i % 3 == 0:
            input[i] = Float32(1.5)  # Positive
        else:
            input[i] = Float32(-0.5)  # Negative

    # Apply ReLU
    var output = relu(input)

    # Verify shape preserved
    assert_shape(output, input.shape())

    # Verify values: positive should stay, negative should be zero
    for i in range(4 * 64 * 32 * 32):
        if i % 3 == 0:
            assert_almost_equal(
                output[i], Float32(1.5), tolerance=Float32(1e-5)
            )
        else:
            assert_almost_equal(output[i], 0.0, tolerance=1e-5)


def test_vgg16_relu_backward() raises:
    """Test ReLU backward pass.

    Gradient only flows for positive input values.
    """
    var shape = List[Int]()
    shape.append(4)
    shape.append(64)
    shape.append(8)
    shape.append(8)
    var input = zeros(shape, DType.float32)

    # Create mixed values
    for i in range(4 * 64 * 8 * 8):
        if i % 2 == 0:
            input[i] = Float32(2.0)
        else:
            input[i] = Float32(-1.0)

    # Forward pass
    var output = relu(input)

    # Create gradient w.r.t. output
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var grad_input = relu_backward(input, grad_output)

    # Verify gradient shape
    assert_shape(grad_input, input.shape())


def test_vgg16_fc_forward() raises:
    """Test fully connected layer forward pass.

    VGG-16 has 3 FC layers with ReLU between them (except final).
    Testing: 4096 -> 4096 (mid-layer FC)
    """
    var batch_size = 4
    var in_features = 4096
    var out_features = 4096

    # Create input: (batch_size, in_features)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights: (out_features, in_features)
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Create bias: (out_features,)
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)

    # Verify output shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], out_features)


def test_vgg16_fc_backward() raises:
    """Test fully connected layer backward pass."""
    var batch_size = 2
    var in_features = 4096
    var out_features = 4096

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)

    # Create gradient
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var backward_result = linear_backward(grad_output, input, weights)

    # Verify gradient shapes
    assert_shape(backward_result.grad_input, input.shape())


def test_vgg16_output_layer_forward() raises:
    """Test output layer forward pass (4096 -> 10 classes).

    This is the final fully connected layer producing logits.
    """
    var batch_size = 4
    var in_features = 4096
    var out_features = 10  # CIFAR-10 classes

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)

    # Verify output shape (logits for 10 classes)
    var output_shape = output.shape()
    assert_equal(output_shape[0], batch_size)
    assert_equal(output_shape[1], 10)


def test_vgg16_output_layer_backward() raises:
    """Test output layer backward pass."""
    var batch_size = 2
    var in_features = 4096
    var out_features = 10

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_features)
    var input = ones(input_shape, DType.float32)

    # Create weights
    var weights_shape = List[Int]()
    weights_shape.append(out_features)
    weights_shape.append(in_features)
    var weights = ones(weights_shape, DType.float32)

    # Create bias
    var bias_shape = List[Int]()
    bias_shape.append(out_features)
    var bias = zeros(bias_shape, DType.float32)

    # Forward pass
    var output = linear(input, weights, bias)

    # Create gradient
    var grad_output = ones(output.shape(), DType.float32)

    # Backward pass
    var backward_result = linear_backward(grad_output, input, weights)

    # Verify gradient shape
    assert_shape(backward_result.grad_input, input.shape())


def main() raises:
    """Run all test_vgg16_layers tests."""
    print("Running test_vgg16_layers tests...")

    test_vgg16_conv64_forward()
    print("✓ test_vgg16_conv64_forward")

    test_vgg16_conv64_backward()
    print("✓ test_vgg16_conv64_backward")

    test_vgg16_conv128_forward()
    print("✓ test_vgg16_conv128_forward")

    test_vgg16_conv128_backward()
    print("✓ test_vgg16_conv128_backward")

    test_vgg16_conv256_forward()
    print("✓ test_vgg16_conv256_forward")

    test_vgg16_conv256_backward()
    print("✓ test_vgg16_conv256_backward")

    test_vgg16_conv512_forward()
    print("✓ test_vgg16_conv512_forward")

    test_vgg16_conv512_backward()
    print("✓ test_vgg16_conv512_backward")

    test_vgg16_maxpool_forward()
    print("✓ test_vgg16_maxpool_forward")

    test_vgg16_maxpool_backward()
    print("✓ test_vgg16_maxpool_backward")

    test_vgg16_relu_forward()
    print("✓ test_vgg16_relu_forward")

    test_vgg16_relu_backward()
    print("✓ test_vgg16_relu_backward")

    test_vgg16_fc_forward()
    print("✓ test_vgg16_fc_forward")

    test_vgg16_fc_backward()
    print("✓ test_vgg16_fc_backward")

    test_vgg16_output_layer_forward()
    print("✓ test_vgg16_output_layer_forward")

    test_vgg16_output_layer_backward()
    print("✓ test_vgg16_output_layer_backward")

    print("\nAll test_vgg16_layers tests passed!")
