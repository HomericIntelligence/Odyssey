# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_vgg16_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for VGG-16 pooling, activation, and FC layer operations (Part 2 of 2).

VGG-16 Architecture Overview (~25 layer operations):
- 13 Conv2D layers (3x3 kernels, various channel depths)
- 5 MaxPool2D layers (2x2, stride 2)
- 3 Fully Connected (FC) layers
- ReLU activation after each conv and FC
- Dropout (0.5) in FC layers during training

This file contains Part 2: MaxPool, ReLU, and FC layer tests.
See test_vgg16_layers_part1.mojo for Conv layer tests (64, 128, 256, 512 channels).

All tests use pure functional API - no internal state or parameters.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
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


# ============================================================================
# MaxPool2D Tests
# ============================================================================
# VGG-16 uses 5 MaxPool layers: 2x2 kernel, stride 2


fn test_vgg16_maxpool_forward() raises:
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


fn test_vgg16_maxpool_backward() raises:
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


# ============================================================================
# ReLU Activation Tests
# ============================================================================
# VGG-16 uses ReLU after each conv and FC layer


fn test_vgg16_relu_forward() raises:
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


fn test_vgg16_relu_backward() raises:
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


# ============================================================================
# Fully Connected Layer Tests
# ============================================================================
# VGG-16 has 3 FC layers: 512*1*1 -> 4096 -> 4096 -> 10 (CIFAR-10)


fn test_vgg16_fc_forward() raises:
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


fn test_vgg16_fc_backward() raises:
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


# ============================================================================
# Final Output Layer Test
# ============================================================================
# VGG-16 final FC layer: 4096 -> 10 (CIFAR-10 classes)


fn test_vgg16_output_layer_forward() raises:
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


fn test_vgg16_output_layer_backward() raises:
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


fn main() raises:
    """Run VGG-16 pooling, activation, and FC layer tests (Part 2 of 2)."""
    print(
        "Starting VGG-16 Layerwise Tests (Part 2: MaxPool, ReLU, FC Layers)..."
    )

    # MaxPool tests
    print("  test_vgg16_maxpool_forward...", end="")
    test_vgg16_maxpool_forward()
    print(" OK")

    print("  test_vgg16_maxpool_backward...", end="")
    test_vgg16_maxpool_backward()
    print(" OK")

    # ReLU tests
    print("  test_vgg16_relu_forward...", end="")
    test_vgg16_relu_forward()
    print(" OK")

    print("  test_vgg16_relu_backward...", end="")
    test_vgg16_relu_backward()
    print(" OK")

    # FC layer tests
    print("  test_vgg16_fc_forward...", end="")
    test_vgg16_fc_forward()
    print(" OK")

    print("  test_vgg16_fc_backward...", end="")
    test_vgg16_fc_backward()
    print(" OK")

    # Output layer tests
    print("  test_vgg16_output_layer_forward...", end="")
    test_vgg16_output_layer_forward()
    print(" OK")

    print("  test_vgg16_output_layer_backward...", end="")
    test_vgg16_output_layer_backward()
    print(" OK")

    print("All VGG-16 Part 2 (MaxPool, ReLU, FC Layer) tests passed!")
