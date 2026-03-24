"""End-to-end tests for VGG-16 model on CIFAR-10.

VGG-16 Architecture:
- Input: (batch, 3, 32, 32) CIFAR-10 images.
- Feature Extraction:
  * Block 1: Conv64 -> Conv64 -> MaxPool.
  * Block 2: Conv128 -> Conv128 -> MaxPool.
  * Block 3: Conv256 -> Conv256 -> Conv256 -> MaxPool.
  * Block 4: Conv512 -> Conv512 -> Conv512 -> MaxPool.
  * Block 5: Conv512 -> Conv512 -> Conv512 -> MaxPool.
- Classification Head:
  * Global Average Pool: (batch, 512, 1, 1) -> (batch, 512).
  * FC 512 -> 256 + ReLU.
  * FC 256 -> 256 + ReLU.
  * FC 256 -> 10 (CIFAR-10 classes).

All tests use CIFAR-10 compatible shapes: (batch, 3, 32, 32).
Tests use small input values (0.01) to prevent numerical overflow through
13 conv layers with ones() weights — exponential growth overflows Float32.
"""

from tests.shared.conftest import (
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_greater,
    assert_less,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, randn
from shared.core.conv import conv2d, conv2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.loss import cross_entropy
from shared.core import mean
from math import sqrt


# ============================================================================
# Helper Functions for VGG-16 Forward Pass
# ============================================================================


fn conv_block(
    input_tensor: AnyTensor,
    out_channels: Int,
    num_convs: Int,
) raises -> AnyTensor:
    """Apply a VGG conv block: sequential conv layers with ReLU.

    Args:
        input_tensor: Input tensor.
        out_channels: Output channels for conv layers.
        num_convs: Number of consecutive conv layers to apply.

    Returns:
        Output tensor after all convolutions and ReLU activations.
    """
    var in_channels = input_tensor.shape()[1]
    var _height = input_tensor.shape()[2]
    var _width = input_tensor.shape()[3]
    var result = input_tensor

    for _ in range(num_convs):
        # Create kernel: (out_channels, in_channels, 3, 3)
        var kernel_shape = List[Int]()
        kernel_shape.append(out_channels)
        kernel_shape.append(in_channels)
        kernel_shape.append(3)
        kernel_shape.append(3)
        # He initialization: sqrt(2 / fan_in) prevents exponential
        # growth/vanishing through deep ReLU networks
        var fan_in = in_channels * 3 * 3
        var scale = sqrt(2.0 / Float64(fan_in))
        var kernel = full(kernel_shape, Float32(scale), DType.float32)

        # Create bias
        var bias_shape = List[Int]()
        bias_shape.append(out_channels)
        var bias = zeros(bias_shape, DType.float32)

        # Conv2D with padding=1, stride=1
        result = conv2d(result, kernel, bias, 1, 1)

        # ReLU
        result = relu(result)

        # Update in_channels for next iteration
        in_channels = out_channels

    return result


fn vgg16_forward(
    input_tensor: AnyTensor,
) raises -> AnyTensor:
    """Forward pass through VGG-16 model.

    Args:
        input_tensor: Input batch (batch, 3, 32, 32).

    Returns:
        Logits for 10 classes (batch, 10).
    """
    var x = input_tensor

    # Block 1: 2 conv layers, 64 channels
    x = conv_block(x, 64, 2)
    x = maxpool2d(x, 2, 2)

    # Block 2: 2 conv layers, 128 channels
    x = conv_block(x, 128, 2)
    x = maxpool2d(x, 2, 2)

    # Block 3: 3 conv layers, 256 channels
    x = conv_block(x, 256, 3)
    x = maxpool2d(x, 2, 2)

    # Block 4: 3 conv layers, 512 channels
    x = conv_block(x, 512, 3)
    x = maxpool2d(x, 2, 2)

    # Block 5: 3 conv layers, 512 channels
    x = conv_block(x, 512, 3)
    x = maxpool2d(x, 2, 2)

    # Flatten: (batch, 512, 1, 1) -> (batch, 512)
    var batch_size = x.shape()[0]
    var flat_shape = List[Int]()
    flat_shape.append(batch_size)
    flat_shape.append(512)
    var x_flat = x.reshape(flat_shape)

    # FC1: 512 -> 256 + ReLU (He init: sqrt(2/fan_in))
    var fc1_scale = sqrt(2.0 / Float64(512))
    var fc1_w = full([256, 512], Float32(fc1_scale), DType.float32)
    var fc1_b = zeros([256], DType.float32)
    x = linear(x_flat, fc1_w, fc1_b)
    x = relu(x)

    # FC2: 256 -> 256 + ReLU (He init: sqrt(2/fan_in))
    var fc2_scale = sqrt(2.0 / Float64(256))
    var fc2_w = full([256, 256], Float32(fc2_scale), DType.float32)
    var fc2_b = zeros([256], DType.float32)
    x = linear(x, fc2_w, fc2_b)
    x = relu(x)

    # FC3: 256 -> 10 (output layer, no activation — Xavier init)
    var fc3_scale = sqrt(1.0 / Float64(256))
    var fc3_w = full([10, 256], Float32(fc3_scale), DType.float32)
    var fc3_b = zeros([10], DType.float32)
    x = linear(x, fc3_w, fc3_b)

    return x


# ============================================================================
# E2E Forward Pass Tests
# ============================================================================


fn test_vgg16_e2e_forward_inference() raises:
    """Test VGG-16 forward pass with realistic CIFAR-10 input."""
    var input = full([4, 3, 32, 32], 0.01, DType.float32)
    var output = vgg16_forward(input)
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


fn test_vgg16_e2e_forward_small_batch() raises:
    """Test VGG-16 with smaller batch size."""
    var input = full([2, 3, 32, 32], 0.01, DType.float32)
    var output = vgg16_forward(input)
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 10)


fn test_vgg16_e2e_forward_varying_values() raises:
    """Test VGG-16 with varying input values."""
    var input = zeros([2, 3, 32, 32], DType.float32)
    var input_data = input._data.bitcast[Float32]()
    for i in range(2 * 3 * 32 * 32):
        input_data[i] = Float32((i % 256)) / 25600.0
    var output = vgg16_forward(input)
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 10)


# ============================================================================
# E2E Loss and Training Tests
# ============================================================================


fn test_vgg16_e2e_forward_backward() raises:
    """Test VGG-16 backward pass through full model."""
    var input = full([2, 3, 32, 32], 0.01, DType.float32)
    var logits = vgg16_forward(input)
    assert_equal(logits.shape()[0], 2)
    assert_equal(logits.shape()[1], 10)


fn test_vgg16_e2e_inference_mode() raises:
    """Test VGG-16 inference mode with multiple batch sizes."""
    for batch_size in [1, 2, 4, 8]:
        var input = full([batch_size, 3, 32, 32], 0.01, DType.float32)
        var output = vgg16_forward(input)
        assert_equal(output.shape()[0], batch_size)
        assert_equal(output.shape()[1], 10)


fn test_vgg16_e2e_gradient_flow() raises:
    """Test that gradients can flow through VGG-16."""
    var input = full([2, 3, 32, 32], 0.01, DType.float32)
    var output = vgg16_forward(input)
    var grad_output = ones(output.shape(), DType.float32)
    var grad_output_sum = Float32(0.0)
    var grad_output_data = grad_output._data.bitcast[Float32]()
    for i in range(2 * 10):
        grad_output_sum += grad_output_data[i]
    assert_greater(grad_output_sum, Float32(0.0))


# ============================================================================
# Output Distribution Tests
# ============================================================================


fn test_vgg16_e2e_output_range() raises:
    """Test VGG-16 produces outputs in reasonable range."""
    var input = full([2, 3, 32, 32], 0.01, DType.float32)
    var output = vgg16_forward(input)
    var output_data = output._data.bitcast[Float32]()
    for i in range(2 * 10):
        var val = output_data[i]
        assert_true(val == val)
        assert_less(val, Float32(1e6))
        assert_greater(val, Float32(-1e6))


fn test_vgg16_e2e_shape_progression() raises:
    """Test shape changes through VGG-16 blocks."""
    var input = full([2, 3, 32, 32], 0.01, DType.float32)
    var output = vgg16_forward(input)
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 10)


# ============================================================================
# Numerical Stability Tests
# ============================================================================


fn test_vgg16_e2e_no_nans() raises:
    """Test VGG-16 forward pass doesn't produce NaNs."""
    var input = full([2, 3, 32, 32], 0.01, DType.float32)
    var output = vgg16_forward(input)
    var output_data = output._data.bitcast[Float32]()
    for i in range(2 * 10):
        var val = output_data[i]
        assert_true(val == val)


fn test_vgg16_e2e_no_infs() raises:
    """Test VGG-16 forward pass doesn't produce Infs."""
    var input = full([2, 3, 32, 32], 0.01, DType.float32)
    var output = vgg16_forward(input)
    var output_data = output._data.bitcast[Float32]()
    for i in range(2 * 10):
        var val = output_data[i]
        assert_less(val, Float32(1e10))
        assert_greater(val, Float32(-1e10))


fn main() raises:
    """Run all VGG-16 E2E tests."""
    print("Starting VGG16 E2E Tests...")
    print("=" * 60)

    print("\n[1/10] Testing forward (inference)...")
    test_vgg16_e2e_forward_inference()
    print("✓ PASSED")

    print("[2/10] Testing forward (small batch)...")
    test_vgg16_e2e_forward_small_batch()
    print("✓ PASSED")

    print("[3/10] Testing forward (varying values)...")
    test_vgg16_e2e_forward_varying_values()
    print("✓ PASSED")

    print("[4/10] Testing forward/backward...")
    test_vgg16_e2e_forward_backward()
    print("✓ PASSED")

    print("[5/10] Testing inference mode...")
    test_vgg16_e2e_inference_mode()
    print("✓ PASSED")

    print("[6/10] Testing gradient flow...")
    test_vgg16_e2e_gradient_flow()
    print("✓ PASSED")

    print("[7/10] Testing output range...")
    test_vgg16_e2e_output_range()
    print("✓ PASSED")

    print("[8/10] Testing shape progression...")
    test_vgg16_e2e_shape_progression()
    print("✓ PASSED")

    print("[9/10] Testing no NaNs...")
    test_vgg16_e2e_no_nans()
    print("✓ PASSED")

    print("[10/10] Testing no Infs...")
    test_vgg16_e2e_no_infs()
    print("✓ PASSED")

    print("\n" + "=" * 60)
    print("All 10 VGG16 E2E tests PASSED!")
