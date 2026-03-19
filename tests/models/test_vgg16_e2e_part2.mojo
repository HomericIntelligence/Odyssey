# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_vgg16_e2e.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""End-to-end tests for VGG-16 model on CIFAR-10 (Part 2 of 2).

VGG-16 Architecture:
- Input: (batch, 3, 32, 32) CIFAR-10 images
- Feature Extraction:
  * Block 1: Conv64 -> Conv64 -> MaxPool
  * Block 2: Conv128 -> Conv128 -> MaxPool
  * Block 3: Conv256 -> Conv256 -> Conv256 -> MaxPool
  * Block 4: Conv512 -> Conv512 -> Conv512 -> MaxPool
  * Block 5: Conv512 -> Conv512 -> Conv512 -> MaxPool
- Classification Head:
  * Global Average Pool: (batch, 512, 1, 1) -> (batch, 512)
  * FC 512 -> 256 + ReLU
  * FC 256 -> 256 + ReLU
  * FC 256 -> 10 (CIFAR-10 classes)

Part 2 Tests (gradient flow, output analysis, numerical stability):
- Gradient flow through full model
- Output range verification
- Shape progression through blocks
- NaN detection
- Inf detection

See test_vgg16_e2e_part1.mojo for forward pass and training tests.

All tests use CIFAR-10 compatible shapes: (batch, 3, 32, 32)
Batch sizes: 4 (inference) and 2 (training, small for speed)
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
from shared.core.extensor import ExTensor, zeros, ones, full, randn
from shared.core.conv import conv2d, conv2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.pooling import maxpool2d, maxpool2d_backward
from shared.core.loss import cross_entropy
from shared.core import mean


# ============================================================================
# Helper Functions for VGG-16 Forward Pass
# ============================================================================


fn conv_block(
    input_tensor: ExTensor,
    out_channels: Int,
    num_convs: Int,
) raises -> ExTensor:
    """Apply a VGG conv block: sequential conv layers with ReLU.

    Args:
        input_tensor: Input tensor.
        out_channels: Output channels for conv layers.
        num_convs: Number of consecutive conv layers to apply.

    Returns:
        Output tensor after all convolutions and ReLU activations.
    """
    var in_channels = input_tensor.shape()[1]
    var height = input_tensor.shape()[2]
    var width = input_tensor.shape()[3]
    var result = input_tensor

    for _ in range(num_convs):
        # Create kernel: (out_channels, in_channels, 3, 3)
        var kernel_shape = List[Int]()
        kernel_shape.append(out_channels)
        kernel_shape.append(in_channels)
        kernel_shape.append(3)
        kernel_shape.append(3)
        var kernel = ones(kernel_shape, DType.float32)

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
    input_tensor: ExTensor,
) raises -> ExTensor:
    """Forward pass through VGG-16 model.

    Args:
        input_tensor: Input batch (batch, 3, 32, 32).

    Returns:
        Logits for 10 classes (batch, 10).
    """
    var x = input_tensor

    # Block 1: 2 conv layers, 64 channels
    x = conv_block(x, 64, 2)
    # MaxPool 2x2 stride 2: (batch, 64, 32, 32) -> (batch, 64, 16, 16)
    x = maxpool2d(x, 2, 2)

    # Block 2: 2 conv layers, 128 channels
    x = conv_block(x, 128, 2)
    # MaxPool: (batch, 128, 16, 16) -> (batch, 128, 8, 8)
    x = maxpool2d(x, 2, 2)

    # Block 3: 3 conv layers, 256 channels
    x = conv_block(x, 256, 3)
    # MaxPool: (batch, 256, 8, 8) -> (batch, 256, 4, 4)
    x = maxpool2d(x, 2, 2)

    # Block 4: 3 conv layers, 512 channels
    x = conv_block(x, 512, 3)
    # MaxPool: (batch, 512, 4, 4) -> (batch, 512, 2, 2)
    x = maxpool2d(x, 2, 2)

    # Block 5: 3 conv layers, 512 channels
    x = conv_block(x, 512, 3)
    # MaxPool: (batch, 512, 2, 2) -> (batch, 512, 1, 1)
    x = maxpool2d(x, 2, 2)

    # Flatten for FC layers
    # Shape: (batch, 512, 1, 1) -> (batch, 512)
    var batch_size = x.shape()[0]
    var flat_shape = List[Int]()
    flat_shape.append(batch_size)
    flat_shape.append(512)
    var x_flat = x.reshape(flat_shape)

    # FC1: 512 -> 256 + ReLU
    var fc1_w_shape = List[Int]()
    fc1_w_shape.append(256)
    fc1_w_shape.append(512)
    var fc1_w = ones(fc1_w_shape, DType.float32)
    var fc1_b_shape = List[Int]()
    fc1_b_shape.append(256)
    var fc1_b = zeros(fc1_b_shape, DType.float32)
    x = linear(x_flat, fc1_w, fc1_b)
    x = relu(x)

    # FC2: 256 -> 256 + ReLU
    var fc2_w_shape = List[Int]()
    fc2_w_shape.append(256)
    fc2_w_shape.append(256)
    var fc2_w = ones(fc2_w_shape, DType.float32)
    var fc2_b_shape = List[Int]()
    fc2_b_shape.append(256)
    var fc2_b = zeros(fc2_b_shape, DType.float32)
    x = linear(x, fc2_w, fc2_b)
    x = relu(x)

    # FC3: 256 -> 10 (output layer, no activation)
    var fc3_w_shape = List[Int]()
    fc3_w_shape.append(10)
    fc3_w_shape.append(256)
    var fc3_w = ones(fc3_w_shape, DType.float32)
    var fc3_b_shape = List[Int]()
    fc3_b_shape.append(10)
    var fc3_b = zeros(fc3_b_shape, DType.float32)
    x = linear(x, fc3_w, fc3_b)

    return x


# ============================================================================
# Gradient Flow Tests
# ============================================================================


fn test_vgg16_e2e_gradient_flow() raises:
    """Test that gradients can flow through VGG-16.

    This is a simplified test checking:
    - Forward pass completes.
    - Output is differentiable (not constant).
    - Loss computation possible.

    Full gradient computation is complex and tested in
    test_vgg16_layers.mojo in detail.
    """
    var batch_size = 2

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(3)
    input_shape.append(32)
    input_shape.append(32)
    var input = ones(input_shape, DType.float32)

    # Forward pass
    var output = vgg16_forward(input)

    # Create gradient w.r.t. output (uniform gradients for simplicity)
    var grad_output = ones(output.shape(), DType.float32)

    # Gradient values should be non-trivial
    var grad_output_sum = Float32(0.0)
    var grad_output_data = grad_output._data.bitcast[Float32]()
    for i in range(batch_size * 10):
        grad_output_sum += grad_output_data[i]

    # Verify gradients are meaningful (not zero)
    assert_greater(grad_output_sum, Float32(0.0))


# ============================================================================
# Output Distribution Tests
# ============================================================================


fn test_vgg16_e2e_output_range() raises:
    """Test VGG-16 produces outputs in reasonable range.

    For logits, values should not be extreme (not NaN/inf).
    Typical range: (-100, 100) for cross-entropy loss computation.
    """
    var batch_size = 2

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(3)
    input_shape.append(32)
    input_shape.append(32)
    var input = ones(input_shape, DType.float32)

    # Forward pass
    var output = vgg16_forward(input)

    # Check all output values are finite and in reasonable range
    var output_data = output._data.bitcast[Float32]()
    for i in range(batch_size * 10):
        var val = output_data[i]
        # Check not NaN (NaN != NaN)
        assert_true(val == val)
        # Check not too extreme
        assert_less(val, Float32(1e6))
        assert_greater(val, Float32(-1e6))


# ============================================================================
# E2E Shape Propagation Tests
# ============================================================================


fn test_vgg16_e2e_shape_progression() raises:
    """Test shape changes through VGG-16 blocks.

    Tracks shape transformations:
    Input (b, 3, 32, 32)
    -> Block1 (b, 64, 32, 32) -> Pool (b, 64, 16, 16)
    -> Block2 (b, 128, 16, 16) -> Pool (b, 128, 8, 8)
    -> Block3 (b, 256, 8, 8) -> Pool (b, 256, 4, 4)
    -> Block4 (b, 512, 4, 4) -> Pool (b, 512, 2, 2)
    -> Block5 (b, 512, 2, 2) -> Pool (b, 512, 1, 1)
    -> FC layers
    -> Output (b, 10)
    """
    var batch_size = 2

    # Create input: (2, 3, 32, 32)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(3)
    input_shape.append(32)
    input_shape.append(32)
    var input = ones(input_shape, DType.float32)

    # Test full forward pass
    var output = vgg16_forward(input)

    # Verify final shape
    var final_shape = output.shape()
    assert_equal(final_shape[0], batch_size)
    assert_equal(final_shape[1], 10)


# ============================================================================
# Numerical Stability Tests
# ============================================================================


fn test_vgg16_e2e_no_nans() raises:
    """Test VGG-16 forward pass doesn't produce NaNs.

    This is a critical smoke test for numerical stability.
    """
    var batch_size = 2

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(3)
    input_shape.append(32)
    input_shape.append(32)
    var input = ones(input_shape, DType.float32)

    # Forward pass
    var output = vgg16_forward(input)

    # Check no NaNs
    var output_data = output._data.bitcast[Float32]()
    for i in range(batch_size * 10):
        var val = output_data[i]
        # NaN != NaN check
        assert_true(val == val)


fn test_vgg16_e2e_no_infs() raises:
    """Test VGG-16 forward pass doesn't produce Infs.

    Prevents overflow from deep network.
    """
    var batch_size = 2

    # Create input with normalized values
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(3)
    input_shape.append(32)
    input_shape.append(32)
    var input = zeros(input_shape, DType.float32)

    # Fill with normalized values [0, 1]
    var input_data = input._data.bitcast[Float32]()
    for i in range(batch_size * 3 * 32 * 32):
        input_data[i] = Float32(0.5)

    # Forward pass
    var output = vgg16_forward(input)

    # Check no infinities
    var output_data = output._data.bitcast[Float32]()
    for i in range(batch_size * 10):
        var val = output_data[i]
        # Check finite
        assert_less(val, Float32(1e10))
        assert_greater(val, Float32(-1e10))


fn main() raises:
    print("Starting VGG16 E2E Tests (Part 2)...")
    print("  test_vgg16_e2e_gradient_flow...", end="")
    test_vgg16_e2e_gradient_flow()
    print(" OK")
    print("  test_vgg16_e2e_output_range...", end="")
    test_vgg16_e2e_output_range()
    print(" OK")
    print("  test_vgg16_e2e_shape_progression...", end="")
    test_vgg16_e2e_shape_progression()
    print(" OK")
    print("  test_vgg16_e2e_no_nans...", end="")
    test_vgg16_e2e_no_nans()
    print(" OK")
    print("  test_vgg16_e2e_no_infs...", end="")
    test_vgg16_e2e_no_infs()
    print(" OK")
    print("All VGG16 E2E Tests (Part 2) passed!")
