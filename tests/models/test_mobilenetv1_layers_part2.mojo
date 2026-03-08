# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_mobilenetv1_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for MobileNetV1 depthwise separable blocks, BatchNorm, and ReLU layers.

Tests cover:
- Depthwise separable blocks: combined depthwise + pointwise
- BatchNorm2D: batch normalization with training and inference modes
- ReLU activation: non-linear activation function

Split from test_mobilenetv1_layers.mojo per ADR-009 (≤10 fn test_ per file).
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    full,
    zeros_like,
    ones_like,
)
from shared.core.conv import (
    depthwise_conv2d,
    depthwise_conv2d_backward,
    depthwise_conv2d_no_bias,
    conv2d,
    conv2d_backward,
)
from shared.core.activation import relu, relu_backward
from shared.core.layers.batchnorm import BatchNorm2dLayer
from shared.core.pooling import global_avgpool2d, global_avgpool2d_backward


# ============================================================================
# Depthwise Separable Block Tests
# ============================================================================


fn test_depthwise_separable_block_basic() raises:
    """Test depthwise separable block: complete forward pass.

    Block structure:
    1. Depthwise Conv 3x3 + ReLU
    2. Pointwise Conv 1x1 + ReLU

    This tests the combined operation without BatchNorm to isolate the core logic.
    """
    var batch_size = 1
    var in_channels = 32
    var out_channels = 64
    var height = 8
    var width = 8

    # Create input: (1, 32, 8, 8)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Stage 1: Depthwise conv
    var depthwise_kernel_shape = List[Int]()
    depthwise_kernel_shape.append(in_channels)
    depthwise_kernel_shape.append(1)
    depthwise_kernel_shape.append(3)
    depthwise_kernel_shape.append(3)
    var depthwise_kernel = ones(depthwise_kernel_shape, DType.float32)

    var depthwise_bias_shape = List[Int]()
    depthwise_bias_shape.append(in_channels)
    var depthwise_bias = zeros(depthwise_bias_shape, DType.float32)

    var depthwise_output = depthwise_conv2d(
        input, depthwise_kernel, depthwise_bias, stride=1, padding=1
    )

    # Apply ReLU
    var depthwise_relu = relu(depthwise_output)

    # Stage 2: Pointwise conv
    var pointwise_kernel_shape = List[Int]()
    pointwise_kernel_shape.append(out_channels)
    pointwise_kernel_shape.append(in_channels)
    pointwise_kernel_shape.append(1)
    pointwise_kernel_shape.append(1)
    var pointwise_kernel = ones(pointwise_kernel_shape, DType.float32)

    var pointwise_bias_shape = List[Int]()
    pointwise_bias_shape.append(out_channels)
    var pointwise_bias = zeros(pointwise_bias_shape, DType.float32)

    var output = conv2d(
        depthwise_relu, pointwise_kernel, pointwise_bias, stride=1, padding=0
    )

    # Apply ReLU
    var output_relu = relu(output)

    # Verify final output shape
    var out_shape = output_relu.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


fn test_depthwise_separable_block_with_stride() raises:
    """Test depthwise separable block with stride 2 for downsampling.

    MobileNetV1 uses stride=2 in blocks 2, 4, and 6 for spatial reduction.
    """
    var batch_size = 1
    var in_channels = 64
    var out_channels = 128
    var in_height = 14
    var in_width = 14
    var stride = 2

    # Create input: (1, 64, 14, 14)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(in_height)
    input_shape.append(in_width)
    var input = ones(input_shape, DType.float32)

    # Depthwise conv with stride 2
    var depthwise_kernel_shape = List[Int]()
    depthwise_kernel_shape.append(in_channels)
    depthwise_kernel_shape.append(1)
    depthwise_kernel_shape.append(3)
    depthwise_kernel_shape.append(3)
    var depthwise_kernel = ones(depthwise_kernel_shape, DType.float32)

    var depthwise_bias_shape = List[Int]()
    depthwise_bias_shape.append(in_channels)
    var depthwise_bias = zeros(depthwise_bias_shape, DType.float32)

    var depthwise_output = depthwise_conv2d(
        input, depthwise_kernel, depthwise_bias, stride=stride, padding=1
    )

    # Verify depthwise output shape after stride
    var dw_shape = depthwise_output.shape()
    assert_equal(dw_shape[2], 7)  # (14 + 2*1 - 3) // 2 + 1 = 7
    assert_equal(dw_shape[3], 7)

    # Pointwise conv
    var pointwise_kernel_shape = List[Int]()
    pointwise_kernel_shape.append(out_channels)
    pointwise_kernel_shape.append(in_channels)
    pointwise_kernel_shape.append(1)
    pointwise_kernel_shape.append(1)
    var pointwise_kernel = ones(pointwise_kernel_shape, DType.float32)

    var pointwise_bias_shape = List[Int]()
    pointwise_bias_shape.append(out_channels)
    var pointwise_bias = zeros(pointwise_bias_shape, DType.float32)

    var output = conv2d(
        depthwise_output, pointwise_kernel, pointwise_bias, stride=1, padding=0
    )

    # Verify final output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], 7)
    assert_equal(out_shape[3], 7)


# ============================================================================
# BatchNorm Tests
# ============================================================================


fn test_batchnorm2d_initialization() raises:
    """Test BatchNorm2D layer initialization.

    Gamma (scale) should be initialized to 1.0
    Beta (shift) should be initialized to 0.0
    Running mean/variance for inference mode
    """
    var num_channels = 64

    # Initialize BatchNorm2D layer
    var bn = BatchNorm2dLayer(num_channels, momentum=0.1, eps=1e-5)

    # Verify parameters shape
    var gamma_shape = bn.gamma.shape()
    var beta_shape = bn.beta.shape()
    var running_mean_shape = bn.running_mean.shape()
    var running_var_shape = bn.running_var.shape()

    assert_equal(gamma_shape[0], num_channels)
    assert_equal(beta_shape[0], num_channels)
    assert_equal(running_mean_shape[0], num_channels)
    assert_equal(running_var_shape[0], num_channels)

    # Verify gamma is initialized to 1.0
    var gamma_data = bn.gamma._data.bitcast[Float32]()
    assert_almost_equal(gamma_data[0], 1.0, tolerance=1e-5)

    # Verify beta is initialized to 0.0
    var beta_data = bn.beta._data.bitcast[Float32]()
    assert_almost_equal(beta_data[0], 0.0, tolerance=1e-5)


fn test_batchnorm2d_forward_training() raises:
    """Test BatchNorm2D forward pass in training mode.

    In training mode:
    - Uses batch statistics (mean, variance)
    - Updates running statistics with exponential moving average
    """
    var batch_size = 4
    var num_channels = 32
    var height = 8
    var width = 8

    # Create input: (4, 32, 8, 8)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(num_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Initialize BatchNorm2D
    var bn = BatchNorm2dLayer(num_channels)

    # Forward pass in training mode
    var output = bn.forward(input, training=True)

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], num_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


fn test_batchnorm2d_forward_inference() raises:
    """Test BatchNorm2D forward pass in inference mode.

    In inference mode:
    - Uses running statistics (mean, variance)
    - Does not update statistics
    """
    var batch_size = 4
    var num_channels = 32
    var height = 8
    var width = 8

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(num_channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Initialize BatchNorm2D
    var bn = BatchNorm2dLayer(num_channels)

    # Forward pass in inference mode
    var output = bn.forward(input, training=False)

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], num_channels)


# ============================================================================
# ReLU Activation Tests
# ============================================================================


fn test_relu_activation_basic() raises:
    """Test ReLU activation: clamps negative values to 0.

    ReLU(x) = max(0, x)
    Used after every convolution in MobileNetV1.
    """
    var batch_size = 1
    var channels = 32
    var height = 4
    var width = 4

    # Create input with mixed positive and negative values
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Set some values to negative
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -1.0
    input_data[1] = 0.5
    input_data[2] = -0.5
    input_data[3] = 2.0

    # Apply ReLU
    var output = relu(input)

    # Verify shape is preserved
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)

    # Verify values: negatives become 0, positives stay same
    var output_data = output._data.bitcast[Float32]()
    assert_almost_equal(output_data[0], 0.0, tolerance=1e-5)  # -1.0 -> 0.0
    assert_almost_equal(output_data[1], 0.5, tolerance=1e-5)  # 0.5 -> 0.5
    assert_almost_equal(output_data[2], 0.0, tolerance=1e-5)  # -0.5 -> 0.0
    assert_almost_equal(output_data[3], 2.0, tolerance=1e-5)  # 2.0 -> 2.0


fn test_relu_multiple_applications() raises:
    """Test ReLU applied multiple times in a forward pass sequence.

    Verifies that ReLU can be applied repeatedly without issues.
    """
    var batch_size = 1
    var channels = 4
    var height = 2
    var width = 2

    # Create input with mixed values
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var input = ones(input_shape, DType.float32)

    # Set some values negative
    var input_data = input._data.bitcast[Float32]()
    input_data[0] = -1.0
    input_data[1] = 0.5
    input_data[2] = 2.0
    input_data[3] = -0.5

    # First ReLU
    var output1 = relu(input)

    # Second ReLU (idempotent after first ReLU)
    var output2 = relu(output1)

    # Verify shapes are preserved
    var shape1 = output1.shape()
    var shape2 = output2.shape()
    assert_equal(shape1[0], batch_size)
    assert_equal(shape2[0], batch_size)

    # After ReLU, all negative values should be 0
    var out_data = output2._data.bitcast[Float32]()
    assert_almost_equal(out_data[0], 0.0, tolerance=1e-5)  # -1.0 -> 0.0
    assert_almost_equal(out_data[1], 0.5, tolerance=1e-5)  # 0.5 -> 0.5
    assert_almost_equal(out_data[2], 2.0, tolerance=1e-5)  # 2.0 -> 2.0
    assert_almost_equal(out_data[3], 0.0, tolerance=1e-5)  # -0.5 -> 0.0


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    print("Starting MobileNetV1 Layerwise Tests Part 2 (Separable Blocks + BatchNorm + ReLU)...")

    print("  test_depthwise_separable_block_basic...", end="")
    test_depthwise_separable_block_basic()
    print(" OK")

    print("  test_depthwise_separable_block_with_stride...", end="")
    test_depthwise_separable_block_with_stride()
    print(" OK")

    print("  test_batchnorm2d_initialization...", end="")
    test_batchnorm2d_initialization()
    print(" OK")

    print("  test_batchnorm2d_forward_training...", end="")
    test_batchnorm2d_forward_training()
    print(" OK")

    print("  test_batchnorm2d_forward_inference...", end="")
    test_batchnorm2d_forward_inference()
    print(" OK")

    print("  test_relu_activation_basic...", end="")
    test_relu_activation_basic()
    print(" OK")

    print("  test_relu_multiple_applications...", end="")
    test_relu_multiple_applications()
    print(" OK")

    print("All MobileNetV1 layerwise tests part 2 passed!")
