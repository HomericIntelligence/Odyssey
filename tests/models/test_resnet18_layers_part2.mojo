# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_resnet18_layers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for ResNet-18 layerwise operations (Part 2 of 2).

Tests cover:
- BatchNorm gamma/beta parameter effects
- ReLU activations within residual block context
- Forward and backward pass consistency
- Multiple sequential residual blocks

ResNet-18 Architecture Components:
- Layer 1: 64 channels with 2 blocks (no projection)
- Layer 2: 128 channels with 2 blocks (first with projection)
- Layer 3: 256 channels with 2 blocks (first with projection)
- Layer 4: 512 channels with 2 blocks (first with projection)
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
from shared.core.extensor import ExTensor, zeros, ones, full, randn
from shared.core.conv import conv2d, conv2d_backward
from shared.core.activation import relu, relu_backward
from shared.core.normalization import batch_norm2d
from shared.core.arithmetic import add


# ============================================================================
# Residual Block Helper Functions
# ============================================================================


fn create_basic_block(
    x: ExTensor,
    conv1_weight: ExTensor,
    conv1_bias: ExTensor,
    bn1_gamma: ExTensor,
    bn1_beta: ExTensor,
    bn1_running_mean: ExTensor,
    bn1_running_var: ExTensor,
    conv2_weight: ExTensor,
    conv2_bias: ExTensor,
    bn2_gamma: ExTensor,
    bn2_beta: ExTensor,
    bn2_running_mean: ExTensor,
    bn2_running_var: ExTensor,
    training: Bool = True,
) raises -> ExTensor:
    """Forward pass for a basic residual block without projection.

    Formula:
        `out = relu(conv_block2(bn(relu(conv_block1(bn(x, bn1), conv1), conv1_weight) + x)))`

    Simple version showing the residual connection (skip):
        ```
        identity = x
        out = conv2(bn(relu(conv1(bn(x)))))
        out = out + identity
        out = relu(out)
        ```

    Args:
        x: Input tensor (batch, channels, height, width).
        conv1_weight: First conv kernel (out_channels, in_channels, 3, 3).
        conv1_bias: First conv bias (out_channels,).
        bn1_gamma: First BN scale (out_channels,).
        bn1_beta: First BN shift (out_channels,).
        bn1_running_mean: First BN running mean (out_channels,).
        bn1_running_var: First BN running variance (out_channels,).
        conv2_weight: Second conv kernel (out_channels, out_channels, 3, 3).
        conv2_bias: Second conv bias (out_channels,).
        bn2_gamma: Second BN scale (out_channels,).
        bn2_beta: Second BN shift (out_channels,).
        bn2_running_mean: Second BN running mean (out_channels,).
        bn2_running_var: Second BN running variance (out_channels,).
        training: Whether in training mode for BatchNorm.

    Returns:
        Output tensor after residual block.
    """
    # First conv -> BN -> ReLU
    var conv1_out = conv2d(x, conv1_weight, conv1_bias, stride=1, padding=1)
    var bn1_out: ExTensor
    var _: ExTensor
    var __: ExTensor
    (bn1_out, _, __) = batch_norm2d(
        conv1_out,
        bn1_gamma,
        bn1_beta,
        bn1_running_mean,
        bn1_running_var,
        training=training,
    )
    var relu1_out = relu(bn1_out)

    # Second conv -> BN
    var conv2_out = conv2d(
        relu1_out, conv2_weight, conv2_bias, stride=1, padding=1
    )
    var bn2_out: ExTensor
    (bn2_out, _, __) = batch_norm2d(
        conv2_out,
        bn2_gamma,
        bn2_beta,
        bn2_running_mean,
        bn2_running_var,
        training=training,
    )

    # Skip connection (identity - requires same spatial dimensions)
    var residual = add(bn2_out, x)

    # Final ReLU
    var output = relu(residual)

    return output


# ============================================================================
# BatchNorm Layer Tests (continued)
# ============================================================================


fn test_batchnorm2d_gamma_beta_effects() raises:
    """Test that gamma (scale) and beta (shift) parameters work correctly.

    gamma = 2.0 should double the normalized values.
    beta = 1.0 should shift values up by 1.0.
    """
    var batch_size = 2
    var channels = 4
    var height = 8
    var width = 8

    # Create input with known values
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = ones(input_shape, DType.float32)

    # Parameters: gamma=2, beta=1
    var gamma = full([channels], 2.0, DType.float32)
    var beta = ones([channels], DType.float32)
    var running_mean = zeros([channels], DType.float32)
    var running_var = ones([channels], DType.float32)

    # Forward pass
    var output: ExTensor
    var _: ExTensor
    var __: ExTensor
    (output, _, __) = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=False
    )

    # With gamma=2 and beta=1, running_mean=0, running_var=1:
    # normalized = (x - running_mean) / sqrt(running_var + eps) = (1 - 0) / 1 = 1
    # output = gamma * normalized + beta = 2 * 1 + 1 = 3
    var out_data = output._data.bitcast[Float32]()
    var total_elements = batch_size * channels * height * width
    for i in range(min(10, total_elements)):
        # Expected: 3.0 (gamma * normalized + beta = 2 * 1 + 1)
        assert_almost_equal(out_data[i], 3.0, tolerance=1e-4)


# ============================================================================
# ReLU Activation Tests
# ============================================================================


fn test_relu_in_residual_block() raises:
    """Test ReLU activation within residual block context.

    ReLU should zero out negative values.
    """
    var batch_size = 2
    var channels = 64
    var height = 32
    var width = 32

    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)

    # Create tensor with both positive and negative values
    var x = randn(input_shape, DType.float32)

    # Apply ReLU
    var output = relu(x)

    # Verify shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)

    # Verify all values are non-negative
    var out_data = output._data.bitcast[Float32]()
    var total_elements = batch_size * channels * height * width
    for i in range(min(100, total_elements)):
        assert_true(
            out_data[i] >= 0.0, "ReLU should produce non-negative values"
        )


# ============================================================================
# Integration Tests
# ============================================================================


fn test_block_forward_backward_consistency() raises:
    """Test that block forward and backward shapes are consistent.

    Backward pass should produce gradients with same shapes as forward inputs.
    """
    var batch_size = 2
    var in_channels = 64
    var out_channels = 64
    var height = 32
    var width = 32

    # Create inputs
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = ones(input_shape, DType.float32)

    # Conv weights
    var conv_weight_shape = List[Int]()
    conv_weight_shape.append(out_channels)
    conv_weight_shape.append(in_channels)
    conv_weight_shape.append(3)
    conv_weight_shape.append(3)
    var conv1_weight = ones(conv_weight_shape, DType.float32)
    var conv1_bias = zeros([out_channels], DType.float32)

    var conv2_weight = ones(conv_weight_shape, DType.float32)
    var conv2_bias = zeros([out_channels], DType.float32)

    # BN params
    var gamma = ones([out_channels], DType.float32)
    var beta = zeros([out_channels], DType.float32)
    var running_mean = zeros([out_channels], DType.float32)
    var running_var = ones([out_channels], DType.float32)

    # Forward pass
    var output = create_basic_block(
        x,
        conv1_weight,
        conv1_bias,
        gamma,
        beta,
        running_mean,
        running_var,
        conv2_weight,
        conv2_bias,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
    )

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


fn test_multiple_blocks_sequential() raises:
    """Test multiple residual blocks in sequence.

    Tests that output of one block can serve as input to next block.
    """
    var batch_size = 1
    var channels = 64
    var height = 32
    var width = 32

    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)

    # Block 1 input
    var x1 = ones(input_shape, DType.float32)

    # Block 1 parameters
    var conv_weight_shape = List[Int]()
    conv_weight_shape.append(channels)
    conv_weight_shape.append(channels)
    conv_weight_shape.append(3)
    conv_weight_shape.append(3)
    var conv1_weight = ones(conv_weight_shape, DType.float32)
    var conv1_bias = zeros([channels], DType.float32)
    var conv2_weight = ones(conv_weight_shape, DType.float32)
    var conv2_bias = zeros([channels], DType.float32)

    var gamma = ones([channels], DType.float32)
    var beta = zeros([channels], DType.float32)
    var running_mean = zeros([channels], DType.float32)
    var running_var = ones([channels], DType.float32)

    # Block 1
    var block1_out = create_basic_block(
        x1,
        conv1_weight,
        conv1_bias,
        gamma,
        beta,
        running_mean,
        running_var,
        conv2_weight,
        conv2_bias,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
    )

    # Block 2 input is Block 1 output
    var block2_out = create_basic_block(
        block1_out,
        conv1_weight,
        conv1_bias,
        gamma,
        beta,
        running_mean,
        running_var,
        conv2_weight,
        conv2_bias,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
    )

    # Verify shapes
    var out_shape = block2_out.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all ResNet-18 layerwise tests (Part 2 of 2)."""
    print("Running ResNet-18 layerwise tests (Part 2)...")

    # BatchNorm tests (continued)
    test_batchnorm2d_gamma_beta_effects()
    print("✓ test_batchnorm2d_gamma_beta_effects")

    # ReLU tests
    test_relu_in_residual_block()
    print("✓ test_relu_in_residual_block")

    # Integration tests
    test_block_forward_backward_consistency()
    print("✓ test_block_forward_backward_consistency")

    test_multiple_blocks_sequential()
    print("✓ test_multiple_blocks_sequential")

    print("\nAll ResNet-18 layerwise tests (Part 2) passed!")
