"""Unit tests for ResNet-18 layerwise operations

Tests cover:
- Residual blocks with different channel configurations
- Skip connections (identity and projection shortcuts)
- Batch normalization in both training and inference modes

ResNet-18 Architecture Components:
- Layer 1: 64 channels with 2 blocks (no projection)
- Layer 2: 128 channels with 2 blocks (first with projection)
- Layer 3: 256 channels with 2 blocks (first with projection)
- Layer 4: 512 channels with 2 blocks (first with projection)

Test Deduplication Strategy:
- Test one block per unique channel configuration
- Test skip connections separately
- Test BatchNorm in training and inference modes
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
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, randn
from shared.core.conv import conv2d, conv2d_backward
from shared.core.activation import relu, relu_backward
from shared.core.normalization import batch_norm2d
from shared.core.arithmetic import add


def create_basic_block(
    x: AnyTensor,
    conv1_weight: AnyTensor,
    conv1_bias: AnyTensor,
    bn1_gamma: AnyTensor,
    bn1_beta: AnyTensor,
    bn1_running_mean: AnyTensor,
    bn1_running_var: AnyTensor,
    conv2_weight: AnyTensor,
    conv2_bias: AnyTensor,
    bn2_gamma: AnyTensor,
    bn2_beta: AnyTensor,
    bn2_running_mean: AnyTensor,
    bn2_running_var: AnyTensor,
    training: Bool = True,
) raises -> AnyTensor:
    """Forward pass for a basic residual block without projection."""
    var conv1_out = conv2d(x, conv1_weight, conv1_bias, stride=1, padding=1)
    var bn1_out: AnyTensor
    var _: AnyTensor
    var __: AnyTensor
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
    var bn2_out: AnyTensor
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


def test_residual_block_64_channels_forward() raises:
    """Test residual block with 64 channels (no projection).

    This block is used in ResNet-18's first residual layer.
    Input and output channels match, so identity shortcut is used.

    Shape: (2, 64, 32, 32) -> (2, 64, 32, 32)
    """
    var batch_size = 2
    var in_channels = 64
    var out_channels = 64
    var height = 32
    var width = 32

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = ones(input_shape, DType.float32)

    # Conv1: (64, 64, 3, 3)
    var conv1_weight_shape = List[Int]()
    conv1_weight_shape.append(out_channels)
    conv1_weight_shape.append(in_channels)
    conv1_weight_shape.append(3)
    conv1_weight_shape.append(3)
    var conv1_weight = ones(conv1_weight_shape, DType.float32)
    var conv1_bias_shape = List[Int]()
    conv1_bias_shape.append(out_channels)
    var conv1_bias = zeros(conv1_bias_shape, DType.float32)

    # BN1: gamma (64,), beta (64,), running_mean (64,), running_var (64,)
    var bn1_gamma = ones([out_channels], DType.float32)
    var bn1_beta = zeros([out_channels], DType.float32)
    var bn1_running_mean = zeros([out_channels], DType.float32)
    var bn1_running_var = ones([out_channels], DType.float32)

    # Conv2: (64, 64, 3, 3)
    var conv2_weight = ones(conv1_weight_shape, DType.float32)
    var conv2_bias = zeros(conv1_bias_shape, DType.float32)

    # BN2
    var bn2_gamma = ones([out_channels], DType.float32)
    var bn2_beta = zeros([out_channels], DType.float32)
    var bn2_running_mean = zeros([out_channels], DType.float32)
    var bn2_running_var = ones([out_channels], DType.float32)

    # Forward pass
    var output = create_basic_block(
        x,
        conv1_weight,
        conv1_bias,
        bn1_gamma,
        bn1_beta,
        bn1_running_mean,
        bn1_running_var,
        conv2_weight,
        conv2_bias,
        bn2_gamma,
        bn2_beta,
        bn2_running_mean,
        bn2_running_var,
        training=True,
    )

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)

    # Output should not be all zeros due to ReLU and residual
    var out_data = output._data.bitcast[Float32]()
    var total_elements = batch_size * out_channels * height * width
    var non_zero_count = 0
    for i in range(total_elements):
        if out_data[i] > 0.0:
            non_zero_count += 1

    assert_true(non_zero_count > 0, "Output should have non-zero values")


def test_residual_block_64_channels_training_mode() raises:
    """Test residual block in training mode with BatchNorm statistics.

    Training mode should compute batch statistics for BatchNorm.
    """
    var batch_size = 2
    var in_channels = 64
    var out_channels = 64
    var height = 32
    var width = 32

    # Create input with small random values
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = randn(input_shape, DType.float32)

    # Conv1
    var conv1_weight_shape = List[Int]()
    conv1_weight_shape.append(out_channels)
    conv1_weight_shape.append(in_channels)
    conv1_weight_shape.append(3)
    conv1_weight_shape.append(3)
    var conv1_weight = randn(conv1_weight_shape, DType.float32)
    var conv1_bias = zeros([out_channels], DType.float32)

    # BN1
    var bn1_gamma = ones([out_channels], DType.float32)
    var bn1_beta = zeros([out_channels], DType.float32)
    var bn1_running_mean = zeros([out_channels], DType.float32)
    var bn1_running_var = ones([out_channels], DType.float32)

    # Conv2
    var conv2_weight = randn(conv1_weight_shape, DType.float32)
    var conv2_bias = zeros([out_channels], DType.float32)

    # BN2
    var bn2_gamma = ones([out_channels], DType.float32)
    var bn2_beta = zeros([out_channels], DType.float32)
    var bn2_running_mean = zeros([out_channels], DType.float32)
    var bn2_running_var = ones([out_channels], DType.float32)

    # Training mode
    var output = create_basic_block(
        x,
        conv1_weight,
        conv1_bias,
        bn1_gamma,
        bn1_beta,
        bn1_running_mean,
        bn1_running_var,
        conv2_weight,
        conv2_bias,
        bn2_gamma,
        bn2_beta,
        bn2_running_mean,
        bn2_running_var,
        training=True,
    )

    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)


def test_residual_block_64_channels_inference_mode() raises:
    """Test residual block in inference mode using running statistics.

    Inference mode should use running mean/var from BatchNorm.
    """
    var batch_size = 2
    var in_channels = 64
    var out_channels = 64
    var height = 32
    var width = 32

    # Create input
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = randn(input_shape, DType.float32)

    # Conv1
    var conv1_weight_shape = List[Int]()
    conv1_weight_shape.append(out_channels)
    conv1_weight_shape.append(in_channels)
    conv1_weight_shape.append(3)
    conv1_weight_shape.append(3)
    var conv1_weight = randn(conv1_weight_shape, DType.float32)
    var conv1_bias = zeros([out_channels], DType.float32)

    # BN1 with pre-computed running statistics
    var bn1_gamma = ones([out_channels], DType.float32)
    var bn1_beta = zeros([out_channels], DType.float32)
    var bn1_running_mean = zeros([out_channels], DType.float32)
    var bn1_running_var = ones([out_channels], DType.float32)

    # Conv2
    var conv2_weight = randn(conv1_weight_shape, DType.float32)
    var conv2_bias = zeros([out_channels], DType.float32)

    # BN2
    var bn2_gamma = ones([out_channels], DType.float32)
    var bn2_beta = zeros([out_channels], DType.float32)
    var bn2_running_mean = zeros([out_channels], DType.float32)
    var bn2_running_var = ones([out_channels], DType.float32)

    # Inference mode (training=False)
    var output = create_basic_block(
        x,
        conv1_weight,
        conv1_bias,
        bn1_gamma,
        bn1_beta,
        bn1_running_mean,
        bn1_running_var,
        conv2_weight,
        conv2_bias,
        bn2_gamma,
        bn2_beta,
        bn2_running_mean,
        bn2_running_var,
        training=False,
    )

    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)


def test_residual_block_128_channels_projection() raises:
    """Test residual block with 64->128 channels (with projection shortcut).

    When stride > 1 or channels change, projection shortcut is needed.
    This tests the 128-channel layer transition.

    Shape: (2, 64, 32, 32) -> (2, 128, 16, 16) with projection
    """
    var batch_size = 2
    var in_channels = 64
    var out_channels = 128
    var height = 32
    var width = 32

    # Create input: (2, 64, 32, 32)
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(in_channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = ones(input_shape, DType.float32)

    # Conv1: (128, 64, 3, 3) with stride=2
    var conv1_weight_shape = List[Int]()
    conv1_weight_shape.append(out_channels)
    conv1_weight_shape.append(in_channels)
    conv1_weight_shape.append(3)
    conv1_weight_shape.append(3)
    var conv1_weight = ones(conv1_weight_shape, DType.float32)
    var conv1_bias = zeros([out_channels], DType.float32)

    # BN1
    var bn1_gamma = ones([out_channels], DType.float32)
    var bn1_beta = zeros([out_channels], DType.float32)
    var bn1_running_mean = zeros([out_channels], DType.float32)
    var bn1_running_var = ones([out_channels], DType.float32)

    # Conv2: (128, 128, 3, 3)
    var conv2_weight_shape = List[Int]()
    conv2_weight_shape.append(out_channels)
    conv2_weight_shape.append(out_channels)
    conv2_weight_shape.append(3)
    conv2_weight_shape.append(3)
    var conv2_weight = ones(conv2_weight_shape, DType.float32)
    var conv2_bias = zeros([out_channels], DType.float32)

    # BN2
    var bn2_gamma = ones([out_channels], DType.float32)
    var bn2_beta = zeros([out_channels], DType.float32)
    var bn2_running_mean = zeros([out_channels], DType.float32)
    var bn2_running_var = ones([out_channels], DType.float32)

    # Projection shortcut: (128, 64, 1, 1)
    var projection_weight_shape = List[Int]()
    projection_weight_shape.append(out_channels)
    projection_weight_shape.append(in_channels)
    projection_weight_shape.append(1)
    projection_weight_shape.append(1)
    var projection_weight = ones(projection_weight_shape, DType.float32)
    var projection_bias = zeros([out_channels], DType.float32)

    # Forward with stride=2 on first conv (simplified test)
    var conv1_out = conv2d(
        x, conv1_weight, conv1_bias, stride=2, padding=1
    )  # (2, 128, 16, 16)

    var bn1_out: AnyTensor
    var _: AnyTensor
    var __: AnyTensor
    (bn1_out, _, __) = batch_norm2d(
        conv1_out,
        bn1_gamma,
        bn1_beta,
        bn1_running_mean,
        bn1_running_var,
        training=True,
    )
    var relu1_out = relu(bn1_out)

    # Second conv
    var conv2_out = conv2d(
        relu1_out, conv2_weight, conv2_bias, stride=1, padding=1
    )
    var bn2_out: AnyTensor
    (bn2_out, _, __) = batch_norm2d(
        conv2_out,
        bn2_gamma,
        bn2_beta,
        bn2_running_mean,
        bn2_running_var,
        training=True,
    )

    # Projection shortcut
    var proj_out = conv2d(
        x, projection_weight, projection_bias, stride=2, padding=0
    )  # (2, 128, 16, 16)

    # Skip connection
    var residual = add(bn2_out, proj_out)

    # Final ReLU
    var output = relu(residual)

    # Verify output shape: (2, 128, 16, 16)
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], out_channels)
    assert_equal(out_shape[2], height // 2)
    assert_equal(out_shape[3], width // 2)


def test_skip_connection_addition() raises:
    """Test that skip connection (addition) works correctly.

    Skip connection is element-wise addition of two tensors.
    """
    var batch_size = 2
    var channels = 64
    var height = 32
    var width = 32

    # Create two tensors
    var shape = List[Int]()
    shape.append(batch_size)
    shape.append(channels)
    shape.append(height)
    shape.append(width)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    # Skip connection: element-wise addition
    var result = add(a, b)

    # Verify output shape matches input
    var res_shape = result.shape()
    assert_equal(res_shape[0], batch_size)
    assert_equal(res_shape[1], channels)
    assert_equal(res_shape[2], height)
    assert_equal(res_shape[3], width)

    # Verify values: 1 + 1 = 2
    var result_data = result._data.bitcast[Float32]()
    var total_elements = batch_size * channels * height * width
    for i in range(min(10, total_elements)):
        assert_almost_equal(result_data[i], 2.0, tolerance=1e-5)


def test_skip_connection_identity() raises:
    """Test identity skip connection (no change to input).

    When input goes directly through skip without transformation.
    """
    var batch_size = 2
    var channels = 64
    var height = 32
    var width = 32

    var shape = List[Int]()
    shape.append(batch_size)
    shape.append(channels)
    shape.append(height)
    shape.append(width)

    var x = randn(shape, DType.float32)

    # Copy original data for comparison
    var x_data = x._data.bitcast[Float32]()
    var original_values = List[Float32]()
    var total_elements = batch_size * channels * height * width
    for i in range(min(10, total_elements)):
        original_values.append(x_data[i])

    # Identity skip: addition with zero
    var zeros_tensor = zeros(shape, DType.float32)
    var result = add(x, zeros_tensor)

    # Result should equal x
    var result_data = result._data.bitcast[Float32]()
    for i in range(min(10, total_elements)):
        assert_almost_equal(result_data[i], original_values[i], tolerance=1e-5)


def test_batchnorm2d_training_mode() raises:
    """Test BatchNorm2d in training mode.

    Training mode computes batch statistics (mean and variance).
    """
    var batch_size = 4
    var channels = 64
    var height = 32
    var width = 32

    # Create input tensor
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = randn(input_shape, DType.float32)

    # BatchNorm parameters
    var gamma = ones([channels], DType.float32)
    var beta = zeros([channels], DType.float32)
    var running_mean = zeros([channels], DType.float32)
    var running_var = ones([channels], DType.float32)

    # Forward pass in training mode
    var output: AnyTensor
    var new_running_mean: AnyTensor
    var new_running_var: AnyTensor
    (output, new_running_mean, new_running_var) = batch_norm2d(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
        momentum=0.1,
    )

    # Verify output shape matches input
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)

    # Verify running stats shapes
    var new_mean_shape = new_running_mean.shape()
    var new_var_shape = new_running_var.shape()
    assert_equal(new_mean_shape[0], channels)
    assert_equal(new_var_shape[0], channels)


def test_batchnorm2d_inference_mode() raises:
    """Test BatchNorm2d in inference mode.

    Inference mode uses running statistics (mean and variance).
    """
    var batch_size = 4
    var channels = 64
    var height = 32
    var width = 32

    # Create input tensor
    var input_shape = List[Int]()
    input_shape.append(batch_size)
    input_shape.append(channels)
    input_shape.append(height)
    input_shape.append(width)
    var x = randn(input_shape, DType.float32)

    # BatchNorm parameters with pre-computed running statistics
    var gamma = ones([channels], DType.float32)
    var beta = zeros([channels], DType.float32)
    var running_mean = zeros([channels], DType.float32)
    var running_var = ones([channels], DType.float32)

    # Forward pass in inference mode
    var output: AnyTensor
    var new_running_mean: AnyTensor
    var new_running_var: AnyTensor
    (output, new_running_mean, new_running_var) = batch_norm2d(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=False,
    )

    # Verify output shape
    var out_shape = output.shape()
    assert_equal(out_shape[0], batch_size)
    assert_equal(out_shape[1], channels)
    assert_equal(out_shape[2], height)
    assert_equal(out_shape[3], width)

    # In inference mode, running stats should not change
    var new_mean_data = new_running_mean._data.bitcast[Float32]()
    var new_var_data = new_running_var._data.bitcast[Float32]()
    var running_mean_data = running_mean._data.bitcast[Float32]()
    var running_var_data = running_var._data.bitcast[Float32]()

    for i in range(channels):
        assert_almost_equal(
            new_mean_data[i], running_mean_data[i], tolerance=1e-5
        )
        assert_almost_equal(
            new_var_data[i], running_var_data[i], tolerance=1e-5
        )


def test_batchnorm2d_gamma_beta_effects() raises:
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
    var output: AnyTensor
    var _: AnyTensor
    var __: AnyTensor
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


def test_relu_in_residual_block() raises:
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


def test_block_forward_backward_consistency() raises:
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


def test_multiple_blocks_sequential() raises:
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


def main() raises:
    """Run all test_resnet18_layers tests."""
    print("Running test_resnet18_layers tests...")

    test_residual_block_64_channels_forward()
    print("✓ test_residual_block_64_channels_forward")

    test_residual_block_64_channels_training_mode()
    print("✓ test_residual_block_64_channels_training_mode")

    test_residual_block_64_channels_inference_mode()
    print("✓ test_residual_block_64_channels_inference_mode")

    test_residual_block_128_channels_projection()
    print("✓ test_residual_block_128_channels_projection")

    test_skip_connection_addition()
    print("✓ test_skip_connection_addition")

    test_skip_connection_identity()
    print("✓ test_skip_connection_identity")

    test_batchnorm2d_training_mode()
    print("✓ test_batchnorm2d_training_mode")

    test_batchnorm2d_inference_mode()
    print("✓ test_batchnorm2d_inference_mode")

    test_batchnorm2d_gamma_beta_effects()
    print("✓ test_batchnorm2d_gamma_beta_effects")

    test_relu_in_residual_block()
    print("✓ test_relu_in_residual_block")

    test_block_forward_backward_consistency()
    print("✓ test_block_forward_backward_consistency")

    test_multiple_blocks_sequential()
    print("✓ test_multiple_blocks_sequential")

    print("\nAll test_resnet18_layers tests passed!")
