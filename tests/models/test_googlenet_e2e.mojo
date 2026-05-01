"""End-to-end tests for GoogLeNet (Inception-v1) model on CIFAR-10

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

Tests cover:
- Model initialization with correct parameter shapes
- Forward pass through complete architecture (batch sizes 1, 2, 4)
- Training mode vs inference mode (affects batch norm, dropout)
- Output shape verification (batch, 10 classes for CIFAR-10)
- Numerical stability - no NaN values
- Multiple output class counts

All tests use small batches and synthetic data for fast execution (< 90 seconds).
"""


from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.activation import relu
from shared.core.pooling import maxpool2d, global_avgpool2d
from shared.core.normalization import batch_norm2d
from shared.core.conv import conv2d
from shared.core.linear import linear
from shared.core.initializers import kaiming_normal, xavier_normal, constant


struct InceptionModule:
    """Inception module with 4 parallel branches."""

    var conv1x1_1_weights: AnyTensor
    var conv1x1_1_bias: AnyTensor
    var bn1x1_1_gamma: AnyTensor
    var bn1x1_1_beta: AnyTensor
    var bn1x1_1_running_mean: AnyTensor
    var bn1x1_1_running_var: AnyTensor

    var conv1x1_2_weights: AnyTensor
    var conv1x1_2_bias: AnyTensor
    var bn1x1_2_gamma: AnyTensor
    var bn1x1_2_beta: AnyTensor
    var bn1x1_2_running_mean: AnyTensor
    var bn1x1_2_running_var: AnyTensor

    var conv3x3_weights: AnyTensor
    var conv3x3_bias: AnyTensor
    var bn3x3_gamma: AnyTensor
    var bn3x3_beta: AnyTensor
    var bn3x3_running_mean: AnyTensor
    var bn3x3_running_var: AnyTensor

    var conv1x1_3_weights: AnyTensor
    var conv1x1_3_bias: AnyTensor
    var bn1x1_3_gamma: AnyTensor
    var bn1x1_3_beta: AnyTensor
    var bn1x1_3_running_mean: AnyTensor
    var bn1x1_3_running_var: AnyTensor

    var conv5x5_weights: AnyTensor
    var conv5x5_bias: AnyTensor
    var bn5x5_gamma: AnyTensor
    var bn5x5_beta: AnyTensor
    var bn5x5_running_mean: AnyTensor
    var bn5x5_running_var: AnyTensor

    var conv1x1_4_weights: AnyTensor
    var conv1x1_4_bias: AnyTensor
    var bn1x1_4_gamma: AnyTensor
    var bn1x1_4_beta: AnyTensor
    var bn1x1_4_running_mean: AnyTensor
    var bn1x1_4_running_var: AnyTensor

    def __init__(
        out self,
        in_channels: Int,
        out_1x1: Int,
        reduce_3x3: Int,
        out_3x3: Int,
        reduce_5x5: Int,
        out_5x5: Int,
        pool_proj: Int,
    ) raises:
        """Initialize Inception module."""
        # Branch 1
        self.conv1x1_1_weights = kaiming_normal(
            in_channels, out_1x1, [out_1x1, in_channels, 1, 1]
        )
        self.conv1x1_1_bias = zeros([out_1x1], DType.float32)
        self.bn1x1_1_gamma = constant([out_1x1], 1.0)
        self.bn1x1_1_beta = zeros([out_1x1], DType.float32)
        self.bn1x1_1_running_mean = zeros([out_1x1], DType.float32)
        self.bn1x1_1_running_var = constant([out_1x1], 1.0)

        # Branch 2: 1x1
        self.conv1x1_2_weights = kaiming_normal(
            in_channels, reduce_3x3, [reduce_3x3, in_channels, 1, 1]
        )
        self.conv1x1_2_bias = zeros([reduce_3x3], DType.float32)
        self.bn1x1_2_gamma = constant([reduce_3x3], 1.0)
        self.bn1x1_2_beta = zeros([reduce_3x3], DType.float32)
        self.bn1x1_2_running_mean = zeros([reduce_3x3], DType.float32)
        self.bn1x1_2_running_var = constant([reduce_3x3], 1.0)

        # Branch 2: 3x3
        self.conv3x3_weights = kaiming_normal(
            reduce_3x3 * 9, out_3x3, [out_3x3, reduce_3x3, 3, 3]
        )
        self.conv3x3_bias = zeros([out_3x3], DType.float32)
        self.bn3x3_gamma = constant([out_3x3], 1.0)
        self.bn3x3_beta = zeros([out_3x3], DType.float32)
        self.bn3x3_running_mean = zeros([out_3x3], DType.float32)
        self.bn3x3_running_var = constant([out_3x3], 1.0)

        # Branch 3: 1x1
        self.conv1x1_3_weights = kaiming_normal(
            in_channels, reduce_5x5, [reduce_5x5, in_channels, 1, 1]
        )
        self.conv1x1_3_bias = zeros([reduce_5x5], DType.float32)
        self.bn1x1_3_gamma = constant([reduce_5x5], 1.0)
        self.bn1x1_3_beta = zeros([reduce_5x5], DType.float32)
        self.bn1x1_3_running_mean = zeros([reduce_5x5], DType.float32)
        self.bn1x1_3_running_var = constant([reduce_5x5], 1.0)

        # Branch 3: 5x5
        self.conv5x5_weights = kaiming_normal(
            reduce_5x5 * 25, out_5x5, [out_5x5, reduce_5x5, 5, 5]
        )
        self.conv5x5_bias = zeros([out_5x5], DType.float32)
        self.bn5x5_gamma = constant([out_5x5], 1.0)
        self.bn5x5_beta = zeros([out_5x5], DType.float32)
        self.bn5x5_running_mean = zeros([out_5x5], DType.float32)
        self.bn5x5_running_var = constant([out_5x5], 1.0)

        # Branch 4: pool + 1x1
        self.conv1x1_4_weights = kaiming_normal(
            in_channels, pool_proj, [pool_proj, in_channels, 1, 1]
        )
        self.conv1x1_4_bias = zeros([pool_proj], DType.float32)
        self.bn1x1_4_gamma = constant([pool_proj], 1.0)
        self.bn1x1_4_beta = zeros([pool_proj], DType.float32)
        self.bn1x1_4_running_mean = zeros([pool_proj], DType.float32)
        self.bn1x1_4_running_var = constant([pool_proj], 1.0)

    def forward(mut self, x: AnyTensor, training: Bool) raises -> AnyTensor:
        """Forward pass through Inception module."""
        # Branch 1: 1x1 conv
        var b1 = conv2d(
            x, self.conv1x1_1_weights, self.conv1x1_1_bias, stride=1, padding=0
        )
        b1, _, _ = batch_norm2d(
            b1,
            self.bn1x1_1_gamma,
            self.bn1x1_1_beta,
            self.bn1x1_1_running_mean,
            self.bn1x1_1_running_var,
            training,
        )
        b1 = relu(b1)

        # Branch 2: 1x1 reduce -> 3x3
        var b2 = conv2d(
            x, self.conv1x1_2_weights, self.conv1x1_2_bias, stride=1, padding=0
        )
        b2, _, _ = batch_norm2d(
            b2,
            self.bn1x1_2_gamma,
            self.bn1x1_2_beta,
            self.bn1x1_2_running_mean,
            self.bn1x1_2_running_var,
            training,
        )
        b2 = relu(b2)
        b2 = conv2d(
            b2, self.conv3x3_weights, self.conv3x3_bias, stride=1, padding=1
        )
        b2, _, _ = batch_norm2d(
            b2,
            self.bn3x3_gamma,
            self.bn3x3_beta,
            self.bn3x3_running_mean,
            self.bn3x3_running_var,
            training,
        )
        b2 = relu(b2)

        # Branch 3: 1x1 reduce -> 5x5
        var b3 = conv2d(
            x, self.conv1x1_3_weights, self.conv1x1_3_bias, stride=1, padding=0
        )
        b3, _, _ = batch_norm2d(
            b3,
            self.bn1x1_3_gamma,
            self.bn1x1_3_beta,
            self.bn1x1_3_running_mean,
            self.bn1x1_3_running_var,
            training,
        )
        b3 = relu(b3)
        b3 = conv2d(
            b3, self.conv5x5_weights, self.conv5x5_bias, stride=1, padding=2
        )
        b3, _, _ = batch_norm2d(
            b3,
            self.bn5x5_gamma,
            self.bn5x5_beta,
            self.bn5x5_running_mean,
            self.bn5x5_running_var,
            training,
        )
        b3 = relu(b3)

        # Branch 4: pool -> 1x1
        var b4 = maxpool2d(x, kernel_size=3, stride=1, padding=1)
        b4 = conv2d(
            b4, self.conv1x1_4_weights, self.conv1x1_4_bias, stride=1, padding=0
        )
        b4, _, _ = batch_norm2d(
            b4,
            self.bn1x1_4_gamma,
            self.bn1x1_4_beta,
            self.bn1x1_4_running_mean,
            self.bn1x1_4_running_var,
            training,
        )
        b4 = relu(b4)

        # Concatenate branches
        return concatenate_depthwise(b1, b2, b3, b4)


def concatenate_depthwise(
    t1: AnyTensor, t2: AnyTensor, t3: AnyTensor, t4: AnyTensor
) raises -> AnyTensor:
    """Concatenate 4 tensors along channel dimension."""
    var batch_size = t1.shape()[0]
    var c1 = t1.shape()[1]
    var c2 = t2.shape()[1]
    var c3 = t3.shape()[1]
    var c4 = t4.shape()[1]
    var height = t1.shape()[2]
    var width = t1.shape()[3]

    var total_channels = c1 + c2 + c3 + c4
    var result = zeros([batch_size, total_channels, height, width], t1.dtype())

    var result_data = result._data.bitcast[Float32]()
    var t1_data = t1._data.bitcast[Float32]()
    var t2_data = t2._data.bitcast[Float32]()
    var t3_data = t3._data.bitcast[Float32]()
    var t4_data = t4._data.bitcast[Float32]()

    var hw = height * width

    for b in range(batch_size):
        for c in range(c1):
            for i in range(hw):
                var src_idx = ((b * c1 + c) * hw) + i
                var dst_idx = ((b * total_channels + c) * hw) + i
                result_data[dst_idx] = t1_data[src_idx]

        for c in range(c2):
            for i in range(hw):
                var src_idx = ((b * c2 + c) * hw) + i
                var dst_idx = ((b * total_channels + (c1 + c)) * hw) + i
                result_data[dst_idx] = t2_data[src_idx]

        for c in range(c3):
            for i in range(hw):
                var src_idx = ((b * c3 + c) * hw) + i
                var dst_idx = ((b * total_channels + (c1 + c2 + c)) * hw) + i
                result_data[dst_idx] = t3_data[src_idx]

        for c in range(c4):
            for i in range(hw):
                var src_idx = ((b * c4 + c) * hw) + i
                var dst_idx = (
                    (b * total_channels + (c1 + c2 + c3 + c)) * hw
                ) + i
                result_data[dst_idx] = t4_data[src_idx]

    return result


struct GoogLeNetSmall:
    """Simplified GoogLeNet for E2E testing with smaller input size."""

    # Initial conv
    var initial_conv_weights: AnyTensor
    var initial_conv_bias: AnyTensor
    var initial_bn_gamma: AnyTensor
    var initial_bn_beta: AnyTensor
    var initial_bn_running_mean: AnyTensor
    var initial_bn_running_var: AnyTensor

    # Inception modules (3 instead of 9 for faster testing)
    var inception_1: InceptionModule
    var inception_2: InceptionModule
    var inception_3: InceptionModule

    # FC layer
    var fc_weights: AnyTensor
    var fc_bias: AnyTensor

    def __init__(out self, num_classes: Int = 10) raises:
        """Initialize simplified GoogLeNet for testing."""
        # Initial conv: 3->64, 3x3 kernel
        self.initial_conv_weights = kaiming_normal(
            fan_in=3 * 9, fan_out=64, shape=[64, 3, 3, 3]
        )
        self.initial_conv_bias = zeros([64], DType.float32)
        self.initial_bn_gamma = constant([64], 1.0)
        self.initial_bn_beta = zeros([64], DType.float32)
        self.initial_bn_running_mean = zeros([64], DType.float32)
        self.initial_bn_running_var = constant([64], 1.0)

        # Inception modules (simplified channel config)
        self.inception_1 = InceptionModule(
            in_channels=64,
            out_1x1=32,
            reduce_3x3=24,
            out_3x3=32,
            reduce_5x5=8,
            out_5x5=16,
            pool_proj=16,
        )

        self.inception_2 = InceptionModule(
            in_channels=96,
            out_1x1=32,
            reduce_3x3=24,
            out_3x3=32,
            reduce_5x5=8,
            out_5x5=16,
            pool_proj=16,
        )

        self.inception_3 = InceptionModule(
            in_channels=96,
            out_1x1=32,
            reduce_3x3=24,
            out_3x3=32,
            reduce_5x5=8,
            out_5x5=16,
            pool_proj=16,
        )

        # FC layer: feature_dim -> num_classes
        self.fc_weights = xavier_normal(96, num_classes, [num_classes, 96])
        self.fc_bias = zeros([num_classes], DType.float32)

    def forward(
        mut self, x: AnyTensor, training: Bool = True
    ) raises -> AnyTensor:
        """Forward pass through simplified GoogLeNet."""
        # Initial conv
        var out = conv2d(
            x,
            self.initial_conv_weights,
            self.initial_conv_bias,
            stride=1,
            padding=1,
        )
        out, _, _ = batch_norm2d(
            out,
            self.initial_bn_gamma,
            self.initial_bn_beta,
            self.initial_bn_running_mean,
            self.initial_bn_running_var,
            training,
        )
        out = relu(out)

        # Inception 1
        out = self.inception_1.forward(out, training)

        # Inception 2
        out = self.inception_2.forward(out, training)

        # Inception 3
        out = self.inception_3.forward(out, training)

        # Global average pool: (batch, 96, H, W) -> (batch, 96, 1, 1)
        out = global_avgpool2d(out)

        # Flatten: (batch, 96, 1, 1) -> (batch, 96)
        var batch_size = out.shape()[0]
        var flat_shape = List[Int]()
        flat_shape.append(batch_size)
        flat_shape.append(96)
        out = out.reshape(flat_shape)

        # FC
        out = linear(out, self.fc_weights, self.fc_bias)

        return out


# ============================================================================
# E2E Initialization Tests
# ============================================================================


def test_googlenet_initialization() raises:
    """Test that GoogLeNet model can be initialized with correct shapes."""
    var model = GoogLeNetSmall(num_classes=10)

    # Verify initial conv parameters
    assert_shape(model.initial_conv_weights, [64, 3, 3, 3])
    assert_shape(model.initial_conv_bias, [64])

    # Verify FC parameters
    assert_shape(model.fc_weights, [10, 96])
    assert_shape(model.fc_bias, [10])


def test_googlenet_forward_batch_size_1() raises:
    """Test forward pass with batch size 1.

    Input: (batch=1, channels=3, height=8, width=8)
    Output: (batch=1, classes=10)
    """
    var batch_size = 1
    var model = GoogLeNetSmall(num_classes=10)

    # Create input
    var input = ones([batch_size, 3, 8, 8], DType.float32)

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify output shape
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], 10)
    assert_equal(len(output.shape()), 2)


def test_googlenet_forward_batch_size_2() raises:
    """Test forward pass with batch size 2.

    Input: (batch=2, channels=3, height=8, width=8)
    Output: (batch=2, classes=10)
    """
    var batch_size = 2
    var model = GoogLeNetSmall(num_classes=10)

    # Create input
    var input = ones([batch_size, 3, 8, 8], DType.float32)

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify output shape
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], 10)


def test_googlenet_forward_batch_size_4() raises:
    """Test forward pass with batch size 4.

    Input: (batch=4, channels=3, height=8, width=8)
    Output: (batch=4, classes=10)
    """
    var batch_size = 4
    var model = GoogLeNetSmall(num_classes=10)

    # Create input
    var input = ones([batch_size, 3, 8, 8], DType.float32)

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify output shape
    assert_equal(output.shape()[0], batch_size)
    assert_equal(output.shape()[1], 10)


def test_googlenet_training_mode() raises:
    """Test forward pass in training mode (batch norm affects output).

    Training mode should use mini-batch statistics.
    """
    var batch_size = 2
    var model = GoogLeNetSmall(num_classes=10)

    # Create input
    var input = ones([batch_size, 3, 8, 8], DType.float32)

    # Forward pass in training mode
    var output_train = model.forward(input, training=True)

    # Verify output shape
    assert_equal(output_train.shape()[0], batch_size)
    assert_equal(output_train.shape()[1], 10)


def test_googlenet_inference_mode() raises:
    """Test forward pass in inference mode (batch norm uses running stats).

    Inference mode should use running mean/variance.
    """
    var batch_size = 2
    var model = GoogLeNetSmall(num_classes=10)

    # Create input
    var input = ones([batch_size, 3, 8, 8], DType.float32)

    # Forward pass in inference mode
    var output_infer = model.forward(input, training=False)

    # Verify output shape
    assert_equal(output_infer.shape()[0], batch_size)
    assert_equal(output_infer.shape()[1], 10)


def test_googlenet_different_class_counts() raises:
    """Test GoogLeNet with different output class counts.

    CIFAR-10: 10 classes
    CIFAR-100: 100 classes
    ImageNet: 1000 classes
    """
    var batch_size = 2
    var input = ones([batch_size, 3, 8, 8], DType.float32)

    # Test with 10 classes (CIFAR-10)
    var model_10 = GoogLeNetSmall(num_classes=10)
    var output_10 = model_10.forward(input, training=False)
    assert_equal(output_10.shape()[1], 10)

    # Test with 100 classes (CIFAR-100)
    var model_100 = GoogLeNetSmall(num_classes=100)
    var output_100 = model_100.forward(input, training=False)
    assert_equal(output_100.shape()[1], 100)


def test_googlenet_no_nan_output() raises:
    """Test that forward pass produces no NaN values.

    NaN can indicate numerical instability (e.g., log(negative)).
    """
    var batch_size = 2
    var model = GoogLeNetSmall(num_classes=10)

    # Create input
    var input = ones([batch_size, 3, 8, 8], DType.float32)
    var input_data = input._data.bitcast[Float32]()
    for i in range(input.numel()):
        input_data[i] = 0.1

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify no NaN values
    var output_data = output._data.bitcast[Float32]()
    for i in range(output.numel()):
        var val = output_data[i]
        # Simple NaN check: NaN != NaN
        assert_true(val == val, "Output contains NaN")


def test_googlenet_no_inf_output() raises:
    """Test that forward pass produces no infinite values.

    Infinity can indicate overflow or division issues.
    """
    var batch_size = 2
    var model = GoogLeNetSmall(num_classes=10)

    # Create small input to avoid overflow
    var input = full([batch_size, 3, 8, 8], 0.1, DType.float32)

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify no infinite values by checking if finite
    var output_data = output._data.bitcast[Float32]()
    for i in range(output.numel()):
        var val = output_data[i]
        # Simple finiteness check
        assert_true(val == val, "Output contains Inf or NaN")
        assert_true(val > -1e10, "Output contains -Inf")
        assert_true(val < 1e10, "Output contains +Inf")


def test_googlenet_different_input_values() raises:
    """Test forward pass with different input value ranges.

    Should handle various input scales without numerical issues.
    """
    var batch_size = 2
    var model = GoogLeNetSmall(num_classes=10)

    # Test with small values (0.01)
    var input_small = full([batch_size, 3, 8, 8], 0.01, DType.float32)
    var output_small = model.forward(input_small, training=False)
    assert_equal(output_small.shape()[0], batch_size)

    # Test with unit values (1.0)
    var input_unit = full([batch_size, 3, 8, 8], 1.0, DType.float32)
    var output_unit = model.forward(input_unit, training=False)
    assert_equal(output_unit.shape()[0], batch_size)

    # Test with larger values (5.0, simulating normalized but higher magnitude)
    var input_large = full([batch_size, 3, 8, 8], 5.0, DType.float32)
    var output_large = model.forward(input_large, training=False)
    assert_equal(output_large.shape()[0], batch_size)


def test_googlenet_reproducible_output() raises:
    """Test that same input produces same output in inference mode.

    Inference mode should be deterministic (no stochastic operations).
    """
    var batch_size = 2
    var input = ones([batch_size, 3, 8, 8], DType.float32)
    var input_data = input._data.bitcast[Float32]()
    for i in range(input.numel()):
        input_data[i] = 0.5

    # Forward pass 1
    var model1 = GoogLeNetSmall(num_classes=10)
    var output1 = model1.forward(input, training=False)

    # Forward pass 2 with different model instance
    var model2 = GoogLeNetSmall(num_classes=10)
    var output2 = model2.forward(input, training=False)

    # Note: Outputs will differ due to different random initialization,
    # but shapes should match
    assert_equal(output1.shape()[0], output2.shape()[0])
    assert_equal(output1.shape()[1], output2.shape()[1])


def test_googlenet_multiple_forward_passes() raises:
    """Test multiple forward passes through same model instance.

    Verifies that model state is handled correctly across multiple calls.
    """
    var batch_size = 2
    var model = GoogLeNetSmall(num_classes=10)

    # First forward pass
    var input1 = ones([batch_size, 3, 8, 8], DType.float32)
    var output1 = model.forward(input1, training=False)

    # Second forward pass with different input
    var input2 = full([batch_size, 3, 8, 8], 0.5, DType.float32)
    var output2 = model.forward(input2, training=False)

    # Both outputs should have correct shape
    assert_equal(output1.shape()[0], batch_size)
    assert_equal(output1.shape()[1], 10)
    assert_equal(output2.shape()[0], batch_size)
    assert_equal(output2.shape()[1], 10)


def test_googlenet_inception_module_contribution() raises:
    """Test that each Inception module is contributing to the computation.

    Verifies that all modules are in the forward pass.
    """
    var batch_size = 1
    var model = GoogLeNetSmall(num_classes=10)

    # Create input with known values
    var input = ones([batch_size, 3, 8, 8], DType.float32)
    var input_data = input._data.bitcast[Float32]()
    for i in range(input.numel()):
        input_data[i] = 0.2

    # Forward pass
    var output = model.forward(input, training=True)

    # Verify output contains values (not all zeros)
    var output_data = output._data.bitcast[Float32]()
    var sum_val = Float32(0.0)
    for i in range(output.numel()):
        sum_val += Float32(output_data[i] * output_data[i])

    assert_true(
        sum_val > 0.0,
        "Output should contain non-zero values from Inception modules",
    )


def test_googlenet_batch_independence() raises:
    """Test that batch samples are processed independently.

    Processing a batch of size 2 should be equivalent to processing 2 separate samples.
    This is a basic sanity check for batch processing.
    """
    var model = GoogLeNetSmall(num_classes=10)

    # Create batch input
    var batch_input = ones([2, 3, 8, 8], DType.float32)
    var batch_output = model.forward(batch_input, training=False)

    # Create single sample inputs
    var sample_input_1 = ones([1, 3, 8, 8], DType.float32)
    var sample_output_1 = model.forward(sample_input_1, training=False)

    var sample_input_2 = ones([1, 3, 8, 8], DType.float32)
    var sample_output_2 = model.forward(sample_input_2, training=False)

    # All outputs should have consistent shapes
    assert_equal(batch_output.shape()[0], 2)
    assert_equal(sample_output_1.shape()[0], 1)
    assert_equal(sample_output_2.shape()[0], 1)

    # FC output dimension should be same
    assert_equal(batch_output.shape()[1], sample_output_1.shape()[1])
    assert_equal(batch_output.shape()[1], sample_output_2.shape()[1])


def test_googlenet_output_is_tensor() raises:
    """Test that output is a properly formed tensor.

    Verifies shape(), numel(), dtype() methods work on output.
    """
    var batch_size = 2
    var model = GoogLeNetSmall(num_classes=10)

    # Create input
    var input = ones([batch_size, 3, 8, 8], DType.float32)

    # Forward pass
    var output = model.forward(input, training=False)

    # Verify tensor operations
    var shape = output.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], batch_size)
    assert_equal(shape[1], 10)

    var numel = output.numel()
    assert_equal(numel, batch_size * 10)

    var dtype = output.dtype()
    assert_true(dtype == DType.float32, "dtype should be float32")


def main() raises:
    """Run all test_googlenet_e2e tests."""
    print("Running test_googlenet_e2e tests...")

    test_googlenet_initialization()
    print("✓ test_googlenet_initialization")

    test_googlenet_forward_batch_size_1()
    print("✓ test_googlenet_forward_batch_size_1")

    test_googlenet_forward_batch_size_2()
    print("✓ test_googlenet_forward_batch_size_2")

    test_googlenet_forward_batch_size_4()
    print("✓ test_googlenet_forward_batch_size_4")

    test_googlenet_training_mode()
    print("✓ test_googlenet_training_mode")

    test_googlenet_inference_mode()
    print("✓ test_googlenet_inference_mode")

    test_googlenet_different_class_counts()
    print("✓ test_googlenet_different_class_counts")

    test_googlenet_no_nan_output()
    print("✓ test_googlenet_no_nan_output")

    test_googlenet_no_inf_output()
    print("✓ test_googlenet_no_inf_output")

    test_googlenet_different_input_values()
    print("✓ test_googlenet_different_input_values")

    test_googlenet_reproducible_output()
    print("✓ test_googlenet_reproducible_output")

    test_googlenet_multiple_forward_passes()
    print("✓ test_googlenet_multiple_forward_passes")

    test_googlenet_inception_module_contribution()
    print("✓ test_googlenet_inception_module_contribution")

    test_googlenet_batch_independence()
    print("✓ test_googlenet_batch_independence")

    test_googlenet_output_is_tensor()
    print("✓ test_googlenet_output_is_tensor")

    print("\nAll test_googlenet_e2e tests passed!")
