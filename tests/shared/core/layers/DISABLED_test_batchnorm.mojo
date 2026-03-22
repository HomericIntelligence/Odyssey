"""Unit tests for BatchNorm2dLayer class.

Tests cover:
- BatchNorm2dLayer initialization with correct parameter shapes
- Forward pass in training mode (batch statistics, running stats update)
- Forward pass in inference mode (running statistics)
- Parameter extraction and management
- Running statistics management

Following TDD principles - these tests define the expected API for BatchNorm2dLayer.
"""

from shared.testing.assertions import (
    assert_almost_equal,
    assert_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.extensor import AnyTensor, zeros, ones, randn
from shared.core.layers.batchnorm import BatchNorm2dLayer


# ============================================================================
# BatchNorm2dLayer Initialization Tests
# ============================================================================


fn test_batchnorm_initialization() raises:
    """Test BatchNorm2dLayer parameter creation with correct shapes.

    Verifies that gamma, beta, and running statistics are initialized correctly.
    """
    var num_channels = 16

    var layer = BatchNorm2dLayer(num_channels)

    # Check gamma shape: (channels,)
    var gamma_shape = layer.gamma.shape()
    assert_equal_int(len(gamma_shape), 1)
    assert_equal(gamma_shape[0], num_channels)

    # Check beta shape: (channels,)
    var beta_shape = layer.beta.shape()
    assert_equal_int(len(beta_shape), 1)
    assert_equal(beta_shape[0], num_channels)

    # Check running_mean shape: (channels,)
    var running_mean_shape = layer.running_mean.shape()
    assert_equal_int(len(running_mean_shape), 1)
    assert_equal(running_mean_shape[0], num_channels)

    # Check running_var shape: (channels,)
    var running_var_shape = layer.running_var.shape()
    assert_equal_int(len(running_var_shape), 1)
    assert_equal(running_var_shape[0], num_channels)


fn test_batchnorm_gamma_initialized_to_one() raises:
    """Test that gamma (scale) is initialized to 1.0 for each channel.

    Gamma = 1.0 means identity scaling initially.
    """
    var layer = BatchNorm2dLayer(16)

    var gamma_data = layer.gamma._data.bitcast[Float32]()
    for i in range(layer.gamma.numel()):
        assert_almost_equal(gamma_data[i], 1.0, tolerance=1e-6)


fn test_batchnorm_beta_initialized_to_zero() raises:
    """Test that beta (shift) is initialized to 0.0 for each channel.

    Beta = 0.0 means no shift initially.
    """
    var layer = BatchNorm2dLayer(16)

    var beta_data = layer.beta._data.bitcast[Float32]()
    for i in range(layer.beta.numel()):
        assert_almost_equal(beta_data[i], 0.0, tolerance=1e-6)


fn test_batchnorm_running_mean_initialized_to_zero() raises:
    """Test that running_mean is initialized to 0.0."""
    var layer = BatchNorm2dLayer(16)

    var mean_data = layer.running_mean._data.bitcast[Float32]()
    for i in range(layer.running_mean.numel()):
        assert_almost_equal(mean_data[i], 0.0, tolerance=1e-6)


fn test_batchnorm_running_var_initialized_to_one() raises:
    """Test that running_var is initialized to 1.0."""
    var layer = BatchNorm2dLayer(16)

    var var_data = layer.running_var._data.bitcast[Float32]()
    for i in range(layer.running_var.numel()):
        assert_almost_equal(var_data[i], 1.0, tolerance=1e-6)


fn test_batchnorm_initialization_with_momentum_eps() raises:
    """Test BatchNorm2dLayer initialization with custom momentum and epsilon.

    Verifies that momentum and eps parameters are stored correctly.
    """
    var num_channels = 32
    var momentum = Float32(0.05)
    var eps = Float32(1e-3)

    var layer = BatchNorm2dLayer(num_channels, momentum=momentum, eps=eps)

    assert_equal(layer.num_channels, num_channels)
    assert_almost_equal(layer.momentum, momentum, tolerance=1e-6)
    assert_almost_equal(layer.eps, eps, tolerance=1e-6)


# ============================================================================
# BatchNorm2dLayer Forward Pass Tests
# ============================================================================


fn test_batchnorm_forward_output_shape() raises:
    """Test BatchNorm2dLayer forward pass preserves input shape.

    BatchNorm should not change spatial dimensions.
    """
    var layer = BatchNorm2dLayer(16)

    # Input: (batch=2, channels=16, height=32, width=32)
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(16)
    input_shape.append(32)
    input_shape.append(32)
    var input = randn(input_shape, DType.float32)

    var output = layer.forward(input, training=True)

    # Output shape should match input shape
    var output_shape = output.shape()
    assert_equal(output_shape[0], 2)
    assert_equal(output_shape[1], 16)
    assert_equal(output_shape[2], 32)
    assert_equal(output_shape[3], 32)


fn test_batchnorm_forward_training_mode() raises:
    """Test BatchNorm2dLayer forward pass in training mode.

    Training mode should:
    1. Compute batch statistics
    2. Normalize using batch statistics
    3. Update running statistics.
    """
    var layer = BatchNorm2dLayer(4, momentum=0.1)

    # Create input with known values
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(4)
    input_shape.append(2)
    input_shape.append(2)
    var input = zeros(input_shape, DType.float32)

    # Fill with simple values to verify normalization
    # Batch 0: all values = 2.0
    # Batch 1: all values = 4.0
    for i in range(2):  # batch
        for c in range(4):  # channels
            for h in range(2):  # height
                for w in range(2):  # width
                    var idx = i * (4 * 2 * 2) + c * (2 * 2) + h * 2 + w
                    input._data.bitcast[Float32]()[idx] = Float32(2.0 + i * 2.0)

    # Forward in training mode
    var output = layer.forward(input, training=True)

    # Running statistics should be updated
    # After first forward with momentum=0.1:
    # running_mean = 0.9 * 0 + 0.1 * batch_mean = 0.1 * 3.0 = 0.3
    # (batch_mean = (2.0 * 8 + 4.0 * 8) / 16 = 3.0)
    var running_mean_data = layer.running_mean._data.bitcast[Float32]()
    var running_var_data = layer.running_var._data.bitcast[Float32]()

    # Check that running stats changed (not still initial values)
    var mean_changed = Float32(0.0)
    for i in range(4):
        if running_mean_data[i] != 0.0:
            mean_changed = 1.0

    assert_true(mean_changed > 0.5, "Running mean not updated in training mode")


fn test_batchnorm_forward_inference_mode() raises:
    """Test BatchNorm2dLayer forward pass in inference mode.

    Inference mode should:
    1. Use running statistics (not compute batch statistics)
    2. NOT update running statistics.
    """
    var layer = BatchNorm2dLayer(4)

    # Manually set running statistics
    var mean_data = layer.running_mean._data.bitcast[Float32]()
    var var_data = layer.running_var._data.bitcast[Float32]()
    for i in range(4):
        mean_data[i] = 0.5
        var_data[i] = 2.0

    # Create input
    var input_shape: List[Int] = [2, 4, 2, 2]
    var input = randn(input_shape, DType.float32)

    # Forward in inference mode
    var output = layer.forward(input, training=False)

    # Verify output shape unchanged
    var output_shape = output.shape()
    assert_equal(output_shape[0], 2)
    assert_equal(output_shape[1], 4)
    assert_equal(output_shape[2], 2)
    assert_equal(output_shape[3], 2)

    # Running statistics should NOT have changed
    for i in range(4):
        assert_almost_equal(mean_data[i], 0.5, tolerance=1e-6)
        assert_almost_equal(var_data[i], 2.0, tolerance=1e-6)


fn test_batchnorm_forward_without_gamma_beta() raises:
    """Test BatchNorm2dLayer forward pass with gamma=1, beta=0 (identity).

    With gamma=1 and beta=0, output should be normalized input.
    """
    var layer = BatchNorm2dLayer(4)

    # Input with known mean and variance
    var input_shape: List[Int] = [1, 4, 2, 2]
    var input = zeros(input_shape, DType.float32)

    # Fill channel 0 with [1, 2, 3, 4]
    var data = input._data.bitcast[Float32]()
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0

    # Forward pass
    var output = layer.forward(input, training=True)

    # With gamma=1, beta=0, output should be normalized
    # mean = 2.5, std = sqrt(1.25) ≈ 1.118
    var output_data = output._data.bitcast[Float32]()

    # Check that values are roughly normalized (mean ≈ 0)
    var sum = Float32(0.0)
    for i in range(4):
        sum += output_data[i]

    var mean = sum / 4.0
    assert_true(
        mean > -0.1 and mean < 0.1, "Normalized values should have mean near 0"
    )


# ============================================================================
# BatchNorm2dLayer Parameters Tests
# ============================================================================


fn test_batchnorm_parameters_list() raises:
    """Test BatchNorm2dLayer.parameters() returns gamma and beta tensors."""
    var layer = BatchNorm2dLayer(16)

    var params = layer.parameters()

    # Should return [gamma, beta]
    assert_equal(len(params), 2)

    # First parameter is gamma
    var gamma = params[0]
    var gamma_shape = gamma.shape()
    assert_equal(gamma_shape[0], 16)

    # Second parameter is beta
    var beta = params[1]
    var beta_shape = beta.shape()
    assert_equal(beta_shape[0], 16)


# ============================================================================
# BatchNorm2dLayer Running Statistics Tests
# ============================================================================


fn test_batchnorm_get_running_stats() raises:
    """Test BatchNorm2dLayer.get_running_stats() returns current statistics."""
    var layer = BatchNorm2dLayer(16)

    var stats = layer.get_running_stats()
    var mean = stats[0]
    var variance = stats[1]

    # Should return copies of running_mean and running_var
    var mean_shape = mean.shape()
    var variance_shape = variance.shape()

    assert_equal(mean_shape[0], 16)
    assert_equal(variance_shape[0], 16)

    # Verify initial values
    var mean_data = mean._data.bitcast[Float32]()
    var variance_data = variance._data.bitcast[Float32]()

    for i in range(16):
        assert_almost_equal(mean_data[i], 0.0, tolerance=1e-6)
        assert_almost_equal(variance_data[i], 1.0, tolerance=1e-6)


fn test_batchnorm_set_running_stats() raises:
    """Test BatchNorm2dLayer.set_running_stats() updates statistics."""
    var layer = BatchNorm2dLayer(16)

    # Create new running statistics
    var new_mean_shape = List[Int]()
    new_mean_shape.append(16)
    var new_mean = ones(new_mean_shape, DType.float32)

    var new_var_shape = List[Int]()
    new_var_shape.append(16)
    var new_var = zeros(new_var_shape, DType.float32)
    for i in range(16):
        new_var._data.bitcast[Float32]()[i] = 2.0

    # Set the statistics
    layer.set_running_stats(new_mean, new_var)

    # Verify they were set
    var mean_data = layer.running_mean._data.bitcast[Float32]()
    var var_data = layer.running_var._data.bitcast[Float32]()

    for i in range(16):
        assert_almost_equal(mean_data[i], 1.0, tolerance=1e-6)
        assert_almost_equal(var_data[i], 2.0, tolerance=1e-6)


fn test_batchnorm_running_stats_update_over_batches() raises:
    """Test BatchNorm2dLayer running statistics accumulate over multiple batches.

    With momentum=0.1, running_stat = 0.9 * old + 0.1 * batch_stat.
    """
    var layer = BatchNorm2dLayer(1, momentum=0.1)

    # First batch: all 1.0
    var input1_shape = List[Int]()
    input1_shape.append(1)
    input1_shape.append(1)
    input1_shape.append(2)
    input1_shape.append(2)
    var input1 = ones(input1_shape, DType.float32)

    var output1 = layer.forward(input1, training=True)

    var (mean1, var1) = layer.get_running_stats()
    var mean_data1 = mean1._data.bitcast[Float32]()
    var mean_after_first = mean_data1[0]

    # Second batch: all 3.0
    var input2 = zeros(input1_shape, DType.float32)
    var data2 = input2._data.bitcast[Float32]()
    for i in range(4):
        data2[i] = 3.0

    var output2 = layer.forward(input2, training=True)

    var (mean2, var2) = layer.get_running_stats()
    var mean_data2 = mean2._data.bitcast[Float32]()
    var mean_after_second = mean_data2[0]

    # Running mean should have changed (0.9 * 0.1 + 0.1 * 3 ≈ 0.309)
    # But not as much as batch mean (which would be 3.0)
    assert_true(
        mean_after_first < mean_after_second,
        "Running mean should increase after seeing 3.0",
    )
    assert_true(
        mean_after_second < Float32(1.5),
        "Running mean should not jump all the way to batch mean",
    )


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all BatchNorm2dLayer tests."""
    print("Running BatchNorm2dLayer initialization tests...")
    test_batchnorm_initialization()
    test_batchnorm_gamma_initialized_to_one()
    test_batchnorm_beta_initialized_to_zero()
    test_batchnorm_running_mean_initialized_to_zero()
    test_batchnorm_running_var_initialized_to_one()
    test_batchnorm_initialization_with_momentum_eps()

    print("Running BatchNorm2dLayer forward pass tests...")
    test_batchnorm_forward_output_shape()
    test_batchnorm_forward_training_mode()
    test_batchnorm_forward_inference_mode()
    test_batchnorm_forward_without_gamma_beta()

    print("Running BatchNorm2dLayer parameters tests...")
    test_batchnorm_parameters_list()

    print("Running BatchNorm2dLayer running statistics tests...")
    test_batchnorm_get_running_stats()
    test_batchnorm_set_running_stats()
    test_batchnorm_running_stats_update_over_batches()

    print("\nAll BatchNorm2dLayer tests passed! ✓")
