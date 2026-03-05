"""Tests for batch normalization and layer normalization.

Tests cover:
- Batch normalization (training and inference modes)
- Layer normalization (2D and 4D inputs)
- Running statistics updates
- Numerical correctness
- Shape validation
- Backward passes with gradient checking (COMPREHENSIVE)

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_shape_equal,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.testing import (
    compute_numerical_gradient,
    assert_gradients_close,
)
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
    layer_norm,
    layer_norm_backward,
)
from shared.core.arithmetic import add, subtract, multiply
from shared.core.reduction import sum as reduce_sum


# ============================================================================
# Batch Normalization Tests
# ============================================================================


fn test_batch_norm2d_shapes() raises:
    """Test that batch_norm2d returns correct output shape."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(3)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var x = ones(shape, DType.float32)

    # Create gamma, beta, running_mean, running_var for 3 channels
    var param_shape = List[Int]()
    param_shape.append(3)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Training mode
    var result = batch_norm2d(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
        momentum=0.1,
        epsilon=1e-5,
    )
    var output = result[0]
    var new_mean = result[1]
    var new_var = result[2]

    # Check output shape
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 3)
    assert_equal(output.shape()[2], 4)
    assert_equal(output.shape()[3], 4)

    # Check statistics shapes
    assert_equal(new_mean.shape()[0], 3)
    assert_equal(new_var.shape()[0], 3)


fn test_batch_norm2d_training_mode() raises:
    """Test that batch_norm2d computes batch statistics in training mode."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(1)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var x = zeros(shape, DType.float32)

    # Set specific values: [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(8):
        x._data.bitcast[Float32]()[i] = Float32(i)

    # Mean should be 3.5, variance should be computed from data
    var param_shape = List[Int]()
    param_shape.append(1)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var result2 = batch_norm2d(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
        momentum=0.1,
        epsilon=1e-5,
    )
    var output = result2[0]
    var new_mean = result2[1]
    var new_var = result2[2]

    # In training mode, running stats should be updated
    # new_running_mean = (1 - momentum) * old + momentum * batch_mean
    # = 0.9 * 0.0 + 0.1 * 3.5 = 0.35
    assert_almost_equal(
        new_mean._data.bitcast[Float32]()[0], Float32(0.35), tolerance=1e-4
    )


fn test_batch_norm2d_inference_mode() raises:
    """Test that batch_norm2d uses running statistics in inference mode."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(1)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var x = ones(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape.append(1)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    # Set running statistics
    var running_mean = zeros(param_shape, DType.float32)
    running_mean._data.bitcast[Float32]()[0] = 0.5

    var running_var = ones(param_shape, DType.float32)
    running_var._data.bitcast[Float32]()[0] = 0.25

    # Inference mode
    var result3 = batch_norm2d(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=False,
        momentum=0.1,
        epsilon=1e-5,
    )
    var output = result3[0]
    var new_mean = result3[1]
    var new_var = result3[2]

    # Running statistics should be unchanged in inference mode
    assert_almost_equal(
        new_mean._data.bitcast[Float32]()[0], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        new_var._data.bitcast[Float32]()[0], Float32(0.25), tolerance=1e-5
    )

    # Output should use running statistics for normalization
    # normalized = (x - running_mean) / sqrt(running_var + eps)
    # = (1.0 - 0.5) / sqrt(0.25 + 1e-5) ≈ 0.5 / 0.5 = 1.0
    # output = gamma * normalized + beta = 1.0 * 1.0 + 0.0 = 1.0
    for i in range(8):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-3
        )


fn test_batch_norm2d_scale_shift() raises:
    """Test that batch_norm2d applies gamma and beta correctly."""
    var shape = List[Int]()
    shape.append(1)  # batch
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var x = zeros(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape.append(2)

    # Set gamma = [2.0, 3.0], beta = [1.0, -1.0]
    var gamma = zeros(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 2.0
    gamma._data.bitcast[Float32]()[1] = 3.0

    var beta = zeros(param_shape, DType.float32)
    beta._data.bitcast[Float32]()[0] = 1.0
    beta._data.bitcast[Float32]()[1] = -1.0

    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Inference mode with zero input and zero mean
    var result4 = batch_norm2d(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=False,
        momentum=0.1,
        epsilon=1e-5,
    )
    var output = result4[0]

    # For zero input with zero mean: normalized = 0
    # output = gamma * 0 + beta = beta
    # Channel 0: beta[0] = 1.0
    # Channel 1: beta[1] = -1.0

    # Check channel 0 values (indices 0-3)
    for i in range(4):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-4
        )

    # Check channel 1 values (indices 4-7)
    for i in range(4, 8):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i], Float32(-1.0), tolerance=1e-4
        )


fn test_batch_norm2d_zero_variance() raises:
    """Test that batch_norm2d handles zero variance with epsilon."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(1)  # channels
    shape.append(1)  # height
    shape.append(1)  # width
    var x = ones(shape, DType.float32)  # All values are 1.0 - zero variance

    var param_shape = List[Int]()
    param_shape.append(1)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # This should not crash due to division by zero
    var result5 = batch_norm2d(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=True,
        momentum=0.1,
        epsilon=1e-5,
    )
    var output = result5[0]

    # All outputs should be finite
    for i in range(2):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(val == val)  # Not NaN
        assert_true(val > -1e10 and val < 1e10)  # Not infinite


# ============================================================================
# Batch Normalization Backward Pass Tests (GRADIENT CHECKING)
# ============================================================================


fn test_batch_norm2d_backward_gradient_input() raises:
    """Test batch_norm2d_backward gradient w.r.t. input using numerical validation.

    CRITICAL TEST: Validates mathematical correctness of batch norm backpropagation.
    Uses central finite differences for gold-standard gradient validation.

    Uses non-uniform grad_output to avoid the pathological case where
    grad_output=ones gives analytically-zero gradients (sum(x_norm)=0 cancellation),
    which would make the test sensitive to floating-point noise in the forward pass.
    """
    # Small tensor for gradient checking (computational cost is O(n²))
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width

    # Create test input with varying values
    var x = zeros(shape, DType.float32)
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Parameters
    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 1.5
    gamma._data.bitcast[Float32]()[1] = 2.0

    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Forward pass
    var result6 = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )
    var output = result6[0]

    # Use non-uniform grad_output to avoid the pathological cancellation case
    # where grad_output=ones gives dL/dx ~ sum(x_norm)=0 (analytically zero gradient).
    # Non-uniform weights break the symmetry, producing non-zero testable gradients.
    var grad_output = zeros_like(output)
    for i in range(16):
        # Alternating pattern to avoid symmetry: [0.5, -0.3, 0.8, -0.2, ...]
        var val = Float32(i % 4) * Float32(0.25) - Float32(0.3)
        grad_output._data.bitcast[Float32]()[i] = val

    # Backward pass
    var result7 = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=True,
        epsilon=1e-5,
    )
    var grad_input = result7[0]

    # Numerical gradient via finite differences.
    # forward_for_grad computes weighted sum: sum(output * grad_output)
    # so the numerical gradient matches what the backward should compute.
    fn forward_for_grad(inp: ExTensor) raises -> ExTensor:
        var result_nested = batch_norm2d(
            inp,
            gamma,
            beta,
            running_mean,
            running_var,
            training=True,
            epsilon=1e-5,
        )
        var out = result_nested[0]
        # Compute weighted sum: sum(output * grad_output)
        # This matches the backward with our non-uniform grad_output
        var weighted = multiply(out, grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result

    var numerical_grad = compute_numerical_gradient(
        forward_for_grad, x, epsilon=1e-4
    )

    # Validate analytical gradient matches numerical gradient
    # Looser tolerance for batch norm (complex operation with many intermediate steps)
    assert_gradients_close(
        grad_input,
        numerical_grad,
        rtol=1e-2,
        atol=1e-5,
        message="Batch norm gradient w.r.t. input",
    )

    print("✓ Batch norm backward gradient (input) validated numerically")


fn test_batch_norm2d_backward_training_vs_inference() raises:
    """Test that batch_norm2d_backward behaves differently in training vs inference.

    Training mode: Gradients flow through batch statistics
    Inference mode: Gradients bypass statistics (use running stats).
    """
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(1)  # channels
    shape.append(2)  # height
    shape.append(2)  # width

    var x = zeros(shape, DType.float32)
    for i in range(8):
        x._data.bitcast[Float32]()[i] = Float32(i)

    var param_shape = List[Int]()
    param_shape.append(1)
    var gamma = ones(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 2.0

    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Forward passes
    var result8 = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True
    )
    var out_train = result8[0]
    var result9 = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=False
    )
    var out_infer = result9[0]

    var grad_output = ones_like(out_train)

    # Backward passes
    var result10 = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var, training=True
    )
    var grad_train = result10[0]
    var result11 = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var, training=False
    )
    var grad_infer = result11[0]

    # Gradients should differ between training and inference modes
    var diff_found = False
    for i in range(8):
        var diff = abs(
            grad_train._data.bitcast[Float32]()[i]
            - grad_infer._data.bitcast[Float32]()[i]
        )
        if diff > 1e-5:
            diff_found = True
            break

    assert_true(diff_found, "Training and inference gradients should differ")
    print("✓ Batch norm backward: training vs inference modes differ correctly")


fn test_batch_norm2d_backward_shapes() raises:
    """Test that batch_norm2d_backward returns correct gradient shapes."""
    var shape = List[Int]()
    shape.append(3)  # batch
    shape.append(4)  # channels
    shape.append(5)  # height
    shape.append(5)  # width

    var x = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape.append(4)
    var gamma = ones(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var result12 = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var, training=True
    )
    var grad_input = result12[0]
    var grad_gamma = result12[1]
    var grad_beta = result12[2]

    # Validate shapes
    assert_shape_equal(
        grad_input.shape(), x.shape(), "grad_input should match input shape"
    )
    assert_shape_equal(
        grad_gamma.shape(), gamma.shape(), "grad_gamma should match gamma shape"
    )
    assert_shape_equal(
        grad_beta.shape(), gamma.shape(), "grad_beta should match beta shape"
    )

    # Check specific dimensions
    assert_equal(grad_input.shape()[0], 3, "batch dimension")
    assert_equal(grad_input.shape()[1], 4, "channels dimension")
    assert_equal(grad_gamma.shape()[0], 4, "gamma channels")
    assert_equal(grad_beta.shape()[0], 4, "beta channels")


# ============================================================================
# Layer Normalization Tests
# ============================================================================


fn test_layer_norm_shapes_2d() raises:
    """Test that layer_norm returns correct shape for 2D input."""
    var shape = List[Int]()
    shape.append(4)  # batch
    shape.append(10)  # features
    var x = ones(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape.append(10)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # Check output shape
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)


fn test_layer_norm_shapes_4d() raises:
    """Test that layer_norm returns correct shape for 4D input."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(3)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var x = ones(shape, DType.float32)

    # For 4D input, normalize over C*H*W
    var normalized_shape = 3 * 4 * 4  # 48
    var param_shape = List[Int]()
    param_shape.append(normalized_shape)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # Check output shape
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 3)
    assert_equal(output.shape()[2], 4)
    assert_equal(output.shape()[3], 4)


fn test_layer_norm_normalization_2d() raises:
    """Test that layer_norm normalizes each sample independently."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(4)  # features
    var x = zeros(shape, DType.float32)

    # Sample 1: [0, 1, 2, 3]
    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 2.0
    x._data.bitcast[Float32]()[3] = 3.0

    # Sample 2: [4, 5, 6, 7]
    x._data.bitcast[Float32]()[4] = 4.0
    x._data.bitcast[Float32]()[5] = 5.0
    x._data.bitcast[Float32]()[6] = 6.0
    x._data.bitcast[Float32]()[7] = 7.0

    var param_shape = List[Int]()
    param_shape.append(4)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # For each sample, mean should be 0 and variance should be 1 (approximately)
    # Sample 1 mean: (0+1+2+3)/4 = 1.5
    # Sample 1 std: sqrt(((0-1.5)^2 + (1-1.5)^2 + (2-1.5)^2 + (3-1.5)^2)/4) ≈ 1.118

    # After normalization: (x - mean) / std
    # Sample 1[0]: (0 - 1.5) / 1.118 ≈ -1.34
    # Sample 1[1]: (1 - 1.5) / 1.118 ≈ -0.45
    # Sample 1[2]: (2 - 1.5) / 1.118 ≈ 0.45
    # Sample 1[3]: (3 - 1.5) / 1.118 ≈ 1.34

    # Check that first sample has approximately zero mean
    var sum1 = Float32(0.0)
    for i in range(4):
        sum1 += output._data.bitcast[Float32]()[i]
    var mean1 = sum1 / 4.0
    assert_almost_equal(mean1, Float32(0.0), tolerance=1e-5)

    # Check that second sample has approximately zero mean
    var sum2 = Float32(0.0)
    for i in range(4, 8):
        sum2 += output._data.bitcast[Float32]()[i]
    var mean2 = sum2 / 4.0
    assert_almost_equal(mean2, Float32(0.0), tolerance=1e-5)


fn test_layer_norm_scale_shift() raises:
    """Test that layer_norm applies gamma and beta correctly."""
    var shape = List[Int]()
    shape.append(1)  # batch
    shape.append(3)  # features
    var x = zeros(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape.append(3)

    # Set gamma = [2.0, 3.0, 4.0], beta = [1.0, 0.0, -1.0]
    var gamma = zeros(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 2.0
    gamma._data.bitcast[Float32]()[1] = 3.0
    gamma._data.bitcast[Float32]()[2] = 4.0

    var beta = zeros(param_shape, DType.float32)
    beta._data.bitcast[Float32]()[0] = 1.0
    beta._data.bitcast[Float32]()[1] = 0.0
    beta._data.bitcast[Float32]()[2] = -1.0

    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # For zero input with zero mean: normalized = 0
    # output = gamma * 0 + beta = beta
    assert_almost_equal(
        output._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-4
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-4
    )
    assert_almost_equal(
        output._data.bitcast[Float32]()[2], Float32(-1.0), tolerance=1e-4
    )


fn test_layer_norm_zero_variance() raises:
    """Test that layer_norm handles zero variance with epsilon."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(3)  # features
    var x = ones(shape, DType.float32)  # All values are 1.0 - zero variance

    var param_shape = List[Int]()
    param_shape.append(3)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)

    # This should not crash due to division by zero
    var output = layer_norm(x, gamma, beta, epsilon=1e-5)

    # All outputs should be finite
    for i in range(6):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(val == val)  # Not NaN
        assert_true(val > -1e10 and val < 1e10)  # Not infinite


# ============================================================================
# Layer Normalization Backward Pass Tests
# ============================================================================


fn test_layer_norm_backward_shapes_2d() raises:
    """Test that layer_norm_backward returns correct gradient shapes for 2D input.
    """
    var shape = List[Int]()
    shape.append(4)  # batch
    shape.append(10)  # features
    var x = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape.append(10)
    var gamma = ones(param_shape, DType.float32)

    var result13 = layer_norm_backward(grad_output, x, gamma, epsilon=1e-5)
    var grad_input = result13[0]
    var grad_gamma = result13[1]
    var grad_beta = result13[2]

    # Validate shapes
    assert_shape_equal(
        grad_input.shape(), x.shape(), "grad_input should match input shape"
    )
    assert_shape_equal(
        grad_gamma.shape(), gamma.shape(), "grad_gamma should match gamma shape"
    )
    assert_shape_equal(
        grad_beta.shape(), gamma.shape(), "grad_beta should match beta shape"
    )

    # Check specific dimensions
    assert_equal(grad_input.shape()[0], 4, "batch dimension")
    assert_equal(grad_input.shape()[1], 10, "features dimension")
    assert_equal(grad_gamma.shape()[0], 10, "gamma features")
    assert_equal(grad_beta.shape()[0], 10, "beta features")


fn test_layer_norm_backward_shapes_4d() raises:
    """Test that layer_norm_backward returns correct gradient shapes for 4D input.
    """
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(3)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var x = ones(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    # For 4D input, gamma has shape (C * H * W)
    var normalized_size = 3 * 4 * 4  # 48
    var param_shape = List[Int]()
    param_shape.append(normalized_size)
    var gamma = ones(param_shape, DType.float32)

    var result14 = layer_norm_backward(grad_output, x, gamma, epsilon=1e-5)
    var grad_input = result14[0]
    var grad_gamma = result14[1]
    var grad_beta = result14[2]

    # Validate shapes
    assert_shape_equal(
        grad_input.shape(), x.shape(), "grad_input should match input shape"
    )
    assert_shape_equal(
        grad_gamma.shape(), gamma.shape(), "grad_gamma should match gamma shape"
    )
    assert_shape_equal(
        grad_beta.shape(), gamma.shape(), "grad_beta should match beta shape"
    )


fn test_layer_norm_backward_grad_beta() raises:
    """Test that layer_norm_backward grad_beta equals sum of grad_output over batch.
    """
    var shape = List[Int]()
    shape.append(3)  # batch
    shape.append(4)  # features
    var x = zeros(shape, DType.float32)

    # Set varying input values
    for i in range(12):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var param_shape = List[Int]()
    param_shape.append(4)
    var gamma = ones(param_shape, DType.float32)

    # Create varying grad_output
    var grad_output = zeros(shape, DType.float32)
    for b in range(3):
        for f in range(4):
            var idx = b * 4 + f
            grad_output._data.bitcast[Float32]()[idx] = (
                Float32(b + 1) * Float32(f + 1) * 0.1
            )

    var result15 = layer_norm_backward(grad_output, x, gamma, epsilon=1e-5)
    var grad_beta = result15[2]

    # grad_beta should be sum over batch dimension
    # For each feature f: grad_beta[f] = sum over b of grad_output[b, f]
    for f in range(4):
        var expected_sum = Float32(0.0)
        for b in range(3):
            expected_sum += Float32(b + 1) * Float32(f + 1) * 0.1
        assert_almost_equal(
            grad_beta._data.bitcast[Float32]()[f], expected_sum, tolerance=1e-4
        )


fn test_layer_norm_backward_zero_input() raises:
    """Test layer_norm_backward with zero input (edge case)."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(4)  # features
    var x = zeros(shape, DType.float32)  # All zeros - zero variance

    var param_shape = List[Int]()
    param_shape.append(4)
    var gamma = ones(param_shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    # Should not crash with zero variance
    var result16 = layer_norm_backward(grad_output, x, gamma, epsilon=1e-5)
    var grad_input = result16[0]
    var grad_gamma = result16[1]
    var grad_beta = result16[2]

    # All gradients should be finite
    for i in range(8):
        var val = grad_input._data.bitcast[Float32]()[i]
        assert_true(val == val, "grad_input should not be NaN")
        assert_true(val > -1e10 and val < 1e10, "grad_input should be finite")

    for i in range(4):
        var val_gamma = grad_gamma._data.bitcast[Float32]()[i]
        var val_beta = grad_beta._data.bitcast[Float32]()[i]
        assert_true(val_gamma == val_gamma, "grad_gamma should not be NaN")
        assert_true(val_beta == val_beta, "grad_beta should not be NaN")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all normalization tests."""
    print("Running normalization tests...")

    # Batch normalization tests
    test_batch_norm2d_shapes()
    print("✓ test_batch_norm2d_shapes")

    test_batch_norm2d_training_mode()
    print("✓ test_batch_norm2d_training_mode")

    test_batch_norm2d_inference_mode()
    print("✓ test_batch_norm2d_inference_mode")

    test_batch_norm2d_scale_shift()
    print("✓ test_batch_norm2d_scale_shift")

    test_batch_norm2d_zero_variance()
    print("✓ test_batch_norm2d_zero_variance")

    # Batch normalization backward pass tests (gradient checking)
    test_batch_norm2d_backward_gradient_input()
    print("✓ test_batch_norm2d_backward_gradient_input")

    test_batch_norm2d_backward_training_vs_inference()
    print("✓ test_batch_norm2d_backward_training_vs_inference")

    test_batch_norm2d_backward_shapes()
    print("✓ test_batch_norm2d_backward_shapes")

    # Layer normalization tests
    test_layer_norm_shapes_2d()
    print("✓ test_layer_norm_shapes_2d")

    test_layer_norm_shapes_4d()
    print("✓ test_layer_norm_shapes_4d")

    test_layer_norm_normalization_2d()
    print("✓ test_layer_norm_normalization_2d")

    test_layer_norm_scale_shift()
    print("✓ test_layer_norm_scale_shift")

    test_layer_norm_zero_variance()
    print("✓ test_layer_norm_zero_variance")

    # Layer normalization backward pass tests
    test_layer_norm_backward_shapes_2d()
    print("✓ test_layer_norm_backward_shapes_2d")

    test_layer_norm_backward_shapes_4d()
    print("✓ test_layer_norm_backward_shapes_4d")

    test_layer_norm_backward_grad_beta()
    print("✓ test_layer_norm_backward_grad_beta")

    test_layer_norm_backward_zero_input()
    print("✓ test_layer_norm_backward_zero_input")

    print("\nAll normalization tests passed!")
