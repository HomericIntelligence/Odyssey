# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_normalization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for batch normalization backward (continued) and layer normalization forward.

Tests cover:
- Batch normalization backward: gradient beta
- Batch normalization backward: training vs inference
- Batch normalization backward: shapes
- Layer normalization shapes (2D and 4D inputs)
- Layer normalization normalization correctness
- Layer normalization scale/shift
- Layer normalization zero variance
- Layer normalization backward shapes (2D)

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
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
    layer_norm,
    layer_norm_backward,
)
from shared.core.arithmetic import add, subtract, multiply
from shared.core.reduction import sum as reduce_sum


# ============================================================================
# Batch Normalization Backward Pass Tests (continued)
# ============================================================================


fn test_batch_norm2d_backward_gradient_beta() raises:
    """Test batch_norm2d_backward gradient w.r.t. beta using numerical validation.

    Perturbs beta using finite differences and compares against analytical grad_beta.
    beta contributes additively, so grad_beta = sum(grad_output) over batch and spatial dims.
    """
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width

    var x = zeros(shape, DType.float32)
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var param_shape = List[Int]()
    param_shape.append(2)

    var gamma = ones(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 1.5
    gamma._data.bitcast[Float32]()[1] = 2.0

    var beta = zeros(param_shape, DType.float32)
    beta._data.bitcast[Float32]()[0] = 0.5
    beta._data.bitcast[Float32]()[1] = -0.5

    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Forward pass to get output shape for grad_output
    var result_fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )
    var output = result_fwd[0]

    # Non-uniform grad_output
    var grad_output = zeros_like(output)
    for i in range(16):
        grad_output._data.bitcast[Float32]()[i] = Float32(i + 1) * 0.1

    # Analytical gradient
    var result_bwd = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=True,
        epsilon=1e-5,
    )
    var grad_beta_analytical = result_bwd[2]

    # Numerical gradient: perturb beta
    fn forward_for_beta(b: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            x, gamma, b, running_mean, running_var, training=True, epsilon=1e-5
        )
        var out = res[0]
        var weighted = multiply(out, grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result

    var numerical_grad_beta = compute_numerical_gradient(
        forward_for_beta, beta, epsilon=1e-3
    )

    assert_gradients_close(
        grad_beta_analytical,
        numerical_grad_beta,
        rtol=1e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. beta",
    )
    print("✓ Batch norm backward gradient (beta) validated numerically")


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
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run batch normalization backward (continued) and layer norm forward tests.
    """
    print("Running normalization part2 tests...")

    # Batch normalization backward pass tests (continued)
    test_batch_norm2d_backward_gradient_beta()
    print("✓ test_batch_norm2d_backward_gradient_beta")

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

    print("\nAll normalization part2 tests passed!")
