# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_normalization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for layer normalization backward pass.

Tests cover:
- Layer normalization backward shapes (2D and 4D)
- Layer normalization backward grad_beta
- Layer normalization backward zero input edge case
- Layer normalization backward gradient input (numerical validation)

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
from shared.core.extensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
    layer_norm,
    layer_norm_backward,
)
from shared.core.arithmetic import add, subtract, multiply
from shared.core.reduction import sum as reduce_sum


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


fn test_layer_norm_backward_gradient_input() raises:
    """Test layer_norm_backward gradient w.r.t. input using numerical validation.

    CRITICAL TEST: Validates mathematical correctness of layer norm backpropagation.
    Uses central finite differences for gold-standard gradient validation.
    Uses non-uniform grad_output to prevent algebraic cancellation masking bugs.

    When grad_output=ones, sum(grad_output * x_hat) = sum(x_hat) = 0 by normalization,
    making the last term in the backward formula vanish. Non-uniform grad_output
    ensures sum(grad_output * x_hat) != 0, exercising the full backward formula.
    """
    # Small 2D tensor: (batch=2, features=4)
    var shape = List[Int]()
    shape.append(2)
    shape.append(4)

    # Input with varying values (not uniform, not zero)
    var x = zeros(shape, DType.float32)
    for i in range(8):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.05

    # Parameters: non-trivial gamma, zero beta
    var param_shape = List[Int]()
    param_shape.append(4)
    var gamma = ones(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 1.5
    gamma._data.bitcast[Float32]()[1] = 0.8
    gamma._data.bitcast[Float32]()[2] = 1.2
    gamma._data.bitcast[Float32]()[3] = 2.0
    var beta = zeros(param_shape, DType.float32)

    # Non-uniform grad_output: critical to avoid algebraic cancellation
    var grad_output = zeros(shape, DType.float32)
    grad_output._data.bitcast[Float32]()[0] = 0.3
    grad_output._data.bitcast[Float32]()[1] = -0.5
    grad_output._data.bitcast[Float32]()[2] = 1.2
    grad_output._data.bitcast[Float32]()[3] = -0.8
    grad_output._data.bitcast[Float32]()[4] = 0.7
    grad_output._data.bitcast[Float32]()[5] = -0.2
    grad_output._data.bitcast[Float32]()[6] = 0.9
    grad_output._data.bitcast[Float32]()[7] = -1.1

    # Analytical backward pass
    var result = layer_norm_backward(grad_output, x, gamma, epsilon=1e-5)
    var grad_input = result[0]

    # Numerical gradient via finite differences.
    # The scalar loss is sum(layer_norm(x) * grad_output), so the numerical
    # gradient matches what layer_norm_backward(grad_output, x, gamma) computes.
    fn forward_for_grad(inp: AnyTensor) raises -> AnyTensor:
        var out = layer_norm(inp, gamma, beta, epsilon=1e-5)
        # Weighted sum: sum(out * grad_output) matches backward with non-uniform grad_output
        var weighted = multiply(out, grad_output)
        var result_inner = weighted
        while result_inner.dim() > 0:
            result_inner = reduce_sum(result_inner, axis=0, keepdims=False)
        return result_inner

    var numerical_grad = compute_numerical_gradient(
        forward_for_grad, x, epsilon=1e-4
    )

    # Validate analytical gradient matches numerical gradient
    assert_gradients_close(
        grad_input,
        numerical_grad,
        rtol=1e-2,
        atol=1e-5,
        message="Layer norm gradient w.r.t. input",
    )

    print("✓ Layer norm backward gradient (input) validated numerically")


fn test_layer_norm_backward_gradient_input_4d() raises:
    """Test layer_norm_backward gradient w.r.t. input on 4D inputs using numerical validation.

    CRITICAL TEST: Validates mathematical correctness of layer norm backpropagation for
    4D inputs (batch, channels, H, W), where normalization is applied over the last 3
    dimensions. Indexing and reduction logic differs from 2D inputs, so independent
    numerical validation is required.

    Shape: [2, 2, 2, 4] — 2 samples, normalized over [2, 2, 4] = 16 elements each.
    Gamma shape: [16] (flattened last 3 dims), matching 4D implementation convention.

    Uses non-uniform grad_output to prevent algebraic cancellation:
    When grad_output=ones, sum(grad_output * x_hat) = sum(x_hat) = 0 by normalization,
    making the last term in the backward formula vanish. Non-uniform grad_output
    ensures sum(grad_output * x_hat) != 0, exercising the full backward formula.
    """
    # 4D tensor: (batch=2, channels=2, H=2, W=4)
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(4)

    # Input with varying values across all 32 elements (not uniform, not zero)
    var x = zeros(shape, DType.float32)
    for i in range(32):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.05

    # Parameters: gamma shape [16] (flattened last 3 dims), non-trivial values
    var param_shape = List[Int]()
    param_shape.append(16)
    var gamma = ones(param_shape, DType.float32)
    # Non-uniform gamma values cycling through [1.5, 0.8, 1.2, 2.0]
    for i in range(16):
        var cycling_values = List[Float32]()
        cycling_values.append(1.5)
        cycling_values.append(0.8)
        cycling_values.append(1.2)
        cycling_values.append(2.0)
        gamma._data.bitcast[Float32]()[i] = cycling_values[i % 4]
    var beta = zeros(param_shape, DType.float32)

    # Non-uniform grad_output: critical to avoid algebraic cancellation
    # Shape [2, 2, 2, 4] = 32 elements; alternating mixed signs, small magnitudes
    var grad_output = zeros(shape, DType.float32)
    var go_vals = List[Float32]()
    go_vals.append(0.03)
    go_vals.append(-0.07)
    go_vals.append(0.05)
    go_vals.append(-0.02)
    go_vals.append(0.06)
    go_vals.append(-0.04)
    go_vals.append(0.08)
    go_vals.append(-0.01)
    go_vals.append(-0.05)
    go_vals.append(0.09)
    go_vals.append(-0.03)
    go_vals.append(0.07)
    go_vals.append(-0.06)
    go_vals.append(0.02)
    go_vals.append(-0.08)
    go_vals.append(0.04)
    go_vals.append(0.05)
    go_vals.append(-0.09)
    go_vals.append(0.01)
    go_vals.append(-0.06)
    go_vals.append(0.07)
    go_vals.append(-0.03)
    go_vals.append(0.04)
    go_vals.append(-0.08)
    go_vals.append(0.02)
    go_vals.append(-0.05)
    go_vals.append(0.09)
    go_vals.append(-0.01)
    go_vals.append(0.06)
    go_vals.append(-0.07)
    go_vals.append(0.03)
    go_vals.append(-0.04)
    for i in range(32):
        grad_output._data.bitcast[Float32]()[i] = go_vals[i]

    # Analytical backward pass
    var result = layer_norm_backward(grad_output, x, gamma, epsilon=1e-5)
    var grad_input = result[0]

    # Numerical gradient via finite differences.
    # The scalar loss is sum(layer_norm(x) * grad_output), so the numerical
    # gradient matches what layer_norm_backward(grad_output, x, gamma) computes.
    fn forward_for_grad_4d(inp: AnyTensor) raises -> AnyTensor:
        var out = layer_norm(inp, gamma, beta, epsilon=1e-5)
        # Weighted sum: sum(out * grad_output) matches backward with non-uniform grad_output
        var weighted = multiply(out, grad_output)
        var result_inner = weighted
        while result_inner.dim() > 0:
            result_inner = reduce_sum(result_inner, axis=0, keepdims=False)
        return result_inner

    var numerical_grad = compute_numerical_gradient(
        forward_for_grad_4d, x, epsilon=1e-4
    )

    # Validate analytical gradient matches numerical gradient
    assert_gradients_close(
        grad_input,
        numerical_grad,
        rtol=1e-2,
        atol=1e-5,
        message="Layer norm 4D gradient w.r.t. input",
    )

    print("✓ Layer norm backward 4D gradient (input) validated numerically")


# ============================================================================
# Main Test Runner
# ============================================================================


# ============================================================================
# Batch Norm Backward Tests (#3664, #3665)
# ============================================================================


fn test_batch_norm2d_backward_gamma_beta_nonzero() raises:
    """Test batch_norm2d_backward produces non-zero gamma/beta gradients. Closes #3664.
    """
    # NCHW: batch=2, channels=2, height=2, width=2
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var x = zeros(shape, DType.float32)
    # Fill with varying values to get meaningful gradients
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Forward pass to get running stats
    var fwd_result = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )

    # Grad output: ones
    var grad_output = ones(shape, DType.float32)

    var bwd_result = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var, training=True, epsilon=1e-5
    )
    var grad_gamma = bwd_result[1]
    var grad_beta = bwd_result[2]

    # grad_beta should be sum of grad_output over (N,H,W) = 2*2*2 = 8 per channel
    assert_almost_equal(
        grad_beta._data.bitcast[Float32]()[0], Float32(8.0), tolerance=1e-3
    )
    assert_almost_equal(
        grad_beta._data.bitcast[Float32]()[1], Float32(8.0), tolerance=1e-3
    )

    # grad_gamma should be non-zero (sum of grad_output * x_hat over N,H,W)
    var gg0 = grad_gamma._data.bitcast[Float32]()[0]
    var gg1 = grad_gamma._data.bitcast[Float32]()[1]
    assert_true(gg0 == gg0, "grad_gamma[0] should not be NaN")
    assert_true(gg1 == gg1, "grad_gamma[1] should not be NaN")

    print("✓ batch_norm2d_backward gamma/beta gradient test passed")


fn test_batch_norm2d_backward_inference_mode() raises:
    """Test batch_norm2d_backward in inference mode. Closes #3665."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var x = zeros(shape, DType.float32)
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var grad_output = ones(shape, DType.float32)

    # Backward in inference mode (training=False)
    var bwd_result = batch_norm2d_backward(
        grad_output, x, gamma, running_mean, running_var, training=False, epsilon=1e-5
    )
    var grad_input = bwd_result[0]

    # grad_input should be finite and non-zero in inference mode
    for i in range(16):
        var val = grad_input._data.bitcast[Float32]()[i]
        assert_true(val == val, "Inference mode grad should not be NaN")

    print("✓ batch_norm2d_backward inference mode test passed")


fn test_batch_norm2d_backward_gradient_gamma_inference_mode() raises:
    """Test batch_norm2d_backward gradient w.r.t. gamma in inference mode.

    Mirrors test_batch_norm2d_backward_gradient_gamma but with training=False.
    In inference mode, gamma gradient depends on fixed running statistics.
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

    var running_mean = zeros(param_shape, DType.float32)
    running_mean._data.bitcast[Float32]()[0] = 0.3
    running_mean._data.bitcast[Float32]()[1] = 0.7

    var running_var = ones(param_shape, DType.float32)
    running_var._data.bitcast[Float32]()[0] = 0.5
    running_var._data.bitcast[Float32]()[1] = 1.5

    # Forward pass in inference mode
    var result_fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=False, epsilon=1e-5
    )
    var output = result_fwd[0]

    # Non-uniform grad_output to exercise per-channel sensitivity
    var grad_output = zeros_like(output)
    for i in range(16):
        grad_output._data.bitcast[Float32]()[i] = Float32(i + 1) * 0.1

    # Analytical gradient in inference mode
    var result_bwd = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=False,
        epsilon=1e-5,
    )
    var grad_gamma_analytical = result_bwd[1]

    # Numerical gradient: perturb gamma in inference mode
    fn forward_for_gamma_infer(g: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            x, g, beta, running_mean, running_var, training=False, epsilon=1e-5
        )
        var out = res[0]
        var weighted = multiply(out, grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result

    var numerical_grad_gamma = compute_numerical_gradient(
        forward_for_gamma_infer, gamma, epsilon=1e-3
    )

    assert_gradients_close(
        grad_gamma_analytical,
        numerical_grad_gamma,
        rtol=1e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. gamma (inference mode)",
    )
    print("✓ Batch norm backward gradient (gamma) inference mode validated numerically")


fn test_batch_norm2d_backward_gradient_beta_inference_mode() raises:
    """Test batch_norm2d_backward gradient w.r.t. beta in inference mode.

    Mirrors test_batch_norm2d_backward_gradient_beta but with training=False.
    In inference mode, beta gradient is independent of running statistics.
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
    running_mean._data.bitcast[Float32]()[0] = 0.3
    running_mean._data.bitcast[Float32]()[1] = 0.7

    var running_var = ones(param_shape, DType.float32)
    running_var._data.bitcast[Float32]()[0] = 0.5
    running_var._data.bitcast[Float32]()[1] = 1.5

    # Forward pass in inference mode
    var result_fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=False, epsilon=1e-5
    )
    var output = result_fwd[0]

    # Non-uniform grad_output
    var grad_output = zeros_like(output)
    for i in range(16):
        grad_output._data.bitcast[Float32]()[i] = Float32(i + 1) * 0.1

    # Analytical gradient in inference mode
    var result_bwd = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=False,
        epsilon=1e-5,
    )
    var grad_beta_analytical = result_bwd[2]

    # Numerical gradient: perturb beta in inference mode
    fn forward_for_beta_infer(b: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            x, gamma, b, running_mean, running_var, training=False, epsilon=1e-5
        )
        var out = res[0]
        var weighted = multiply(out, grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result

    var numerical_grad_beta = compute_numerical_gradient(
        forward_for_beta_infer, beta, epsilon=1e-3
    )

    assert_gradients_close(
        grad_beta_analytical,
        numerical_grad_beta,
        rtol=1e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. beta (inference mode)",
    )
    print("✓ Batch norm backward gradient (beta) inference mode validated numerically")


fn main() raises:
    """Run layer normalization and batch normalization backward tests."""
    print("Running normalization part3 tests...")

    # Layer normalization backward pass tests
    test_layer_norm_backward_shapes_2d()
    print("✓ test_layer_norm_backward_shapes_2d")

    test_layer_norm_backward_shapes_4d()
    print("✓ test_layer_norm_backward_shapes_4d")

    test_layer_norm_backward_grad_beta()
    print("✓ test_layer_norm_backward_grad_beta")

    test_layer_norm_backward_zero_input()
    print("✓ test_layer_norm_backward_zero_input")

    test_layer_norm_backward_gradient_input()
    print("✓ test_layer_norm_backward_gradient_input")

    test_layer_norm_backward_gradient_input_4d()
    print("✓ test_layer_norm_backward_gradient_input_4d")

    # Batch norm backward tests
    test_batch_norm2d_backward_gamma_beta_nonzero()
    test_batch_norm2d_backward_inference_mode()

    # Batch norm backward gradient checks in inference mode (#3809)
    test_batch_norm2d_backward_gradient_gamma_inference_mode()
    test_batch_norm2d_backward_gradient_beta_inference_mode()

    print("\nAll normalization part3 tests passed!")
