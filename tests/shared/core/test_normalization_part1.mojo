# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_normalization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for batch normalization (forward pass and first backward tests).

Tests cover:
- Batch normalization shapes
- Batch normalization training mode
- Batch normalization inference mode
- Batch normalization scale/shift
- Batch normalization zero variance
- Batch normalization backward: gradient input (training)
- Batch normalization backward: gradient input (inference)
- Batch normalization backward: gradient gamma

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
        forward_for_grad, x, epsilon=1e-3
    )

    # Validate analytical gradient matches numerical gradient
    # Looser tolerance for batch norm (complex operation with many intermediate steps)
    # rtol=2e-2 (2%) to accommodate batch norm's compounding of floating-point errors
    # across normalization, scale, and shift operations.
    assert_gradients_close(
        grad_input,
        numerical_grad,
        rtol=2e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. input",
    )

    print("✓ Batch norm backward gradient (input) validated numerically")


fn test_batch_norm2d_backward_gradient_input_inference_mode() raises:
    """Numerical gradient validation for batch_norm2d_backward in inference mode.

    Inference mode uses fixed running_mean/running_var, giving a simpler
    linear gradient: grad_input = grad_output * gamma / sqrt(running_var + eps).
    Validates the inference code path in batch_norm2d_backward independently
    from training mode.
    """
    # Small tensor: batch=2, channels=2, height=2, width=2 (16 elements)
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(2)  # channels
    shape.append(2)  # height
    shape.append(2)  # width

    # Input with varying values
    var x = zeros(shape, DType.float32)
    for i in range(16):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1

    # Parameters: non-unit gamma, non-zero running stats per channel
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

    # Forward pass in inference mode (running stats frozen)
    var result_fwd = batch_norm2d(
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        training=False,
        epsilon=1e-5,
    )
    var output = result_fwd[0]

    # Use ones grad_output: in inference mode, grad_input = grad_output * gamma / std
    # which is non-zero for grad_output=ones (no cancellation issue unlike training mode)
    var grad_output = ones_like(output)

    # Analytical backward pass in inference mode
    var result_bwd = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=False,
        epsilon=1e-5,
    )
    var grad_input = result_bwd[0]

    # Numerical gradient: closure uses training=False with fixed running stats
    fn forward_for_grad_infer(inp: ExTensor) raises -> ExTensor:
        var res = batch_norm2d(
            inp,
            gamma,
            beta,
            running_mean,
            running_var,
            training=False,
            epsilon=1e-5,
        )
        var out = res[0]
        # Weighted sum matching our grad_output=ones backward
        var weighted = multiply(out, grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result

    var numerical_grad = compute_numerical_gradient(
        forward_for_grad_infer, x, epsilon=3e-4
    )

    # Inference mode gradient is a simple linear rescaling; tolerances can be tight
    assert_gradients_close(
        grad_input,
        numerical_grad,
        rtol=1e-2,
        atol=1e-5,
        message="Batch norm inference mode gradient w.r.t. input",
    )

    print(
        "✓ Batch norm backward gradient (input) inference mode validated"
        " numerically"
    )


fn test_batch_norm2d_backward_gradient_gamma() raises:
    """Test batch_norm2d_backward gradient w.r.t. gamma using numerical validation.

    Perturbs gamma using finite differences and compares against analytical grad_gamma.
    Uses non-uniform grad_output to test sensitivity across channels.
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
    var running_var = ones(param_shape, DType.float32)

    # Forward pass to get output shape for grad_output
    var result_fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )
    var output = result_fwd[0]

    # Non-uniform grad_output to exercise per-channel sensitivity
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
    var grad_gamma_analytical = result_bwd[1]

    # Numerical gradient: perturb gamma
    # The forward closure computes: sum(batch_norm2d(x, g, ...) * grad_output)
    # so the numerical gradient matches what batch_norm2d_backward should produce.
    fn forward_for_gamma(g: ExTensor) raises -> ExTensor:
        var res = batch_norm2d(
            x, g, beta, running_mean, running_var, training=True, epsilon=1e-5
        )
        var out = res[0]
        var weighted = multiply(out, grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result

    var numerical_grad_gamma = compute_numerical_gradient(
        forward_for_gamma, gamma, epsilon=1e-3
    )

    assert_gradients_close(
        grad_gamma_analytical,
        numerical_grad_gamma,
        rtol=1e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. gamma",
    )
    print("✓ Batch norm backward gradient (gamma) validated numerically")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run batch normalization forward and first backward tests."""
    print("Running normalization part1 tests...")

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

    test_batch_norm2d_backward_gradient_input_inference_mode()
    print("✓ test_batch_norm2d_backward_gradient_input_inference_mode")

    test_batch_norm2d_backward_gradient_gamma()
    print("✓ test_batch_norm2d_backward_gradient_gamma")

    print("\nAll normalization part1 tests passed!")
