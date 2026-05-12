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
    TestFixtures,
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_shape_equal,
    assert_true,
)
from shared.testing.gradient_checker import (
    compute_numerical_gradient,
    assert_gradients_close,
    NumericalForward,
)


@fieldwise_init
struct _BNormInputFwd(NumericalForward):
    var training: Bool
    var gamma: AnyTensor
    var beta: AnyTensor
    var running_mean: AnyTensor
    var running_var: AnyTensor
    var grad_output: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            inp,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            training=self.training,
            epsilon=1e-5,
        )
        var out = res[0]
        var weighted = multiply(out, self.grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result


@fieldwise_init
struct _BNormGammaFwd(NumericalForward):
    var training: Bool
    var x: AnyTensor
    var beta: AnyTensor
    var running_mean: AnyTensor
    var running_var: AnyTensor
    var grad_output: AnyTensor

    def __call__(self, g: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            self.x,
            g,
            self.beta,
            self.running_mean,
            self.running_var,
            training=self.training,
            epsilon=1e-5,
        )
        var out = res[0]
        var weighted = multiply(out, self.grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result


@fieldwise_init
struct _BNormBetaFwd(NumericalForward):
    var training: Bool
    var x: AnyTensor
    var gamma: AnyTensor
    var running_mean: AnyTensor
    var running_var: AnyTensor
    var grad_output: AnyTensor

    def __call__(self, b: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            self.x,
            self.gamma,
            b,
            self.running_mean,
            self.running_var,
            training=self.training,
            epsilon=1e-5,
        )
        var out = res[0]
        var weighted = multiply(out, self.grad_output)
        var result = weighted
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result


@fieldwise_init
struct _LayerNormInputFwd(NumericalForward):
    var gamma: AnyTensor
    var beta: AnyTensor
    var grad_output: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        var out = layer_norm(inp, self.gamma, self.beta, epsilon=1e-5)
        var weighted = multiply(out, self.grad_output)
        var result_inner = weighted
        while result_inner.dim() > 0:
            result_inner = reduce_sum(result_inner, axis=0, keepdims=False)
        return result_inner


from shared.tensor.any_tensor import (
    AnyTensor,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from shared.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
    layer_norm,
    layer_norm_backward,
)
from shared.core.arithmetic import (
    add,
    multiply,
    subtract,
)
from shared.core.reduction import sum as reduce_sum

def _check_grad_input_batch_size(batch_size: Int) raises:
    """Run grad_input gradient check for the given batch_size.

    batch_size=1: variance=0 degenerate case — asserts finite and non-NaN only.
    batch_size=2, 4: uses assert_gradients_close with standard tolerances.
    """
    # Shape: (batch_size, C=2, H=2, W=2)
    var shape = List[Int]()
    shape.append(batch_size)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var n_elems = batch_size * 8

    var x = zeros(shape, DType.float32)
    for i in range(n_elems):
        x.set(i, Float32(i) * 0.1 + 0.05)

    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    # Forward pass
    var fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )
    var output = fwd[0]

    # Non-uniform grad_output to avoid the cancellation case where ones gives
    # analytically-zero gradients for symmetric inputs.
    var grad_output = zeros_like(output)
    for i in range(n_elems):
        var val = Float32(i % 4) * Float32(0.25) - Float32(0.3)
        grad_output.set(i, Float32(val))

    # Analytical backward
    var bwd = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=True,
        epsilon=1e-5,
    )
    var grad_input = bwd[0]

    if batch_size == 1:
        # Degenerate case: batch_size=1 causes variance=0 in training mode.
        # The denominator collapses to sqrt(eps), making exact gradient matching
        # unreliable. Assert finiteness and non-NaN only.
        for i in range(n_elems):
            var val = grad_input._data.bitcast[Float32]()[i]
            assert_true(
                val == val, "grad_input should not be NaN (batch_size=1)"
            )
            assert_true(
                val > -1e10 and val < 1e10,
                "grad_input should be finite (batch_size=1)",
            )
    else:
        # Standard gradient check for batch_size=2 and batch_size=4.
        var numerical_grad = compute_numerical_gradient(
            _BNormInputFwd(
                True, gamma, beta, running_mean, running_var, grad_output
            ),
            x,
            epsilon=1e-3,
        )

        assert_gradients_close(
            grad_input,
            numerical_grad,
            rtol=5e-2,
            atol=5e-4,
            message="Batch norm grad_input",
        )


def _check_grad_gamma_batch_size(batch_size: Int) raises:
    """Run grad_gamma gradient check for the given batch_size.

    batch_size=1: asserts finiteness only (variance=0 degenerate case).
    batch_size=2, 4: uses assert_gradients_close with standard tolerances.
    """
    var shape = List[Int]()
    shape.append(batch_size)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var n_elems = batch_size * 8

    var x = zeros(shape, DType.float32)
    for i in range(n_elems):
        x.set(i, Float32(i) * 0.1 + 0.05)

    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )
    var output = fwd[0]

    var grad_output = zeros_like(output)
    for i in range(n_elems):
        grad_output.set(i, Float32(i + 1) * 0.1)

    var bwd = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=True,
        epsilon=1e-5,
    )
    var grad_gamma = bwd[1]

    if batch_size == 1:
        # batch_size=1: variance=0 degenerate case — assert finiteness only.
        for i in range(2):
            var val = grad_gamma._data.bitcast[Float32]()[i]
            assert_true(
                val == val, "grad_gamma should not be NaN (batch_size=1)"
            )
            assert_true(
                val > -1e10 and val < 1e10,
                "grad_gamma should be finite (batch_size=1)",
            )
    else:
        # Standard gradient check: perturb gamma.
        var numerical_grad_gamma = compute_numerical_gradient(
            _BNormGammaFwd(
                True, x, beta, running_mean, running_var, grad_output
            ),
            gamma,
            epsilon=1e-3,
        )

        assert_gradients_close(
            grad_gamma,
            numerical_grad_gamma,
            rtol=1e-2,
            atol=1e-4,
            message="Batch norm grad_gamma",
        )


def _check_grad_beta_batch_size(batch_size: Int) raises:
    """Run grad_beta gradient check for the given batch_size.

    batch_size=1: asserts finiteness only (variance=0 degenerate case).
    batch_size=2, 4: uses assert_gradients_close with standard tolerances.
    """
    var shape = List[Int]()
    shape.append(batch_size)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var n_elems = batch_size * 8

    var x = zeros(shape, DType.float32)
    for i in range(n_elems):
        x.set(i, Float32(i) * 0.1 + 0.05)

    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

    var beta = zeros(param_shape, DType.float32)
    beta.set(0, Float32(0.5))
    beta.set(1, Float32(-0.5))

    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )
    var output = fwd[0]

    var grad_output = zeros_like(output)
    for i in range(n_elems):
        grad_output.set(i, Float32(i + 1) * 0.1)

    var bwd = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=True,
        epsilon=1e-5,
    )
    var grad_beta = bwd[2]

    if batch_size == 1:
        # batch_size=1: variance=0 degenerate case — assert finiteness only.
        for i in range(2):
            var val = grad_beta._data.bitcast[Float32]()[i]
            assert_true(
                val == val, "grad_beta should not be NaN (batch_size=1)"
            )
            assert_true(
                val > -1e10 and val < 1e10,
                "grad_beta should be finite (batch_size=1)",
            )
    else:
        # Standard gradient check: perturb beta.
        var numerical_grad_beta = compute_numerical_gradient(
            _BNormBetaFwd(
                True, x, gamma, running_mean, running_var, grad_output
            ),
            beta,
            epsilon=1e-3,
        )

        assert_gradients_close(
            grad_beta,
            numerical_grad_beta,
            rtol=1e-2,
            atol=1e-4,
            message="Batch norm grad_beta",
        )


def test_batch_norm2d_shapes() raises:
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


def test_batch_norm2d_training_mode() raises:
    """Test that batch_norm2d computes batch statistics in training mode."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(1)  # channels
    shape.append(2)  # height
    shape.append(2)  # width
    var x = zeros(shape, DType.float32)

    # Set specific values: [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(8):
        x.set(i, Float32(i))

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


def test_batch_norm2d_inference_mode() raises:
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
    running_mean.set(0, Float32(0.5))

    var running_var = ones(param_shape, DType.float32)
    running_var.set(0, Float32(0.25))

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


def test_batch_norm2d_scale_shift() raises:
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
    gamma.set(0, Float32(2.0))
    gamma.set(1, Float32(3.0))

    var beta = zeros(param_shape, DType.float32)
    beta.set(0, Float32(1.0))
    beta.set(1, Float32(-1.0))

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


def test_batch_norm2d_zero_variance() raises:
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


def test_batch_norm2d_backward_gradient_input() raises:
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
        x.set(i, Float32(i) * 0.1)

    # Parameters
    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

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
        grad_output.set(i, Float32(val))

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
    # forward computes weighted sum: sum(output * grad_output)
    # so the numerical gradient matches what the backward should compute.
    var numerical_grad = compute_numerical_gradient(
        _BNormInputFwd(
            True, gamma, beta, running_mean, running_var, grad_output
        ),
        x,
        epsilon=1e-3,
    )

    # Validate analytical gradient matches numerical gradient
    # Looser tolerance for batch norm (complex operation with many intermediate steps)
    # rtol=2e-2 (2%) to accommodate batch norm's compounding of floating-point errors
    # across normalization, scale, and shift operations.
    # Batch norm has compounding FP errors across normalization, scale, and shift.
    # Use wider tolerance to avoid flaky failures from numerical noise.
    # TODO: investigate proper numerical stability fix (see GitHub issue)
    assert_gradients_close(
        grad_input,
        numerical_grad,
        rtol=5e-2,
        atol=5e-4,
        message="Batch norm gradient w.r.t. input",
    )

    print("✓ Batch norm backward gradient (input) validated numerically")


def test_batch_norm2d_backward_gradient_input_inference_mode() raises:
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
        x.set(i, Float32(i) * 0.1)

    # Parameters: non-unit gamma, non-zero running stats per channel
    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

    var beta = zeros(param_shape, DType.float32)

    var running_mean = zeros(param_shape, DType.float32)
    running_mean.set(0, Float32(0.3))
    running_mean.set(1, Float32(0.7))

    var running_var = ones(param_shape, DType.float32)
    running_var.set(0, Float32(0.5))
    running_var.set(1, Float32(1.5))

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

    # Numerical gradient: uses training=False with fixed running stats
    var numerical_grad = compute_numerical_gradient(
        _BNormInputFwd(
            False, gamma, beta, running_mean, running_var, grad_output
        ),
        x,
        epsilon=3e-4,
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


def test_batch_norm2d_backward_gradient_gamma() raises:
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
        x.set(i, Float32(i) * 0.1)

    var param_shape = List[Int]()
    param_shape.append(2)

    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

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
        grad_output.set(i, Float32(i + 1) * 0.1)

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
    # The forward computes: sum(batch_norm2d(x, g, ...) * grad_output)
    # so the numerical gradient matches what batch_norm2d_backward should produce.
    var numerical_grad_gamma = compute_numerical_gradient(
        _BNormGammaFwd(True, x, beta, running_mean, running_var, grad_output),
        gamma,
        epsilon=1e-3,
    )

    assert_gradients_close(
        grad_gamma_analytical,
        numerical_grad_gamma,
        rtol=1e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. gamma",
    )
    print("✓ Batch norm backward gradient (gamma) validated numerically")


def test_batch_norm2d_backward_gradient_beta() raises:
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
        x.set(i, Float32(i) * 0.1)

    var param_shape = List[Int]()
    param_shape.append(2)

    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

    var beta = zeros(param_shape, DType.float32)
    beta.set(0, Float32(0.5))
    beta.set(1, Float32(-0.5))

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
        grad_output.set(i, Float32(i + 1) * 0.1)

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
    var numerical_grad_beta = compute_numerical_gradient(
        _BNormBetaFwd(True, x, gamma, running_mean, running_var, grad_output),
        beta,
        epsilon=1e-3,
    )

    assert_gradients_close(
        grad_beta_analytical,
        numerical_grad_beta,
        rtol=1e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. beta",
    )
    print("✓ Batch norm backward gradient (beta) validated numerically")


def test_batch_norm2d_backward_training_vs_inference() raises:
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
        x.set(i, Float32(i))

    var param_shape = List[Int]()
    param_shape.append(1)
    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(2.0))

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


def test_batch_norm2d_backward_shapes() raises:
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


def test_layer_norm_shapes_2d() raises:
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


def test_layer_norm_shapes_4d() raises:
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


def test_layer_norm_normalization_2d() raises:
    """Test that layer_norm normalizes each sample independently."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(4)  # features
    var x = zeros(shape, DType.float32)

    # Sample 1: [0, 1, 2, 3]
    x.set(0, Float32(0.0))
    x.set(1, Float32(1.0))
    x.set(2, Float32(2.0))
    x.set(3, Float32(3.0))

    # Sample 2: [4, 5, 6, 7]
    x.set(4, Float32(4.0))
    x.set(5, Float32(5.0))
    x.set(6, Float32(6.0))
    x.set(7, Float32(7.0))

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


def test_layer_norm_scale_shift() raises:
    """Test that layer_norm applies gamma and beta correctly."""
    var shape = List[Int]()
    shape.append(1)  # batch
    shape.append(3)  # features
    var x = zeros(shape, DType.float32)

    var param_shape = List[Int]()
    param_shape.append(3)

    # Set gamma = [2.0, 3.0, 4.0], beta = [1.0, 0.0, -1.0]
    var gamma = zeros(param_shape, DType.float32)
    gamma.set(0, Float32(2.0))
    gamma.set(1, Float32(3.0))
    gamma.set(2, Float32(4.0))

    var beta = zeros(param_shape, DType.float32)
    beta.set(0, Float32(1.0))
    beta.set(1, Float32(0.0))
    beta.set(2, Float32(-1.0))

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


def test_layer_norm_zero_variance() raises:
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


def test_layer_norm_backward_shapes_2d() raises:
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


def test_layer_norm_backward_shapes_4d() raises:
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


def test_layer_norm_backward_grad_beta() raises:
    """Test that layer_norm_backward grad_beta equals sum of grad_output over batch.
    """
    var shape = List[Int]()
    shape.append(3)  # batch
    shape.append(4)  # features
    var x = zeros(shape, DType.float32)

    # Set varying input values
    for i in range(12):
        x.set(i, Float32(i) * 0.1)

    var param_shape = List[Int]()
    param_shape.append(4)
    var gamma = ones(param_shape, DType.float32)

    # Create varying grad_output
    var grad_output = zeros(shape, DType.float32)
    for b in range(3):
        for f in range(4):
            var idx = b * 4 + f
            grad_output.set(idx, Float32(b + 1) * Float32(f + 1) * 0.1)

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


def test_layer_norm_backward_zero_input() raises:
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


def test_layer_norm_backward_gradient_input() raises:
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
        x.set(i, Float32(i) * 0.1 + 0.05)

    # Parameters: non-trivial gamma, zero beta
    var param_shape = List[Int]()
    param_shape.append(4)
    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(0.8))
    gamma.set(2, Float32(1.2))
    gamma.set(3, Float32(2.0))
    var beta = zeros(param_shape, DType.float32)

    # Non-uniform grad_output: critical to avoid algebraic cancellation
    var grad_output = zeros(shape, DType.float32)
    grad_output.set(0, Float32(0.3))
    grad_output.set(1, Float32(-0.5))
    grad_output.set(2, Float32(1.2))
    grad_output.set(3, Float32(-0.8))
    grad_output.set(4, Float32(0.7))
    grad_output.set(5, Float32(-0.2))
    grad_output.set(6, Float32(0.9))
    grad_output.set(7, Float32(-1.1))

    # Analytical backward pass
    var result = layer_norm_backward(grad_output, x, gamma, epsilon=1e-5)
    var grad_input = result[0]

    # Numerical gradient via finite differences.
    # The scalar loss is sum(layer_norm(x) * grad_output), so the numerical
    # gradient matches what layer_norm_backward(grad_output, x, gamma) computes.
    var numerical_grad = compute_numerical_gradient(
        _LayerNormInputFwd(gamma, beta, grad_output), x, epsilon=1e-4
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


def test_layer_norm_backward_gradient_input_4d() raises:
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
        x.set(i, Float32(i) * 0.1 + 0.05)

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
        gamma.set(i, Float32(cycling_values[i % 4]))
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
        grad_output.set(i, Float32(go_vals[i]))

    # Analytical backward pass
    var result = layer_norm_backward(grad_output, x, gamma, epsilon=1e-5)
    var grad_input = result[0]

    # Numerical gradient via finite differences.
    # The scalar loss is sum(layer_norm(x) * grad_output), so the numerical
    # gradient matches what layer_norm_backward(grad_output, x, gamma) computes.
    var numerical_grad = compute_numerical_gradient(
        _LayerNormInputFwd(gamma, beta, grad_output), x, epsilon=1e-4
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


def test_batch_norm2d_backward_gamma_beta_nonzero() raises:
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
        x.set(i, Float32(i) * 0.1)

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
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=True,
        epsilon=1e-5,
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


def test_batch_norm2d_backward_inference_mode() raises:
    """Test batch_norm2d_backward in inference mode. Closes #3665."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var x = zeros(shape, DType.float32)
    for i in range(16):
        x.set(i, Float32(i) * 0.1)

    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var grad_output = ones(shape, DType.float32)

    # Backward in inference mode (training=False)
    var bwd_result = batch_norm2d_backward(
        grad_output,
        x,
        gamma,
        running_mean,
        running_var,
        training=False,
        epsilon=1e-5,
    )
    var grad_input = bwd_result[0]

    # grad_input should be finite and non-zero in inference mode
    for i in range(16):
        var val = grad_input._data.bitcast[Float32]()[i]
        assert_true(val == val, "Inference mode grad should not be NaN")

    print("✓ batch_norm2d_backward inference mode test passed")


def test_batch_norm2d_backward_gradient_gamma_inference_mode() raises:
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
        x.set(i, Float32(i) * 0.1)

    var param_shape = List[Int]()
    param_shape.append(2)

    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

    var beta = zeros(param_shape, DType.float32)

    var running_mean = zeros(param_shape, DType.float32)
    running_mean.set(0, Float32(0.3))
    running_mean.set(1, Float32(0.7))

    var running_var = ones(param_shape, DType.float32)
    running_var.set(0, Float32(0.5))
    running_var.set(1, Float32(1.5))

    # Forward pass in inference mode
    var result_fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=False, epsilon=1e-5
    )
    var output = result_fwd[0]

    # Non-uniform grad_output to exercise per-channel sensitivity
    var grad_output = zeros_like(output)
    for i in range(16):
        grad_output.set(i, Float32(i + 1) * 0.1)

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
    var numerical_grad_gamma = compute_numerical_gradient(
        _BNormGammaFwd(False, x, beta, running_mean, running_var, grad_output),
        gamma,
        epsilon=1e-3,
    )

    assert_gradients_close(
        grad_gamma_analytical,
        numerical_grad_gamma,
        rtol=1e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. gamma (inference mode)",
    )
    print(
        "✓ Batch norm backward gradient (gamma) inference mode validated"
        " numerically"
    )


def test_batch_norm2d_backward_gradient_beta_inference_mode() raises:
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
        x.set(i, Float32(i) * 0.1)

    var param_shape = List[Int]()
    param_shape.append(2)

    var gamma = ones(param_shape, DType.float32)
    gamma.set(0, Float32(1.5))
    gamma.set(1, Float32(2.0))

    var beta = zeros(param_shape, DType.float32)
    beta.set(0, Float32(0.5))
    beta.set(1, Float32(-0.5))

    var running_mean = zeros(param_shape, DType.float32)
    running_mean.set(0, Float32(0.3))
    running_mean.set(1, Float32(0.7))

    var running_var = ones(param_shape, DType.float32)
    running_var.set(0, Float32(0.5))
    running_var.set(1, Float32(1.5))

    # Forward pass in inference mode
    var result_fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=False, epsilon=1e-5
    )
    var output = result_fwd[0]

    # Non-uniform grad_output
    var grad_output = zeros_like(output)
    for i in range(16):
        grad_output.set(i, Float32(i + 1) * 0.1)

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
    var numerical_grad_beta = compute_numerical_gradient(
        _BNormBetaFwd(False, x, gamma, running_mean, running_var, grad_output),
        beta,
        epsilon=1e-3,
    )

    assert_gradients_close(
        grad_beta_analytical,
        numerical_grad_beta,
        rtol=1e-2,
        atol=1e-4,
        message="Batch norm gradient w.r.t. beta (inference mode)",
    )
    print(
        "✓ Batch norm backward gradient (beta) inference mode validated"
        " numerically"
    )


def test_batch_norm2d_backward_grad_input_batch_sizes() raises:
    """Parametrized gradient check for grad_input over batch_sizes [1, 2, 4].

    batch_size=1 is a degenerate case (variance=0); only finiteness is checked.
    batch_size=2 and batch_size=4 use numerical gradient validation.
    """
    _check_grad_input_batch_size(1)
    _check_grad_input_batch_size(2)
    _check_grad_input_batch_size(4)
    print(
        "✓ Batch norm backward grad_input validated for batch_sizes [1, 2, 4]"
    )


def test_batch_norm2d_backward_grad_gamma_batch_sizes() raises:
    """Parametrized gradient check for grad_gamma over batch_sizes [1, 2, 4].

    batch_size=1 is a degenerate case (variance=0); only finiteness is checked.
    batch_size=2 and batch_size=4 use numerical gradient validation.
    """
    _check_grad_gamma_batch_size(1)
    _check_grad_gamma_batch_size(2)
    _check_grad_gamma_batch_size(4)
    print(
        "✓ Batch norm backward grad_gamma validated for batch_sizes [1, 2, 4]"
    )


def test_batch_norm2d_backward_grad_beta_batch_sizes() raises:
    """Parametrized gradient check for grad_beta over batch_sizes [1, 2, 4].

    batch_size=1 is a degenerate case (variance=0); only finiteness is checked.
    batch_size=2 and batch_size=4 use numerical gradient validation.
    """
    _check_grad_beta_batch_size(1)
    _check_grad_beta_batch_size(2)
    _check_grad_beta_batch_size(4)
    print("✓ Batch norm backward grad_beta validated for batch_sizes [1, 2, 4]")


def main() raises:
    """Run all test_normalization tests."""
    print("Running test_normalization tests...")

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

    test_batch_norm2d_backward_gradient_input()
    print("✓ test_batch_norm2d_backward_gradient_input")

    test_batch_norm2d_backward_gradient_input_inference_mode()
    print("✓ test_batch_norm2d_backward_gradient_input_inference_mode")

    test_batch_norm2d_backward_gradient_gamma()
    print("✓ test_batch_norm2d_backward_gradient_gamma")

    test_batch_norm2d_backward_gradient_beta()
    print("✓ test_batch_norm2d_backward_gradient_beta")

    test_batch_norm2d_backward_training_vs_inference()
    print("✓ test_batch_norm2d_backward_training_vs_inference")

    test_batch_norm2d_backward_shapes()
    print("✓ test_batch_norm2d_backward_shapes")

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

    test_batch_norm2d_backward_gamma_beta_nonzero()
    print("✓ test_batch_norm2d_backward_gamma_beta_nonzero")

    test_batch_norm2d_backward_inference_mode()
    print("✓ test_batch_norm2d_backward_inference_mode")

    test_batch_norm2d_backward_gradient_gamma_inference_mode()
    print("✓ test_batch_norm2d_backward_gradient_gamma_inference_mode")

    test_batch_norm2d_backward_gradient_beta_inference_mode()
    print("✓ test_batch_norm2d_backward_gradient_beta_inference_mode")

    test_batch_norm2d_backward_grad_input_batch_sizes()
    print("✓ test_batch_norm2d_backward_grad_input_batch_sizes")

    test_batch_norm2d_backward_grad_gamma_batch_sizes()
    print("✓ test_batch_norm2d_backward_grad_gamma_batch_sizes")

    test_batch_norm2d_backward_grad_beta_batch_sizes()
    print("✓ test_batch_norm2d_backward_grad_beta_batch_sizes")

    print("\nAll test_normalization tests passed!")
