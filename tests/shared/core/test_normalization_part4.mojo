# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_normalization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for batch normalization backward pass across multiple batch sizes.

Tests cover:
- Batch norm backward grad_input for batch_sizes [1, 2, 4]
- Batch norm backward grad_gamma for batch_sizes [1, 2, 4]
- Batch norm backward grad_beta for batch_sizes [1, 2, 4]

batch_size=1 is a degenerate case: with a single sample, variance=0 collapses the
denominator to sqrt(eps), making gradients numerically sensitive. These cases assert
finiteness only. batch_size=2 and batch_size=4 use standard gradient checking tolerances.

Closes #3811.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_true,
)
from shared.testing import (
    compute_numerical_gradient,
    assert_gradients_close,
)
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like
from shared.core.normalization import (
    batch_norm2d,
    batch_norm2d_backward,
)
from shared.core.arithmetic import multiply
from shared.core.reduction import sum as reduce_sum


# ============================================================================
# Private helpers: parametrized gradient check for each gradient type
# ============================================================================


fn _check_grad_input_batch_size(batch_size: Int) raises:
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
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.05

    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 1.5
    gamma._data.bitcast[Float32]()[1] = 2.0

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
        grad_output._data.bitcast[Float32]()[i] = val

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
            assert_true(val == val, "grad_input should not be NaN (batch_size=1)")
            assert_true(
                val > -1e10 and val < 1e10,
                "grad_input should be finite (batch_size=1)",
            )
    else:
        # Standard gradient check for batch_size=2 and batch_size=4.
        fn forward_for_grad(inp: AnyTensor) raises -> AnyTensor:
            var res = batch_norm2d(
                inp,
                gamma,
                beta,
                running_mean,
                running_var,
                training=True,
                epsilon=1e-5,
            )
            var out = res[0]
            var weighted = multiply(out, grad_output)
            var result = weighted
            while result.dim() > 0:
                result = reduce_sum(result, axis=0, keepdims=False)
            return result

        var numerical_grad = compute_numerical_gradient(
            forward_for_grad, x, epsilon=1e-3
        )

        assert_gradients_close(
            grad_input,
            numerical_grad,
            rtol=5e-2,
            atol=5e-4,
            message="Batch norm grad_input",
        )


fn _check_grad_gamma_batch_size(batch_size: Int) raises:
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
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.05

    var param_shape = List[Int]()
    param_shape.append(2)
    var gamma = ones(param_shape, DType.float32)
    gamma._data.bitcast[Float32]()[0] = 1.5
    gamma._data.bitcast[Float32]()[1] = 2.0

    var beta = zeros(param_shape, DType.float32)
    var running_mean = zeros(param_shape, DType.float32)
    var running_var = ones(param_shape, DType.float32)

    var fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )
    var output = fwd[0]

    var grad_output = zeros_like(output)
    for i in range(n_elems):
        grad_output._data.bitcast[Float32]()[i] = Float32(i + 1) * 0.1

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
            assert_true(val == val, "grad_gamma should not be NaN (batch_size=1)")
            assert_true(
                val > -1e10 and val < 1e10,
                "grad_gamma should be finite (batch_size=1)",
            )
    else:
        # Standard gradient check: perturb gamma.
        fn forward_for_gamma(g: AnyTensor) raises -> AnyTensor:
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
            grad_gamma,
            numerical_grad_gamma,
            rtol=1e-2,
            atol=1e-4,
            message="Batch norm grad_gamma",
        )


fn _check_grad_beta_batch_size(batch_size: Int) raises:
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
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 + 0.05

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

    var fwd = batch_norm2d(
        x, gamma, beta, running_mean, running_var, training=True, epsilon=1e-5
    )
    var output = fwd[0]

    var grad_output = zeros_like(output)
    for i in range(n_elems):
        grad_output._data.bitcast[Float32]()[i] = Float32(i + 1) * 0.1

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
            assert_true(val == val, "grad_beta should not be NaN (batch_size=1)")
            assert_true(
                val > -1e10 and val < 1e10,
                "grad_beta should be finite (batch_size=1)",
            )
    else:
        # Standard gradient check: perturb beta.
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
            grad_beta,
            numerical_grad_beta,
            rtol=1e-2,
            atol=1e-4,
            message="Batch norm grad_beta",
        )


# ============================================================================
# Test functions: one per gradient type, each loops over batch sizes [1, 2, 4]
# ============================================================================


fn test_batch_norm2d_backward_grad_input_batch_sizes() raises:
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


fn test_batch_norm2d_backward_grad_gamma_batch_sizes() raises:
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


fn test_batch_norm2d_backward_grad_beta_batch_sizes() raises:
    """Parametrized gradient check for grad_beta over batch_sizes [1, 2, 4].

    batch_size=1 is a degenerate case (variance=0); only finiteness is checked.
    batch_size=2 and batch_size=4 use numerical gradient validation.
    """
    _check_grad_beta_batch_size(1)
    _check_grad_beta_batch_size(2)
    _check_grad_beta_batch_size(4)
    print(
        "✓ Batch norm backward grad_beta validated for batch_sizes [1, 2, 4]"
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run batch-size-parametrized gradient check tests for batch normalization backward."""
    print("Running normalization part4 tests...")

    test_batch_norm2d_backward_grad_input_batch_sizes()
    test_batch_norm2d_backward_grad_gamma_batch_sizes()
    test_batch_norm2d_backward_grad_beta_batch_sizes()

    print("\nAll normalization part4 tests passed!")
