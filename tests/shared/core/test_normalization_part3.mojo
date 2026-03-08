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
    fn forward_for_grad(inp: ExTensor) raises -> ExTensor:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run layer normalization backward tests."""
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

    print("\nAll normalization part3 tests passed!")
