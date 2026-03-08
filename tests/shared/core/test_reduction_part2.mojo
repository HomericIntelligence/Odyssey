# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for variance and standard deviation reduction operations with gradient checking.

Tests cover:
- Variance reduction along axes (population and sample)
- Standard deviation reduction along axes
- Numerical gradient checking using finite differences

Gradient checking formula:
    numerical_grad ≈ (f(x + ε) - f(x - ε)) / (2ε)

All tests validate backward passes produce correct gradient values.
"""

from tests.shared.conftest import (
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
from shared.core.reduction import (
    variance,
    std,
    variance_backward,
    std_backward,
)
from shared.testing import check_gradient


# ============================================================================
# Variance Reduction Tests
# ============================================================================


fn test_var_forward_uniform() raises:
    """Test variance of uniform values (should be 0)."""
    var shape = List[Int]()
    shape.append(5)
    var x = ones(shape, DType.float32)

    var result = variance(x, axis=-1)
    assert_close_float(result._get_float64(0), 0.0, rtol=1e-5, atol=1e-7)


fn test_var_forward_simple() raises:
    """Test variance with known result."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0

    # Mean = 2.0, var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    var result = variance(x, axis=-1, ddof=0)
    assert_close_float(result._get_float64(0), 2.0 / 3.0, rtol=1e-5, atol=1e-7)


fn test_var_forward_with_ddof() raises:
    """Test sample variance with ddof=1."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0

    # Sample variance with ddof=1: var = 2 / 2 = 1.0
    var result = variance(x, axis=-1, ddof=1)
    assert_close_float(result._get_float64(0), 1.0, rtol=1e-5, atol=1e-7)


fn test_var_forward_axis() raises:
    """Test variance along specific axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0
    x._data.bitcast[Float32]()[3] = 4.0
    x._data.bitcast[Float32]()[4] = 5.0
    x._data.bitcast[Float32]()[5] = 6.0

    var result = variance(x, axis=1, ddof=0)
    var result_shape = result.shape()
    assert_equal(result_shape[0], 2)


fn test_var_backward_shapes() raises:
    """Test that var_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    for i in range(6):
        x._data.bitcast[Float32]()[i] = Float32(i) + 1.0

    var result = variance(x, axis=1, ddof=0)
    var grad_output = ones_like(result)
    var grad_input = variance_backward(grad_output, x, axis=1, ddof=0)

    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)


fn test_var_backward_gradient() raises:
    """Test var_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = -0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = -0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return variance(inp, axis=1, ddof=0)

    var y = forward(x)
    var grad_out = ones_like(y)

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return variance_backward(grad, inp, axis=1, ddof=0)

    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


# ============================================================================
# Standard Deviation Tests
# ============================================================================


fn test_std_forward_simple() raises:
    """Test standard deviation with known result."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 3.0

    # std = sqrt(var) = sqrt(2/3)
    var result = std(x, axis=-1, ddof=0)
    var expected = (2.0 / 3.0) ** 0.5
    assert_close_float(result._get_float64(0), expected, rtol=1e-5, atol=1e-7)


fn test_std_backward_gradient() raises:
    """Test std_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = zeros(shape, DType.float32)
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = 0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = 0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return std(inp, axis=1, ddof=0)

    var y = forward(x)
    var grad_out = ones_like(y)

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return std_backward(grad, inp, axis=1, ddof=0)

    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run variance and standard deviation reduction tests."""
    print("Running reduction part 2 tests (variance, std)...")

    # Variance tests
    test_var_forward_uniform()
    print("✓ test_var_forward_uniform")

    test_var_forward_simple()
    print("✓ test_var_forward_simple")

    test_var_forward_with_ddof()
    print("✓ test_var_forward_with_ddof")

    test_var_forward_axis()
    print("✓ test_var_forward_axis")

    test_var_backward_shapes()
    print("✓ test_var_backward_shapes")

    test_var_backward_gradient()
    print("✓ test_var_backward_gradient")

    # Standard deviation tests
    test_std_forward_simple()
    print("✓ test_std_forward_simple")

    test_std_backward_gradient()
    print("✓ test_std_backward_gradient")

    print("\nAll reduction part 2 tests passed!")
