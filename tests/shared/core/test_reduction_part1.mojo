# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_reduction.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for sum and mean reduction operations with gradient checking.

Tests cover:
- Sum reduction along axes
- Mean reduction along axes
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
from shared.core.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.reduction import (
    sum,
    mean,
    max_reduce,
    min_reduce,
    sum_backward,
    mean_backward,
    max_reduce_backward,
    min_reduce_backward,
)
from shared.testing import check_gradient


# ============================================================================
# Sum Reduction Tests
# ============================================================================


fn test_sum_backward_shapes() raises:
    """Test that sum_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var x = ones(shape, DType.float32)

    # Reduce along axis 1
    var result = sum(x, axis=1)

    # Gradient matching output shape
    var grad_output = ones_like(result)

    # Backward pass
    var grad_input = sum_backward(grad_output, x, axis=1)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)
    assert_equal(gi_shape[2], 4)


fn test_sum_backward_gradient() raises:
    """Test sum_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = -0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = -0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    # Forward function wrapper (sum along axis 1)
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return sum(inp, axis=1)

    var y = forward(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return sum_backward(grad, inp, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=2e-3 accounts for Float32 precision in sum accumulation
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


# ============================================================================
# Mean Reduction Tests
# ============================================================================


fn test_mean_backward_shapes() raises:
    """Test that mean_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var x = ones(shape, DType.float32)

    # Reduce along axis 1
    var result = mean(x, axis=1)

    # Gradient matching output shape
    var grad_output = ones_like(result)

    # Backward pass
    var grad_input = mean_backward(grad_output, x, axis=1)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 2)
    assert_equal(gi_shape[1], 3)
    assert_equal(gi_shape[2], 4)


fn test_mean_backward_gradient() raises:
    """Test mean_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = -0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = -0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    # Forward function wrapper (mean along axis 1)
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return mean(inp, axis=1)

    var y = forward(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return mean_backward(grad, inp, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=2e-3 accounts for Float32 precision in division
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


# ============================================================================
# Max Reduction Tests
# ============================================================================


fn test_max_reduce_backward_shapes() raises:
    """Test that max_reduce_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var x = ones(shape, DType.float32)

    # Reduce along axis 1
    var result = max_reduce(x, axis=1)

    # Gradient matching output shape
    var grad_output = ones_like(result)

    # Backward pass
    var grad_input = max_reduce_backward(grad_output, x, axis=1)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 3)
    assert_equal(gi_shape[1], 4)


fn test_max_reduce_backward_gradient() raises:
    """Test max_reduce_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = -0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = -0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    # Forward function wrapper (max along axis 1)
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return max_reduce(inp, axis=1)

    var y = forward(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return max_reduce_backward(grad, inp, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=2e-3 accounts for Float32 precision
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


# ============================================================================
# Min Reduction Tests
# ============================================================================


fn test_min_reduce_backward_shapes() raises:
    """Test that min_reduce_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var x = ones(shape, DType.float32)

    # Reduce along axis 1
    var result = min_reduce(x, axis=1)

    # Gradient matching output shape
    var grad_output = ones_like(result)

    # Backward pass
    var grad_input = min_reduce_backward(grad_output, x, axis=1)

    # Check shape matches input
    var gi_shape = grad_input.shape()
    assert_equal(gi_shape[0], 3)
    assert_equal(gi_shape[1], 4)


fn test_min_reduce_backward_gradient() raises:
    """Test min_reduce_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = -0.3
    x._data.bitcast[Float32]()[2] = 1.2
    x._data.bitcast[Float32]()[3] = -0.8
    x._data.bitcast[Float32]()[4] = 0.1
    x._data.bitcast[Float32]()[5] = 0.7

    # Forward function wrapper (min along axis 1)
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return min_reduce(inp, axis=1)

    var y = forward(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return min_reduce_backward(grad, inp, axis=1)

    # Use numerical gradient checking (gold standard)
    # Note: rtol=2e-3 accounts for Float32 precision
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run sum, mean, max, and min reduction tests."""
    print("Running reduction part 1 tests (sum, mean, max, min)...")

    # Sum reduction tests
    test_sum_backward_shapes()
    print("✓ test_sum_backward_shapes")

    test_sum_backward_gradient()
    print("✓ test_sum_backward_gradient")

    # Mean reduction tests
    test_mean_backward_shapes()
    print("✓ test_mean_backward_shapes")

    test_mean_backward_gradient()
    print("✓ test_mean_backward_gradient")

    # Max reduction tests
    test_max_reduce_backward_shapes()
    print("✓ test_max_reduce_backward_shapes")

    test_max_reduce_backward_gradient()
    print("✓ test_max_reduce_backward_gradient")

    # Min reduction tests
    test_min_reduce_backward_shapes()
    print("✓ test_min_reduce_backward_shapes")

    test_min_reduce_backward_gradient()
    print("✓ test_min_reduce_backward_gradient")

    print("\nAll reduction part 1 tests passed!")
