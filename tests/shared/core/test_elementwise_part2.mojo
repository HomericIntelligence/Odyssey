# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operations - Part 2: Exponential and Logarithm.

Tests cover:
- Mathematical functions: exp, log, log10, log2
- Backward passes for differentiable functions
- Numerical correctness and edge cases

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.elementwise import (
    abs,
    sign,
    exp,
    log,
    sqrt,
    sin,
    cos,
    clip,
    ceil,
    floor,
    round,
    trunc,
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    log10,
    log2,
    exp_backward,
    log_backward,
    sqrt_backward,
    abs_backward,
    clip_backward,
    log10_backward,
    log2_backward,
    sin_backward,
    cos_backward,
)
from shared.testing import check_gradient
from math import sqrt as math_sqrt, pi


# ============================================================================
# Exponential Tests
# ============================================================================


fn test_exp_shapes() raises:
    """Test that exp returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = exp(x)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_exp_values() raises:
    """Test that exp computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 2.0

    var result = exp(x)

    # exp(0) = 1, exp(1) ≈ 2.718, exp(2) ≈ 7.389
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(2.718), tolerance=0.01
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(7.389), tolerance=0.01
    )


fn test_exp_backward() raises:
    """Test exp backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 1.0

    var grad_input = exp_backward(grad_output, x)

    # d/dx[exp(x)] = exp(x)
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(2.718), tolerance=0.01
    )


fn test_exp_backward_gradient() raises:
    """Test exp backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return exp(inp)

    var y = exp(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return exp_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Logarithm Tests
# ============================================================================


fn test_log_shapes() raises:
    """Test that log returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = log(x)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_log_values() raises:
    """Test that log computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.718
    x._data.bitcast[Float32]()[2] = 7.389

    var result = log(x)

    # log(1) = 0, log(e) ≈ 1, log(e^2) ≈ 2
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=0.01
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=0.01
    )


fn test_log_backward() raises:
    """Test log backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0

    var grad_input = log_backward(grad_output, x)

    # d/dx[log(x)] = 1/x
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.5), tolerance=1e-5
    )


fn test_log_backward_gradient() raises:
    """Test log backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set positive non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 2.0

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return log(inp)

    var y = log(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return log_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run exp and log elementwise operation tests."""
    print("Running elementwise operation tests (part 2: exp, log)...")

    # Exponential tests
    test_exp_shapes()
    print("✓ test_exp_shapes")

    test_exp_values()
    print("✓ test_exp_values")

    test_exp_backward()
    print("✓ test_exp_backward")

    test_exp_backward_gradient()
    print("✓ test_exp_backward_gradient")

    # Logarithm tests
    test_log_shapes()
    print("✓ test_log_shapes")

    test_log_values()
    print("✓ test_log_values")

    test_log_backward()
    print("✓ test_log_backward")

    test_log_backward_gradient()
    print("✓ test_log_backward_gradient")

    print("\nAll elementwise operation tests (part 2) passed!")
