# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operations - Part 4: Trigonometric Functions.

Tests cover:
- Mathematical functions: sin, cos
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
from shared.core.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
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
# Trigonometric Tests
# ============================================================================


fn test_sin_values() raises:
    """Test that sin computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = Float32(pi / 2.0)
    x._data.bitcast[Float32]()[2] = Float32(pi)

    var result = sin(x)

    # sin(0) = 0, sin(π/2) = 1, sin(π) = 0
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )


fn test_cos_values() raises:
    """Test that cos computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = Float32(pi / 2.0)
    x._data.bitcast[Float32]()[2] = Float32(pi)

    var result = cos(x)

    # cos(0) = 1, cos(π/2) = 0, cos(π) = -1
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(-1.0), tolerance=1e-5
    )


fn test_sin_backward() raises:
    """Test sin backward pass."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = Float32(pi / 2.0)
    x._data.bitcast[Float32]()[2] = Float32(pi)

    var grad_input = sin_backward(grad_output, x)

    # d/dx[sin(x)] = cos(x)
    # cos(0) = 1, cos(π/2) = 0, cos(π) = -1
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(-1.0), tolerance=1e-5
    )


fn test_sin_backward_gradient() raises:
    """Test sin backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return sin(inp)

    var y = sin(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return sin_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


fn test_cos_backward() raises:
    """Test cos backward pass."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = Float32(pi / 2.0)
    x._data.bitcast[Float32]()[2] = Float32(pi)

    var grad_input = cos_backward(grad_output, x)

    # d/dx[cos(x)] = -sin(x)
    # -sin(0) = 0, -sin(π/2) = -1, -sin(π) = 0
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )


fn test_cos_backward_gradient() raises:
    """Test cos backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return cos(inp)

    var y = cos(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return cos_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run trigonometric elementwise operation tests."""
    print("Running elementwise operation tests (part 4: sin, cos)...")

    # Trigonometric tests
    test_sin_values()
    print("✓ test_sin_values")

    test_cos_values()
    print("✓ test_cos_values")

    test_sin_backward()
    print("✓ test_sin_backward")

    test_sin_backward_gradient()
    print("✓ test_sin_backward_gradient")

    test_cos_backward()
    print("✓ test_cos_backward")

    test_cos_backward_gradient()
    print("✓ test_cos_backward_gradient")

    print("\nAll elementwise operation tests (part 4) passed!")
