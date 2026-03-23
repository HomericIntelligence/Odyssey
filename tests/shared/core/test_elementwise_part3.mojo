# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operations - Part 3: Log10, Log2, and Square Root.

Tests cover:
- Mathematical functions: log10, log2, sqrt
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


fn test_log10_values() raises:
    """Test that log10 computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 10.0
    x._data.bitcast[Float32]()[2] = 100.0

    var result = log10(x)

    # log10(1) = 0, log10(10) = 1, log10(100) = 2
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )


fn test_log10_backward_gradient() raises:
    """Test log10 backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set positive non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 2.0

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return log10(inp)

    var y = log10(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return log10_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=5e-3, atol=1e-5)


fn test_log2_values() raises:
    """Test that log2 computes correct values."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 2.0
    x._data.bitcast[Float32]()[2] = 8.0

    var result = log2(x)

    # log2(1) = 0, log2(2) = 1, log2(8) = 3
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(3.0), tolerance=1e-5
    )


fn test_log2_backward_gradient() raises:
    """Test log2 backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set positive non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 2.0

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return log2(inp)

    var y = log2(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return log2_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=5e-3, atol=1e-5)


# ============================================================================
# Square Root Tests
# ============================================================================


fn test_sqrt_shapes() raises:
    """Test that sqrt returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = sqrt(x)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_sqrt_values() raises:
    """Test that sqrt computes correct values."""
    var shape = List[Int]()
    shape.append(4)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 4.0
    x._data.bitcast[Float32]()[3] = 9.0

    var result = sqrt(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(3.0), tolerance=1e-5
    )


fn test_sqrt_backward() raises:
    """Test sqrt backward pass."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 1.0
    x._data.bitcast[Float32]()[1] = 4.0

    var grad_input = sqrt_backward(grad_output, x)

    # d/dx[sqrt(x)] = 1/(2*sqrt(x))
    # x=1: 1/(2*1) = 0.5
    # x=4: 1/(2*2) = 0.25
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )


fn test_sqrt_backward_gradient() raises:
    """Test sqrt backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set positive non-uniform values
    x._data.bitcast[Float32]()[0] = 0.5
    x._data.bitcast[Float32]()[1] = 1.0
    x._data.bitcast[Float32]()[2] = 2.0

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return sqrt(inp)

    var y = sqrt(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return sqrt_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=5e-3, atol=1e-5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run log10, log2, and sqrt elementwise operation tests."""
    print("Running elementwise operation tests (part 3: log10, log2, sqrt)...")

    test_log10_values()
    print("✓ test_log10_values")

    test_log10_backward_gradient()
    print("✓ test_log10_backward_gradient")

    test_log2_values()
    print("✓ test_log2_values")

    test_log2_backward_gradient()
    print("✓ test_log2_backward_gradient")

    # Square root tests
    test_sqrt_shapes()
    print("✓ test_sqrt_shapes")

    test_sqrt_values()
    print("✓ test_sqrt_values")

    test_sqrt_backward()
    print("✓ test_sqrt_backward")

    test_sqrt_backward_gradient()
    print("✓ test_sqrt_backward_gradient")

    print("\nAll elementwise operation tests (part 3) passed!")
