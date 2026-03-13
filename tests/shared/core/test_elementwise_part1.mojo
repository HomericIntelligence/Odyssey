# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operations - Part 1: Absolute Value and Sign.

Tests cover:
- Mathematical functions: abs, sign
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
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like
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
# Absolute Value Tests
# ============================================================================


fn test_abs_shapes() raises:
    """Test that abs returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = abs(x)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_abs_values() raises:
    """Test that abs computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -5.0
    x._data.bitcast[Float32]()[1] = -2.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 3.0
    x._data.bitcast[Float32]()[4] = 7.0

    var result = abs(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(3.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(7.0), tolerance=1e-5
    )


fn test_abs_backward() raises:
    """Test abs backward pass."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.0
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 3.0

    var grad_input = abs_backward(grad_output, x)

    # Gradient: -1 for x < 0, +1 for x > 0, 0 for x == 0
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )


fn test_abs_backward_gradient() raises:
    """Test abs backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Use non-zero values to avoid discontinuity at x=0
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.2
    x._data.bitcast[Float32]()[2] = 1.5

    # Forward function wrapper
    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return abs(inp)

    var y = abs(x)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return abs_backward(grad, inp)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Sign Tests
# ============================================================================


fn test_sign_values() raises:
    """Test that sign returns correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -5.0
    x._data.bitcast[Float32]()[1] = -0.1
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 0.1
    x._data.bitcast[Float32]()[4] = 7.0

    var result = sign(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(1.0), tolerance=1e-5
    )


# ============================================================================
# Main Test Runner
# ============================================================================


# ============================================================================
# Logical XOR Tests (#4145)
# ============================================================================


fn test_logical_xor_basic() raises:
    """Test logical_xor basic functionality. Closes #4145."""
    var shape = List[Int]()
    shape.append(4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # a: [0, 1, 0, 1], b: [0, 0, 1, 1]
    a._data.bitcast[Float32]()[1] = 1.0
    a._data.bitcast[Float32]()[3] = 1.0
    b._data.bitcast[Float32]()[2] = 1.0
    b._data.bitcast[Float32]()[3] = 1.0

    var result = logical_xor(a, b)

    # XOR: [0^0=0, 1^0=1, 0^1=1, 1^1=0]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(0.0), tolerance=1e-5
    )


fn test_logical_xor_same_inputs() raises:
    """Test logical_xor with identical inputs returns all zeros. Closes #4145."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)

    var result = logical_xor(a, a)

    # XOR of identical inputs should be all 0
    for i in range(3):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            Float32(0.0),
            tolerance=1e-5,
        )


fn main() raises:
    """Run abs, sign, and logical_xor elementwise operation tests."""
    print("Running elementwise operation tests (part 1: abs, sign, xor)...")

    # Absolute value tests
    test_abs_shapes()
    print("✓ test_abs_shapes")

    test_abs_values()
    print("✓ test_abs_values")

    test_abs_backward()
    print("✓ test_abs_backward")

    test_abs_backward_gradient()
    print("✓ test_abs_backward_gradient")

    # Sign tests
    test_sign_values()
    print("✓ test_sign_values")

    # Logical XOR tests
    test_logical_xor_basic()
    print("✓ test_logical_xor_basic")

    test_logical_xor_same_inputs()
    print("✓ test_logical_xor_same_inputs")

    print("\nAll elementwise operation tests (part 1) passed!")
