# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise operations - Part 5: Clip, Rounding, and Logical Operations.

Tests cover:
- Utility functions: clip, ceil, floor, round
- Logical operations: logical_and, logical_or, logical_not
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
from shared.core.extensor import AnyTensor, zeros, ones, zeros_like, ones_like
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
# Clip Tests
# ============================================================================


fn test_clip_shapes() raises:
    """Test that clip returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var result = clip(x, min_val=-1.0, max_val=1.0)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 10)


fn test_clip_values() raises:
    """Test that clip computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -5.0
    x._data.bitcast[Float32]()[1] = -1.0
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.0
    x._data.bitcast[Float32]()[4] = 5.0

    var result = clip(x, min_val=-2.0, max_val=2.0)

    # Clip to [-2, 2]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-2.0), tolerance=1e-5
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
        result._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


fn test_clip_backward() raises:
    """Test clip backward pass."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)
    var grad_output = ones(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -5.0  # Below min
    x._data.bitcast[Float32]()[1] = -1.0  # Within range
    x._data.bitcast[Float32]()[2] = 0.0  # Within range
    x._data.bitcast[Float32]()[3] = 1.0  # Within range
    x._data.bitcast[Float32]()[4] = 5.0  # Above max

    var grad_input = clip_backward(grad_output, x, min_val=-2.0, max_val=2.0)

    # Gradient is 0 outside range, 1 inside range
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[1], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[2], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        grad_input._data.bitcast[Float32]()[4], Float32(0.0), tolerance=1e-5
    )


fn test_clip_backward_gradient() raises:
    """Test clip backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values within the clipping range
    x._data.bitcast[Float32]()[0] = -0.5
    x._data.bitcast[Float32]()[1] = 0.0
    x._data.bitcast[Float32]()[2] = 0.5

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return clip(inp, min_val=-1.0, max_val=1.0)

    var y = clip(x, min_val=-1.0, max_val=1.0)
    var grad_out = ones_like(y)

    # Backward function wrapper
    fn backward_fn(grad: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return clip_backward(grad, inp, min_val=-1.0, max_val=1.0)

    # Use numerical gradient checking (gold standard)
    check_gradient(forward, backward_fn, x, grad_out, rtol=1e-3, atol=1e-6)


# ============================================================================
# Rounding Tests
# ============================================================================


fn test_ceil_values() raises:
    """Test that ceil computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.5
    x._data.bitcast[Float32]()[1] = -1.1
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.1
    x._data.bitcast[Float32]()[4] = 2.5

    var result = ceil(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(-1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(3.0), tolerance=1e-5
    )


fn test_floor_values() raises:
    """Test that floor computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.5
    x._data.bitcast[Float32]()[1] = -1.1
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.1
    x._data.bitcast[Float32]()[4] = 2.5

    var result = floor(x)

    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-3.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(-2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


fn test_round_values() raises:
    """Test that round computes correct values."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = -2.5
    x._data.bitcast[Float32]()[1] = -1.4
    x._data.bitcast[Float32]()[2] = 0.0
    x._data.bitcast[Float32]()[3] = 1.4
    x._data.bitcast[Float32]()[4] = 2.5

    var result = round(x)

    # Round to nearest even (banker's rounding)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(-2.0), tolerance=1e-5
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
        result._data.bitcast[Float32]()[4], Float32(2.0), tolerance=1e-5
    )


# ============================================================================
# Logical Operations Tests
# ============================================================================


fn test_logical_and_values() raises:
    """Test that logical_and computes correct values."""
    var shape = List[Int]()
    shape.append(4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # Test all combinations: (0, 0), (0, 1), (1, 0), (1, 1)
    a._data.bitcast[Float32]()[0] = 0.0
    a._data.bitcast[Float32]()[1] = 0.0
    a._data.bitcast[Float32]()[2] = 1.0
    a._data.bitcast[Float32]()[3] = 1.0

    b._data.bitcast[Float32]()[0] = 0.0
    b._data.bitcast[Float32]()[1] = 1.0
    b._data.bitcast[Float32]()[2] = 0.0
    b._data.bitcast[Float32]()[3] = 1.0

    var result = logical_and(a, b)

    # AND truth table: 0, 0, 0, 1
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )


fn test_logical_or_values() raises:
    """Test that logical_or computes correct values."""
    var shape = List[Int]()
    shape.append(4)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a._data.bitcast[Float32]()[0] = 0.0
    a._data.bitcast[Float32]()[1] = 0.0
    a._data.bitcast[Float32]()[2] = 1.0
    a._data.bitcast[Float32]()[3] = 1.0

    b._data.bitcast[Float32]()[0] = 0.0
    b._data.bitcast[Float32]()[1] = 1.0
    b._data.bitcast[Float32]()[2] = 0.0
    b._data.bitcast[Float32]()[3] = 1.0

    var result = logical_or(a, b)

    # OR truth table: 0, 1, 1, 1
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
        result._data.bitcast[Float32]()[3], Float32(1.0), tolerance=1e-5
    )


fn test_logical_not_values() raises:
    """Test that logical_not computes correct values."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.float32)

    x._data.bitcast[Float32]()[0] = 0.0
    x._data.bitcast[Float32]()[1] = 1.0

    var result = logical_not(x)

    # NOT truth table: 1, 0
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(0.0), tolerance=1e-5
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run clip, rounding, and logical elementwise operation tests."""
    print(
        "Running elementwise operation tests (part 5: clip, rounding,"
        " logical)..."
    )

    # Clip tests
    test_clip_shapes()
    print("✓ test_clip_shapes")

    test_clip_values()
    print("✓ test_clip_values")

    test_clip_backward()
    print("✓ test_clip_backward")

    test_clip_backward_gradient()
    print("✓ test_clip_backward_gradient")

    # Rounding tests
    test_ceil_values()
    print("✓ test_ceil_values")

    test_floor_values()
    print("✓ test_floor_values")

    test_round_values()
    print("✓ test_round_values")

    # Logical operations tests
    test_logical_and_values()
    print("✓ test_logical_and_values")

    test_logical_or_values()
    print("✓ test_logical_or_values")

    test_logical_not_values()
    print("✓ test_logical_not_values")

    print("\nAll elementwise operation tests (part 5) passed!")
