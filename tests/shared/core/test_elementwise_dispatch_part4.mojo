# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise dispatch - Part 4: PowerOp, MaxOp, MinOp, comparison ops."""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.elementwise_dispatch import (
    ElementwiseUnaryOp,
    ElementwiseBinaryOp,
    apply_unary,
    apply_binary,
    ExpOp,
    LogOp,
    SqrtOp,
    SinOp,
    CosOp,
    TanhOp,
    AbsOp,
    NegateOp,
    ReciprocalOp,
    SquareOp,
    SignOp,
    AddOp,
    SubtractOp,
    MultiplyOp,
    DivideOp,
    PowerOp,
    MaxOp,
    MinOp,
    EqualOp,
    GreaterOp,
    GreaterEqualOp,
    LessOp,
    LessEqualOp,
    LogicalAndOp,
    LogicalOrOp,
)


# ============================================================================
# Binary Operation Tests - PowerOp (continued)
# ============================================================================


fn test_apply_binary_power_square() raises:
    """Test power operation: 5^2 = 25."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[PowerOp](a, b)

    assert_almost_equal(result._get_float64(0), 25.0, tolerance=1e-4)


# ============================================================================
# Binary Operation Tests - MaxOp
# ============================================================================


fn test_apply_binary_max_a_greater() raises:
    """Test max operation where a > b."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[MaxOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


fn test_apply_binary_max_b_greater() raises:
    """Test max operation where b > a."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[MaxOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - MinOp
# ============================================================================


fn test_apply_binary_min_a_less() raises:
    """Test min operation where a < b."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[MinOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 2.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - Comparison Operations
# ============================================================================


fn test_apply_binary_equal_true() raises:
    """Test equality when values are equal."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[EqualOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_binary_equal_false() raises:
    """Test equality when values are not equal."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[EqualOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_binary_greater_true() raises:
    """Test greater than when true."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[GreaterOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_binary_greater_false() raises:
    """Test greater than when false."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[GreaterOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run elementwise dispatch tests - Part 4."""
    test_apply_binary_power_square()
    test_apply_binary_max_a_greater()
    test_apply_binary_max_b_greater()
    test_apply_binary_min_a_less()
    test_apply_binary_equal_true()
    test_apply_binary_equal_false()
    test_apply_binary_greater_true()
    test_apply_binary_greater_false()
