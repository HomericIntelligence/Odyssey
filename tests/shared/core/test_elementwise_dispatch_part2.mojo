# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise dispatch - Part 2: TanhOp, AbsOp, NegateOp, SquareOp, SignOp."""

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
# Unary Operation Tests - TrigonometricOps (continued)
# ============================================================================


fn test_apply_unary_tanh_zero() raises:
    """Test tanh(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[TanhOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - AbsOp
# ============================================================================


fn test_apply_unary_abs_negative() raises:
    """Test abs(-5) = 5."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, -5.0, DType.float32)

    var result = apply_unary[AbsOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


fn test_apply_unary_abs_positive() raises:
    """Test abs(5) = 5."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[AbsOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - NegateOp
# ============================================================================


fn test_apply_unary_negate() raises:
    """Test negate operation."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[NegateOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), -5.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - SquareOp
# ============================================================================


fn test_apply_unary_square() raises:
    """Test square operation: x^2."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 3.0, DType.float32)

    var result = apply_unary[SquareOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 9.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - SignOp
# ============================================================================


fn test_apply_unary_sign_positive() raises:
    """Test sign of positive number."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[SignOp](tensor)

    assert_almost_equal(result._get_float64(0), 1.0, tolerance=1e-6)


fn test_apply_unary_sign_negative() raises:
    """Test sign of negative number."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = full(shape, -5.0, DType.float32)

    var result = apply_unary[SignOp](tensor)

    assert_almost_equal(result._get_float64(0), -1.0, tolerance=1e-6)


fn test_apply_unary_sign_zero() raises:
    """Test sign of zero."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = zeros(shape, DType.float32)

    var result = apply_unary[SignOp](tensor)

    assert_almost_equal(result._get_float64(0), 0.0, tolerance=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run elementwise dispatch tests - Part 2."""
    test_apply_unary_tanh_zero()
    test_apply_unary_abs_negative()
    test_apply_unary_abs_positive()
    test_apply_unary_negate()
    test_apply_unary_square()
    test_apply_unary_sign_positive()
    test_apply_unary_sign_negative()
    test_apply_unary_sign_zero()
