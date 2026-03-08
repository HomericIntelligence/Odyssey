# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise dispatch - Part 5: LessOp, logical ops, custom binary op, dtype preservation."""

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
# Custom Operations for Testing
# ============================================================================


struct AverageOp(ElementwiseBinaryOp):
    """Custom operation: (a + b) / 2."""

    fn __init__(out self):
        pass

    fn apply(self, a: Float64, b: Float64) -> Float64:
        return (a + b) / 2.0


# ============================================================================
# Binary Operation Tests - Comparison Operations (continued)
# ============================================================================


fn test_apply_binary_less_true() raises:
    """Test less than when true."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = apply_binary[LessOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - Logical Operations
# ============================================================================


fn test_apply_binary_logical_and_both_true() raises:
    """Test logical AND when both are non-zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[LogicalAndOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_binary_logical_and_one_false() raises:
    """Test logical AND when one is zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = zeros(shape, DType.float32)

    var result = apply_binary[LogicalAndOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_binary_logical_or_both_true() raises:
    """Test logical OR when both are non-zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[LogicalOrOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_binary_logical_or_one_true() raises:
    """Test logical OR when only one is non-zero."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = zeros(shape, DType.float32)

    var result = apply_binary[LogicalOrOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - Custom Operations
# ============================================================================


fn test_apply_binary_custom_average() raises:
    """Test custom average operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 4.0, DType.float32)

    var result = apply_binary[AverageOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 3.0, tolerance=1e-6)


# ============================================================================
# Shape and Dtype Tests
# ============================================================================


fn test_apply_unary_preserves_dtype_float32() raises:
    """Test that unary operations preserve float32 dtype."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = ones(shape, DType.float32)

    var result = apply_unary[ExpOp](tensor)

    assert_true(
        result.dtype() == DType.float32, "Output dtype should match input"
    )


fn test_apply_unary_preserves_dtype_float64() raises:
    """Test that unary operations preserve float64 dtype."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = ones(shape, DType.float64)

    var result = apply_unary[ExpOp](tensor)

    assert_true(
        result.dtype() == DType.float64, "Output dtype should match input"
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run elementwise dispatch tests - Part 5."""
    test_apply_binary_less_true()
    test_apply_binary_logical_and_both_true()
    test_apply_binary_logical_and_one_false()
    test_apply_binary_logical_or_both_true()
    test_apply_binary_logical_or_one_true()
    test_apply_binary_custom_average()
    test_apply_unary_preserves_dtype_float32()
    test_apply_unary_preserves_dtype_float64()
