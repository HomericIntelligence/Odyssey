# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise dispatch - Part 3: Custom unary ops, AddOp, SubtractOp, MultiplyOp, DivideOp, PowerOp."""

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


struct DoubleOp(ElementwiseUnaryOp):
    """Custom operation: 2 * x."""

    fn __init__(out self):
        pass

    fn apply(self, value: Float64) -> Float64:
        return value * 2.0


struct IncrementOp(ElementwiseUnaryOp):
    """Custom operation: x + 1."""

    fn __init__(out self):
        pass

    fn apply(self, value: Float64) -> Float64:
        return value + 1.0


# ============================================================================
# Unary Operation Tests - Custom Operations
# ============================================================================


fn test_apply_unary_custom_double() raises:
    """Test custom double operation."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[DoubleOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 10.0, tolerance=1e-6)


fn test_apply_unary_custom_increment() raises:
    """Test custom increment operation."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 5.0, DType.float32)

    var result = apply_unary[IncrementOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 6.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - AddOp
# ============================================================================


fn test_apply_binary_add() raises:
    """Test addition operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[AddOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - SubtractOp
# ============================================================================


fn test_apply_binary_subtract() raises:
    """Test subtraction operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[SubtractOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 3.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - MultiplyOp
# ============================================================================


fn test_apply_binary_multiply() raises:
    """Test multiplication operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[MultiplyOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 6.0, tolerance=1e-6)


# ============================================================================
# Binary Operation Tests - DivideOp
# ============================================================================


fn test_apply_binary_divide() raises:
    """Test division operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 6.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = apply_binary[DivideOp](a, b)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 3.0, tolerance=1e-6)


fn test_apply_binary_divide_by_zero_error() raises:
    """Test division by zero raises error."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 6.0, DType.float32)
    var b = zeros(shape, DType.float32)

    try:
        var result = apply_binary[DivideOp](a, b)
        assert_true(False, "Expected error for division by zero")
    except e:
        assert_true(True, "Correctly raised error")


# ============================================================================
# Binary Operation Tests - PowerOp
# ============================================================================


fn test_apply_binary_power_base_2() raises:
    """Test power operation: 2^3 = 8."""
    var shape = List[Int]()
    shape.append(1)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[PowerOp](a, b)

    assert_almost_equal(result._get_float64(0), 8.0, tolerance=1e-4)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run elementwise dispatch tests - Part 3."""
    test_apply_unary_custom_double()
    test_apply_unary_custom_increment()
    test_apply_binary_add()
    test_apply_binary_subtract()
    test_apply_binary_multiply()
    test_apply_binary_divide()
    test_apply_binary_divide_by_zero_error()
    test_apply_binary_power_base_2()
