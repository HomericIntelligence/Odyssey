# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise dispatch - Part 6: dtype/shape validation, 2D tensors, ReciprocalOp."""

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


# ============================================================================
# Shape and Dtype Tests (continued)
# ============================================================================


fn test_apply_binary_preserves_dtype() raises:
    """Test that binary operations preserve dtype."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = apply_binary[AddOp](a, b)

    assert_true(
        result.dtype() == DType.float32, "Output dtype should match input"
    )


fn test_apply_binary_shape_mismatch_error() raises:
    """Test error when binary operands have different shapes."""
    var shape1 = List[Int]()
    shape1.append(3)
    var shape2 = List[Int]()
    shape2.append(4)
    var a = ones(shape1, DType.float32)
    var b = ones(shape2, DType.float32)

    try:
        var result = apply_binary[AddOp](a, b)
        assert_true(False, "Expected error for shape mismatch")
    except e:
        assert_true(True, "Correctly raised error")


fn test_apply_binary_dtype_mismatch_error() raises:
    """Test error when binary operands have different dtypes."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)

    try:
        var result = apply_binary[AddOp](a, b)
        assert_true(False, "Expected error for dtype mismatch")
    except e:
        assert_true(True, "Correctly raised error")


# ============================================================================
# Multi-dimensional Tensor Tests
# ============================================================================


fn test_apply_unary_2d_tensor() raises:
    """Test unary operation on 2D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var tensor = ones(shape, DType.float32)

    var result = apply_unary[DoubleOp](tensor)

    assert_equal_int(result.numel(), 6)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 2.0, tolerance=1e-6)


fn test_apply_binary_2d_tensor() raises:
    """Test binary operation on 2D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = apply_binary[AddOp](a, b)

    assert_equal_int(result.numel(), 6)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 5.0, tolerance=1e-6)


# ============================================================================
# Reciprocal and Edge Cases
# ============================================================================


fn test_apply_unary_reciprocal() raises:
    """Test reciprocal operation."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 2.0, DType.float32)

    var result = apply_unary[ReciprocalOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.5, tolerance=1e-6)


fn test_apply_unary_reciprocal_zero_error() raises:
    """Test reciprocal of zero raises error."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = zeros(shape, DType.float32)

    try:
        var result = apply_unary[ReciprocalOp](tensor)
        assert_true(False, "Expected error for reciprocal of zero")
    except e:
        assert_true(True, "Correctly raised error")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run elementwise dispatch tests - Part 6."""
    test_apply_binary_preserves_dtype()
    test_apply_binary_shape_mismatch_error()
    test_apply_binary_dtype_mismatch_error()
    test_apply_unary_2d_tensor()
    test_apply_binary_2d_tensor()
    test_apply_unary_reciprocal()
    test_apply_unary_reciprocal_zero_error()
