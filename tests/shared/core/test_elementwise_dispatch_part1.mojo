# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_elementwise_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for elementwise dispatch - Part 1: Unary ExpOp, LogOp, SqrtOp, SinOp, CosOp."""

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
# Unary Operation Tests - ExpOp
# ============================================================================


fn test_apply_unary_exp_zeros() raises:
    """Test exp(0) = 1."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[ExpOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


fn test_apply_unary_exp_ones() raises:
    """Test exp(1) ≈ 2.71828."""
    var shape = List[Int]()
    shape.append(3)
    var ones_tensor = ones(shape, DType.float32)

    var result = apply_unary[ExpOp](ones_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 2.71828, tolerance=1e-4)


# ============================================================================
# Unary Operation Tests - LogOp
# ============================================================================


fn test_apply_unary_log_ones() raises:
    """Test log(1) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var ones_tensor = ones(shape, DType.float32)

    var result = apply_unary[LogOp](ones_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_unary_log_error_negative() raises:
    """Test log of negative number raises error."""
    var shape = List[Int]()
    shape.append(1)
    var neg_tensor = full(shape, -1.0, DType.float32)

    try:
        var result = apply_unary[LogOp](neg_tensor)
        assert_true(False, "Expected error for log of negative number")
    except e:
        assert_true(True, "Correctly raised error")


# ============================================================================
# Unary Operation Tests - SqrtOp
# ============================================================================


fn test_apply_unary_sqrt_four() raises:
    """Test sqrt(4) = 2."""
    var shape = List[Int]()
    shape.append(3)
    var tensor = full(shape, 4.0, DType.float32)

    var result = apply_unary[SqrtOp](tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 2.0, tolerance=1e-6)


fn test_apply_unary_sqrt_zero() raises:
    """Test sqrt(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[SqrtOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


# ============================================================================
# Unary Operation Tests - TrigonometricOps
# ============================================================================


fn test_apply_unary_sin_zero() raises:
    """Test sin(0) = 0."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[SinOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 0.0, tolerance=1e-6)


fn test_apply_unary_cos_zero() raises:
    """Test cos(0) = 1."""
    var shape = List[Int]()
    shape.append(3)
    var zeros_tensor = zeros(shape, DType.float32)

    var result = apply_unary[CosOp](zeros_tensor)

    assert_equal_int(result.numel(), 3)
    for i in range(result.numel()):
        assert_almost_equal(result._get_float64(i), 1.0, tolerance=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run elementwise dispatch tests - Part 1."""
    test_apply_unary_exp_zeros()
    test_apply_unary_exp_ones()
    test_apply_unary_log_ones()
    test_apply_unary_log_error_negative()
    test_apply_unary_sqrt_four()
    test_apply_unary_sqrt_zero()
    test_apply_unary_sin_zero()
    test_apply_unary_cos_zero()
