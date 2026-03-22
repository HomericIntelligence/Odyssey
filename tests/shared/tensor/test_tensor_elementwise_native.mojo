"""Tests for native Tensor[dtype] elementwise: typed vs AnyTensor equivalence.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- exp_typed vs AnyTensor exp equivalence
- log_typed vs AnyTensor log equivalence
- sqrt_typed vs AnyTensor sqrt equivalence
- abs_typed vs AnyTensor abs equivalence
- sin_typed, cos_typed vs AnyTensor equivalence
- Edge cases: exp(0)=1, log(1)=0, sqrt(4)=2, abs(-3)=3
- Float64 precision preservation
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.factories import zeros, ones, full
from shared.core.any_tensor import AnyTensor, full as any_full, zeros as any_zeros
from shared.core.elementwise import (
    exp,
    log,
    sqrt,
    abs,
    sin,
    cos,
    exp_typed,
    log_typed,
    sqrt_typed,
    abs_typed,
    sin_typed,
    cos_typed,
)


fn test_exp_typed_matches_anytensor() raises:
    """Typed exp produces identical results to AnyTensor exp."""
    var a_any = any_full([6], 0.5, DType.float32)
    var result_any = exp(a_any)

    var a_typed = full[DType.float32]([6], 0.5)
    var result_typed = exp_typed(a_typed)

    for i in range(result_any.numel()):
        assert_almost_equal(
            Float64(result_any[i]),
            Float64(result_typed[i]),
            atol=1e-6,
            msg="exp: typed should match AnyTensor",
        )
    print("PASS: test_exp_typed_matches_anytensor")


fn test_log_typed_matches_anytensor() raises:
    """Typed log produces identical results to AnyTensor log."""
    var a_any = any_full([6], 1.5, DType.float32)
    var result_any = log(a_any)

    var a_typed = full[DType.float32]([6], 1.5)
    var result_typed = log_typed(a_typed)

    for i in range(result_any.numel()):
        assert_almost_equal(
            Float64(result_any[i]),
            Float64(result_typed[i]),
            atol=1e-6,
            msg="log: typed should match AnyTensor",
        )
    print("PASS: test_log_typed_matches_anytensor")


fn test_sqrt_abs_typed_match_anytensor() raises:
    """Typed sqrt and abs match AnyTensor equivalents."""
    # sqrt
    var s_any = any_full([4], 0.25, DType.float32)
    var sr_any = sqrt(s_any)
    var s_typed = full[DType.float32]([4], 0.25)
    var sr_typed = sqrt_typed(s_typed)
    for i in range(4):
        assert_almost_equal(
            Float64(sr_any[i]),
            Float64(sr_typed[i]),
            atol=1e-6,
            msg="sqrt: typed should match AnyTensor",
        )

    # abs
    var a_any = any_full([4], -1.5, DType.float32)
    var ar_any = abs(a_any)
    var a_typed = full[DType.float32]([4], -1.5)
    var ar_typed = abs_typed(a_typed)
    for i in range(4):
        assert_almost_equal(
            Float64(ar_any[i]),
            Float64(ar_typed[i]),
            atol=1e-6,
            msg="abs: typed should match AnyTensor",
        )
    print("PASS: test_sqrt_abs_typed_match_anytensor")


fn test_sin_cos_typed_match_anytensor() raises:
    """Typed sin and cos match AnyTensor equivalents."""
    var a_any = any_full([4], 0.5, DType.float32)
    var sin_any = sin(a_any)
    var cos_any = cos(a_any)

    var a_typed = full[DType.float32]([4], 0.5)
    var sin_t = sin_typed(a_typed)
    var cos_t = cos_typed(a_typed)

    for i in range(4):
        assert_almost_equal(
            Float64(sin_any[i]),
            Float64(sin_t[i]),
            atol=1e-6,
            msg="sin: typed should match AnyTensor",
        )
        assert_almost_equal(
            Float64(cos_any[i]),
            Float64(cos_t[i]),
            atol=1e-6,
            msg="cos: typed should match AnyTensor",
        )
    print("PASS: test_sin_cos_typed_match_anytensor")


fn test_exp_edge_cases() raises:
    """exp(0)=1 and exp(1)~2.71828 for typed tensors."""
    var t = zeros[DType.float32]([2])
    t._data[1] = Scalar[DType.float32](1.0)
    var r = exp_typed(t)
    assert_almost_equal(
        Float64(r[0]), 1.0, atol=1e-6, msg="exp(0) = 1"
    )
    assert_almost_equal(
        Float64(r[1]), 2.718281828, atol=1e-4, msg="exp(1) ~ e"
    )
    print("PASS: test_exp_edge_cases")


fn test_log_sqrt_edge_cases() raises:
    """log(1)=0 and sqrt(4)=2 for typed tensors."""
    # log(1) = 0
    var t1 = full[DType.float32]([2], 1.0)
    var r1 = log_typed(t1)
    for i in range(2):
        assert_almost_equal(
            Float64(r1[i]), 0.0, atol=1e-6, msg="log(1) = 0"
        )

    # sqrt(4) = 2
    var t2 = full[DType.float32]([2], 4.0)
    var r2 = sqrt_typed(t2)
    for i in range(2):
        assert_almost_equal(
            Float64(r2[i]), 2.0, atol=1e-6, msg="sqrt(4) = 2"
        )
    print("PASS: test_log_sqrt_edge_cases")


fn test_abs_edge_cases() raises:
    """abs(-3)=3, abs(0)=0, abs(1.5)=1.5 for typed tensors."""
    var t = Tensor[DType.float32]([3])
    t._data[0] = Scalar[DType.float32](-3.0)
    t._data[1] = Scalar[DType.float32](0.0)
    t._data[2] = Scalar[DType.float32](1.5)
    var r = abs_typed(t)
    assert_almost_equal(
        Float64(r[0]), 3.0, atol=1e-6, msg="abs(-3) = 3"
    )
    assert_almost_equal(
        Float64(r[1]), 0.0, atol=1e-6, msg="abs(0) = 0"
    )
    assert_almost_equal(
        Float64(r[2]), 1.5, atol=1e-6, msg="abs(1.5) = 1.5"
    )
    print("PASS: test_abs_edge_cases")


fn test_exp_typed_float64_precision() raises:
    """Typed float64 exp preserves full precision."""
    var t = zeros[DType.float64]([2])
    t._data[0] = Scalar[DType.float64](1.0)
    t._data[1] = Scalar[DType.float64](0.5)
    var r = exp_typed(t)
    assert_almost_equal(
        Float64(r[0]),
        2.718281828459045,
        atol=1e-14,
        msg="float64 exp(1) should preserve full precision",
    )
    assert_almost_equal(
        Float64(r[1]),
        1.6487212707001282,
        atol=1e-14,
        msg="float64 exp(0.5) should preserve full precision",
    )
    print("PASS: test_exp_typed_float64_precision")


fn test_log_typed_float64_precision() raises:
    """Typed float64 log preserves full precision."""
    var t = zeros[DType.float64]([1])
    t._data[0] = Scalar[DType.float64](2.718281828459045)
    var r = log_typed(t)
    assert_almost_equal(
        Float64(r[0]),
        1.0,
        atol=1e-14,
        msg="float64 log(e) should be 1.0",
    )
    print("PASS: test_log_typed_float64_precision")


fn main() raises:
    test_exp_typed_matches_anytensor()
    test_log_typed_matches_anytensor()
    test_sqrt_abs_typed_match_anytensor()
    test_sin_cos_typed_match_anytensor()
    test_exp_edge_cases()
    test_log_sqrt_edge_cases()
    test_abs_edge_cases()
    test_exp_typed_float64_precision()
    test_log_typed_float64_precision()
    print("All test_tensor_elementwise_native tests passed!")
