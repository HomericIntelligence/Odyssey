"""Tests for native Tensor[dtype] matrix ops: typed vs AnyTensor equivalence.

# ADR-009: This file is intentionally limited to <=10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- matmul_typed vs AnyTensor matmul equivalence
- transpose_typed vs AnyTensor transpose equivalence
- dot_typed vs AnyTensor dot equivalence
- Shape validation for matmul
- Float64 precision preservation
"""

from testing import assert_true, assert_almost_equal
from shared.tensor.tensor import Tensor
from shared.tensor.factories import zeros, ones, full
from shared.core.any_tensor import AnyTensor, ones as any_ones, full as any_full
from shared.core.matrix import (
    matmul,
    transpose,
    dot,
    matmul_typed,
    transpose_typed,
    dot_typed,
)


fn test_matmul_typed_matches_anytensor() raises:
    """Typed matmul produces identical results to AnyTensor matmul."""
    var a_any = any_ones([2, 3], DType.float32)
    var b_any = any_ones([3, 4], DType.float32)
    var result_any = matmul(a_any, b_any)

    var a_typed = ones[DType.float32]([2, 3])
    var b_typed = ones[DType.float32]([3, 4])
    var result_typed = matmul_typed(a_typed, b_typed)

    # Verify shape
    var s = result_typed.shape()
    assert_true(s[0] == 2, "output rows should be 2")
    assert_true(s[1] == 4, "output cols should be 4")

    for i in range(result_any.numel()):
        assert_almost_equal(
            Float64(result_any[i]),
            Float64(result_typed[i]),
            atol=1e-6,
            msg="matmul: typed should match AnyTensor",
        )
    print("PASS: test_matmul_typed_matches_anytensor")


fn test_matmul_typed_shape_validation() raises:
    """Typed matmul [2,3] x [3,4] = [2,4]."""
    var a = ones[DType.float32]([2, 3])
    var b = ones[DType.float32]([3, 4])
    var result = matmul_typed(a, b)
    var shape = result.shape()
    assert_true(shape[0] == 2, "result rows should be 2")
    assert_true(shape[1] == 4, "result cols should be 4")
    # Each element = sum of 3 ones = 3.0
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]), 3.0, atol=1e-6, msg="ones @ ones = 3"
        )
    print("PASS: test_matmul_typed_shape_validation")


fn test_matmul_typed_identity() raises:
    """Typed matmul with identity matrix preserves values."""
    var a = full[DType.float32]([2, 2], 0.5)
    var eye = Tensor[DType.float32]([2, 2])
    eye._data[0] = Scalar[DType.float32](1.0)
    eye._data[1] = Scalar[DType.float32](0.0)
    eye._data[2] = Scalar[DType.float32](0.0)
    eye._data[3] = Scalar[DType.float32](1.0)
    var c = matmul_typed(a, eye)
    for i in range(4):
        assert_almost_equal(
            Float64(c[i]), 0.5, atol=1e-6, msg="A @ I = A"
        )
    print("PASS: test_matmul_typed_identity")


fn test_transpose_typed_matches_anytensor() raises:
    """Typed transpose produces identical results to AnyTensor transpose."""
    var a_any = any_full([2, 3], 0.0, DType.float32)
    # Fill with sequential values via AnyTensor
    for i in range(6):
        a_any[i] = Float32(i)
    var result_any = transpose(a_any)

    var a_typed = Tensor[DType.float32]([2, 3])
    for i in range(6):
        a_typed._data[i] = Scalar[DType.float32](Float32(i))
    var result_typed = transpose_typed(a_typed)

    # Verify transposed shape
    var s = result_typed.shape()
    assert_true(s[0] == 3, "transposed rows should be 3")
    assert_true(s[1] == 2, "transposed cols should be 2")

    for i in range(result_any.numel()):
        assert_almost_equal(
            Float64(result_any[i]),
            Float64(result_typed[i]),
            atol=1e-6,
            msg="transpose: typed should match AnyTensor",
        )
    print("PASS: test_transpose_typed_matches_anytensor")


fn test_dot_typed_matches_anytensor() raises:
    """Typed dot produces identical results to AnyTensor dot."""
    var a_any = any_full([4], 0.5, DType.float32)
    var b_any = any_full([4], 1.5, DType.float32)
    var result_any = dot(a_any, b_any)

    var a_typed = full[DType.float32]([4], 0.5)
    var b_typed = full[DType.float32]([4], 1.5)
    var result_typed = dot_typed(a_typed, b_typed)

    # dot([0.5]*4, [1.5]*4) = 4 * 0.75 = 3.0
    assert_almost_equal(
        Float64(result_any[0]),
        Float64(result_typed[0]),
        atol=1e-6,
        msg="dot: typed should match AnyTensor",
    )
    assert_almost_equal(
        Float64(result_typed[0]),
        3.0,
        atol=1e-6,
        msg="dot([0.5]*4, [1.5]*4) = 3.0",
    )
    print("PASS: test_dot_typed_matches_anytensor")


fn test_matmul_typed_float64() raises:
    """Typed float64 matmul preserves precision."""
    var a = Tensor[DType.float64]([2, 2])
    a._data[0] = Scalar[DType.float64](1.0)
    a._data[1] = Scalar[DType.float64](0.5)
    a._data[2] = Scalar[DType.float64](0.0)
    a._data[3] = Scalar[DType.float64](1.0)

    var b = Tensor[DType.float64]([2, 1])
    b._data[0] = Scalar[DType.float64](3.141592653589793)
    b._data[1] = Scalar[DType.float64](2.718281828459045)

    var result = matmul_typed(a, b)
    var shape = result.shape()
    assert_true(shape[0] == 2, "result rows should be 2")
    assert_true(shape[1] == 1, "result cols should be 1")

    # result[0] = 1.0 * pi + 0.5 * e
    var expected0 = 3.141592653589793 + 0.5 * 2.718281828459045
    assert_almost_equal(
        Float64(result[0]),
        expected0,
        atol=1e-12,
        msg="float64 matmul should preserve full precision",
    )
    # result[1] = 0.0 * pi + 1.0 * e = e
    assert_almost_equal(
        Float64(result[1]),
        2.718281828459045,
        atol=1e-12,
        msg="float64 matmul should preserve full precision",
    )
    print("PASS: test_matmul_typed_float64")


fn test_transpose_typed_preserves_dtype() raises:
    """Typed transpose preserves the dtype."""
    var t = ones[DType.float64]([3, 2])
    var r = transpose_typed(t)
    assert_true(
        r.get_dtype() == DType.float64,
        "transpose should preserve float64 dtype",
    )
    var s = r.shape()
    assert_true(s[0] == 2, "transposed rows should be 2")
    assert_true(s[1] == 3, "transposed cols should be 3")
    print("PASS: test_transpose_typed_preserves_dtype")


fn test_dot_typed_known_values() raises:
    """Typed dot product with known values."""
    var a = Tensor[DType.float32]([3])
    a._data[0] = Scalar[DType.float32](1.0)
    a._data[1] = Scalar[DType.float32](0.0)
    a._data[2] = Scalar[DType.float32](-1.0)

    var b = Tensor[DType.float32]([3])
    b._data[0] = Scalar[DType.float32](1.0)
    b._data[1] = Scalar[DType.float32](1.0)
    b._data[2] = Scalar[DType.float32](1.0)

    var result = dot_typed(a, b)
    # dot([1, 0, -1], [1, 1, 1]) = 1 + 0 + (-1) = 0
    assert_almost_equal(
        Float64(result[0]),
        0.0,
        atol=1e-6,
        msg="dot([1,0,-1], [1,1,1]) = 0",
    )
    print("PASS: test_dot_typed_known_values")


fn main() raises:
    test_matmul_typed_matches_anytensor()
    test_matmul_typed_shape_validation()
    test_matmul_typed_identity()
    test_transpose_typed_matches_anytensor()
    test_dot_typed_matches_anytensor()
    test_matmul_typed_float64()
    test_transpose_typed_preserves_dtype()
    test_dot_typed_known_values()
    print("All test_tensor_matrix_native tests passed!")
