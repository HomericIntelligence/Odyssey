"""Tests for AnyTensor matrix operations and precision.

Tests cover:
- AnyTensor matmul correctness and shape validation
- AnyTensor transpose correctness
- AnyTensor dot product correctness
- Float64 precision preservation
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    ones as any_ones,
    full as any_full,
    zeros as any_zeros,
)
from projectodyssey.core.matrix import (
    matmul,
    transpose,
    dot,
)


def test_matmul_correctness() raises:
    """AnyTensor matmul produces correct results."""
    var a = any_ones([2, 3], DType.float32)
    var b = any_ones([3, 4], DType.float32)
    var result = matmul(a, b)

    # Verify shape
    var s = result.shape()
    assert_true(s[0] == 2, "output rows should be 2")
    assert_true(s[1] == 4, "output cols should be 4")

    # Each element = sum of 3 ones = 3.0
    for i in range(result.numel()):
        assert_almost_equal(
            Float64(result[i]),
            3.0,
            atol=1e-6,
            msg="ones @ ones = 3",
        )
    print("PASS: test_matmul_correctness")


def test_matmul_shape_validation() raises:
    """AnyTensor matmul [2,3] x [3,4] = [2,4]."""
    var a = any_ones([2, 3], DType.float32)
    var b = any_ones([3, 4], DType.float32)
    var result = matmul(a, b)
    var shape = result.shape()
    assert_true(shape[0] == 2, "result rows should be 2")
    assert_true(shape[1] == 4, "result cols should be 4")
    print("PASS: test_matmul_shape_validation")


def test_matmul_identity() raises:
    """AnyTensor matmul with identity matrix preserves values."""
    var a = any_full([2, 2], 0.5, DType.float32)
    var eye = any_zeros([2, 2], DType.float32)
    eye[0] = 1.0
    eye[3] = 1.0
    var c = matmul(a, eye)
    for i in range(4):
        assert_almost_equal(Float64(c[i]), 0.5, atol=1e-6, msg="A @ I = A")
    print("PASS: test_matmul_identity")


def test_transpose_correctness() raises:
    """AnyTensor transpose produces correct results."""
    var a = any_zeros([2, 3], DType.float32)
    for i in range(6):
        a[i] = Float32(i)
    var result = transpose(a)

    # Verify transposed shape
    var s = result.shape()
    assert_true(s[0] == 3, "transposed rows should be 3")
    assert_true(s[1] == 2, "transposed cols should be 2")

    # Original [2,3]: [[0,1,2],[3,4,5]]
    # Transposed [3,2]: [[0,3],[1,4],[2,5]]
    assert_almost_equal(Float64(result[0]), 0.0, atol=1e-6, msg="[0,0]=0")
    assert_almost_equal(Float64(result[1]), 3.0, atol=1e-6, msg="[0,1]=3")
    assert_almost_equal(Float64(result[2]), 1.0, atol=1e-6, msg="[1,0]=1")
    assert_almost_equal(Float64(result[3]), 4.0, atol=1e-6, msg="[1,1]=4")
    print("PASS: test_transpose_correctness")


def test_dot_correctness() raises:
    """AnyTensor dot product produces correct results."""
    var a = any_full([4], 0.5, DType.float32)
    var b = any_full([4], 1.5, DType.float32)
    var result = dot(a, b)

    # dot([0.5]*4, [1.5]*4) = 4 * 0.75 = 3.0
    assert_almost_equal(
        Float64(result[0]),
        3.0,
        atol=1e-6,
        msg="dot([0.5]*4, [1.5]*4) = 3.0",
    )
    print("PASS: test_dot_correctness")


def test_matmul_float64() raises:
    """Float64 matmul preserves precision."""
    var a = any_zeros([2, 2], DType.float64)
    a[0] = 1.0
    a[1] = 0.5
    a[2] = 0.0
    a[3] = 1.0

    var b = any_zeros([2, 1], DType.float64)
    b[0] = 3.141592653589793
    b[1] = 2.718281828459045

    var result = matmul(a, b)
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
    print("PASS: test_matmul_float64")


def test_transpose_preserves_dtype() raises:
    """Transpose preserves the dtype."""
    var t = any_ones([3, 2], DType.float64)
    var r = transpose(t)
    assert_true(
        r.dtype() == DType.float64,
        "transpose should preserve float64 dtype",
    )
    var s = r.shape()
    assert_true(s[0] == 2, "transposed rows should be 2")
    assert_true(s[1] == 3, "transposed cols should be 3")
    print("PASS: test_transpose_preserves_dtype")


def test_dot_known_values() raises:
    """Dot product with known values."""
    var a = any_zeros([3], DType.float32)
    a[0] = 1.0
    a[1] = 0.0
    a[2] = -1.0

    var b = any_full([3], 1.0, DType.float32)

    var result = dot(a, b)
    # dot([1, 0, -1], [1, 1, 1]) = 1 + 0 + (-1) = 0
    assert_almost_equal(
        Float64(result[0]),
        0.0,
        atol=1e-6,
        msg="dot([1,0,-1], [1,1,1]) = 0",
    )
    print("PASS: test_dot_known_values")


def main() raises:
    test_matmul_correctness()
    test_matmul_shape_validation()
    test_matmul_identity()
    test_transpose_correctness()
    test_dot_correctness()
    test_matmul_float64()
    test_transpose_preserves_dtype()
    test_dot_known_values()
    print("All test_tensor_matrix_native tests passed!")
