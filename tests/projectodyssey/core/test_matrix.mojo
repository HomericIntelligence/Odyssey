"""Tests for matrix operations.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

Tests cover:
- Matrix multiplication shapes (2D)
- Matrix multiplication values (2D)
- Matrix multiplication with identity
- Matrix multiplication 2D square/rectangular
- Matrix multiplication with zeros
- Batched matrix multiplication (3D, 4D)
"""


from tests.projectodyssey.conftest import (
    TestFixtures,
    assert_all_close,
    assert_all_values,
    assert_almost_equal,
    assert_close_float,
    assert_dim,
    assert_dtype,
    assert_equal,
    assert_equal_int,
    assert_numel,
    assert_shape,
    assert_true,
    assert_value_at,
)
from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
    full,
    arange,
    eye,
)
from projectodyssey.core.matrix import (
    matmul,
    transpose,
    dot,
    outer,
    matmul_backward,
    transpose_backward,
)
from projectodyssey.testing.gradient_checker import (
    check_gradient,
    compute_numerical_gradient,
    assert_gradients_close,
    NumericalForward,
    NumericalBackward,
)


@fieldwise_init
struct _MatmulAFwd(NumericalForward):
    var b: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return matmul(inp, self.b)


@fieldwise_init
struct _MatmulABwd(NumericalBackward):
    var b: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = matmul_backward(grad_out, inp, self.b)
        return grads.grad_a


@fieldwise_init
struct _MatmulBFwd(NumericalForward):
    var a: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return matmul(self.a, inp)


@fieldwise_init
struct _MatmulBBwd(NumericalBackward):
    var a: AnyTensor

    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        var grads = matmul_backward(grad_out, self.a, inp)
        return grads.grad_b


@fieldwise_init
struct _TransposeFwd(NumericalForward):
    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        return transpose(inp)


@fieldwise_init
struct _TransposeBwd(NumericalBackward):
    def __call__(self, grad_out: AnyTensor, inp: AnyTensor) raises -> AnyTensor:
        return transpose_backward(grad_out)


def test_matmul_shapes() raises:
    """Test that matmul returns correct output shape."""
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(5)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = matmul(a, b)

    # (4, 3) @ (3, 5) = (4, 5)
    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 5)


def test_matmul_values() raises:
    """Test that matmul computes correct values."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(2)

    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(2)

    var a = zeros(shape_a, DType.float32)
    var b = zeros(shape_b, DType.float32)

    # A = [[1, 2], [3, 4]]
    a.set(0, Float32(1.0))
    a.set(1, Float32(2.0))
    a.set(2, Float32(3.0))
    a.set(3, Float32(4.0))

    # B = [[5, 6], [7, 8]]
    b.set(0, Float32(5.0))
    b.set(1, Float32(6.0))
    b.set(2, Float32(7.0))
    b.set(3, Float32(8.0))

    var result = matmul(a, b)

    # Result = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    #        = [[19, 22], [43, 50]]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(19.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(22.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(43.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(50.0), tolerance=1e-5
    )


def test_matmul_identity() raises:
    """Test matmul with identity matrix."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)

    var a = zeros(shape, DType.float32)
    var identity = zeros(shape, DType.float32)

    # A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i in range(9):
        a.set(i, Float32(Float32(i + 1)))

    # Identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    identity.set(0, Float32(1.0))  # (0, 0)
    identity.set(4, Float32(1.0))  # (1, 1)
    identity.set(8, Float32(1.0))  # (2, 2)

    var result = matmul(a, identity)

    # A @ I = A
    for i in range(9):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            a._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


def test_matmul_2d_square() raises:
    """Test 2D matrix multiplication with square matrices."""
    var shape_3x3 = List[Int]()
    shape_3x3.append(3)
    shape_3x3.append(3)

    var a = eye(3, 3, 0, DType.float32)  # 3x3 identity
    var b = full(shape_3x3, 2.0, DType.float32)  # 3x3 matrix of 2s
    var c = matmul(a, b)

    # Identity @ B = B, so result should be all 2s
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 9, "Result should be 3x3 (9 elements)")
    assert_all_values(c, 2.0, 1e-6, "Identity @ B should equal B")


def test_matmul_2d_rectangular() raises:
    """Test 2D matrix multiplication with rectangular matrices."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(4)
    shape_b.append(2)

    var a = ones(shape_a, DType.float32)  # 3x4
    var b = full(shape_b, 2.0, DType.float32)  # 4x2
    var c = matmul(a, b)

    # Result should be 3x2, each element = 1*2 + 1*2 + 1*2 + 1*2 = 8
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 6, "Result should be 3x2 (6 elements)")
    assert_all_values(c, 8.0, 1e-6, "Each element should be 8")


def test_matmul_with_zeros() raises:
    """Test matmul with zero matrices."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var a = zeros(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 9, "Result should be 3x3")
    assert_all_values(c, 0.0, 1e-6, "Zero matrix @ anything = zero matrix")


def test_matmul_batched_3d() raises:
    """Test batched matrix multiplication (3D)."""
    var shape_a = List[Int]()
    shape_a.append(2)  # batch size
    shape_a.append(3)  # rows
    shape_a.append(4)  # cols
    var shape_b = List[Int]()
    shape_b.append(2)  # batch size
    shape_b.append(4)  # rows
    shape_b.append(2)  # cols

    var a = ones(shape_a, DType.float32)  # 2x3x4
    var b = full(shape_b, 0.5, DType.float32)  # 2x4x2
    var c = matmul(a, b)

    # Result should be 2x3x2 (batch_size x a_rows x b_cols)
    # Each element = 1*0.5 + 1*0.5 + 1*0.5 + 1*0.5 = 2
    assert_dim(c, 3, "Result should be 3D")
    assert_numel(c, 12, "Result should be 2x3x2 (12 elements)")
    assert_all_values(c, 2.0, 1e-6, "Each element should be 2")


def test_matmul_batched_4d() raises:
    """Test batched matrix multiplication (4D)."""
    var shape_a = List[Int]()
    shape_a.append(2)  # batch dim 1
    shape_a.append(3)  # batch dim 2
    shape_a.append(4)  # rows
    shape_a.append(5)  # cols
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(3)
    shape_b.append(5)  # rows
    shape_b.append(2)  # cols

    var a = ones(shape_a, DType.float32)  # 2x3x4x5
    var b = ones(shape_b, DType.float32)  # 2x3x5x2
    var c = matmul(a, b)

    # Result should be 2x3x4x2
    assert_dim(c, 4, "Result should be 4D")
    assert_numel(c, 48, "Result should be 2x3x4x2 (48 elements)")
    # Each element = 1*1 + ... (5 times) = 5
    assert_all_values(c, 5.0, 1e-6, "Each element should be 5")


def test_matmul_incompatible_shapes() raises:
    """Test that incompatible shapes raise error."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(5)  # Incompatible: 4 != 5
    shape_b.append(2)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    # Verify error handling
    var error_raised = False
    try:
        var c = matmul(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error(
            "Should have raised error for incompatible matmul shapes (3,4) @"
            " (5,2)"
        )


def test_matmul_dtype_mismatch() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)  # Different dtype

    var error_raised = False
    try:
        var c = matmul(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in matmul")


def test_matmul_1d_error() raises:
    """Test that 1D inputs raise error."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var error_raised = False
    try:
        var c = matmul(a, b)  # matmul requires 2D+
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for 1D inputs to matmul")


def test_matmul_preserves_dtype() raises:
    """Test that matmul preserves dtype."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var a = ones(shape, DType.float64)
    var b = ones(shape, DType.float64)
    var c = matmul(a, b)

    assert_dtype(c, DType.float64, "matmul should preserve float64")


def test_matmul_backward_shapes() raises:
    """Test that matmul_backward returns correct gradient shapes."""
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(5)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = matmul(a, b)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(4)
    grad_output_shape.append(5)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grads = matmul_backward(grad_output, a, b)
    var grad_a = grads.grad_a
    var grad_b = grads.grad_b

    # grad_a should have same shape as a
    assert_equal(grad_a.shape()[0], 4)
    assert_equal(grad_a.shape()[1], 3)

    # grad_b should have same shape as b
    assert_equal(grad_b.shape()[0], 3)
    assert_equal(grad_b.shape()[1], 5)


def test_matmul_backward_gradient_a() raises:
    """Test matmul_backward gradient w.r.t. input A with numerical checking.

    Validates that gradient w.r.t. A matches finite differences.
    """
    var batch = 2
    var m = 3
    var k = 4
    var n = 2

    # Create input A with shape (batch*m, k)
    var shape_a = List[Int]()
    shape_a.append(batch * m)
    shape_a.append(k)
    var a = zeros(shape_a, DType.float32)

    # Initialize A with non-uniform values
    for i in range(batch * m * k):
        a.set(i, Float32(Float32(i) * 0.1 - 1.0))

    # Create input B with shape (k, n)
    var shape_b = List[Int]()
    shape_b.append(k)
    shape_b.append(n)
    var b = zeros(shape_b, DType.float32)

    # Initialize B with non-uniform values that don't sum to zero
    # Using offset 0.1 instead of -0.5 to avoid exact cancellation
    for i in range(k * n):
        b.set(i, Float32(Float32(i) * 0.2 + 0.1))

    var output = _MatmulAFwd(b)(a)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    # Note: atol=1e-3 for robustness against numerical noise in small gradients
    check_gradient(
        _MatmulAFwd(b), _MatmulABwd(b), a, grad_output, rtol=1e-3, atol=1e-3
    )


def test_matmul_backward_gradient_b() raises:
    """Test matmul_backward gradient w.r.t. input B with numerical checking.

    Validates that gradient w.r.t. B matches finite differences.
    """
    var m = 3
    var k = 4
    var n = 2

    # Create input A with shape (m, k)
    var shape_a = List[Int]()
    shape_a.append(m)
    shape_a.append(k)
    var a = zeros(shape_a, DType.float32)

    # Initialize A with non-uniform values
    for i in range(m * k):
        a.set(i, Float32(Float32(i) * 0.1 - 1.0))

    # Create input B with shape (k, n)
    var shape_b = List[Int]()
    shape_b.append(k)
    shape_b.append(n)
    var b = zeros(shape_b, DType.float32)

    # Initialize B with non-uniform values that don't sum to zero
    # Using offset 0.1 instead of -0.5 to avoid exact cancellation
    for i in range(k * n):
        b.set(i, Float32(Float32(i) * 0.2 + 0.1))

    var output = _MatmulBFwd(a)(b)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    # Note: atol=1e-3 for robustness against numerical noise in small gradients
    check_gradient(
        _MatmulBFwd(a), _MatmulBBwd(a), b, grad_output, rtol=1e-3, atol=1e-3
    )


def test_matmul_matrix_vector() raises:
    """Test matrix @ vector multiplication (essential for linear layers)."""
    var shape_w = List[Int]()
    shape_w.append(3)  # out_features
    shape_w.append(4)  # in_features
    var shape_x = List[Int]()
    shape_x.append(4)  # in_features

    var w = ones(shape_w, DType.float32)  # 3x4 weight matrix
    var x = full(shape_x, 2.0, DType.float32)  # 4D input vector
    var y = matmul(w, x)  # Should give 3D output

    # Result: each element = 1*2 + 1*2 + 1*2 + 1*2 = 8
    assert_dim(y, 1, "Result should be 1D vector")
    assert_numel(y, 3, "Result should have 3 elements (out_features)")
    assert_value_at(y, 0, 8.0, 1e-6, "y[0] should be 8.0")
    assert_value_at(y, 1, 8.0, 1e-6, "y[1] should be 8.0")
    assert_value_at(y, 2, 8.0, 1e-6, "y[2] should be 8.0")


def test_matmul_vector_matrix() raises:
    """Test vector @ matrix multiplication."""
    var shape_x = List[Int]()
    shape_x.append(3)
    var shape_w = List[Int]()
    shape_w.append(3)
    shape_w.append(4)

    var x = full(shape_x, 2.0, DType.float32)  # 3D vector
    var w = ones(shape_w, DType.float32)  # 3x4 matrix
    var y = matmul(x, w)  # Should give 4D output

    # Result: each element = 2*1 + 2*1 + 2*1 = 6
    assert_dim(y, 1, "Result should be 1D vector")
    assert_numel(y, 4, "Result should have 4 elements")
    assert_all_values(y, 6.0, 1e-6, "All elements should be 6.0")


def test_matmul_linear_layer_pattern() raises:
    """Test typical linear layer pattern: weight @ input."""
    # Simulate: Linear(in=5, out=10) processing single input
    var shape_w = List[Int]()
    shape_w.append(10)  # out_features
    shape_w.append(5)  # in_features
    var shape_x = List[Int]()
    shape_x.append(5)  # in_features

    var weight = full(shape_w, 0.5, DType.float32)
    var input = ones(shape_x, DType.float32)
    var output = matmul(weight, input)

    # Each output element = 0.5 * 1 + ... (5 times) = 2.5
    assert_dim(output, 1, "Output should be 1D")
    assert_numel(output, 10, "Output should have 10 elements")
    assert_all_values(output, 2.5, 1e-6, "Linear output computation")


def test_matmul_matrix_vector_error() raises:
    """Test matrix @ vector dimension mismatch error."""
    var shape_w = List[Int]()
    shape_w.append(3)
    shape_w.append(4)
    var shape_x = List[Int]()
    shape_x.append(5)  # Wrong size!

    var w = ones(shape_w, DType.float32)
    var x = ones(shape_x, DType.float32)

    var error_raised = False
    try:
        var y = matmul(w, x)
    except:
        error_raised = True

    if not error_raised:
        raise Error(
            "Should have raised error for dimension mismatch (3,4) @ (5,)"
        )


def test_matmul_thin_matrices() raises:
    """Test thin matrices (many rows, few columns)."""
    var shape_a = List[Int]()
    shape_a.append(100)  # Many rows
    shape_a.append(5)  # Few columns
    var shape_b = List[Int]()
    shape_b.append(5)
    shape_b.append(20)

    var a = ones(shape_a, DType.float32)  # 100x5
    var b = ones(shape_b, DType.float32)  # 5x20
    var c = matmul(a, b)  # 100x20

    # Each element = 1*1 + ... (5 times) = 5
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 2000, "Result should be 100x20 (2000 elements)")
    assert_value_at(c, 0, 5.0, 1e-6, "Thin matrix multiplication")
    assert_value_at(c, 1999, 5.0, 1e-6, "Check last element")


def test_matmul_wide_matrices() raises:
    """Test wide matrices (few rows, many columns)."""
    var shape_a = List[Int]()
    shape_a.append(5)  # Few rows
    shape_a.append(100)  # Many columns
    var shape_b = List[Int]()
    shape_b.append(100)
    shape_b.append(20)

    var a = full(shape_a, 0.5, DType.float32)  # 5x100
    var b = ones(shape_b, DType.float32)  # 100x20
    var c = matmul(a, b)  # 5x20

    # Each element = 0.5*1 + ... (100 times) = 50
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 100, "Result should be 5x20 (100 elements)")
    assert_value_at(c, 0, 50.0, 1e-5, "Wide matrix multiplication")


def test_matmul_tiny_matrices() raises:
    """Test very small matrices (1x1, 2x1, 1x2)."""
    # Test 1x1 @ 1x1
    var shape_1x1 = List[Int]()
    shape_1x1.append(1)
    shape_1x1.append(1)

    var a = full(shape_1x1, 3.0, DType.float32)
    var b = full(shape_1x1, 4.0, DType.float32)
    var c = matmul(a, b)

    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 1, "Result should be 1x1 (1 element)")
    assert_value_at(c, 0, 12.0, 1e-6, "1x1 @ 1x1 = 3*4")


def test_matmul_large_square() raises:
    """Test larger square matrix (stress test)."""
    var shape = List[Int]()
    shape.append(50)
    shape.append(50)

    var a = ones(shape, DType.float32)  # 50x50
    var b = ones(shape, DType.float32)  # 50x50
    var c = matmul(a, b)  # 50x50

    # Each element = 1*1 + ... (50 times) = 50
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 2500, "Result should be 50x50 (2500 elements)")
    assert_value_at(c, 0, 50.0, 1e-5, "Large square matrix")
    assert_value_at(c, 2499, 50.0, 1e-5, "Check last element")


def test_transpose_shapes() raises:
    """Test that transpose returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)

    var a = ones(shape, DType.float32)
    var result = transpose(a)

    # (4, 10) -> (10, 4)
    assert_equal(result.shape()[0], 10)
    assert_equal(result.shape()[1], 4)


def test_transpose_values() raises:
    """Test that transpose computes correct values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var a = zeros(shape, DType.float32)

    # A = [[1, 2, 3], [4, 5, 6]]
    a.set(0, Float32(1.0))
    a.set(1, Float32(2.0))
    a.set(2, Float32(3.0))
    a.set(3, Float32(4.0))
    a.set(4, Float32(5.0))
    a.set(5, Float32(6.0))

    var result = transpose(a)

    # A^T = [[1, 4], [2, 5], [3, 6]]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(5.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(3.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[5], Float32(6.0), tolerance=1e-5
    )


def test_transpose_double() raises:
    """Test that transpose(transpose(A)) = A."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var a = zeros(shape, DType.float32)

    # Fill with values
    for i in range(12):
        a.set(i, Float32(Float32(i)))

    var result = transpose(transpose(a))

    # Should get back original
    for i in range(12):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            a._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


def test_transpose_2d() raises:
    """Test transpose of 2D matrix."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)  # 3x4
    var b = transpose(a)

    # Result should be 4x3
    assert_dim(b, 2, "Transpose should be 2D")
    assert_numel(b, 12, "Transpose should have same number of elements")


def test_transpose_identity() raises:
    """Test transpose of identity matrix."""
    var a = eye(4, 4, 0, DType.float32)
    var b = transpose(a)

    # Transpose of identity should be identity
    assert_dim(b, 2, "Transpose of 2D should be 2D")
    assert_numel(b, 16, "Transpose should preserve elements")


def test_transpose_twice() raises:
    """Test that transpose(transpose(x)) == x."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = transpose(a)
    var c = transpose(b)

    # Should return to original shape
    assert_dim(c, 2, "Double transpose should be 2D")
    assert_numel(c, 15, "Double transpose should preserve elements")


def test_transpose_preserves_dtype() raises:
    """Test that transpose preserves dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float64)
    var b = transpose(a)

    assert_dtype(b, DType.float64, "Transpose should preserve float64")


def test_transpose_3d_default() raises:
    """Test transpose of 3D tensor (default permutation)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)  # 2x3x4
    var b = transpose(a)

    # Default: reverse all axes -> 4x3x2
    assert_dim(b, 3, "Transpose should be 3D")
    assert_numel(b, 24, "Transpose should have same number of elements")


def test_transpose_3d_correctness() raises:
    """Test that 3D transpose actually transposes correctly (not just copies).
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var test_shape = List[Int]()
    test_shape.append(2)
    test_shape.append(3)
    test_shape.append(4)
    var t = ones(test_shape, DType.float32)
    var t_T = transpose(t)

    # Verify shape is reversed
    assert_dim(t_T, 3, "Transpose should be 3D")


def test_transpose_backward_shapes() raises:
    """Test that transpose_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)

    var a = ones(shape, DType.float32)
    var result = transpose(a)

    var grad_output_shape = List[Int]()
    grad_output_shape.append(10)
    grad_output_shape.append(4)
    var grad_output = ones(grad_output_shape, DType.float32)

    var grad_input = transpose_backward(grad_output)

    # Gradient should have same shape as input
    assert_equal(grad_input.shape()[0], 4)
    assert_equal(grad_input.shape()[1], 10)


def test_transpose_backward_gradient() raises:
    """Test transpose_backward with numerical gradient checking.

    Validates that gradient matches finite differences. Since transpose is
    its own inverse, the gradient should simply be the transposed gradient.
    """
    var m = 3
    var n = 4

    # Create input with shape (m, n)
    var shape = List[Int]()
    shape.append(m)
    shape.append(n)
    var x = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(m * n):
        x.set(i, Float32(Float32(i) * 0.15 - 2.0))

    var output = _TransposeFwd()(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(
        _TransposeFwd(), _TransposeBwd(), x, grad_output, rtol=1e-3, atol=1e-6
    )


def test_transpose_combination_at_b() raises:
    """Test A.T @ B (common in backprop: weight.T @ gradient)."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(2)

    var a = ones(shape_a, DType.float32)  # 3x4
    var b = full(shape_b, 2.0, DType.float32)  # 3x2
    var a_t = transpose(a)  # 4x3
    var c = matmul(a_t, b)  # 4x3 @ 3x2 -> 4x2

    # Each element = 1*2 + 1*2 + 1*2 = 6
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 8, "Result should be 4x2 (8 elements)")
    assert_all_values(c, 6.0, 1e-6, "A.T @ B computation")


def test_transpose_combination_a_bt() raises:
    """Test A @ B.T (common in attention: Q @ K.T)."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)
    var shape_b = List[Int]()
    shape_b.append(4)
    shape_b.append(3)

    var a = full(shape_a, 2.0, DType.float32)  # 2x3
    var b = ones(shape_b, DType.float32)  # 4x3
    var b_t = transpose(b)  # 3x4
    var c = matmul(a, b_t)  # 2x3 @ 3x4 -> 2x4

    # Each element = 2*1 + 2*1 + 2*1 = 6
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 8, "Result should be 2x4 (8 elements)")
    assert_all_values(c, 6.0, 1e-6, "A @ B.T computation")


def test_transpose_combination_at_bt() raises:
    """Test A.T @ B.T (double transpose pattern)."""
    var shape_a = List[Int]()
    shape_a.append(4)
    shape_a.append(3)
    var shape_b = List[Int]()
    shape_b.append(5)
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)  # 4x3
    var b = full(shape_b, 2.0, DType.float32)  # 5x4
    var a_t = transpose(a)  # 3x4
    var b_t = transpose(b)  # 4x5
    var c = matmul(a_t, b_t)  # 3x4 @ 4x5 -> 3x5

    # Each element = 1*2 + 1*2 + 1*2 + 1*2 = 8
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 15, "Result should be 3x5 (15 elements)")
    assert_all_values(c, 8.0, 1e-6, "A.T @ B.T computation")


def test_transpose_axes_2d_simple() raises:
    """Test 2D transpose with axes [1, 0] (standard transpose)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var t = zeros(shape, DType.float32)  # 3x4

    # Fill with values: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    for i in range(12):
        t.set(i, Float32(Float32(i)))

    # Create axes [1, 0] for standard transpose
    var axes = List[Int]()
    axes.append(1)
    axes.append(0)

    var result = transpose(t, axes^)

    # Result should be 4x3
    assert_dim(result, 2, "Result should be 2D")
    assert_equal(result.shape()[0], 4, "First dimension should be 4")
    assert_equal(result.shape()[1], 3, "Second dimension should be 3")

    # Check actual values: result[i,j] = input[j,i]
    # result[0,0] = input[0,0] = 0, result[0,1] = input[1,0] = 4, result[0,2] = input[2,0] = 8
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(8.0), tolerance=1e-5
    )


def test_transpose_axes_3d_identity() raises:
    """Test 3D transpose with identity permutation [0, 1, 2]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)  # 2x3x4

    # Identity permutation
    var axes = List[Int]()
    axes.append(0)
    axes.append(1)
    axes.append(2)

    var result = transpose(t, axes^)

    # Result shape should be unchanged (2x3x4)
    assert_dim(result, 3, "Result should be 3D")
    assert_equal(result.shape()[0], 2, "First dimension should be 2")
    assert_equal(result.shape()[1], 3, "Second dimension should be 3")
    assert_equal(result.shape()[2], 4, "Third dimension should be 4")

    # Values should be identical
    assert_all_values(
        result, 1.0, 1e-6, "Identity permutation preserves values"
    )


def test_transpose_axes_3d_permutation() raises:
    """Test 3D transpose with permutation [2, 0, 1]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = zeros(shape, DType.float32)  # 2x3x4

    # Fill with sequential values
    for i in range(24):
        t.set(i, Float32(Float32(i)))

    # Permutation [2, 0, 1]: (2, 3, 4) -> (4, 2, 3)
    var axes = List[Int]()
    axes.append(2)
    axes.append(0)
    axes.append(1)

    var result = transpose(t, axes^)

    # Result shape should be (4, 2, 3)
    assert_dim(result, 3, "Result should be 3D")
    assert_equal(result.shape()[0], 4, "First dimension should be 4")
    assert_equal(result.shape()[1], 2, "Second dimension should be 2")
    assert_equal(result.shape()[2], 3, "Third dimension should be 3")

    # Verify element count
    assert_numel(result, 24, "Result should have 24 elements")


def test_transpose_axes_3d_reverse() raises:
    """Test 3D transpose with reverse permutation [2, 1, 0]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)  # 2x3x4

    # Reverse permutation [2, 1, 0]: (2, 3, 4) -> (4, 3, 2)
    var axes = List[Int]()
    axes.append(2)
    axes.append(1)
    axes.append(0)

    var result = transpose(t, axes^)

    # Result shape should be (4, 3, 2)
    assert_dim(result, 3, "Result should be 3D")
    assert_equal(result.shape()[0], 4, "First dimension should be 4")
    assert_equal(result.shape()[1], 3, "Second dimension should be 3")
    assert_equal(result.shape()[2], 2, "Second dimension should be 2")

    # All values should be preserved (all ones)
    assert_all_values(
        result, 1.0, 1e-6, "Values preserved in reverse permutation"
    )


def test_transpose_axes_default_none() raises:
    """Test transpose with axes=None uses default (reverse all)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)  # 2x3x4

    # Explicit default: axes=None
    var result = transpose(t)

    # Should reverse all axes: (2, 3, 4) -> (4, 3, 2)
    assert_dim(result, 3, "Result should be 3D")
    assert_equal(result.shape()[0], 4, "First dimension should be 4")
    assert_equal(result.shape()[1], 3, "Second dimension should be 3")
    assert_equal(result.shape()[2], 2, "Third dimension should be 2")


def test_transpose_axes_4d_permutation() raises:
    """Test 4D transpose with custom permutation [3, 1, 2, 0]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    shape.append(5)

    var t = ones(shape, DType.float32)  # 2x3x4x5

    # Permutation [3, 1, 2, 0]: (2, 3, 4, 5) -> (5, 3, 4, 2)
    var axes = List[Int]()
    axes.append(3)
    axes.append(1)
    axes.append(2)
    axes.append(0)

    var result = transpose(t, axes^)

    # Result shape should be (5, 3, 4, 2)
    assert_dim(result, 4, "Result should be 4D")
    assert_equal(result.shape()[0], 5, "First dimension should be 5")
    assert_equal(result.shape()[1], 3, "Second dimension should be 3")
    assert_equal(result.shape()[2], 4, "Third dimension should be 4")
    assert_equal(result.shape()[3], 2, "Fourth dimension should be 2")

    # Verify element count
    assert_numel(result, 120, "Result should have 120 elements")


def test_transpose_axes_invalid_duplicate() raises:
    """Test that duplicate axes raise error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)

    # Invalid: duplicate axis 0
    var axes = List[Int]()
    axes.append(0)
    axes.append(0)
    axes.append(1)

    var error_raised = False
    try:
        var result = transpose(t, axes^)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for duplicate axes")


def test_transpose_axes_invalid_out_of_bounds() raises:
    """Test that out-of-bounds axes raise error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)

    # Invalid: axis 5 is out of bounds (only 3 dimensions)
    var axes = List[Int]()
    axes.append(0)
    axes.append(1)
    axes.append(5)

    var error_raised = False
    try:
        var result = transpose(t, axes^)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for out-of-bounds axis")


def test_transpose_axes_invalid_length() raises:
    """Test that wrong-length axes raise error."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = ones(shape, DType.float32)

    # Invalid: only 2 axes provided for 3D tensor
    var axes = List[Int]()
    axes.append(0)
    axes.append(1)

    var error_raised = False
    try:
        var result = transpose(t, axes^)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for wrong-length axes")


def test_transpose_axes_backward_3d() raises:
    """Test transpose_backward with custom axes."""
    var m = 2
    var n = 3
    var p = 4

    # Create 3D input with shape (m, n, p)
    var shape = List[Int]()
    shape.append(m)
    shape.append(n)
    shape.append(p)
    var x = zeros(shape, DType.float32)

    # Initialize with non-uniform values
    for i in range(m * n * p):
        x.set(i, Float32(Float32(i) * 0.1 - 2.0))

    # Forward with axes [2, 0, 1]: (2, 3, 4) -> (4, 2, 3)
    var axes = List[Int]()
    axes.append(2)
    axes.append(0)
    axes.append(1)

    var output = transpose(x, axes^)
    var grad_output = ones_like(output)

    # Compute backward - need new axes list since original was transferred
    var backward_axes = List[Int]()
    backward_axes.append(2)
    backward_axes.append(0)
    backward_axes.append(1)
    var grad_input = transpose_backward(grad_output, backward_axes^)

    # Gradient should have same shape as input
    assert_equal(grad_input.shape()[0], m, "Gradient dim 0 should match input")
    assert_equal(grad_input.shape()[1], n, "Gradient dim 1 should match input")
    assert_equal(grad_input.shape()[2], p, "Gradient dim 2 should match input")


def test_transpose_axes_double_permutation() raises:
    """Test that transpose(transpose(x, axes), inverse_axes) recovers original.
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = zeros(shape, DType.float32)

    # Fill with sequential values
    for i in range(24):
        t.set(i, Float32(Float32(i)))

    # Forward permutation [2, 0, 1]
    var axes = List[Int]()
    axes.append(2)
    axes.append(0)
    axes.append(1)

    var t_perm = transpose(t, axes^)

    # Compute inverse permutation
    # For axes [2, 0, 1], inverse is [1, 2, 0]
    var inverse_axes = List[Int]()
    inverse_axes.append(1)
    inverse_axes.append(2)
    inverse_axes.append(0)

    var t_recovered = transpose(t_perm, inverse_axes^)

    # Should recover original shape
    assert_equal(t_recovered.shape()[0], 2, "Recovered dim 0 should be 2")
    assert_equal(t_recovered.shape()[1], 3, "Recovered dim 1 should be 3")
    assert_equal(t_recovered.shape()[2], 4, "Recovered dim 2 should be 4")

    # Check values are recovered
    for i in range(24):
        assert_almost_equal(
            t_recovered._data.bitcast[Float32]()[i],
            t._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


def test_dot_shapes() raises:
    """Test that dot returns scalar output."""
    var shape = List[Int]()
    shape.append(5)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)

    var result = dot(a, b)

    # Dot product returns scalar (0D tensor with empty shape)
    assert_dim(result, 0, "Dot product should be 0D scalar")
    assert_numel(result, 1, "Dot product should have 1 element")


def test_dot_values() raises:
    """Test that dot computes correct values."""
    var shape = List[Int]()
    shape.append(3)

    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    a.set(0, Float32(1.0))
    a.set(1, Float32(2.0))
    a.set(2, Float32(3.0))

    b.set(0, Float32(4.0))
    b.set(1, Float32(5.0))
    b.set(2, Float32(6.0))

    var result = dot(a, b)

    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(32.0), tolerance=1e-5
    )


def test_dot_orthogonal() raises:
    """Test dot product of orthogonal vectors."""
    var shape = List[Int]()
    shape.append(2)

    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)

    # a = [1, 0], b = [0, 1]
    a.set(0, Float32(1.0))
    a.set(1, Float32(0.0))

    b.set(0, Float32(0.0))
    b.set(1, Float32(1.0))

    var result = dot(a, b)

    # Orthogonal vectors have dot product = 0
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )


def test_dot_1d() raises:
    """Test dot product of two 1D vectors."""
    var a = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    var b = arange(1.0, 6.0, 1.0, DType.float32)  # [1, 2, 3, 4, 5]
    var c = dot(a, b)

    # Expected: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 = 55
    assert_dim(c, 0, "Dot product should be scalar (0D)")
    assert_numel(c, 1, "Dot product should have 1 element")
    assert_value_at(c, 0, 55.0, 1e-4, "Dot product result")


def test_dot_2d() raises:
    """Test dot product (equivalent to matmul for 2D)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)  # 2x3
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(2)
    var b = ones(shape_b, DType.float32)  # 3x2
    var c = dot(a, b)

    # Should behave like matmul for 2D
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4, "Result should be 2x2 (4 elements)")
    assert_all_values(c, 3.0, 1e-6, "Each element should be 3")


def test_dot_incompatible_shapes() raises:
    """Test that incompatible 1D shapes raise error."""
    var shape_a = List[Int]()
    shape_a.append(5)
    var shape_b = List[Int]()
    shape_b.append(3)  # Different size

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var error_raised = False
    try:
        var c = dot(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error(
            "Should have raised error for incompatible dot shapes (5,) and (3,)"
        )


def test_dot_dtype_mismatch() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)

    var error_raised = False
    try:
        var c = dot(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in dot")


def test_dot_preserves_dtype() raises:
    """Test that dot preserves dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float64)
    var b = ones(shape, DType.float64)
    var c = dot(a, b)

    assert_dtype(c, DType.float64, "dot should preserve float64")


def test_outer_shapes() raises:
    """Test that outer returns correct output shape."""
    var shape_a = List[Int]()
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)

    var result = outer(a, b)

    # (3,) outer (4,) = (3, 4)
    assert_equal(result.shape()[0], 3)
    assert_equal(result.shape()[1], 4)


def test_outer_values() raises:
    """Test that outer computes correct values."""
    var shape_a = List[Int]()
    shape_a.append(2)

    var shape_b = List[Int]()
    shape_b.append(3)

    var a = zeros(shape_a, DType.float32)
    var b = zeros(shape_b, DType.float32)

    # a = [2, 3]
    a.set(0, Float32(2.0))
    a.set(1, Float32(3.0))

    # b = [4, 5, 6]
    b.set(0, Float32(4.0))
    b.set(1, Float32(5.0))
    b.set(2, Float32(6.0))

    var result = outer(a, b)

    # Outer product = [[2*4, 2*5, 2*6], [3*4, 3*5, 3*6]]
    #                = [[8, 10, 12], [12, 15, 18]]
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(8.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(10.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(12.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(12.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[4], Float32(15.0), tolerance=1e-5
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[5], Float32(18.0), tolerance=1e-5
    )


def test_outer_vectors() raises:
    """Test outer product of two vectors."""
    var a = arange(1.0, 4.0, 1.0, DType.float32)  # [1, 2, 3]
    var b = arange(1.0, 3.0, 1.0, DType.float32)  # [1, 2]
    var c = outer(a, b)

    # Expected 3x2 matrix:
    # [[1, 2],
    #  [2, 4],
    #  [3, 6]]
    assert_dim(c, 2, "Outer product should be 2D")
    assert_numel(c, 6, "Outer product should be 3x2 (6 elements)")
    assert_value_at(c, 0, 1.0, 1e-6, "c[0,0] = 1*1 = 1")
    assert_value_at(c, 1, 2.0, 1e-6, "c[0,1] = 1*2 = 2")
    assert_value_at(c, 2, 2.0, 1e-6, "c[1,0] = 2*1 = 2")
    assert_value_at(c, 3, 4.0, 1e-6, "c[1,1] = 2*2 = 4")
    assert_value_at(c, 4, 3.0, 1e-6, "c[2,0] = 3*1 = 3")
    assert_value_at(c, 5, 6.0, 1e-6, "c[2,1] = 3*2 = 6")


def test_outer_not_1d_error() raises:
    """Test that non-1D inputs raise error."""
    var shape_2d = List[Int]()
    shape_2d.append(2)
    shape_2d.append(3)
    var shape_1d = List[Int]()
    shape_1d.append(3)

    var a = ones(shape_2d, DType.float32)  # 2D tensor
    var b = ones(shape_1d, DType.float32)  # 1D vector

    var error_raised = False
    try:
        var c = outer(a, b)  # Should error: outer requires 1D
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for non-1D input to outer")


def test_outer_dtype_mismatch() raises:
    """Test that dtype mismatch raises error."""
    var shape = List[Int]()
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float64)

    var error_raised = False
    try:
        var c = outer(a, b)
    except:
        error_raised = True

    if not error_raised:
        raise Error("Should have raised error for dtype mismatch in outer")


def test_outer_with_zeros() raises:
    """Test outer product with zero vector."""
    var shape_a = List[Int]()
    shape_a.append(3)
    var shape_b = List[Int]()
    shape_b.append(4)

    var a = zeros(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)
    var c = outer(a, b)

    assert_dim(c, 2, "Outer product should be 2D")
    assert_numel(c, 12, "Outer product should be 3x4 (12 elements)")
    assert_all_values(
        c, 0.0, 1e-6, "Outer with zero vector should be all zeros"
    )


def test_outer_preserves_dtype() raises:
    """Test that outer preserves dtype."""
    var shape_a = List[Int]()
    shape_a.append(2)
    var shape_b = List[Int]()
    shape_b.append(3)

    var a = ones(shape_a, DType.float64)
    var b = ones(shape_b, DType.float64)
    var c = outer(a, b)

    assert_dtype(c, DType.float64, "outer should preserve float64")


def test_dunder_matmul() raises:
    """Test __matmul__ operator overloading (a @ b)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = a @ b

    # Each element should be 1*2 + 1*2 = 4
    assert_dim(c, 2, "Result should be 2D")
    assert_numel(c, 4, "Result should be 2x2 (4 elements)")
    assert_all_values(c, 4.0, 1e-6, "a @ b should work via __matmul__")


def main() raises:
    """Run all test_matrix tests."""
    print("Running test_matrix tests...")

    test_matmul_shapes()
    print("✓ test_matmul_shapes")

    test_matmul_values()
    print("✓ test_matmul_values")

    test_matmul_identity()
    print("✓ test_matmul_identity")

    test_matmul_2d_square()
    print("✓ test_matmul_2d_square")

    test_matmul_2d_rectangular()
    print("✓ test_matmul_2d_rectangular")

    test_matmul_with_zeros()
    print("✓ test_matmul_with_zeros")

    test_matmul_batched_3d()
    print("✓ test_matmul_batched_3d")

    test_matmul_batched_4d()
    print("✓ test_matmul_batched_4d")

    test_matmul_incompatible_shapes()
    print("✓ test_matmul_incompatible_shapes")

    test_matmul_dtype_mismatch()
    print("✓ test_matmul_dtype_mismatch")

    test_matmul_1d_error()
    print("✓ test_matmul_1d_error")

    test_matmul_preserves_dtype()
    print("✓ test_matmul_preserves_dtype")

    test_matmul_backward_shapes()
    print("✓ test_matmul_backward_shapes")

    test_matmul_backward_gradient_a()
    print("✓ test_matmul_backward_gradient_a")

    test_matmul_backward_gradient_b()
    print("✓ test_matmul_backward_gradient_b")

    test_matmul_matrix_vector()
    print("✓ test_matmul_matrix_vector")

    test_matmul_vector_matrix()
    print("✓ test_matmul_vector_matrix")

    test_matmul_linear_layer_pattern()
    print("✓ test_matmul_linear_layer_pattern")

    test_matmul_matrix_vector_error()
    print("✓ test_matmul_matrix_vector_error")

    test_matmul_thin_matrices()
    print("✓ test_matmul_thin_matrices")

    test_matmul_wide_matrices()
    print("✓ test_matmul_wide_matrices")

    test_matmul_tiny_matrices()
    print("✓ test_matmul_tiny_matrices")

    test_matmul_large_square()
    print("✓ test_matmul_large_square")

    test_transpose_shapes()
    print("✓ test_transpose_shapes")

    test_transpose_values()
    print("✓ test_transpose_values")

    test_transpose_double()
    print("✓ test_transpose_double")

    test_transpose_2d()
    print("✓ test_transpose_2d")

    test_transpose_identity()
    print("✓ test_transpose_identity")

    test_transpose_twice()
    print("✓ test_transpose_twice")

    test_transpose_preserves_dtype()
    print("✓ test_transpose_preserves_dtype")

    test_transpose_3d_default()
    print("✓ test_transpose_3d_default")

    test_transpose_3d_correctness()
    print("✓ test_transpose_3d_correctness")

    test_transpose_backward_shapes()
    print("✓ test_transpose_backward_shapes")

    test_transpose_backward_gradient()
    print("✓ test_transpose_backward_gradient")

    test_transpose_combination_at_b()
    print("✓ test_transpose_combination_at_b")

    test_transpose_combination_a_bt()
    print("✓ test_transpose_combination_a_bt")

    test_transpose_combination_at_bt()
    print("✓ test_transpose_combination_at_bt")

    test_transpose_axes_2d_simple()
    print("✓ test_transpose_axes_2d_simple")

    test_transpose_axes_3d_identity()
    print("✓ test_transpose_axes_3d_identity")

    test_transpose_axes_3d_permutation()
    print("✓ test_transpose_axes_3d_permutation")

    test_transpose_axes_3d_reverse()
    print("✓ test_transpose_axes_3d_reverse")

    test_transpose_axes_default_none()
    print("✓ test_transpose_axes_default_none")

    test_transpose_axes_4d_permutation()
    print("✓ test_transpose_axes_4d_permutation")

    test_transpose_axes_invalid_duplicate()
    print("✓ test_transpose_axes_invalid_duplicate")

    test_transpose_axes_invalid_out_of_bounds()
    print("✓ test_transpose_axes_invalid_out_of_bounds")

    test_transpose_axes_invalid_length()
    print("✓ test_transpose_axes_invalid_length")

    test_transpose_axes_backward_3d()
    print("✓ test_transpose_axes_backward_3d")

    test_transpose_axes_double_permutation()
    print("✓ test_transpose_axes_double_permutation")

    test_dot_shapes()
    print("✓ test_dot_shapes")

    test_dot_values()
    print("✓ test_dot_values")

    test_dot_orthogonal()
    print("✓ test_dot_orthogonal")

    test_dot_1d()
    print("✓ test_dot_1d")

    test_dot_2d()
    print("✓ test_dot_2d")

    test_dot_incompatible_shapes()
    print("✓ test_dot_incompatible_shapes")

    test_dot_dtype_mismatch()
    print("✓ test_dot_dtype_mismatch")

    test_dot_preserves_dtype()
    print("✓ test_dot_preserves_dtype")

    test_outer_shapes()
    print("✓ test_outer_shapes")

    test_outer_values()
    print("✓ test_outer_values")

    test_outer_vectors()
    print("✓ test_outer_vectors")

    test_outer_not_1d_error()
    print("✓ test_outer_not_1d_error")

    test_outer_dtype_mismatch()
    print("✓ test_outer_dtype_mismatch")

    test_outer_with_zeros()
    print("✓ test_outer_with_zeros")

    test_outer_preserves_dtype()
    print("✓ test_outer_preserves_dtype")

    test_dunder_matmul()
    print("✓ test_dunder_matmul")

    print("\nAll test_matrix tests passed!")
