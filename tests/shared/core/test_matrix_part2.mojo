"""Tests for matrix operations - Part 2: Matmul Error Cases & Backward Pass.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Matmul incompatible shapes error
- Matmul dtype mismatch error
- Matmul 1D error
- Matmul dtype preservation
- Matmul backward shapes
- Matmul backward gradient w.r.t. A
- Matmul backward gradient w.r.t. B
- Matmul matrix @ vector
"""

from tests.shared.conftest import (
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
from tests.shared.conftest import TestFixtures
from shared.core.extensor import (
    ExTensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
    full,
    arange,
    eye,
)
from shared.core.matrix import (
    matmul,
    transpose,
    dot,
    outer,
    matmul_backward,
    transpose_backward,
)
from shared.testing import (
    check_gradient,
    compute_numerical_gradient,
    assert_gradients_close,
)


# ============================================================================
# Matrix Multiplication Tests - Error Cases
# ============================================================================


fn test_matmul_incompatible_shapes() raises:
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


fn test_matmul_dtype_mismatch() raises:
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


fn test_matmul_1d_error() raises:
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


fn test_matmul_preserves_dtype() raises:
    """Test that matmul preserves dtype."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var a = ones(shape, DType.float64)
    var b = ones(shape, DType.float64)
    var c = matmul(a, b)

    assert_dtype(c, DType.float64, "matmul should preserve float64")


# ============================================================================
# Matrix Multiplication Tests - Backward Pass
# ============================================================================


fn test_matmul_backward_shapes() raises:
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


fn test_matmul_backward_gradient_a() raises:
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
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.0

    # Create input B with shape (k, n)
    var shape_b = List[Int]()
    shape_b.append(k)
    shape_b.append(n)
    var b = zeros(shape_b, DType.float32)

    # Initialize B with non-uniform values that don't sum to zero
    # Using offset 0.1 instead of -0.5 to avoid exact cancellation
    for i in range(k * n):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.1

    # Forward function wrapper
    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return matmul(inp, b)

    # Backward function wrapper for grad_a
    fn backward(grad_out: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        var grads = matmul_backward(grad_out, inp, b)
        return grads.grad_a

    var output = forward(a)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    # Note: atol=1e-3 for robustness against numerical noise in small gradients
    check_gradient(forward, backward, a, grad_output, rtol=1e-3, atol=1e-3)


fn test_matmul_backward_gradient_b() raises:
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
        a._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 1.0

    # Create input B with shape (k, n)
    var shape_b = List[Int]()
    shape_b.append(k)
    shape_b.append(n)
    var b = zeros(shape_b, DType.float32)

    # Initialize B with non-uniform values that don't sum to zero
    # Using offset 0.1 instead of -0.5 to avoid exact cancellation
    for i in range(k * n):
        b._data.bitcast[Float32]()[i] = Float32(i) * 0.2 + 0.1

    # Forward function wrapper
    fn forward(inp: ExTensor) raises escaping -> ExTensor:
        return matmul(a, inp)

    # Backward function wrapper for grad_b
    fn backward(grad_out: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        var grads = matmul_backward(grad_out, a, inp)
        return grads.grad_b

    var output = forward(b)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    # Note: atol=1e-3 for robustness against numerical noise in small gradients
    check_gradient(forward, backward, b, grad_output, rtol=1e-3, atol=1e-3)


# ============================================================================
# Matrix Multiplication Tests - Matrix @ Vector Operations
# ============================================================================


fn test_matmul_matrix_vector() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run matmul error cases and backward pass tests."""
    print(
        "Running matrix operation tests - Part 2: Matmul Errors & Backward..."
    )

    print("\n=== Matrix Multiplication: Error Handling ===")
    test_matmul_incompatible_shapes()
    print("✓ test_matmul_incompatible_shapes")
    test_matmul_dtype_mismatch()
    print("✓ test_matmul_dtype_mismatch")
    test_matmul_1d_error()
    print("✓ test_matmul_1d_error")
    test_matmul_preserves_dtype()
    print("✓ test_matmul_preserves_dtype")

    print("\n=== Matrix Multiplication: Backward Pass ===")
    test_matmul_backward_shapes()
    print("✓ test_matmul_backward_shapes")
    test_matmul_backward_gradient_a()
    print("✓ test_matmul_backward_gradient_a")
    test_matmul_backward_gradient_b()
    print("✓ test_matmul_backward_gradient_b")

    print("\n=== Matrix Multiplication: Matrix @ Vector ===")
    test_matmul_matrix_vector()
    print("✓ test_matmul_matrix_vector")

    print("\n" + "=" * 60)
    print("All 8 tests passed! (Part 2)")
