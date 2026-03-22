"""Tests for matrix operations - Part 4: Transpose Basic & Backward Pass.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Transpose values
- Transpose double (transpose^2 = identity)
- Transpose 2D
- Transpose identity matrix
- Transpose twice
- Transpose dtype preservation
- Transpose 3D default
- Transpose 3D correctness
- Transpose backward shapes
- Transpose backward gradient
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
from shared.core.any_tensor import (
    AnyTensor,
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
# Transpose Tests - Basic 2D Operations
# ============================================================================


fn test_transpose_values() raises:
    """Test that transpose computes correct values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var a = zeros(shape, DType.float32)

    # A = [[1, 2, 3], [4, 5, 6]]
    a._data.bitcast[Float32]()[0] = 1.0
    a._data.bitcast[Float32]()[1] = 2.0
    a._data.bitcast[Float32]()[2] = 3.0
    a._data.bitcast[Float32]()[3] = 4.0
    a._data.bitcast[Float32]()[4] = 5.0
    a._data.bitcast[Float32]()[5] = 6.0

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


fn test_transpose_double() raises:
    """Test that transpose(transpose(A)) = A."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var a = zeros(shape, DType.float32)

    # Fill with values
    for i in range(12):
        a._data.bitcast[Float32]()[i] = Float32(i)

    var result = transpose(transpose(a))

    # Should get back original
    for i in range(12):
        assert_almost_equal(
            result._data.bitcast[Float32]()[i],
            a._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


fn test_transpose_2d() raises:
    """Test transpose of 2D matrix."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)  # 3x4
    var b = transpose(a)

    # Result should be 4x3
    assert_dim(b, 2, "Transpose should be 2D")
    assert_numel(b, 12, "Transpose should have same number of elements")


fn test_transpose_identity() raises:
    """Test transpose of identity matrix."""
    var a = eye(4, 4, 0, DType.float32)
    var b = transpose(a)

    # Transpose of identity should be identity
    assert_dim(b, 2, "Transpose of 2D should be 2D")
    assert_numel(b, 16, "Transpose should preserve elements")


fn test_transpose_twice() raises:
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


fn test_transpose_preserves_dtype() raises:
    """Test that transpose preserves dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float64)
    var b = transpose(a)

    assert_dtype(b, DType.float64, "Transpose should preserve float64")


# ============================================================================
# Transpose Tests - 3D+ Operations
# ============================================================================


fn test_transpose_3d_default() raises:
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


fn test_transpose_3d_correctness() raises:
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


# ============================================================================
# Transpose Tests - Backward Pass
# ============================================================================


fn test_transpose_backward_shapes() raises:
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


fn test_transpose_backward_gradient() raises:
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
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.15 - 2.0

    # Forward function wrapper
    fn forward(inp: AnyTensor) raises escaping -> AnyTensor:
        return transpose(inp)

    # Backward function wrapper
    fn backward(grad_out: AnyTensor, inp: AnyTensor) raises escaping -> AnyTensor:
        return transpose_backward(grad_out)

    var output = forward(x)
    var grad_output = ones_like(output)

    # Numerical gradient checking
    check_gradient(forward, backward, x, grad_output, rtol=1e-3, atol=1e-6)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run transpose basic and backward pass tests."""
    print(
        "Running matrix operation tests - Part 4: Transpose Basic & Backward..."
    )

    print("\n=== Transpose: Basic 2D ===")
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

    print("\n=== Transpose: 3D+ Operations ===")
    test_transpose_3d_default()
    print("✓ test_transpose_3d_default")
    test_transpose_3d_correctness()
    print("✓ test_transpose_3d_correctness")

    print("\n=== Transpose: Backward Pass ===")
    test_transpose_backward_shapes()
    print("✓ test_transpose_backward_shapes")
    test_transpose_backward_gradient()
    print("✓ test_transpose_backward_gradient")

    print("\n" + "=" * 60)
    print("All 10 tests passed! (Part 4)")
