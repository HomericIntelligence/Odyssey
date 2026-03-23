"""Tests for matrix operations - Part 3: Matmul Vector & Shape Variants.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Vector @ matrix multiplication
- Linear layer pattern
- Matrix @ vector dimension mismatch error
- Thin matrices
- Wide matrices
- Tiny matrices
- Large square matrix
- Transpose shapes (basic)
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
from shared.tensor.any_tensor import (
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
# Matrix Multiplication Tests - Matrix @ Vector Operations (continued)
# ============================================================================


fn test_matmul_vector_matrix() raises:
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


fn test_matmul_linear_layer_pattern() raises:
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


fn test_matmul_matrix_vector_error() raises:
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


# ============================================================================
# Matrix Multiplication Tests - Shape Variants
# ============================================================================


fn test_matmul_thin_matrices() raises:
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


fn test_matmul_wide_matrices() raises:
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


fn test_matmul_tiny_matrices() raises:
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


fn test_matmul_large_square() raises:
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


# ============================================================================
# Transpose Tests - Basic 2D Operations
# ============================================================================


fn test_transpose_shapes() raises:
    """Test that transpose returns correct output shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)

    var a = ones(shape, DType.float32)
    var result = transpose(a)

    # (4, 10) -> (10, 4)
    assert_equal(result.shape()[0], 10)
    assert_equal(result.shape()[1], 4)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run matmul vector/shape variant and transpose shapes tests."""
    print(
        "Running matrix operation tests - Part 3: Matmul Vector & Shape"
        " Variants..."
    )

    print("\n=== Matrix Multiplication: Matrix @ Vector (continued) ===")
    test_matmul_vector_matrix()
    print("✓ test_matmul_vector_matrix")
    test_matmul_linear_layer_pattern()
    print("✓ test_matmul_linear_layer_pattern")
    test_matmul_matrix_vector_error()
    print("✓ test_matmul_matrix_vector_error")

    print("\n=== Matrix Multiplication: Shape Variants ===")
    test_matmul_thin_matrices()
    print("✓ test_matmul_thin_matrices")
    test_matmul_wide_matrices()
    print("✓ test_matmul_wide_matrices")
    test_matmul_tiny_matrices()
    print("✓ test_matmul_tiny_matrices")
    test_matmul_large_square()
    print("✓ test_matmul_large_square")

    print("\n=== Transpose: Basic 2D ===")
    test_transpose_shapes()
    print("✓ test_transpose_shapes")

    print("\n" + "=" * 60)
    print("All 8 tests passed! (Part 3)")
