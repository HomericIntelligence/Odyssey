"""Tests for matrix operations - Part 5: Transpose Combinations & Axes.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Transpose combination A.T @ B (backprop pattern)
- Transpose combination A @ B.T (attention pattern)
- Transpose combination A.T @ B.T (double transpose)
- Transpose axes 2D simple [1, 0]
- Transpose axes 3D identity [0, 1, 2]
- Transpose axes 3D permutation [2, 0, 1]
- Transpose axes 3D reverse [2, 1, 0]
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
# Transpose Tests - Combinations (BLAS Patterns)
# ============================================================================


fn test_transpose_combination_at_b() raises:
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


fn test_transpose_combination_a_bt() raises:
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


fn test_transpose_combination_at_bt() raises:
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


# ============================================================================
# Transpose Tests - Custom Axes Permutation (Issue #2389)
# ============================================================================


fn test_transpose_axes_2d_simple() raises:
    """Test 2D transpose with axes [1, 0] (standard transpose)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)

    var t = zeros(shape, DType.float32)  # 3x4

    # Fill with values: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    for i in range(12):
        t._data.bitcast[Float32]()[i] = Float32(i)

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


fn test_transpose_axes_3d_identity() raises:
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


fn test_transpose_axes_3d_permutation() raises:
    """Test 3D transpose with permutation [2, 0, 1]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)

    var t = zeros(shape, DType.float32)  # 2x3x4

    # Fill with sequential values
    for i in range(24):
        t._data.bitcast[Float32]()[i] = Float32(i)

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


fn test_transpose_axes_3d_reverse() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run transpose combinations and axes permutation tests."""
    print(
        "Running matrix operation tests - Part 5: Transpose Combinations &"
        " Axes..."
    )

    print("\n=== Transpose: Combinations (BLAS Patterns) ===")
    test_transpose_combination_at_b()
    print("✓ test_transpose_combination_at_b")
    test_transpose_combination_a_bt()
    print("✓ test_transpose_combination_a_bt")
    test_transpose_combination_at_bt()
    print("✓ test_transpose_combination_at_bt")

    print("\n=== Transpose: Custom Axes Permutation (Issue #2389) ===")
    test_transpose_axes_2d_simple()
    print("✓ test_transpose_axes_2d_simple")
    test_transpose_axes_3d_identity()
    print("✓ test_transpose_axes_3d_identity")
    test_transpose_axes_3d_permutation()
    print("✓ test_transpose_axes_3d_permutation")
    test_transpose_axes_3d_reverse()
    print("✓ test_transpose_axes_3d_reverse")

    print("\n" + "=" * 60)
    print("All 7 tests passed! (Part 5)")
