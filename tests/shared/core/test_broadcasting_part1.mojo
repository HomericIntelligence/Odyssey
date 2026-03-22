"""Tests for AnyTensor broadcasting operations - Part 1: Scalar and vector-to-matrix.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_broadcasting.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests NumPy-style broadcasting rules for scalar and vector-to-matrix cases.
"""

# Import AnyTensor and operations
from shared.core.extensor import AnyTensor, zeros, ones, full
from shared.core.arithmetic import add, multiply
from testing import assert_true

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test scalar broadcasting
# ============================================================================


fn test_broadcast_scalar_to_1d() raises:
    """Test broadcasting scalar to 1D tensor."""
    var shape_vec = List[Int]()
    shape_vec.append(5)
    var shape_scalar = List[Int]()

    var a = full(shape_vec, 3.0, DType.float32)  # [3, 3, 3, 3, 3]
    var b = full(shape_scalar, 2.0, DType.float32)  # scalar 2
    var c = add(a, b)  # Expected: [5, 5, 5, 5, 5]

    assert_numel(c, 5, "Result should have 5 elements")
    assert_all_values(c, 5.0, 1e-6, "3 + 2 should broadcast to [5, 5, 5, 5, 5]")


fn test_broadcast_scalar_to_2d() raises:
    """Test broadcasting scalar to 2D tensor."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_scalar = List[Int]()

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix of ones
    var b = full(shape_scalar, 5.0, DType.float32)  # scalar 5
    var c = multiply(a, b)  # Expected: 3x4 matrix of fives

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 5.0, 1e-6, "1 * 5 should broadcast to all 5s")


fn test_broadcast_scalar_to_3d() raises:
    """Test broadcasting scalar to 3D tensor."""
    var shape_3d = List[Int]()
    shape_3d.append(2)
    shape_3d.append(3)
    shape_3d.append(4)
    var shape_scalar = List[Int]()

    var a = full(shape_3d, 2.0, DType.float32)  # 2x3x4 tensor
    var b = full(shape_scalar, 3.0, DType.float32)  # scalar 3
    var c = add(a, b)  # Expected: 2x3x4 tensor of fives

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "2 + 3 should broadcast to all 5s")


# ============================================================================
# Test vector-to-matrix broadcasting
# ============================================================================


fn test_broadcast_vector_to_matrix_row() raises:
    """Test broadcasting row vector to matrix."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_vec = List[Int]()
    shape_vec.append(1)
    shape_vec.append(4)

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix
    var b = full(shape_vec, 2.0, DType.float32)  # 1x4 vector
    var c = add(a, b)  # Expected: 3x4 matrix, each row is [3, 3, 3, 3]

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 3.0, 1e-6, "Broadcasting 1x4 vector to 3x4 matrix")


fn test_broadcast_vector_to_matrix_column() raises:
    """Test broadcasting column vector to matrix."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_vec = List[Int]()
    shape_vec.append(3)
    shape_vec.append(1)

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix
    var b = full(shape_vec, 2.0, DType.float32)  # 3x1 vector
    var c = multiply(a, b)  # Expected: 3x4 matrix, each column multiplied by 2

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 2.0, 1e-6, "Broadcasting 3x1 vector to 3x4 matrix")


fn test_broadcast_1d_to_2d() raises:
    """Test broadcasting 1D vector to 2D matrix."""
    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_vec = List[Int]()
    shape_vec.append(4)

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix
    var b = full(shape_vec, 3.0, DType.float32)  # 4-element vector
    var c = add(a, b)  # Expected: 3x4 matrix, each row is [4, 4, 4, 4]

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 4.0, 1e-6, "Broadcasting 1D(4) to 2D(3,4)")


# ============================================================================
# Test dimension size 1 broadcasting
# ============================================================================


fn test_broadcast_size_one_dim_leading() raises:
    """Test broadcasting with leading dimension of size 1."""
    var shape_a = List[Int]()
    shape_a.append(1)
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(3)
    shape_b.append(4)

    var a = full(shape_a, 2.0, DType.float32)  # 1x3x4
    var b = ones(shape_b, DType.float32)  # 2x3x4
    var c = add(a, b)  # Expected: 2x3x4, all 3s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 3.0, 1e-6, "Broadcasting 1x3x4 to 2x3x4")


fn test_broadcast_size_one_dim_middle() raises:
    """Test broadcasting with middle dimension of size 1."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(1)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(3)
    shape_b.append(4)

    var a = full(shape_a, 5.0, DType.float32)  # 2x1x4
    var b = ones(shape_b, DType.float32)  # 2x3x4
    var c = multiply(a, b)  # Expected: 2x3x4, all 5s

    assert_numel(c, 24, "Result should have 24 elements")
    assert_all_values(c, 5.0, 1e-6, "Broadcasting 2x1x4 to 2x3x4")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run broadcasting part 1 tests."""
    print("Running AnyTensor broadcasting tests - Part 1...")

    # Scalar broadcasting
    print("  Testing scalar broadcasting...")
    test_broadcast_scalar_to_1d()
    test_broadcast_scalar_to_2d()
    test_broadcast_scalar_to_3d()

    # Vector-to-matrix broadcasting
    print("  Testing vector-to-matrix broadcasting...")
    test_broadcast_vector_to_matrix_row()
    test_broadcast_vector_to_matrix_column()
    test_broadcast_1d_to_2d()

    # Size-1 dimension broadcasting
    print("  Testing size-1 dimension broadcasting...")
    test_broadcast_size_one_dim_leading()
    test_broadcast_size_one_dim_middle()

    print("All broadcasting part 1 tests completed!")
