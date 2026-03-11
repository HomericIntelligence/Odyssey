"""Tests for ExTensor broadcasting operations - Part 3: Output shapes, dtype, and integration.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_broadcasting.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests broadcast output shapes, dtype preservation, and integration with comparison/arithmetic ops.
"""

# Import ExTensor and operations
from shared.core import ExTensor, zeros, ones, full, add, multiply
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
# Test broadcast output shape
# ============================================================================


fn test_broadcast_output_shape_1d_2d() raises:
    """Test broadcast output shape for 1D + 2D."""
    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)
    var shape_1d = List[Int]()
    shape_1d.append(4)

    var a = ones(shape_2d, DType.float32)
    var b = ones(shape_1d, DType.float32)
    var c = add(a, b)

    assert_dim(c, 2, "Output should be 2D")
    assert_numel(c, 12, "Output should have 12 elements")


fn test_broadcast_output_shape_3d_complex() raises:
    """Test broadcast output shape for complex 3D case."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(1)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(1)
    shape_b.append(3)
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)
    var b = ones(shape_b, DType.float32)
    var c = add(a, b)

    assert_dim(c, 3, "Output should be 3D")
    assert_numel(c, 24, "Output should have 24 elements (2*3*4)")


# ============================================================================
# Test dtype preservation in broadcasting
# ============================================================================


fn test_broadcast_preserves_dtype() raises:
    """Test that broadcasting preserves dtype."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(4)
    var shape_b = List[Int]()
    shape_b.append(4)

    var a = ones(shape_a, DType.float64)
    var b = ones(shape_b, DType.float64)
    var c = add(a, b)

    assert_dtype(c, DType.float64, "Broadcast should preserve float64 dtype")


# ============================================================================
# Test broadcasting integration with comparison operations
# ============================================================================


fn test_broadcast_with_comparison_scalar() raises:
    """Test broadcasting scalar with comparison operations."""
    from shared.core import greater

    var shape_vec = List[Int]()
    shape_vec.append(5)
    var shape_scalar = List[Int]()

    var a = full(shape_vec, 3.0, DType.float32)  # [3, 3, 3, 3, 3]
    var b = full(shape_scalar, 2.0, DType.float32)  # scalar 2
    var c = greater(a, b)  # Should broadcast: [True, True, True, True, True]

    assert_numel(c, 5, "Result should have 5 elements")
    assert_dtype(c, DType.bool, "Comparison should return bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "3 > 2 should be True")


fn test_broadcast_with_comparison_vector_matrix() raises:
    """Test broadcasting vector to matrix with comparison."""
    from shared.core import less_equal

    var shape_mat = List[Int]()
    shape_mat.append(3)
    shape_mat.append(4)
    var shape_vec = List[Int]()
    shape_vec.append(4)

    var a = ones(shape_mat, DType.float32)  # 3x4 matrix of ones
    var b = full(shape_vec, 2.0, DType.float32)  # vector [2, 2, 2, 2]
    var c = less_equal(a, b)  # 1 <= 2 broadcasts to 3x4

    assert_numel(c, 12, "Result should have 12 elements")
    assert_dtype(c, DType.bool, "Comparison should return bool dtype")
    for i in range(12):
        assert_value_at(c, i, 1.0, 1e-6, "1 <= 2 should be True")


fn test_broadcast_chained_operations() raises:
    """Test chained operations with broadcasting."""
    var shape_mat = List[Int]()
    shape_mat.append(2)
    shape_mat.append(3)
    var shape_scalar = List[Int]()

    var a = full(shape_mat, 5.0, DType.float32)  # 2x3 matrix
    var b = full(shape_scalar, 2.0, DType.float32)  # scalar
    var c = full(shape_scalar, 3.0, DType.float32)  # scalar

    # (a + b) * c = (5 + 2) * 3 = 7 * 3 = 21
    var result = multiply(add(a, b), c)

    assert_numel(result, 6, "Result should have 6 elements")
    assert_all_values(result, 21.0, 1e-6, "(5 + 2) * 3 should be 21")


fn test_broadcast_with_subtract() raises:
    """Test broadcasting with subtraction."""
    from shared.core import subtract

    var shape_2d = List[Int]()
    shape_2d.append(3)
    shape_2d.append(4)
    var shape_1d = List[Int]()
    shape_1d.append(4)

    var a = full(shape_2d, 10.0, DType.float32)  # 3x4 matrix of 10s
    var b = full(shape_1d, 3.0, DType.float32)  # vector [3, 3, 3, 3]
    var c = subtract(a, b)  # 10 - 3 = 7, broadcast to 3x4

    assert_numel(c, 12, "Result should have 12 elements")
    assert_all_values(c, 7.0, 1e-6, "10 - 3 should broadcast to all 7s")


fn test_broadcast_with_divide() raises:
    """Test broadcasting with division."""
    from shared.core import divide

    var shape_mat = List[Int]()
    shape_mat.append(2)
    shape_mat.append(5)
    var shape_scalar = List[Int]()

    var a = full(shape_mat, 20.0, DType.float32)  # 2x5 matrix of 20s
    var b = full(shape_scalar, 4.0, DType.float32)  # scalar 4
    var c = divide(a, b)  # 20 / 4 = 5, broadcast

    assert_numel(c, 10, "Result should have 10 elements")
    assert_all_values(c, 5.0, 1e-6, "20 / 4 should broadcast to all 5s")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run broadcasting part 3 tests."""
    print("Running ExTensor broadcasting tests - Part 3...")

    # Output shape verification
    print("  Testing broadcast output shapes...")
    test_broadcast_output_shape_1d_2d()
    test_broadcast_output_shape_3d_complex()

    # Dtype preservation
    print("  Testing dtype preservation...")
    test_broadcast_preserves_dtype()

    # Integration tests
    print("  Testing broadcasting integration with operations...")
    test_broadcast_with_comparison_scalar()
    test_broadcast_with_comparison_vector_matrix()
    test_broadcast_chained_operations()
    test_broadcast_with_subtract()
    test_broadcast_with_divide()

    print("All broadcasting part 3 tests completed!")
