"""Tests for matrix operation edge cases - Part 2.

Tests edge cases for matmul including:
- Numerical stability with extreme values
- Operations with different dtypes
- Correctness with known values

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_matrix_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full, zeros_like, eye
from shared.core.matrix import matmul

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_true,
)


# ============================================================================
# Test matmul numerical stability
# ============================================================================


fn test_matmul_small_values() raises:
    """Test matmul with very small values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var t = full(shape, 1e-10, DType.float32)
    var c = matmul(t, t)

    # Should still compute correctly
    assert_dim(c, 2, "Result should be 2D")


fn test_matmul_large_values() raises:
    """Test matmul with large values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var t = full(shape, 1e6, DType.float32)
    var c = matmul(t, t)

    # Should compute (may overflow)
    assert_dim(c, 2, "Result should be 2D")


fn test_matmul_mixed_signs() raises:
    """Test matmul with mixed positive/negative values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, -1.0, DType.float32)

    var c = matmul(a, b)

    # 2*(-1) + 2*(-1) = -4
    assert_all_values(c, -4.0, 1e-5, "Mixed sign matmul")


# ============================================================================
# Test matmul with different dtypes
# ============================================================================


fn test_matmul_float64() raises:
    """Test matmul with float64 dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = ones(shape, DType.float64)
    var b = ones(shape, DType.float64)
    var c = matmul(a, b)

    assert_dtype(c, DType.float64, "Result should be float64")
    assert_all_values(c, 2.0, 1e-10, "Float64 matmul")


fn test_matmul_float32() raises:
    """Test matmul with float32 dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)

    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = matmul(a, b)

    assert_dtype(c, DType.float32, "Result should be float32")
    assert_all_values(c, 2.0, 1e-5, "Float32 matmul")


# ============================================================================
# Test matmul correctness with known values
# ============================================================================


fn test_matmul_known_result() raises:
    """Test matmul with known result values."""
    # [1, 2] @ [3, 4] = 1*3 + 2*4 = 11
    var shape_a = List[Int]()
    shape_a.append(1)
    shape_a.append(2)

    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(1)

    var a = AnyTensor(shape_a, DType.float32)
    a._set_float32(0, 1.0)
    a._set_float32(1, 2.0)

    var b = AnyTensor(shape_b, DType.float32)
    b._set_float32(0, 3.0)
    b._set_float32(1, 4.0)

    var c = matmul(a, b)

    assert_value_at(c, 0, 11.0, 1e-5, "[1,2]@[3,4]^T = 11")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run matrix operation edge case tests - Part 2."""
    print("Running matrix operation edge case tests (Part 2)...")

    # Numerical stability
    print("  Testing numerical stability...")
    test_matmul_small_values()
    test_matmul_large_values()
    test_matmul_mixed_signs()

    # Different dtypes
    print("  Testing different dtypes...")
    test_matmul_float64()
    test_matmul_float32()

    # Known results
    print("  Testing known results...")
    test_matmul_known_result()

    print("All matrix operation edge case tests (Part 2) completed!")
