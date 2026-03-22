"""Integration tests for AnyTensor operations - Part 1: Chained and Creation Patterns.

Tests chained arithmetic operations and creation + arithmetic patterns.
Split from test_integration.mojo per ADR-009 to avoid heap corruption.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_integration.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full, arange, eye, linspace
from shared.core.arithmetic import add, subtract, multiply

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
# Test chained arithmetic operations
# ============================================================================


fn test_chained_add_operations() raises:
    """Test chaining multiple add operations."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)  # [1, 1, 1, 1, 1]
    var b = full(shape, 2.0, DType.float32)  # [2, 2, 2, 2, 2]
    var c = full(shape, 3.0, DType.float32)  # [3, 3, 3, 3, 3]

    var result = add(add(a, b), c)  # (1+2)+3 = 6

    assert_all_values(result, 6.0, 1e-6, "Chained additions should work")


fn test_mixed_arithmetic_operations() raises:
    """Test mixing different arithmetic operations."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = full(shape, 4.0, DType.float32)

    # (a + b) * c = (2 + 3) * 4 = 20
    var sum_ab = add(a, b)
    var result = multiply(sum_ab, c)

    assert_all_values(result, 20.0, 1e-6, "Mixed operations should work")


fn test_arithmetic_with_operator_overloading() raises:
    """Test using operator overloading for complex expressions."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = full(shape, 3.0, DType.float32)

    # a + b * c = 1 + 2 * 3 = 1 + 6 = 7
    var result = a + b * c

    assert_all_values(result, 7.0, 1e-6, "Operator precedence should work")


fn test_complex_expression() raises:
    """Test complex arithmetic expression."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = full(shape, 3.0, DType.float32)
    var d = full(shape, 4.0, DType.float32)

    # ((a + b) * c) - d = ((1 + 2) * 3) - 4 = 9 - 4 = 5
    var result = ((a + b) * c) - d

    assert_all_values(result, 5.0, 1e-6, "Complex expressions should work")


# ============================================================================
# Test creation + arithmetic patterns
# ============================================================================


fn test_identity_matrix_operations() raises:
    """Test operations with identity matrix."""
    var I = eye(3, 3, 0, DType.float32)
    var A = full(List[Int](), 2.0, DType.float32)  # Will need reshaping
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var B = full(shape, 2.0, DType.float32)

    # I + B should give all 3s on diagonal, 2s elsewhere
    var result = add(I, B)

    assert_numel(result, 9, "Result should be 3x3")
    # Check diagonal
    assert_value_at(result, 0, 3.0, 1e-6, "Diagonal [0,0]")
    assert_value_at(result, 4, 3.0, 1e-6, "Diagonal [1,1]")
    assert_value_at(result, 8, 3.0, 1e-6, "Diagonal [2,2]")
    # Check off-diagonal
    assert_value_at(result, 1, 2.0, 1e-6, "Off-diagonal [0,1]")
    assert_value_at(result, 3, 2.0, 1e-6, "Off-diagonal [1,0]")


fn test_arange_arithmetic() raises:
    """Test arithmetic with arange-created tensors."""
    var a = arange(0.0, 5.0, 1.0, DType.float32)  # [0, 1, 2, 3, 4]
    var shape = List[Int]()
    shape.append(5)
    var b = ones(shape, DType.float32)  # [1, 1, 1, 1, 1]

    var result = add(a, b)  # [1, 2, 3, 4, 5]

    assert_value_at(result, 0, 1.0, 1e-6, "0 + 1 = 1")
    assert_value_at(result, 2, 3.0, 1e-6, "2 + 1 = 3")
    assert_value_at(result, 4, 5.0, 1e-6, "4 + 1 = 5")


fn test_linspace_operations() raises:
    """Test operations with linspace-created tensors."""
    var a = linspace(0.0, 4.0, 5, DType.float32)  # [0, 1, 2, 3, 4]
    var b = linspace(5.0, 9.0, 5, DType.float32)  # [5, 6, 7, 8, 9]

    var result = add(a, b)  # [5, 7, 9, 11, 13]

    assert_value_at(result, 0, 5.0, 1e-6, "0 + 5 = 5")
    assert_value_at(result, 2, 9.0, 1e-6, "2 + 7 = 9")
    assert_value_at(result, 4, 13.0, 1e-6, "4 + 9 = 13")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run integration tests part 1: chained and creation patterns."""
    print("Running AnyTensor integration tests (part 1)...")

    # Chained operations
    print("  Testing chained operations...")
    test_chained_add_operations()
    test_mixed_arithmetic_operations()
    test_arithmetic_with_operator_overloading()
    test_complex_expression()

    # Creation + arithmetic
    print("  Testing creation + arithmetic patterns...")
    test_identity_matrix_operations()
    test_arange_arithmetic()
    test_linspace_operations()

    print("Integration tests part 1 completed!")
