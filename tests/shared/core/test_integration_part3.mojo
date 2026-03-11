"""Integration tests for ExTensor operations - Part 3: Identity, Scalar, and Large Tensors.

Tests zero and identity element behavior, scalar patterns, and large tensor operations.
Split from test_integration.mojo per ADR-009 to avoid heap corruption.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_integration.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    eye,
    linspace,
    add,
    subtract,
    multiply,
)

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
# Test zero and identity element behavior
# ============================================================================


fn test_additive_identity() raises:
    """Test that adding zero doesn't change values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 7.5, DType.float32)
    var zero = zeros(shape, DType.float32)

    var result = add(a, zero)

    assert_all_values(result, 7.5, 1e-6, "x + 0 = x")


fn test_multiplicative_identity() raises:
    """Test that multiplying by one doesn't change values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 7.5, DType.float32)
    var one = ones(shape, DType.float32)

    var result = multiply(a, one)

    assert_all_values(result, 7.5, 1e-6, "x * 1 = x")


fn test_multiplicative_zero() raises:
    """Test that multiplying by zero gives zero."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 99.9, DType.float32)
    var zero = zeros(shape, DType.float32)

    var result = multiply(a, zero)

    assert_all_values(result, 0.0, 1e-8, "x * 0 = 0")


# ============================================================================
# Test scalar patterns
# ============================================================================


fn test_scalar_operations() raises:
    """Test operations with scalar tensors."""
    var shape_scalar = List[Int]()
    var a = full(shape_scalar, 5.0, DType.float32)
    var b = full(shape_scalar, 3.0, DType.float32)

    var result = add(a, b)

    assert_dim(result, 0, "Result should be scalar")
    assert_value_at(result, 0, 8.0, 1e-6, "5 + 3 = 8")


# ============================================================================
# Test large tensor operations
# ============================================================================


fn test_large_tensor_operations() raises:
    """Test operations on large tensors."""
    var shape = List[Int]()
    shape.append(10000)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = multiply(a, b)

    assert_numel(result, 10000, "Result should have 10000 elements")
    # Spot check a few values
    assert_value_at(result, 0, 2.0, 1e-6, "First element")
    assert_value_at(result, 5000, 2.0, 1e-6, "Middle element")
    assert_value_at(result, 9999, 2.0, 1e-6, "Last element")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run integration tests part 3: identity elements, scalar, and large tensor ops.
    """
    print("Running ExTensor integration tests (part 3)...")

    # Identity elements
    print("  Testing identity element behavior...")
    test_additive_identity()
    test_multiplicative_identity()
    test_multiplicative_zero()

    # Scalar operations
    print("  Testing scalar operations...")
    test_scalar_operations()

    # Large tensors
    print("  Testing large tensor operations...")
    test_large_tensor_operations()

    print("Integration tests part 3 completed!")
