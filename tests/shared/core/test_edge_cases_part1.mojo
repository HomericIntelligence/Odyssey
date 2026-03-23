# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_edge_cases.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor edge cases - Part 1: Empty tensors, 0D scalars, large tensors.

Split from test_edge_cases.mojo per ADR-009 (≤10 fn test_ functions per file).
"""

from math import isnan, isinf

# NOTE(#3330): This file intentionally uses a package-level import to serve as a
# reproducible test case for the JIT crash root cause. `from shared.core import`
# forces the JIT to compile all 37,401 lines across 60+ modules via __init__.mojo,
# which intermittently overflows a JIT-internal buffer and triggers __fortify_fail_abort.
# Run 20+ times to observe: `for i in $(seq 1 20); do pixi run mojo run tests/shared/core/test_edge_cases_part1.mojo 2>&1 | grep -E "OK|crashed|PASS"; done`
# Import AnyTensor and operations
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    arange,
    nan_tensor,
    inf_tensor,
    neg_inf_tensor,
)
from shared.core import (
    add,
    subtract,
    multiply,
    divide,
    floor_divide,
    modulo,
    power,
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_equal_int,
    assert_true,
)


# ============================================================================
# Test empty tensors (0 elements)
# ============================================================================


fn test_empty_tensor_creation() raises:
    """Test creating empty tensor with 0 elements."""
    var shape = List[Int]()
    shape.append(0)
    var t = zeros(shape, DType.float32)

    assert_numel(t, 0, "Empty tensor should have 0 elements")
    assert_dim(t, 1, "Empty tensor should have 1 dimension")


fn test_empty_tensor_operations() raises:
    """Test operations on empty tensors."""
    var shape = List[Int]()
    shape.append(0)
    var a = zeros(shape, DType.float32)
    var b = zeros(shape, DType.float32)
    var c = add(a, b)

    assert_numel(c, 0, "Operations on empty tensors should give empty tensor")


fn test_empty_tensor_2d() raises:
    """Test 2D empty tensor (0 rows or 0 cols)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(0)  # 3x0 matrix
    var t = zeros(shape, DType.float32)

    assert_numel(t, 0, "3x0 matrix should have 0 elements")
    assert_dim(t, 2, "Should preserve 2D structure")


# ============================================================================
# Test 0D scalar tensors
# ============================================================================


fn test_scalar_tensor_creation() raises:
    """Test creating 0D scalar tensor."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)

    assert_numel(t, 1, "0D tensor should have 1 element")
    assert_dim(t, 0, "0D tensor should have 0 dimensions")
    assert_value_at(t, 0, 42.0, 1e-6, "Scalar value should be 42.0")


fn test_scalar_tensor_operations() raises:
    """Test operations on 0D scalar tensors."""
    var shape = List[Int]()
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = add(a, b)

    assert_numel(c, 1, "Scalar + scalar should give scalar")
    assert_dim(c, 0, "Result should be 0D")
    assert_value_at(c, 0, 5.0, 1e-6, "3 + 2 should be 5")


fn test_scalar_to_vector_broadcast() raises:
    """Test broadcasting scalar to vector."""
    var shape_scalar = List[Int]()
    var shape_vec = List[Int]()
    shape_vec.append(5)

    var a = full(shape_scalar, 10.0, DType.float32)  # scalar
    var b = full(shape_vec, 2.0, DType.float32)  # vector
    var c = multiply(a, b)

    assert_numel(c, 5, "Result should have 5 elements")
    assert_all_values(c, 20.0, 1e-6, "10 * 2 broadcast to [20, 20, 20, 20, 20]")


# ============================================================================
# Test very large tensors
# ============================================================================


fn test_large_1d_tensor() raises:
    """Test creating and operating on large 1D tensor."""
    var shape = List[Int]()
    shape.append(10000000)  # 10 million elements
    var t = zeros(shape, DType.float32)

    assert_numel(t, 10000000, "Large tensor should have 10M elements")
    # Spot check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "First element should be 0")
    assert_value_at(t, 9999999, 0.0, 1e-8, "Last element should be 0")


fn test_large_dimension_count() raises:
    """Test tensor with many dimensions (10D)."""
    var shape = List[Int](length=10, fill=2)
    var t = zeros(shape, DType.float32)

    assert_dim(t, 10, "Should have 10 dimensions")
    assert_numel(t, 1024, "2^10 = 1024 elements")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run edge case tests - Part 1."""
    print("Running AnyTensor edge case tests (Part 1)...")

    # Empty tensors
    print("  Testing empty tensors...")
    test_empty_tensor_creation()
    test_empty_tensor_operations()
    test_empty_tensor_2d()

    # 0D scalar tensors
    print("  Testing 0D scalar tensors...")
    test_scalar_tensor_creation()
    test_scalar_tensor_operations()
    test_scalar_to_vector_broadcast()

    # Very large tensors
    print("  Testing very large tensors...")
    test_large_1d_tensor()
    test_large_dimension_count()

    print("All edge case tests (Part 1) completed!")
