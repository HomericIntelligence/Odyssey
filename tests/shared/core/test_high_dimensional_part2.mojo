"""Tests for high-dimensional tensor operations (Part 2: 6D/7D, broadcasting, large tensors).

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_high_dimensional.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests 6D/7D tensor operations, broadcasting, and large tensors.
"""

# Import AnyTensor and operations
from shared.core.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core.arithmetic import add, multiply, subtract
from shared.core.reduction import sum, mean, max_reduce, min_reduce

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
# Test 6D tensor operations
# ============================================================================


fn test_6d_tensor_creation() raises:
    """Create and verify 6D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = zeros(shape, DType.float32)

    assert_dim(t, 6, "Tensor should be 6D")
    assert_numel(t, 64, "2^6 = 64 elements")


fn test_6d_tensor_arithmetic() raises:
    """Arithmetic on 6D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var result = add(a, b)

    assert_dim(result, 6, "Result should be 6D")
    assert_all_values(result, 2.0, 1e-5, "1 + 1 = 2 for all elements")


fn test_6d_tensor_reduction() raises:
    """Reduction of 6D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    assert_value_at(result, 0, 64.0, 1e-5, "Sum of 64 ones = 64")


# ============================================================================
# Test 7D tensor operations
# ============================================================================


fn test_7d_tensor_creation() raises:
    """Create and verify 7D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)

    assert_dim(t, 7, "Tensor should be 7D")
    assert_numel(t, 128, "2^7 = 128 elements")


fn test_7d_tensor_sum() raises:
    """Sum of 7D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    assert_value_at(result, 0, 128.0, 1e-5, "Sum of 128 ones = 128")


# ============================================================================
# Test broadcasting with high dimensions
# ============================================================================


fn test_5d_broadcasting_scalar() raises:
    """Broadcasting scalar to 5D tensor."""
    var shape_scalar = List[Int]()
    var shape_5d = List[Int]()
    shape_5d.append(2)
    shape_5d.append(3)
    shape_5d.append(2)
    shape_5d.append(3)
    shape_5d.append(2)

    var scalar = full(shape_scalar, 10.0, DType.float32)
    var t5d = ones(shape_5d, DType.float32)
    var result = multiply(scalar, t5d)

    assert_dim(result, 5, "Result should be 5D")
    assert_all_values(result, 10.0, 1e-5, "Broadcast scalar correctly")


fn test_6d_broadcasting_1d() raises:
    """Broadcasting 1D to 6D tensor."""
    var shape_1d = List[Int]()
    shape_1d.append(2)
    var shape_6d = List[Int]()
    shape_6d.append(2)
    shape_6d.append(3)
    shape_6d.append(2)
    shape_6d.append(2)
    shape_6d.append(2)
    shape_6d.append(2)

    var t1d = full(shape_1d, 2.0, DType.float32)
    var t6d = ones(shape_6d, DType.float32)
    var result = multiply(t1d, t6d)

    assert_dim(result, 6, "Result should be 6D")
    assert_all_values(result, 2.0, 1e-5, "Broadcast 1D to 6D correctly")


# ============================================================================
# Test large tensor operations
# ============================================================================


fn test_large_1d_tensor() raises:
    """Create and operate on large 1D tensor (50 million elements)."""
    var shape = List[Int]()
    shape.append(50000000)
    var t = zeros(shape, DType.float32)

    assert_numel(t, 50000000, "Should create 50M element tensor")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run Part 2 high-dimensional tensor tests (6D/7D, broadcasting, large)."""
    print(
        "Running high-dimensional tensor tests (Part 2: 6D/7D, broadcasting,"
        " large)..."
    )

    print("  Testing 6D tensor operations...")
    test_6d_tensor_creation()
    test_6d_tensor_arithmetic()
    test_6d_tensor_reduction()

    print("  Testing 7D tensor operations...")
    test_7d_tensor_creation()
    test_7d_tensor_sum()

    print("  Testing broadcasting with high dimensions...")
    test_5d_broadcasting_scalar()
    test_6d_broadcasting_1d()

    print("  Testing large tensor operations...")
    test_large_1d_tensor()

    print("All Part 2 high-dimensional tensor tests completed!")
