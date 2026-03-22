"""Tests for high-dimensional tensor operations (Part 1: 5D tensors).

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_high_dimensional.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests 5D tensor creation and arithmetic operations.
"""

# Import AnyTensor and operations
from shared.core.extensor import AnyTensor, zeros, ones, full, arange
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
# Test 5D tensor operations
# ============================================================================


fn test_5d_tensor_creation() raises:
    """Create and verify 5D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)

    assert_dim(t, 5, "Tensor should be 5D")
    assert_numel(t, 32, "2^5 = 32 elements")


fn test_5d_tensor_arithmetic() raises:
    """Arithmetic operations on 5D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var result = add(a, b)

    assert_dim(result, 5, "Result should be 5D")
    assert_all_values(result, 3.0, 1e-5, "1 + 2 = 3 for all elements")


fn test_5d_tensor_multiply() raises:
    """Multiplication on 5D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 4.0, DType.float32)
    var result = multiply(a, b)

    assert_all_values(result, 12.0, 1e-5, "3 * 4 = 12 for all elements")


fn test_5d_tensor_subtract() raises:
    """Subtraction on 5D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var a = full(shape, 10.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var result = subtract(a, b)

    assert_all_values(result, 7.0, 1e-5, "10 - 3 = 7 for all elements")


fn test_5d_tensor_reduction() raises:
    """Reduction of 5D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    assert_dim(result, 0, "Result should be scalar")
    assert_value_at(result, 0, 32.0, 1e-5, "Sum of 32 ones = 32")


fn test_5d_tensor_mean() raises:
    """Mean of 5D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = full(shape, 5.0, DType.float32)
    var result = mean(t)

    assert_dim(result, 0, "Result should be scalar")
    assert_value_at(result, 0, 5.0, 1e-5, "Mean of constant tensor")


fn test_5d_float64() raises:
    """5D tensor with float64."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float64)

    assert_dtype(t, DType.float64, "Tensor should be float64")
    var result = sum(t)
    assert_value_at(result, 0, 32.0, 1e-10, "Float64 precision maintained")


fn test_5d_int32() raises:
    """5D tensor with int32."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = full(shape, 5.0, DType.int32)

    assert_dtype(t, DType.int32, "Tensor should be int32")
    var result = sum(t)
    assert_value_at(result, 0, 160.0, 1e-6, "Int32 arithmetic correct")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run Part 1 high-dimensional tensor tests (5D tensors)."""
    print("Running high-dimensional tensor tests (Part 1: 5D tensors)...")

    print("  Testing 5D tensor operations...")
    test_5d_tensor_creation()
    test_5d_tensor_arithmetic()
    test_5d_tensor_multiply()
    test_5d_tensor_subtract()
    test_5d_tensor_reduction()
    test_5d_tensor_mean()

    print("  Testing different dtypes...")
    test_5d_float64()
    test_5d_int32()

    print("All Part 1 high-dimensional tensor tests completed!")
