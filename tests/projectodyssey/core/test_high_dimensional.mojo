"""Tests for high-dimensional tensor operations (Part 1: 5D tensors).

Tests 5D tensor creation and arithmetic operations.
"""


from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones, full, arange
from projectodyssey.core.arithmetic import add, multiply, subtract
from projectodyssey.core.reduction import sum, mean, max_reduce, min_reduce
from tests.projectodyssey.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


def test_5d_tensor_creation() raises:
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


def test_5d_tensor_arithmetic() raises:
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


def test_5d_tensor_multiply() raises:
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


def test_5d_tensor_subtract() raises:
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


def test_5d_tensor_reduction() raises:
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


def test_5d_tensor_mean() raises:
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


def test_5d_float64() raises:
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


def test_5d_int32() raises:
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


def test_6d_tensor_creation() raises:
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


def test_6d_tensor_arithmetic() raises:
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


def test_6d_tensor_reduction() raises:
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


def test_7d_tensor_creation() raises:
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


def test_7d_tensor_sum() raises:
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


def test_5d_broadcasting_scalar() raises:
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


def test_6d_broadcasting_1d() raises:
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


def test_large_1d_tensor() raises:
    """Create and operate on large 1D tensor (50 million elements)."""
    var shape = List[Int]()
    shape.append(50000000)
    var t = zeros(shape, DType.float32)

    assert_numel(t, 50000000, "Should create 50M element tensor")


def test_large_multidimensional() raises:
    """Large multidimensional tensor [100, 100, 100, 100]."""
    var shape = List[Int]()
    shape.append(100)
    shape.append(100)
    shape.append(100)
    shape.append(100)
    var t = zeros(shape, DType.float32)

    assert_dim(t, 4, "Tensor should be 4D")
    assert_numel(t, 100000000, "100^4 = 100M elements")


def test_5d_sum_precision() raises:
    """High-dimensional sum should maintain precision."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    shape.append(3)
    shape.append(3)
    shape.append(3)
    var t = full(shape, 1.0, DType.float32)
    var result = sum(t)

    # 3^5 = 243
    assert_value_at(result, 0, 243.0, 1e-3, "Sum preserves precision")


def test_5d_mean_precision() raises:
    """High-dimensional mean should maintain precision."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = full(shape, 7.5, DType.float32)
    var result = mean(t)

    assert_value_at(result, 0, 7.5, 1e-5, "Mean of constant tensor")


def test_6d_max_reduce() raises:
    """Max reduction on 6D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = full(shape, 5.0, DType.float32)
    var result = max_reduce(t)

    assert_value_at(result, 0, 5.0, 1e-6, "Max of constant tensor")


def test_6d_min_reduce() raises:
    """Min reduction on 6D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = full(shape, 5.0, DType.float32)
    var result = min_reduce(t)

    assert_value_at(result, 0, 5.0, 1e-6, "Min of constant tensor")


def test_5d_accumulation() raises:
    """Accumulation in 5D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    # Create tensor with values 1.0 each
    var t = ones(shape, DType.float32)

    # Sum should accumulate correctly
    var result = sum(t)
    assert_value_at(result, 0, 32.0, 1e-4, "Accumulation preserves precision")


def test_6d_mixed_arithmetic() raises:
    """Mixed arithmetic operations on 6D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    # a + b = 3
    var result1 = add(a, b)
    # result1 * 2 = 6
    var result2 = multiply(result1, full(shape, 2.0, DType.float32))

    assert_all_values(result2, 6.0, 1e-5, "Mixed arithmetic correct")


def main() raises:
    """Run all test_high_dimensional tests."""
    print("Running test_high_dimensional tests...")

    test_5d_tensor_creation()
    print("✓ test_5d_tensor_creation")

    test_5d_tensor_arithmetic()
    print("✓ test_5d_tensor_arithmetic")

    test_5d_tensor_multiply()
    print("✓ test_5d_tensor_multiply")

    test_5d_tensor_subtract()
    print("✓ test_5d_tensor_subtract")

    test_5d_tensor_reduction()
    print("✓ test_5d_tensor_reduction")

    test_5d_tensor_mean()
    print("✓ test_5d_tensor_mean")

    test_5d_float64()
    print("✓ test_5d_float64")

    test_5d_int32()
    print("✓ test_5d_int32")

    test_6d_tensor_creation()
    print("✓ test_6d_tensor_creation")

    test_6d_tensor_arithmetic()
    print("✓ test_6d_tensor_arithmetic")

    test_6d_tensor_reduction()
    print("✓ test_6d_tensor_reduction")

    test_7d_tensor_creation()
    print("✓ test_7d_tensor_creation")

    test_7d_tensor_sum()
    print("✓ test_7d_tensor_sum")

    test_5d_broadcasting_scalar()
    print("✓ test_5d_broadcasting_scalar")

    test_6d_broadcasting_1d()
    print("✓ test_6d_broadcasting_1d")

    test_large_1d_tensor()
    print("✓ test_large_1d_tensor")

    test_large_multidimensional()
    print("✓ test_large_multidimensional")

    test_5d_sum_precision()
    print("✓ test_5d_sum_precision")

    test_5d_mean_precision()
    print("✓ test_5d_mean_precision")

    test_6d_max_reduce()
    print("✓ test_6d_max_reduce")

    test_6d_min_reduce()
    print("✓ test_6d_min_reduce")

    test_5d_accumulation()
    print("✓ test_5d_accumulation")

    test_6d_mixed_arithmetic()
    print("✓ test_6d_mixed_arithmetic")

    print("\nAll test_high_dimensional tests passed!")
