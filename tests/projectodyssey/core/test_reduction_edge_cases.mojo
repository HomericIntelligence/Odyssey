"""Tests for reduction operation edge cases.

Tests edge cases for sum, mean, max_reduce, min_reduce operations including:
- Empty tensor reductions
- Scalar tensor reductions
- Single-element tensor reductions
- Reductions with all same values

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""


from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    arange,
)
from projectodyssey.core.reduction import sum, mean, max_reduce, min_reduce
from tests.projectodyssey.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_true,
)


def test_sum_empty_tensor() raises:
    """Sum of empty tensor should return 0 (identity element)."""
    var shape = List[Int]()
    shape.append(0)
    var t = zeros(shape, DType.float32)
    var result = sum(t)

    assert_numel(result, 1, "Sum of empty should return scalar")
    assert_value_at(result, 0, 0.0, 1e-6, "Sum of empty should be 0")


def test_mean_empty_tensor() raises:
    """Mean of empty tensor should handle division by 0."""
    var shape = List[Int]()
    shape.append(0)
    var t = zeros(shape, DType.float32)
    var result = mean(t)

    # Mean of empty: 0/0 = NaN or special handling
    # Verify it doesn't crash
    assert_dim(result, 0, "Mean of empty should return scalar")


def test_sum_scalar_tensor() raises:
    """Sum of 0D scalar tensor should return the scalar itself."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)
    var result = sum(t)

    assert_numel(result, 1, "Sum of scalar should return 1 element")
    assert_value_at(
        result, 0, 42.0, 1e-6, "Sum of scalar should equal the scalar"
    )


def test_mean_scalar_tensor() raises:
    """Mean of 0D scalar tensor should return the scalar itself."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)
    var result = mean(t)

    assert_numel(result, 1, "Mean of scalar should return 1 element")
    assert_value_at(
        result, 0, 42.0, 1e-6, "Mean of scalar should equal the scalar"
    )


def test_sum_single_element() raises:
    """Sum of [x] should equal x."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 7.5, DType.float32)
    var result = sum(t)

    assert_value_at(result, 0, 7.5, 1e-6, "Sum of [7.5] should be 7.5")


def test_mean_single_element() raises:
    """Mean of [x] should equal x."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 7.5, DType.float32)
    var result = mean(t)

    assert_value_at(result, 0, 7.5, 1e-6, "Mean of [7.5] should be 7.5")


def test_sum_all_ones() raises:
    """Sum of tensor with all ones."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    assert_value_at(result, 0, 10.0, 1e-5, "Sum of 10 ones should be 10")


def test_mean_all_zeros() raises:
    """Mean of all zeros should be 0."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    var result = mean(t)

    assert_value_at(result, 0, 0.0, 1e-8, "Mean of zeros should be 0")


def test_mean_all_same() raises:
    """Mean of tensor with all same values."""
    var shape = List[Int]()
    shape.append(10)
    var t = full(shape, 3.14159, DType.float32)
    var result = mean(t)

    assert_value_at(result, 0, 3.14159, 1e-5, "Mean of constant tensor")


def test_sum_2d() raises:
    """Sum of 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = full(shape, 2.0, DType.float32)
    var result = sum(t)

    # Sum of all 2.0 in 3x4 = 24.0
    assert_value_at(result, 0, 24.0, 1e-5, "Sum of 3x4 tensor of 2.0")


def test_mean_2d() raises:
    """Mean of 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = full(shape, 8.0, DType.float32)
    var result = mean(t)

    # Mean of all 8.0 = 8.0
    assert_value_at(result, 0, 8.0, 1e-5, "Mean of constant 2D tensor")


def test_sum_3d_tensor() raises:
    """Sum reduction on 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    # Sum of all ones in 2x3x4 = 24
    assert_value_at(result, 0, 24.0, 1e-5, "Sum of 3D ones")


def test_mean_3d_tensor() raises:
    """Mean reduction on 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = full(shape, 5.0, DType.float32)
    var result = mean(t)

    # Mean of all 5.0 = 5.0
    assert_value_at(result, 0, 5.0, 1e-5, "Mean of 3D constant tensor")


def test_sum_4d_tensor() raises:
    """Sum reduction on 4D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    # Sum of all ones = 2^4 = 16
    assert_value_at(result, 0, 16.0, 1e-5, "Sum of 4D ones")


def test_sum_5d_tensor() raises:
    """Sum reduction on 5D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var t = ones(shape, DType.float32)
    var result = sum(t)

    # Sum of all ones = 2^5 = 32
    assert_value_at(result, 0, 32.0, 1e-5, "Sum of 5D ones")


def main() raises:
    """Run all test_reduction_edge_cases tests."""
    print("Running test_reduction_edge_cases tests...")

    test_sum_empty_tensor()
    print("✓ test_sum_empty_tensor")

    test_mean_empty_tensor()
    print("✓ test_mean_empty_tensor")

    test_sum_scalar_tensor()
    print("✓ test_sum_scalar_tensor")

    test_mean_scalar_tensor()
    print("✓ test_mean_scalar_tensor")

    test_sum_single_element()
    print("✓ test_sum_single_element")

    test_mean_single_element()
    print("✓ test_mean_single_element")

    test_sum_all_ones()
    print("✓ test_sum_all_ones")

    test_mean_all_zeros()
    print("✓ test_mean_all_zeros")

    test_mean_all_same()
    print("✓ test_mean_all_same")

    test_sum_2d()
    print("✓ test_sum_2d")

    test_mean_2d()
    print("✓ test_mean_2d")

    test_sum_3d_tensor()
    print("✓ test_sum_3d_tensor")

    test_mean_3d_tensor()
    print("✓ test_mean_3d_tensor")

    test_sum_4d_tensor()
    print("✓ test_sum_4d_tensor")

    test_sum_5d_tensor()
    print("✓ test_sum_5d_tensor")

    print("\nAll test_reduction_edge_cases tests passed!")
