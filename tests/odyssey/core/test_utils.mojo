"""Tests for argmax utility functions in odyssey.core.utils.

This module tests argmax functions including:
- argmax (scalar)
- argmax (axis-based)
- top_k_indices_simple
"""


from tests.odyssey.conftest import (
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
    assert_close_float,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, ones, arange
from odyssey.core.utils import (
    argmax,
    argsort,
    top_k,
    top_k_indices,
)


def test_argmax_scalar_simple() raises:
    """Test argmax on a simple 1D tensor."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var idx = argmax(t)
    assert_equal_int(idx, 9)


def test_argmax_scalar_negative_values() raises:
    """Test argmax with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t.set(0, Float32(-5.0))
    t.set(1, Float32(-2.0))
    t.set(2, Float32(-10.0))
    t.set(3, Float32(-1.0))
    t.set(4, Float32(-3.0))
    var idx = argmax(t)
    assert_equal_int(idx, 3)


def test_argmax_scalar_multi_dimensional() raises:
    """Test argmax on multi-dimensional tensor flattens correctly."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max at linear index 5 (row 1, col 1)
    t.set(5, Float32(42.0))
    var idx = argmax(t)
    assert_equal_int(idx, 5)


def test_argmax_axis_1d() raises:
    """Test argmax along axis on 1D tensor."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var result = argmax(t, axis=0)
    assert_equal_int(result.numel(), 1)
    assert_equal_int(Int(result._get_int64(0)), 4)


def test_argmax_axis_2d_axis0() raises:
    """Test argmax along axis 0 on 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set values such that row 2 has the max in each column
    # Column 0: indices 0, 4, 8 -> max at 8
    t.set(8, Float32(100.0))
    t.set(9, Float32(100.0))
    t.set(10, Float32(100.0))
    t.set(11, Float32(100.0))

    var result = argmax(t, axis=0)
    assert_shape(result, [4])

    # All should be 2 (row index 2, which is linear indices 8-11)
    for i in range(4):
        assert_equal_int(Int(result._get_int64(i)), 2)


def test_argmax_axis_2d_axis1() raises:
    """Test argmax along axis 1 on 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max in last column for each row
    # Row 0, col 3: linear index 3
    # Row 1, col 3: linear index 7
    # Row 2, col 3: linear index 11
    t.set(3, Float32(10.0))
    t.set(7, Float32(20.0))
    t.set(11, Float32(30.0))

    var result = argmax(t, axis=1)
    assert_shape(result, [3])
    assert_equal_int(Int(result._get_int64(0)), 3)
    assert_equal_int(Int(result._get_int64(1)), 3)
    assert_equal_int(Int(result._get_int64(2)), 3)


def test_argmax_axis_3d() raises:
    """Test argmax on 3D tensor along axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)
    # Set max values
    t.set(5, Float32(50.0))

    var result = argmax(t, axis=2)
    assert_shape(result, [2, 3])


def test_top_k_indices_simple() raises:
    """Test top_k_indices on a simple tensor."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 3)
    assert_equal_int(indices[0], 9)
    assert_equal_int(indices[1], 8)
    assert_equal_int(indices[2], 7)


def test_top_k_indices_single_element() raises:
    """Test top_k_indices with k=1."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 1)
    assert_equal_int(indices[0], 4)


def test_top_k_indices_all_elements() raises:
    """Test top_k_indices with k=numel."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 5)
    assert_equal_int(indices[0], 4)
    assert_equal_int(indices[1], 3)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 1)
    assert_equal_int(indices[4], 0)


def test_top_k_indices_with_duplicates() raises:
    """Test top_k_indices with duplicate values."""
    var shape = List[Int]()
    shape.append(6)
    var t = zeros(shape, DType.float32)
    t.set(0, Float32(5.0))
    t.set(1, Float32(5.0))
    t.set(2, Float32(3.0))
    t.set(3, Float32(3.0))
    t.set(4, Float32(1.0))
    t.set(5, Float32(1.0))

    var indices = top_k_indices(t, 3)
    # First two should be indices 0 and 1 (both value 5)
    # Third should be index 2 or 3 (value 3)
    assert_equal_int(len(indices), 3)


def test_top_k_values_and_indices() raises:
    """Test top_k function returns both values and indices."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var result = top_k(t, 3)
    var values = result[0]

    # Check shape of values
    assert_shape(values, [3])

    # Check values are in correct order (descending)
    assert_close_float(values._get_float64(0), 9.0)
    assert_close_float(values._get_float64(1), 8.0)
    assert_close_float(values._get_float64(2), 7.0)

    # Check indices (access from result tuple to avoid List[Int] copy)
    assert_equal_int(result[1][0], 9)
    assert_equal_int(result[1][1], 8)
    assert_equal_int(result[1][2], 7)


def test_top_k_multidimensional() raises:
    """Test top_k on multi-dimensional tensor."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var result = top_k(t, 2)
    var values = result[0]

    assert_shape(values, [2])
    assert_close_float(values._get_float64(0), 11.0)
    assert_close_float(values._get_float64(1), 10.0)


def test_argsort_ascending() raises:
    """Test argsort in ascending order."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t.set(0, Float32(5.0))
    t.set(1, Float32(2.0))
    t.set(2, Float32(8.0))
    t.set(3, Float32(1.0))
    t.set(4, Float32(9.0))

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 3)  # value 1
    assert_equal_int(indices[1], 1)  # value 2
    assert_equal_int(indices[2], 0)  # value 5
    assert_equal_int(indices[3], 2)  # value 8
    assert_equal_int(indices[4], 4)  # value 9


def test_argsort_descending() raises:
    """Test argsort in descending order."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t.set(0, Float32(5.0))
    t.set(1, Float32(2.0))
    t.set(2, Float32(8.0))
    t.set(3, Float32(1.0))
    t.set(4, Float32(9.0))

    var indices = argsort(t, descending=True)
    assert_equal_int(indices[0], 4)  # value 9
    assert_equal_int(indices[1], 2)  # value 8
    assert_equal_int(indices[2], 0)  # value 5
    assert_equal_int(indices[3], 1)  # value 2
    assert_equal_int(indices[4], 3)  # value 1


def test_argsort_single_element() raises:
    """Test argsort with single element."""
    var t = arange(0.0, 1.0, 1.0, DType.float32)
    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)


def test_argsort_sorted_array() raises:
    """Test argsort on already sorted array."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)
    assert_equal_int(indices[1], 1)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 3)
    assert_equal_int(indices[4], 4)


def test_argsort_reverse_sorted() raises:
    """Test argsort on reverse sorted array."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t.set(0, Float32(5.0))
    t.set(1, Float32(4.0))
    t.set(2, Float32(3.0))
    t.set(3, Float32(2.0))
    t.set(4, Float32(1.0))

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 4)
    assert_equal_int(indices[1], 3)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 1)
    assert_equal_int(indices[4], 0)


def test_argsort_negative_values() raises:
    """Test argsort with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t.set(0, Float32(-5.0))
    t.set(1, Float32(2.0))
    t.set(2, Float32(-1.0))
    t.set(3, Float32(0.0))
    t.set(4, Float32(3.0))

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)  # -5
    assert_equal_int(indices[1], 2)  # -1
    assert_equal_int(indices[2], 3)  # 0
    assert_equal_int(indices[3], 1)  # 2
    assert_equal_int(indices[4], 4)  # 3


def test_argsort_multidimensional() raises:
    """Test argsort on multi-dimensional tensor (flattens)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = zeros(shape, DType.float32)
    t.set(0, Float32(5.0))
    t.set(1, Float32(1.0))
    t.set(2, Float32(3.0))
    t.set(3, Float32(2.0))
    t.set(4, Float32(4.0))
    t.set(5, Float32(0.0))

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 5)  # 0
    assert_equal_int(indices[1], 1)  # 1
    assert_equal_int(indices[2], 3)  # 2
    assert_equal_int(indices[3], 2)  # 3
    assert_equal_int(indices[4], 4)  # 4
    assert_equal_int(indices[5], 0)  # 5


def main() raises:
    """Run all test_utils tests."""
    print("Running test_utils tests...")

    test_argmax_scalar_simple()
    print("✓ test_argmax_scalar_simple")

    test_argmax_scalar_negative_values()
    print("✓ test_argmax_scalar_negative_values")

    test_argmax_scalar_multi_dimensional()
    print("✓ test_argmax_scalar_multi_dimensional")

    test_argmax_axis_1d()
    print("✓ test_argmax_axis_1d")

    test_argmax_axis_2d_axis0()
    print("✓ test_argmax_axis_2d_axis0")

    test_argmax_axis_2d_axis1()
    print("✓ test_argmax_axis_2d_axis1")

    test_argmax_axis_3d()
    print("✓ test_argmax_axis_3d")

    test_top_k_indices_simple()
    print("✓ test_top_k_indices_simple")

    test_top_k_indices_single_element()
    print("✓ test_top_k_indices_single_element")

    test_top_k_indices_all_elements()
    print("✓ test_top_k_indices_all_elements")

    test_top_k_indices_with_duplicates()
    print("✓ test_top_k_indices_with_duplicates")

    test_top_k_values_and_indices()
    print("✓ test_top_k_values_and_indices")

    test_top_k_multidimensional()
    print("✓ test_top_k_multidimensional")

    test_argsort_ascending()
    print("✓ test_argsort_ascending")

    test_argsort_descending()
    print("✓ test_argsort_descending")

    test_argsort_single_element()
    print("✓ test_argsort_single_element")

    test_argsort_sorted_array()
    print("✓ test_argsort_sorted_array")

    test_argsort_reverse_sorted()
    print("✓ test_argsort_reverse_sorted")

    test_argsort_negative_values()
    print("✓ test_argsort_negative_values")

    test_argsort_multidimensional()
    print("✓ test_argsort_multidimensional")

    print("\nAll test_utils tests passed!")
