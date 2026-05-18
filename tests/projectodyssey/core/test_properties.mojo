"""Tests for AnyTensor shape and dtype properties.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""


from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones, full, arange, eye
from tests.projectodyssey.conftest import (
    assert_true,
    assert_false,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_equal_int,
    assert_value_at,
)


def test_shape_1d() raises:
    """Test shape property for 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 1, "1D tensor should have 1 dimension in shape")
    assert_equal_int(s[0], 10, "First dimension should be 10")


def test_shape_2d() raises:
    """Test shape property for 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 2, "2D tensor should have 2 dimensions")
    assert_equal_int(s[0], 3, "First dimension should be 3")
    assert_equal_int(s[1], 4, "Second dimension should be 4")


def test_shape_3d() raises:
    """Test shape property for 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 3, "3D tensor should have 3 dimensions")
    assert_equal_int(s[0], 2, "Dim 0 should be 2")
    assert_equal_int(s[1], 3, "Dim 1 should be 3")
    assert_equal_int(s[2], 4, "Dim 2 should be 4")


def test_shape_scalar() raises:
    """Test shape property for scalar (0D) tensor."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)

    var s = t.shape()
    assert_equal_int(len(s), 0, "Scalar tensor should have 0 dimensions")


def test_dtype_float32() raises:
    """Test dtype property for float32 tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float32)

    assert_dtype(t, DType.float32, "Should be float32")


def test_dtype_float64() raises:
    """Test dtype property for float64 tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = ones(shape, DType.float64)

    assert_dtype(t, DType.float64, "Should be float64")


def test_dtype_int32() raises:
    """Test dtype property for int32 tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.int32)

    assert_dtype(t, DType.int32, "Should be int32")


def test_dtype_int64() raises:
    """Test dtype property for int64 tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.int64)

    assert_dtype(t, DType.int64, "Should be int64")


def test_dtype_bool() raises:
    """Test dtype property for bool tensor."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.bool)

    assert_dtype(t, DType.bool, "Should be bool")


def test_numel_1d() raises:
    """Test numel for 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    assert_numel(t, 10, "1D tensor with 10 elements")


def test_numel_2d() raises:
    """Test numel for 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_numel(t, 12, "2D tensor with 12 elements (3*4)")


def test_numel_3d() raises:
    """Test numel for 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_numel(t, 24, "3D tensor with 24 elements (2*3*4)")


def test_numel_scalar() raises:
    """Test numel for scalar tensor."""
    var shape = List[Int]()
    var t = full(shape, 1.0, DType.float32)

    assert_numel(t, 1, "Scalar tensor has 1 element")


def test_numel_empty() raises:
    """Test numel for empty tensor."""
    var shape = List[Int]()
    shape.append(0)
    var t = zeros(shape, DType.float32)

    assert_numel(t, 0, "Empty tensor has 0 elements")


def test_strides_1d() raises:
    """Test stride calculation for 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    var strides = t._strides.copy()
    assert_equal_int(len(strides), 1, "1D tensor should have 1 stride")
    assert_equal_int(strides[0], 1, "1D stride should be 1")


def test_strides_2d_row_major() raises:
    """Test stride calculation for 2D tensor (row-major)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var strides = t._strides.copy()
    assert_equal_int(len(strides), 2, "2D tensor should have 2 strides")
    assert_equal_int(strides[0], 4, "Outer stride should be 4 (row length)")
    assert_equal_int(strides[1], 1, "Inner stride should be 1")


def test_strides_3d_row_major() raises:
    """Test stride calculation for 3D tensor (row-major)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var strides = t._strides.copy()
    assert_equal_int(len(strides), 3, "3D tensor should have 3 strides")
    assert_equal_int(strides[0], 12, "Stride 0 should be 12 (3*4)")
    assert_equal_int(strides[1], 4, "Stride 1 should be 4")
    assert_equal_int(strides[2], 1, "Stride 2 should be 1")


def test_contiguous_new_tensor() raises:
    """Test that newly created tensors are contiguous."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_true(t.is_contiguous(), "Newly created tensor should be contiguous")


def test_contiguous_1d() raises:
    """Test that 1D tensors are contiguous."""
    var shape = List[Int]()
    shape.append(100)
    var t = arange(0.0, 100.0, 1.0, DType.float32)

    assert_true(t.is_contiguous(), "1D tensor should be contiguous")


def test_contiguous_scalar() raises:
    """Test that scalar tensors are contiguous."""
    var shape = List[Int]()
    var t = full(shape, 5.0, DType.float32)

    assert_true(t.is_contiguous(), "Scalar tensor should be contiguous")


def test_dim_1d() raises:
    """Test dim for 1D tensor."""
    var shape = List[Int]()
    shape.append(10)
    var t = ones(shape, DType.float32)

    assert_dim(t, 1, "1D tensor should have dim=1")


def test_dim_2d() raises:
    """Test dim for 2D tensor."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_dim(t, 2, "2D tensor should have dim=2")


def test_dim_3d() raises:
    """Test dim for 3D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_dim(t, 3, "3D tensor should have dim=3")


def test_dim_scalar() raises:
    """Test dim for scalar (0D) tensor."""
    var shape = List[Int]()
    var t = full(shape, 1.0, DType.float32)

    assert_dim(t, 0, "Scalar tensor should have dim=0")


def test_value_access_1d() raises:
    """Test accessing values in 1D tensor."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)

    assert_value_at(t, 0, 0.0, 1e-6, "First element")
    assert_value_at(t, 2, 2.0, 1e-6, "Middle element")
    assert_value_at(t, 4, 4.0, 1e-6, "Last element")


def test_value_access_2d_row_major() raises:
    """Test accessing values in 2D tensor (row-major order)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = arange(0.0, 6.0, 1.0, DType.float32)
    # Should be: [[0, 1, 2], [3, 4, 5]]

    assert_value_at(t, 0, 0.0, 1e-6, "Element [0,0]")
    assert_value_at(t, 2, 2.0, 1e-6, "Element [0,2]")
    assert_value_at(t, 3, 3.0, 1e-6, "Element [1,0]")
    assert_value_at(t, 5, 5.0, 1e-6, "Element [1,2]")


def test_value_access_identity() raises:
    """Test accessing values in identity matrix."""
    var t = eye(3, 3, 0, DType.float32)

    # Diagonal elements should be 1.0
    assert_value_at(t, 0, 1.0, 1e-6, "Diagonal [0,0]")
    assert_value_at(t, 4, 1.0, 1e-6, "Diagonal [1,1]")
    assert_value_at(t, 8, 1.0, 1e-6, "Diagonal [2,2]")

    # Off-diagonal should be 0.0
    assert_value_at(t, 1, 0.0, 1e-6, "Off-diagonal [0,1]")
    assert_value_at(t, 3, 0.0, 1e-6, "Off-diagonal [1,0]")


def test_all_zeros_pattern() raises:
    """Test that zeros creates all zero values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var t = zeros(shape, DType.float32)

    for i in range(9):
        assert_value_at(t, i, 0.0, 1e-8, "All elements should be 0")


def test_all_ones_pattern() raises:
    """Test that ones creates all one values."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var t = ones(shape, DType.float32)

    for i in range(9):
        assert_value_at(t, i, 1.0, 1e-8, "All elements should be 1")


def test_full_pattern() raises:
    """Test that full creates uniform values."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(4)
    var t = full(shape, 7.5, DType.float32)

    for i in range(8):
        assert_value_at(t, i, 7.5, 1e-6, "All elements should be 7.5")


def test_arange_sequential_pattern() raises:
    """Test that arange creates sequential values."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    for i in range(10):
        assert_value_at(t, i, Float64(i), 1e-6, "Sequential values")


def test_eye_identity_pattern() raises:
    """Test that eye creates proper identity pattern."""
    var t = eye(4, 4, 0, DType.float32)

    for i in range(4):
        for j in range(4):
            var idx = i * 4 + j
            if i == j:
                assert_value_at(t, idx, 1.0, 1e-6, "Diagonal should be 1")
            else:
                assert_value_at(t, idx, 0.0, 1e-6, "Off-diagonal should be 0")


def test_is_view_false_for_new_tensors() raises:
    """Test that newly created tensors are not views."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    assert_false(t._is_view, "Newly created tensor should not be a view")


def test_dtype_size_float32() raises:
    """Test dtype size for float32."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float32)

    var size = t._get_dtype_size()
    assert_equal_int(size, 4, "float32 should be 4 bytes")


def test_dtype_size_float64() raises:
    """Test dtype size for float64."""
    var shape = List[Int]()
    shape.append(1)
    var t = ones(shape, DType.float64)

    var size = t._get_dtype_size()
    assert_equal_int(size, 8, "float64 should be 8 bytes")


def test_dtype_size_int32() raises:
    """Test dtype size for int32."""
    var shape = List[Int]()
    shape.append(1)
    var t = zeros(shape, DType.int32)

    var size = t._get_dtype_size()
    assert_equal_int(size, 4, "int32 should be 4 bytes")


def main() raises:
    """Run all test_properties tests."""
    print("Running test_properties tests...")

    test_shape_1d()
    print("✓ test_shape_1d")

    test_shape_2d()
    print("✓ test_shape_2d")

    test_shape_3d()
    print("✓ test_shape_3d")

    test_shape_scalar()
    print("✓ test_shape_scalar")

    test_dtype_float32()
    print("✓ test_dtype_float32")

    test_dtype_float64()
    print("✓ test_dtype_float64")

    test_dtype_int32()
    print("✓ test_dtype_int32")

    test_dtype_int64()
    print("✓ test_dtype_int64")

    test_dtype_bool()
    print("✓ test_dtype_bool")

    test_numel_1d()
    print("✓ test_numel_1d")

    test_numel_2d()
    print("✓ test_numel_2d")

    test_numel_3d()
    print("✓ test_numel_3d")

    test_numel_scalar()
    print("✓ test_numel_scalar")

    test_numel_empty()
    print("✓ test_numel_empty")

    test_strides_1d()
    print("✓ test_strides_1d")

    test_strides_2d_row_major()
    print("✓ test_strides_2d_row_major")

    test_strides_3d_row_major()
    print("✓ test_strides_3d_row_major")

    test_contiguous_new_tensor()
    print("✓ test_contiguous_new_tensor")

    test_contiguous_1d()
    print("✓ test_contiguous_1d")

    test_contiguous_scalar()
    print("✓ test_contiguous_scalar")

    test_dim_1d()
    print("✓ test_dim_1d")

    test_dim_2d()
    print("✓ test_dim_2d")

    test_dim_3d()
    print("✓ test_dim_3d")

    test_dim_scalar()
    print("✓ test_dim_scalar")

    test_value_access_1d()
    print("✓ test_value_access_1d")

    test_value_access_2d_row_major()
    print("✓ test_value_access_2d_row_major")

    test_value_access_identity()
    print("✓ test_value_access_identity")

    test_all_zeros_pattern()
    print("✓ test_all_zeros_pattern")

    test_all_ones_pattern()
    print("✓ test_all_ones_pattern")

    test_full_pattern()
    print("✓ test_full_pattern")

    test_arange_sequential_pattern()
    print("✓ test_arange_sequential_pattern")

    test_eye_identity_pattern()
    print("✓ test_eye_identity_pattern")

    test_is_view_false_for_new_tensors()
    print("✓ test_is_view_false_for_new_tensors")

    test_dtype_size_float32()
    print("✓ test_dtype_size_float32")

    test_dtype_size_float64()
    print("✓ test_dtype_size_float64")

    test_dtype_size_int32()
    print("✓ test_dtype_size_int32")

    print("\nAll test_properties tests passed!")
