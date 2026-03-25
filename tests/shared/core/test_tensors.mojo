"""Tests for tensor creation operations

Tests cover:
- Tensor creation: zeros, ones, full, empty, arange, eye, linspace

All tests use pure functional API.
"""


from tests.shared.conftest import (
    assert_true,
    assert_equal_int,
    assert_close_float,
    assert_equal,
    assert_almost_equal,
    assert_dtype_equal,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    empty,
    arange,
    eye,
    linspace,
    zeros_like,
    ones_like,
    full_like,
)


fn test_zeros_creation() raises:
    """Test zeros tensor creation."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    # Check shape
    assert_equal_int(t.shape()[0], 3)
    assert_equal_int(t.shape()[1], 4)

    # Check all values are zero
    for i in range(12):
        assert_close_float(
            Float64(t._data.bitcast[Float32]()[i]), Float64(0.0), atol=1e-5
        )


fn test_ones_creation() raises:
    """Test ones tensor creation."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = ones(shape, DType.float32)

    # Check shape
    assert_equal(t.shape()[0], 2)
    assert_equal(t.shape()[1], 3)

    # Check all values are one
    for i in range(6):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


fn test_full_creation() raises:
    """Test full tensor creation with specified value."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var t = full(shape, 3.14, DType.float32)

    # Check shape
    assert_equal(t.shape()[0], 2)
    assert_equal(t.shape()[1], 2)

    # Check all values are 3.14
    for i in range(4):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(3.14), tolerance=1e-5
        )


fn test_empty_creation() raises:
    """Test empty tensor creation (uninitialized memory)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var t = empty(shape, DType.float32)

    # Check shape is correct (values are uninitialized)
    assert_equal(t.shape()[0], 3)
    assert_equal(t.shape()[1], 3)
    assert_equal(t.numel(), 9)


fn test_arange_creation() raises:
    """Test arange tensor creation (sequential values)."""
    var t = arange(0, 10, 1, DType.float32)

    # Check values: [0, 1, 2, ..., 9]
    assert_equal(t.numel(), 10)
    for i in range(10):
        assert_almost_equal(
            t._data.bitcast[Float32]()[i], Float32(i), tolerance=1e-5
        )


fn test_arange_with_step() raises:
    """Test arange with non-unit step."""
    var t = arange(0, 10, 2, DType.float32)

    # Check values: [0, 2, 4, 6, 8]
    assert_equal(t.numel(), 5)
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[2], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[3], Float32(6.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[4], Float32(8.0), tolerance=1e-5
    )


fn test_eye_creation() raises:
    """Test identity matrix creation."""
    var t = eye(3, 3, 0, DType.float32)

    # Check shape (3x3)
    assert_equal(t.shape()[0], 3)
    assert_equal(t.shape()[1], 3)

    # Check diagonal is 1, off-diagonal is 0
    for i in range(3):
        for j in range(3):
            var idx = i * 3 + j
            var val = t._data.bitcast[Float32]()[idx]
            if i == j:
                assert_almost_equal(val, Float32(1.0), tolerance=1e-5)
            else:
                assert_almost_equal(val, Float32(0.0), tolerance=1e-5)


fn test_linspace_creation() raises:
    """Test linspace tensor creation (evenly spaced values)."""
    var t = linspace(0.0, 1.0, 5, DType.float32)

    # Check values: [0.0, 0.25, 0.5, 0.75, 1.0]
    assert_equal(t.numel(), 5)
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(0.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[1], Float32(0.25), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[2], Float32(0.5), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[3], Float32(0.75), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[4], Float32(1.0), tolerance=1e-5
    )


fn test_zeros_like() raises:
    """Test zeros_like creates tensor with same shape and dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = full(shape, 5.0, DType.float32)
    var y = zeros_like(x)

    # Check shape matches
    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 3)

    # Check dtype matches
    assert_dtype_equal(y.dtype(), DType.float32)

    # Check all values are zero
    for i in range(6):
        assert_almost_equal(
            y._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-5
        )


fn test_ones_like() raises:
    """Test ones_like creates tensor with same shape and dtype."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    shape.append(2)
    var x = zeros(shape, DType.float64)
    var y = ones_like(x)

    # Check shape matches
    assert_equal(y.shape()[0], 2)
    assert_equal(y.shape()[1], 2)
    assert_equal(y.shape()[2], 2)

    # Check dtype matches
    assert_dtype_equal(y.dtype(), DType.float64)

    # Check all values are one
    for i in range(8):
        assert_almost_equal(y._data.bitcast[Float64]()[i], 1.0, tolerance=1e-10)


fn test_full_like() raises:
    """Test full_like creates tensor with same shape and dtype."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(2)
    var x = zeros(shape, DType.float32)
    var y = full_like(x, 7.5)

    # Check shape matches
    assert_equal(y.shape()[0], 3)
    assert_equal(y.shape()[1], 2)

    # Check dtype matches
    assert_dtype_equal(y.dtype(), DType.float32)

    # Check all values are 7.5
    for i in range(6):
        assert_almost_equal(
            y._data.bitcast[Float32]()[i], Float32(7.5), tolerance=1e-5
        )


fn test_tensor_shape() raises:
    """Test tensor shape property."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    var t_shape = t.shape()
    assert_equal(len(t_shape), 3)
    assert_equal(t_shape[0], 2)
    assert_equal(t_shape[1], 3)
    assert_equal(t_shape[2], 4)


fn test_tensor_dtype() raises:
    """Test tensor dtype property."""
    var shape = List[Int]()
    shape.append(5)

    var t_f32 = zeros(shape, DType.float32)
    assert_dtype_equal(t_f32.dtype(), DType.float32)

    var t_f64 = zeros(shape, DType.float64)
    assert_dtype_equal(t_f64.dtype(), DType.float64)

    var t_i32 = zeros(shape, DType.int32)
    assert_dtype_equal(t_i32.dtype(), DType.int32)


fn test_tensor_numel() raises:
    """Test tensor numel (number of elements)."""
    var shape1 = List[Int]()
    shape1.append(10)
    var t1 = zeros(shape1, DType.float32)
    assert_equal(t1.numel(), 10)

    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    var t2 = zeros(shape2, DType.float32)
    assert_equal(t2.numel(), 12)

    var shape3 = List[Int]()
    shape3.append(2)
    shape3.append(3)
    shape3.append(4)
    var t3 = zeros(shape3, DType.float32)
    assert_equal(t3.numel(), 24)


fn test_tensor_dim() raises:
    """Test tensor dim (number of dimensions)."""
    var shape1 = List[Int]()
    shape1.append(10)
    var t1 = zeros(shape1, DType.float32)
    assert_equal(t1.dim(), 1)

    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    var t2 = zeros(shape2, DType.float32)
    assert_equal(t2.dim(), 2)

    var shape3 = List[Int]()
    shape3.append(2)
    shape3.append(3)
    shape3.append(4)
    shape3.append(5)
    var t3 = zeros(shape3, DType.float32)
    assert_equal(t3.dim(), 4)


fn test_float_dtypes() raises:
    """Test tensor creation with different float dtypes."""
    var shape = List[Int]()
    shape.append(3)

    # float16
    var t_f16 = zeros(shape, DType.float16)
    assert_dtype_equal(t_f16.dtype(), DType.float16)
    assert_equal(t_f16.numel(), 3)

    # float32
    var t_f32 = zeros(shape, DType.float32)
    assert_dtype_equal(t_f32.dtype(), DType.float32)
    assert_equal(t_f32.numel(), 3)

    # float64
    var t_f64 = zeros(shape, DType.float64)
    assert_dtype_equal(t_f64.dtype(), DType.float64)
    assert_equal(t_f64.numel(), 3)


fn test_int_dtypes() raises:
    """Test tensor creation with different integer dtypes."""
    var shape = List[Int]()
    shape.append(3)

    # int8
    var t_i8 = zeros(shape, DType.int8)
    assert_dtype_equal(t_i8.dtype(), DType.int8)
    assert_equal(t_i8.numel(), 3)

    # int16
    var t_i16 = zeros(shape, DType.int16)
    assert_dtype_equal(t_i16.dtype(), DType.int16)
    assert_equal(t_i16.numel(), 3)

    # int32
    var t_i32 = zeros(shape, DType.int32)
    assert_dtype_equal(t_i32.dtype(), DType.int32)
    assert_equal(t_i32.numel(), 3)

    # int64
    var t_i64 = zeros(shape, DType.int64)
    assert_dtype_equal(t_i64.dtype(), DType.int64)
    assert_equal(t_i64.numel(), 3)


fn test_uint_dtypes() raises:
    """Test tensor creation with different unsigned integer dtypes."""
    var shape = List[Int]()
    shape.append(3)

    # uint8
    var t_u8 = zeros(shape, DType.uint8)
    assert_dtype_equal(t_u8.dtype(), DType.uint8)
    assert_equal(t_u8.numel(), 3)

    # uint16
    var t_u16 = zeros(shape, DType.uint16)
    assert_dtype_equal(t_u16.dtype(), DType.uint16)
    assert_equal(t_u16.numel(), 3)

    # uint32
    var t_u32 = zeros(shape, DType.uint32)
    assert_dtype_equal(t_u32.dtype(), DType.uint32)
    assert_equal(t_u32.numel(), 3)

    # uint64
    var t_u64 = zeros(shape, DType.uint64)
    assert_dtype_equal(t_u64.dtype(), DType.uint64)
    assert_equal(t_u64.numel(), 3)


fn test_scalar_tensor() raises:
    """Test scalar tensor (0 dimensions, single element)."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 42.0, DType.float32)

    assert_equal(t.numel(), 1)
    assert_equal(t.dim(), 1)
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(42.0), tolerance=1e-5
    )


fn test_large_tensor() raises:
    """Test large tensor creation."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(20)
    shape.append(30)
    var t = zeros(shape, DType.float32)

    assert_equal(t.shape()[0], 10)
    assert_equal(t.shape()[1], 20)
    assert_equal(t.shape()[2], 30)
    assert_equal(t.numel(), 6000)
    assert_equal(t.dim(), 3)


fn test_high_dimensional_tensor() raises:
    """Test high-dimensional tensor (4D, 5D, 6D)."""
    # 4D tensor
    var shape4 = List[Int]()
    shape4.append(2)
    shape4.append(3)
    shape4.append(4)
    shape4.append(5)
    var t4 = zeros(shape4, DType.float32)
    assert_equal(t4.dim(), 4)
    assert_equal(t4.numel(), 120)

    # 5D tensor
    var shape5 = List[Int](length=5, fill=2)
    var t5 = zeros(shape5, DType.float32)
    assert_equal(t5.dim(), 5)
    assert_equal(t5.numel(), 32)


fn test_set_and_get_values() raises:
    """Test setting and getting individual values."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)

    # Set values
    t._data.bitcast[Float32]()[0] = 1.0
    t._data.bitcast[Float32]()[1] = 2.0
    t._data.bitcast[Float32]()[2] = 3.0
    t._data.bitcast[Float32]()[3] = 4.0
    t._data.bitcast[Float32]()[4] = 5.0

    # Get and check values
    assert_almost_equal(
        t._data.bitcast[Float32]()[0], Float32(1.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[1], Float32(2.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[2], Float32(3.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[3], Float32(4.0), tolerance=1e-5
    )
    assert_almost_equal(
        t._data.bitcast[Float32]()[4], Float32(5.0), tolerance=1e-5
    )


fn test_2d_indexing() raises:
    """Test 2D tensor indexing."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    # Set value at (1, 2) -> linear index = 1*4 + 2 = 6
    t._data.bitcast[Float32]()[6] = 42.0

    # Get value at (1, 2)
    var val = t._data.bitcast[Float32]()[6]
    assert_almost_equal(val, Float32(42.0), tolerance=1e-5)


fn test_3d_indexing() raises:
    """Test 3D tensor indexing."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = zeros(shape, DType.float32)

    # Set value at (1, 2, 3) -> linear index = 1*12 + 2*4 + 3 = 23
    t._data.bitcast[Float32]()[23] = 99.0

    # Get value at (1, 2, 3)
    var val = t._data.bitcast[Float32]()[23]
    assert_almost_equal(val, Float32(99.0), tolerance=1e-5)


fn main() raises:
    """Run all test_tensors tests."""
    print("Running test_tensors tests...")

    test_zeros_creation()
    print("✓ test_zeros_creation")

    test_ones_creation()
    print("✓ test_ones_creation")

    test_full_creation()
    print("✓ test_full_creation")

    test_empty_creation()
    print("✓ test_empty_creation")

    test_arange_creation()
    print("✓ test_arange_creation")

    test_arange_with_step()
    print("✓ test_arange_with_step")

    test_eye_creation()
    print("✓ test_eye_creation")

    test_linspace_creation()
    print("✓ test_linspace_creation")

    test_zeros_like()
    print("✓ test_zeros_like")

    test_ones_like()
    print("✓ test_ones_like")

    test_full_like()
    print("✓ test_full_like")

    test_tensor_shape()
    print("✓ test_tensor_shape")

    test_tensor_dtype()
    print("✓ test_tensor_dtype")

    test_tensor_numel()
    print("✓ test_tensor_numel")

    test_tensor_dim()
    print("✓ test_tensor_dim")

    test_float_dtypes()
    print("✓ test_float_dtypes")

    test_int_dtypes()
    print("✓ test_int_dtypes")

    test_uint_dtypes()
    print("✓ test_uint_dtypes")

    test_scalar_tensor()
    print("✓ test_scalar_tensor")

    test_large_tensor()
    print("✓ test_large_tensor")

    test_high_dimensional_tensor()
    print("✓ test_high_dimensional_tensor")

    test_set_and_get_values()
    print("✓ test_set_and_get_values")

    test_2d_indexing()
    print("✓ test_2d_indexing")

    test_3d_indexing()
    print("✓ test_3d_indexing")

    print("\nAll test_tensors tests passed!")
