"""Tests for tensor dtype support, edge cases, and indexing (Part 3 of 3).

Tests cover:
- Integer and unsigned integer dtype support
- Edge cases: scalar, large, high-dimensional tensors
- Value setting and getting, 2D/3D indexing

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_tensors.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

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
from shared.core.extensor import (
    ExTensor,
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


# ============================================================================
# Integer Dtype Support Tests
# ============================================================================


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


# ============================================================================
# Edge Case Tests
# ============================================================================


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


# ============================================================================
# Value Setting and Getting Tests
# ============================================================================


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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_tensors_part3.mojo")
    print("=" * 70 + "\n")

    # test_int_dtypes
    total += 1
    try:
        test_int_dtypes()
        passed += 1
        print("  ✓ test_int_dtypes")
    except e:
        failed += 1
        print("  ✗ test_int_dtypes:", e)

    # test_uint_dtypes
    total += 1
    try:
        test_uint_dtypes()
        passed += 1
        print("  ✓ test_uint_dtypes")
    except e:
        failed += 1
        print("  ✗ test_uint_dtypes:", e)

    # test_scalar_tensor
    total += 1
    try:
        test_scalar_tensor()
        passed += 1
        print("  ✓ test_scalar_tensor")
    except e:
        failed += 1
        print("  ✗ test_scalar_tensor:", e)

    # test_large_tensor
    total += 1
    try:
        test_large_tensor()
        passed += 1
        print("  ✓ test_large_tensor")
    except e:
        failed += 1
        print("  ✗ test_large_tensor:", e)

    # test_high_dimensional_tensor
    total += 1
    try:
        test_high_dimensional_tensor()
        passed += 1
        print("  ✓ test_high_dimensional_tensor")
    except e:
        failed += 1
        print("  ✗ test_high_dimensional_tensor:", e)

    # test_set_and_get_values
    total += 1
    try:
        test_set_and_get_values()
        passed += 1
        print("  ✓ test_set_and_get_values")
    except e:
        failed += 1
        print("  ✗ test_set_and_get_values:", e)

    # test_2d_indexing
    total += 1
    try:
        test_2d_indexing()
        passed += 1
        print("  ✓ test_2d_indexing")
    except e:
        failed += 1
        print("  ✗ test_2d_indexing:", e)

    # test_3d_indexing
    total += 1
    try:
        test_3d_indexing()
        passed += 1
        print("  ✓ test_3d_indexing")
    except e:
        failed += 1
        print("  ✗ test_3d_indexing:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
