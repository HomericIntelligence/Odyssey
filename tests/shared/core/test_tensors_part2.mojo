"""Tests for tensor copying and properties (Part 2 of 3).

Tests cover:
- Tensor copying: zeros_like, ones_like, full_like
- Tensor properties: shape, dtype, numel, dim
- Float dtype support

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
from shared.core.any_tensor import (
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


# ============================================================================
# Tensor Copying Tests
# ============================================================================


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


# ============================================================================
# Tensor Properties Tests
# ============================================================================


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


# ============================================================================
# Float Dtype Support Tests
# ============================================================================


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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_tensors_part2.mojo")
    print("=" * 70 + "\n")

    # test_zeros_like
    total += 1
    try:
        test_zeros_like()
        passed += 1
        print("  ✓ test_zeros_like")
    except e:
        failed += 1
        print("  ✗ test_zeros_like:", e)

    # test_ones_like
    total += 1
    try:
        test_ones_like()
        passed += 1
        print("  ✓ test_ones_like")
    except e:
        failed += 1
        print("  ✗ test_ones_like:", e)

    # test_full_like
    total += 1
    try:
        test_full_like()
        passed += 1
        print("  ✓ test_full_like")
    except e:
        failed += 1
        print("  ✗ test_full_like:", e)

    # test_tensor_shape
    total += 1
    try:
        test_tensor_shape()
        passed += 1
        print("  ✓ test_tensor_shape")
    except e:
        failed += 1
        print("  ✗ test_tensor_shape:", e)

    # test_tensor_dtype
    total += 1
    try:
        test_tensor_dtype()
        passed += 1
        print("  ✓ test_tensor_dtype")
    except e:
        failed += 1
        print("  ✗ test_tensor_dtype:", e)

    # test_tensor_numel
    total += 1
    try:
        test_tensor_numel()
        passed += 1
        print("  ✓ test_tensor_numel")
    except e:
        failed += 1
        print("  ✗ test_tensor_numel:", e)

    # test_tensor_dim
    total += 1
    try:
        test_tensor_dim()
        passed += 1
        print("  ✓ test_tensor_dim")
    except e:
        failed += 1
        print("  ✗ test_tensor_dim:", e)

    # test_float_dtypes
    total += 1
    try:
        test_float_dtypes()
        passed += 1
        print("  ✓ test_float_dtypes")
    except e:
        failed += 1
        print("  ✗ test_float_dtypes:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
