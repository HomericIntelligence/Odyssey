# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_dtype_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for dtype dispatch helpers - Part 2: Binary int/mismatch, Scalar, and Float-unary dispatch.

Tests cover:
- dispatch_binary with int32 and dtype mismatch error
- dispatch_scalar with float32 (add, multiply) and int32
- dispatch_float_unary with float32, float64, and int32 rejection
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.core.dtype_dispatch import (
    dispatch_binary,
    dispatch_scalar,
    dispatch_float_unary,
)


# ============================================================================
# Helper Operations for Testing
# ============================================================================


fn identity_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Identity operation for unary dispatch testing."""
    return x


fn add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
    """Add operation for binary dispatch testing."""
    return x + y


fn mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
    """Multiply operation for binary dispatch testing."""
    return x * y


# ============================================================================
# Binary Dispatch Tests - Int32 and Mismatch
# ============================================================================


fn test_dispatch_binary_int32() raises:
    """Test dispatch_binary with int32."""
    var shape = List[Int]()
    shape.append(2)
    var a = zeros(shape, DType.int32)
    var b = zeros(shape, DType.int32)

    a._data.bitcast[Int32]()[0] = 10
    a._data.bitcast[Int32]()[1] = 20
    b._data.bitcast[Int32]()[0] = 5
    b._data.bitcast[Int32]()[1] = 10

    var result = dispatch_binary[add_op](a, b)

    assert_equal_int(result._numel, 2)
    assert_equal_int(Int(result._data.bitcast[Int32]()[0]), 15)
    assert_equal_int(Int(result._data.bitcast[Int32]()[1]), 30)


fn test_dispatch_binary_dtype_mismatch() raises:
    """Test dispatch_binary error when dtypes don't match."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 1.0, DType.float32)
    var b = full(shape, 2.0, DType.float64)

    var error_caught = False
    try:
        var _ = dispatch_binary[add_op](a, b)
    except Error:
        error_caught = True

    assert_true(error_caught, "Expected error for dtype mismatch")


# ============================================================================
# Scalar Dispatch Tests
# ============================================================================


fn test_dispatch_scalar_float32_add() raises:
    """Test dispatch_scalar with float32 and add operation."""
    var shape = List[Int]()
    shape.append(3)
    var x = full(shape, 10.0, DType.float32)

    var result = dispatch_scalar[add_op](x, 5.0)

    assert_equal_int(result._numel, 3)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(15.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(15.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(15.0), tolerance=1e-6
    )


fn test_dispatch_scalar_float32_mul() raises:
    """Test dispatch_scalar with float32 and multiply operation."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 4.0, DType.float32)

    var result = dispatch_scalar[mul_op](x, 3.0)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(12.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(12.0), tolerance=1e-6
    )


fn test_dispatch_scalar_int32() raises:
    """Test dispatch_scalar with int32."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.int32)

    x._data.bitcast[Int32]()[0] = 5
    x._data.bitcast[Int32]()[1] = 10

    var result = dispatch_scalar[add_op](x, 3.0)

    assert_equal_int(result._numel, 2)
    assert_equal_int(Int(result._data.bitcast[Int32]()[0]), 8)
    assert_equal_int(Int(result._data.bitcast[Int32]()[1]), 13)


# ============================================================================
# Float-only Unary Dispatch Tests
# ============================================================================


fn test_dispatch_float_unary_float32() raises:
    """Test dispatch_float_unary with float32."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 2.0, DType.float32)

    var result = dispatch_float_unary[identity_op](x)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(2.0), tolerance=1e-6
    )


fn test_dispatch_float_unary_float64() raises:
    """Test dispatch_float_unary with float64."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 3.5, DType.float64)

    var result = dispatch_float_unary[identity_op](x)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float64]()[0], Float64(3.5), tolerance=1e-6
    )


fn test_dispatch_float_unary_rejects_int32() raises:
    """Test dispatch_float_unary rejects int32."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.int32)

    var error_caught = False
    try:
        var _ = dispatch_float_unary[identity_op](x)
    except Error:
        error_caught = True

    assert_true(error_caught, "Expected error for non-float dtype")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run dtype_dispatch part2 tests."""
    print("Running dtype_dispatch_part2 tests...")

    test_dispatch_binary_int32()
    print("✓ test_dispatch_binary_int32")

    test_dispatch_binary_dtype_mismatch()
    print("✓ test_dispatch_binary_dtype_mismatch")

    test_dispatch_scalar_float32_add()
    print("✓ test_dispatch_scalar_float32_add")

    test_dispatch_scalar_float32_mul()
    print("✓ test_dispatch_scalar_float32_mul")

    test_dispatch_scalar_int32()
    print("✓ test_dispatch_scalar_int32")

    test_dispatch_float_unary_float32()
    print("✓ test_dispatch_float_unary_float32")

    test_dispatch_float_unary_float64()
    print("✓ test_dispatch_float_unary_float64")

    test_dispatch_float_unary_rejects_int32()
    print("✓ test_dispatch_float_unary_rejects_int32")

    print("\nAll 8 dtype_dispatch_part2 tests passed!")
