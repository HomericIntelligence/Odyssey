# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_dtype_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for dtype dispatch helpers - Part 1: Unary and Binary Float dispatch.

Tests cover:
- dispatch_unary with float32 (identity, double operations)
- dispatch_unary with float64 and int32 and uint8
- dispatch_binary with float32 (add, multiply) and float64
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.dtype_dispatch import (
    dispatch_unary,
    dispatch_binary,
)


# ============================================================================
# Helper Operations for Testing
# ============================================================================


fn identity_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Identity operation for unary dispatch testing."""
    return x


fn double_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Double operation for unary dispatch testing."""
    return x + x


fn add_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
    """Add operation for binary dispatch testing."""
    return x + y


fn mul_op[T: DType](x: Scalar[T], y: Scalar[T]) -> Scalar[T]:
    """Multiply operation for binary dispatch testing."""
    return x * y


# ============================================================================
# Unary Dispatch Tests - Float32
# ============================================================================


fn test_dispatch_unary_float32_identity() raises:
    """Test dispatch_unary with float32 and identity operation."""
    var shape = List[Int]()
    shape.append(3)
    var x = full(shape, 5.0, DType.float32)

    var result = dispatch_unary[identity_op](x)

    assert_equal_int(result._numel, 3)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(5.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(5.0), tolerance=1e-6
    )


fn test_dispatch_unary_float32_double() raises:
    """Test dispatch_unary with float32 and double operation."""
    var shape = List[Int]()
    shape.append(3)
    var x = full(shape, 3.0, DType.float32)

    var result = dispatch_unary[double_op](x)

    assert_equal_int(result._numel, 3)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(6.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(6.0), tolerance=1e-6
    )


fn test_dispatch_unary_float64() raises:
    """Test dispatch_unary with float64."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 7.0, DType.float64)

    var result = dispatch_unary[identity_op](x)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float64]()[0], Float64(7.0), tolerance=1e-6
    )


fn test_dispatch_unary_int32() raises:
    """Test dispatch_unary with int32."""
    var shape = List[Int]()
    shape.append(3)
    var x = zeros(shape, DType.int32)

    x._data.bitcast[Int32]()[0] = 1
    x._data.bitcast[Int32]()[1] = 2
    x._data.bitcast[Int32]()[2] = 3

    var result = dispatch_unary[identity_op](x)

    assert_equal_int(result._numel, 3)
    assert_equal_int(Int(result._data.bitcast[Int32]()[0]), 1)
    assert_equal_int(Int(result._data.bitcast[Int32]()[1]), 2)
    assert_equal_int(Int(result._data.bitcast[Int32]()[2]), 3)


fn test_dispatch_unary_uint8() raises:
    """Test dispatch_unary with uint8."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.uint8)

    x._data.bitcast[UInt8]()[0] = 42
    x._data.bitcast[UInt8]()[1] = 84

    var result = dispatch_unary[identity_op](x)

    assert_equal_int(result._numel, 2)
    assert_equal_int(Int(result._data.bitcast[UInt8]()[0]), 42)
    assert_equal_int(Int(result._data.bitcast[UInt8]()[1]), 84)


# ============================================================================
# Binary Dispatch Tests - Float32 and Float64
# ============================================================================


fn test_dispatch_binary_float32_add() raises:
    """Test dispatch_binary with float32 and add operation."""
    var shape = List[Int]()
    shape.append(3)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = dispatch_binary[add_op](a, b)

    assert_equal_int(result._numel, 3)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(5.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(5.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[2], Float32(5.0), tolerance=1e-6
    )


fn test_dispatch_binary_float32_mul() raises:
    """Test dispatch_binary with float32 and multiply operation."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 4.0, DType.float32)
    var b = full(shape, 5.0, DType.float32)

    var result = dispatch_binary[mul_op](a, b)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(20.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[1], Float32(20.0), tolerance=1e-6
    )


fn test_dispatch_binary_float64() raises:
    """Test dispatch_binary with float64."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 1.5, DType.float64)
    var b = full(shape, 2.5, DType.float64)

    var result = dispatch_binary[add_op](a, b)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float64]()[0], Float64(4.0), tolerance=1e-6
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run dtype_dispatch part1 tests."""
    print("Running dtype_dispatch_part1 tests...")

    test_dispatch_unary_float32_identity()
    print("✓ test_dispatch_unary_float32_identity")

    test_dispatch_unary_float32_double()
    print("✓ test_dispatch_unary_float32_double")

    test_dispatch_unary_float64()
    print("✓ test_dispatch_unary_float64")

    test_dispatch_unary_int32()
    print("✓ test_dispatch_unary_int32")

    test_dispatch_unary_uint8()
    print("✓ test_dispatch_unary_uint8")

    test_dispatch_binary_float32_add()
    print("✓ test_dispatch_binary_float32_add")

    test_dispatch_binary_float32_mul()
    print("✓ test_dispatch_binary_float32_mul")

    test_dispatch_binary_float64()
    print("✓ test_dispatch_binary_float64")

    print("\nAll 8 dtype_dispatch_part1 tests passed!")
