# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_dtype_dispatch.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for dtype dispatch helpers - Part 3: Float-binary, Float-scalar, and 2D tensor dispatch.

Tests cover:
- dispatch_float_binary with float32 and int32 rejection
- dispatch_float_scalar with float32 and int32 rejection
- dispatch_unary and dispatch_binary with 2D tensors
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal_int,
    assert_true,
)
from shared.core.extensor import AnyTensor, zeros, ones, full
from shared.core.dtype_dispatch import (
    dispatch_unary,
    dispatch_binary,
    dispatch_float_binary,
    dispatch_float_scalar,
)


# ============================================================================
# Helper Operations for Testing
# ============================================================================


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
# Float-only Binary Dispatch Tests
# ============================================================================


fn test_dispatch_float_binary_float32() raises:
    """Test dispatch_float_binary with float32."""
    var shape = List[Int]()
    shape.append(2)
    var a = full(shape, 1.5, DType.float32)
    var b = full(shape, 2.5, DType.float32)

    var result = dispatch_float_binary[add_op](a, b)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(4.0), tolerance=1e-6
    )


fn test_dispatch_float_binary_rejects_int32() raises:
    """Test dispatch_float_binary rejects int32."""
    var shape = List[Int]()
    shape.append(2)
    var a = zeros(shape, DType.int32)
    var b = zeros(shape, DType.int32)

    var error_caught = False
    try:
        var _ = dispatch_float_binary[add_op](a, b)
    except Error:
        error_caught = True

    assert_true(error_caught, "Expected error for non-float dtype")


# ============================================================================
# Float-only Scalar Dispatch Tests
# ============================================================================


fn test_dispatch_float_scalar_float32() raises:
    """Test dispatch_float_scalar with float32."""
    var shape = List[Int]()
    shape.append(2)
    var x = full(shape, 5.0, DType.float32)

    var result = dispatch_float_scalar[mul_op](x, 2.0)

    assert_equal_int(result._numel, 2)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(10.0), tolerance=1e-6
    )


fn test_dispatch_float_scalar_rejects_int32() raises:
    """Test dispatch_float_scalar rejects int32."""
    var shape = List[Int]()
    shape.append(2)
    var x = zeros(shape, DType.int32)

    var error_caught = False
    try:
        var _ = dispatch_float_scalar[mul_op](x, 2.0)
    except Error:
        error_caught = True

    assert_true(error_caught, "Expected error for non-float dtype")


# ============================================================================
# 2D Tensor Tests
# ============================================================================


fn test_dispatch_unary_2d_tensor() raises:
    """Test dispatch_unary with 2D tensor."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var x = full(shape, 4.0, DType.float32)

    var result = dispatch_unary[double_op](x)

    assert_equal_int(result._numel, 6)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(8.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[5], Float32(8.0), tolerance=1e-6
    )


fn test_dispatch_binary_2d_tensor() raises:
    """Test dispatch_binary with 2D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(2)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var result = dispatch_binary[mul_op](a, b)

    assert_equal_int(result._numel, 4)
    assert_almost_equal(
        result._data.bitcast[Float32]()[0], Float32(6.0), tolerance=1e-6
    )
    assert_almost_equal(
        result._data.bitcast[Float32]()[3], Float32(6.0), tolerance=1e-6
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run dtype_dispatch part3 tests."""
    print("Running dtype_dispatch_part3 tests...")

    test_dispatch_float_binary_float32()
    print("✓ test_dispatch_float_binary_float32")

    test_dispatch_float_binary_rejects_int32()
    print("✓ test_dispatch_float_binary_rejects_int32")

    test_dispatch_float_scalar_float32()
    print("✓ test_dispatch_float_scalar_float32")

    test_dispatch_float_scalar_rejects_int32()
    print("✓ test_dispatch_float_scalar_rejects_int32")

    test_dispatch_unary_2d_tensor()
    print("✓ test_dispatch_unary_2d_tensor")

    test_dispatch_binary_2d_tensor()
    print("✓ test_dispatch_binary_2d_tensor")

    print("\nAll 6 dtype_dispatch_part3 tests passed!")
