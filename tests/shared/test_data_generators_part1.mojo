"""Tests for data generators module - Part 1: random_tensor.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_data_generators.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests random_tensor function:
- Shape creation (1D, 2D, 3D)
- Dtype handling (float32, float64, int32)
- Value ranges
- Default dtype
"""

from shared.testing import (
    random_tensor,
)

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_dtype,
    assert_numel,
    assert_dim,
)


# ============================================================================
# Test random_tensor()
# ============================================================================


fn test_random_tensor_shape_1d() raises:
    """Test random_tensor creates correct 1D shape."""
    var shape = List[Int]()
    shape.append(10)
    var tensor = random_tensor(shape, DType.float32)

    assert_dim(tensor, 1, "random_tensor 1D should have 1 dimension")
    assert_numel(tensor, 10, "random_tensor 1D should have 10 elements")


fn test_random_tensor_shape_2d() raises:
    """Test random_tensor creates correct 2D shape."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(8)
    var tensor = random_tensor(shape, DType.float32)

    assert_dim(tensor, 2, "random_tensor 2D should have 2 dimensions")
    assert_numel(tensor, 40, "random_tensor 2D(5,8) should have 40 elements")


fn test_random_tensor_shape_3d() raises:
    """Test random_tensor creates correct 3D shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var tensor = random_tensor(shape, DType.float32)

    assert_dim(tensor, 3, "random_tensor 3D should have 3 dimensions")
    assert_numel(tensor, 24, "random_tensor 3D(2,3,4) should have 24 elements")


fn test_random_tensor_dtype_float32() raises:
    """Test random_tensor with float32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var tensor = random_tensor(shape, DType.float32)

    assert_dtype(tensor, DType.float32, "random_tensor should respect dtype")


fn test_random_tensor_dtype_float64() raises:
    """Test random_tensor with float64 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var tensor = random_tensor(shape, DType.float64)

    assert_dtype(tensor, DType.float64, "random_tensor should respect dtype")


fn test_random_tensor_dtype_int32() raises:
    """Test random_tensor with int32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var tensor = random_tensor(shape, DType.int32)

    assert_dtype(tensor, DType.int32, "random_tensor should respect dtype")


fn test_random_tensor_values_in_range() raises:
    """Test random_tensor values are in [0, 1)."""
    var shape = List[Int]()
    shape.append(100)
    var tensor = random_tensor(shape, DType.float32)

    # Check a sample of values are in valid range
    for i in range(0, 100, 10):
        var val = tensor._get_float64(i)
        assert_true(val >= 0.0, "random_tensor value should be >= 0.0")
        assert_true(val < 1.0, "random_tensor value should be < 1.0")


fn test_random_tensor_default_dtype() raises:
    """Test random_tensor uses float32 as default dtype."""
    var shape = List[Int]()
    shape.append(5)
    var tensor = random_tensor(shape)  # No dtype specified

    assert_dtype(
        tensor, DType.float32, "random_tensor default dtype should be float32"
    )


fn main() raises:
    """Run random_tensor tests."""
    print("Running random_tensor tests (Part 1)...")

    test_random_tensor_shape_1d()
    print("✓ random_tensor 1D shape")

    test_random_tensor_shape_2d()
    print("✓ random_tensor 2D shape")

    test_random_tensor_shape_3d()
    print("✓ random_tensor 3D shape")

    test_random_tensor_dtype_float32()
    print("✓ random_tensor float32 dtype")

    test_random_tensor_dtype_float64()
    print("✓ random_tensor float64 dtype")

    test_random_tensor_dtype_int32()
    print("✓ random_tensor int32 dtype")

    test_random_tensor_values_in_range()
    print("✓ random_tensor values in range")

    test_random_tensor_default_dtype()
    print("✓ random_tensor default dtype")

    print("\nAll random_tensor tests passed!")
