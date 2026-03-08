"""Tests for shared.testing.tensor_factory module - Part 4: set_tensor_value (second half) and integration tests.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_tensor_factory.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_equal
from shared.testing.tensor_factory import (
    zeros_tensor,
    ones_tensor,
    full_tensor,
    random_tensor,
    random_normal_tensor,
    set_tensor_value,
)
from shared.testing.assertions import (
    assert_shape_equal,
    assert_dtype_equal,
    assert_almost_equal,
)


# ============================================================================
# Test set_tensor_value (second half)
# ============================================================================


fn test_set_tensor_value_float64() raises:
    """Test set_tensor_value with float64 dtype."""
    var shape = [5]
    var tensor = zeros_tensor(shape, DType.float64)

    set_tensor_value(tensor, 2, 6.28, DType.float64)

    var val = tensor._get_float64(2)
    assert_almost_equal(val, 6.28, tolerance=1e-10)


fn test_set_tensor_value_multiple_indices() raises:
    """Test setting multiple values in same tensor."""
    var shape = [10]
    var tensor = zeros_tensor(shape, DType.float32)

    # Set multiple values
    for i in range(10):
        set_tensor_value(tensor, i, Float64(i) * 1.5, DType.float32)

    # Verify all values
    for i in range(10):
        var val = tensor._get_float64(i)
        var expected = Float64(i) * 1.5
        assert_almost_equal(val, expected, tolerance=1e-4)


# ============================================================================
# Integration Tests
# ============================================================================


fn test_tensor_factory_workflow() raises:
    """Test typical workflow using multiple factory functions."""
    # Create tensors for a simple test scenario
    var shape = [10, 10]

    # Create various tensors
    var zeros = zeros_tensor(shape, DType.float32)
    var ones = ones_tensor(shape, DType.float32)
    var fives = full_tensor(shape, 5.0, DType.float32)
    var random = random_tensor(shape, DType.float32, -1.0, 1.0)
    var normal = random_normal_tensor(shape, DType.float32, 0.0, 1.0)

    # Verify all have correct shape
    assert_shape_equal(zeros.shape(), shape)
    assert_shape_equal(ones.shape(), shape)
    assert_shape_equal(fives.shape(), shape)
    assert_shape_equal(random.shape(), shape)
    assert_shape_equal(normal.shape(), shape)

    # Verify all have correct dtype
    assert_dtype_equal(zeros.dtype(), DType.float32)
    assert_dtype_equal(ones.dtype(), DType.float32)
    assert_dtype_equal(fives.dtype(), DType.float32)
    assert_dtype_equal(random.dtype(), DType.float32)
    assert_dtype_equal(normal.dtype(), DType.float32)


fn test_tensor_factory_all_dtypes() raises:
    """Test tensor factories work with multiple dtypes."""
    var shape = [5]
    var dtypes = List[DType]()
    dtypes.append(DType.float32)
    dtypes.append(DType.float64)
    dtypes.append(DType.int32)
    dtypes.append(DType.int64)

    # Test each dtype
    for dtype in dtypes:
        var zeros = zeros_tensor(shape, dtype)
        var ones = ones_tensor(shape, dtype)
        var full = full_tensor(shape, 3.0, dtype)

        assert_dtype_equal(zeros.dtype(), dtype)
        assert_dtype_equal(ones.dtype(), dtype)
        assert_dtype_equal(full.dtype(), dtype)


fn main() raises:
    """Run all tests."""
    test_set_tensor_value_float64()
    test_set_tensor_value_multiple_indices()

    test_tensor_factory_workflow()
    test_tensor_factory_all_dtypes()
