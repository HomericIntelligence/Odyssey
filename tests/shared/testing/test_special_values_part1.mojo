# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_special_values.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for special_values module - Part 1: Constants and tensor creation

Tests FP-representable test value utilities:
- Special value constants (including negative values)
- Tensor creation with special values (zeros, ones, halves, one-and-half, neg)
- Alternating pattern tensor creation
"""

from shared.testing.special_values import (
    SPECIAL_VALUE_ZERO,
    SPECIAL_VALUE_HALF,
    SPECIAL_VALUE_ONE,
    SPECIAL_VALUE_ONE_HALF,
    SPECIAL_VALUE_NEG_HALF,
    SPECIAL_VALUE_NEG_ONE,
    create_special_value_tensor,
    create_alternating_pattern_tensor,
    verify_special_value_invariants,
)
from shared.testing.assertions import (
    assert_equal_float,
    assert_shape,
    assert_dtype,
)


fn test_special_value_constants() raises:
    """Test that special value constants have correct values."""
    assert_equal_float(
        Float32(SPECIAL_VALUE_ZERO), 0.0, "SPECIAL_VALUE_ZERO should be 0.0"
    )
    assert_equal_float(
        Float32(SPECIAL_VALUE_HALF), 0.5, "SPECIAL_VALUE_HALF should be 0.5"
    )
    assert_equal_float(
        Float32(SPECIAL_VALUE_ONE), 1.0, "SPECIAL_VALUE_ONE should be 1.0"
    )
    assert_equal_float(
        Float32(SPECIAL_VALUE_ONE_HALF),
        1.5,
        "SPECIAL_VALUE_ONE_HALF should be 1.5",
    )
    assert_equal_float(
        Float32(SPECIAL_VALUE_NEG_HALF),
        -0.5,
        "SPECIAL_VALUE_NEG_HALF should be -0.5",
    )
    assert_equal_float(
        Float32(SPECIAL_VALUE_NEG_ONE),
        -1.0,
        "SPECIAL_VALUE_NEG_ONE should be -1.0",
    )


fn test_create_special_value_tensor_zeros() raises:
    """Test creating tensor filled with zeros."""
    var tensor = create_special_value_tensor([3, 3], DType.float32, 0.0)
    assert_shape(tensor, [3, 3], "Shape should be [3, 3]")
    assert_dtype(tensor, DType.float32, "Dtype should be float32")

    # Verify all values are zero
    verify_special_value_invariants(tensor, 0.0)


fn test_create_special_value_tensor_ones() raises:
    """Test creating tensor filled with ones."""
    var tensor = create_special_value_tensor([2, 4], DType.float16, 1.0)
    assert_shape(tensor, [2, 4], "Shape should be [2, 4]")
    assert_dtype(tensor, DType.float16, "Dtype should be float16")

    # Verify all values are one
    verify_special_value_invariants(tensor, 1.0)


fn test_create_special_value_tensor_halves() raises:
    """Test creating tensor filled with 0.5."""
    var tensor = create_special_value_tensor([4, 2], DType.float32, 0.5)
    assert_shape(tensor, [4, 2], "Shape should be [4, 2]")

    # Verify all values are 0.5
    verify_special_value_invariants(tensor, 0.5)


fn test_create_special_value_tensor_one_and_half() raises:
    """Test creating tensor filled with 1.5."""
    var tensor = create_special_value_tensor([2, 2], DType.float64, 1.5)
    assert_shape(tensor, [2, 2], "Shape should be [2, 2]")

    # Verify all values are 1.5
    verify_special_value_invariants(tensor, 1.5)


fn test_create_special_value_tensor_neg_one() raises:
    """Test creating tensor filled with -1.0 (for ReLU gradient testing)."""
    var tensor = create_special_value_tensor([3, 3], DType.float32, -1.0)
    assert_shape(tensor, [3, 3], "Shape should be [3, 3]")
    assert_dtype(tensor, DType.float32, "Dtype should be float32")

    # Verify all values are -1.0
    verify_special_value_invariants(tensor, -1.0)


fn test_create_special_value_tensor_neg_half() raises:
    """Test creating tensor filled with -0.5 (for ReLU gradient testing)."""
    var tensor = create_special_value_tensor([2, 3], DType.float16, -0.5)
    assert_shape(tensor, [2, 3], "Shape should be [2, 3]")

    # Verify all values are -0.5
    verify_special_value_invariants(tensor, -0.5)


fn test_create_alternating_pattern_tensor() raises:
    """Test creating tensor with alternating special values (6-value pattern).
    """
    var tensor = create_alternating_pattern_tensor([2, 3], DType.float32)
    assert_shape(tensor, [2, 3], "Shape should be [2, 3]")
    assert_dtype(tensor, DType.float32, "Dtype should be float32")

    # Verify alternating pattern: -1.0, -0.5, 0.0, 0.5, 1.0, 1.5
    var val0 = tensor._get_float64(0)
    var val1 = tensor._get_float64(1)
    var val2 = tensor._get_float64(2)
    var val3 = tensor._get_float64(3)
    var val4 = tensor._get_float64(4)
    var val5 = tensor._get_float64(5)

    assert_equal_float(Float32(val0), -1.0, "Element 0 should be -1.0")
    assert_equal_float(Float32(val1), -0.5, "Element 1 should be -0.5")
    assert_equal_float(Float32(val2), 0.0, "Element 2 should be 0.0")
    assert_equal_float(Float32(val3), 0.5, "Element 3 should be 0.5")
    assert_equal_float(Float32(val4), 1.0, "Element 4 should be 1.0")
    assert_equal_float(Float32(val5), 1.5, "Element 5 should be 1.5")


fn main() raises:
    print("Testing special_values module - Part 1: Constants and tensor creation...")

    test_special_value_constants()
    print("✓ test_special_value_constants")

    test_create_special_value_tensor_zeros()
    print("✓ test_create_special_value_tensor_zeros")

    test_create_special_value_tensor_ones()
    print("✓ test_create_special_value_tensor_ones")

    test_create_special_value_tensor_halves()
    print("✓ test_create_special_value_tensor_halves")

    test_create_special_value_tensor_one_and_half()
    print("✓ test_create_special_value_tensor_one_and_half")

    test_create_special_value_tensor_neg_one()
    print("✓ test_create_special_value_tensor_neg_one")

    test_create_special_value_tensor_neg_half()
    print("✓ test_create_special_value_tensor_neg_half")

    test_create_alternating_pattern_tensor()
    print("✓ test_create_alternating_pattern_tensor")

    print("\n✅ All special_values Part 1 tests passed!")
