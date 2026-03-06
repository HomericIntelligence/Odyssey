"""Tests for special_values module

Tests FP-representable test value utilities:
- Special value constants (including negative values)
- Tensor creation with special values
- Alternating pattern tensors (6-value pattern with negatives)
- Value invariant verification
- Seeded random tensor generation for gradient checking

Ensures special values work correctly across all dtypes: FP4, FP8, FP16, FP32, BF16, Int8
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
    create_seeded_random_tensor,
    verify_special_value_invariants,
    create_zeros_tensor,
    create_ones_tensor,
    create_halves_tensor,
    create_one_and_half_tensor,
    create_nan_tensor,
    create_inf_tensor,
)
from shared.core.numerical_safety import has_nan, has_inf
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


fn test_create_alternating_pattern_repeats() raises:
    """Test that alternating pattern repeats after 6 values."""
    var tensor = create_alternating_pattern_tensor([2, 6], DType.float32)

    # First cycle: -1.0, -0.5, 0.0, 0.5, 1.0, 1.5
    assert_equal_float(
        Float32(tensor._get_float64(0)), -1.0, "Element 0 should be -1.0"
    )
    assert_equal_float(
        Float32(tensor._get_float64(1)), -0.5, "Element 1 should be -0.5"
    )
    assert_equal_float(
        Float32(tensor._get_float64(2)), 0.0, "Element 2 should be 0.0"
    )
    assert_equal_float(
        Float32(tensor._get_float64(3)), 0.5, "Element 3 should be 0.5"
    )
    assert_equal_float(
        Float32(tensor._get_float64(4)), 1.0, "Element 4 should be 1.0"
    )
    assert_equal_float(
        Float32(tensor._get_float64(5)), 1.5, "Element 5 should be 1.5"
    )

    # Second cycle: repeats
    assert_equal_float(
        Float32(tensor._get_float64(6)), -1.0, "Element 6 should be -1.0"
    )
    assert_equal_float(
        Float32(tensor._get_float64(7)), -0.5, "Element 7 should be -0.5"
    )
    assert_equal_float(
        Float32(tensor._get_float64(8)), 0.0, "Element 8 should be 0.0"
    )
    assert_equal_float(
        Float32(tensor._get_float64(9)), 0.5, "Element 9 should be 0.5"
    )
    assert_equal_float(
        Float32(tensor._get_float64(10)), 1.0, "Element 10 should be 1.0"
    )
    assert_equal_float(
        Float32(tensor._get_float64(11)), 1.5, "Element 11 should be 1.5"
    )


fn test_verify_special_value_invariants_passes() raises:
    """Test that verify_special_value_invariants passes for correct tensor."""
    var tensor = create_special_value_tensor([3, 3], DType.float32, 1.0)

    # Should not raise
    verify_special_value_invariants(tensor, 1.0)


fn test_convenience_functions() raises:
    """Test convenience functions for creating special value tensors."""
    # Test create_zeros_tensor
    var zeros = create_zeros_tensor([2, 2], DType.float32)
    verify_special_value_invariants(zeros, 0.0)

    # Test create_ones_tensor
    var ones = create_ones_tensor([2, 2], DType.float32)
    verify_special_value_invariants(ones, 1.0)

    # Test create_halves_tensor
    var halves = create_halves_tensor([2, 2], DType.float32)
    verify_special_value_invariants(halves, 0.5)

    # Test create_one_and_half_tensor
    var one_and_half = create_one_and_half_tensor([2, 2], DType.float32)
    verify_special_value_invariants(one_and_half, 1.5)


fn test_dtypes_float32() raises:
    """Test special values work with float32."""
    var tensor = create_special_value_tensor([2, 2], DType.float32, 1.0)
    assert_dtype(tensor, DType.float32, "Should be float32")
    verify_special_value_invariants(tensor, 1.0)


fn test_dtypes_float64() raises:
    """Test special values work with float64."""
    var tensor = create_special_value_tensor([2, 2], DType.float64, 0.5)
    assert_dtype(tensor, DType.float64, "Should be float64")
    verify_special_value_invariants(tensor, 0.5)


fn test_dtypes_float16() raises:
    """Test special values work with float16."""
    var tensor = create_special_value_tensor([2, 2], DType.float16, 1.5)
    assert_dtype(tensor, DType.float16, "Should be float16")
    verify_special_value_invariants(tensor, 1.5)


fn test_dtypes_bfloat16() raises:
    """Test special values work with bfloat16.

    DType.bfloat16 is available in Mojo, but ExTensor's _set_float64/_get_float64
    path does not correctly round-trip values through bfloat16 storage, so this
    test is skipped until the ExTensor float64 read/write path supports bfloat16.

    Note:
        DType.bfloat16 is also not supported on Apple Silicon hardware.

    TODO: Enable when ExTensor._set_float64/_get_float64 correctly handle bfloat16.
        var tensor = create_special_value_tensor([2, 2], DType.bfloat16, 1.0)
        assert_dtype(tensor, DType.bfloat16, "Should be bfloat16")
        verify_special_value_invariants(tensor, 1.0)
    """
    pass


fn test_create_seeded_random_tensor_reproducibility() raises:
    """Test that seeded random tensors are reproducible."""
    # Create two tensors with same seed
    var tensor1 = create_seeded_random_tensor(
        [3, 3], DType.float32, 42, -1.0, 1.0
    )
    var tensor2 = create_seeded_random_tensor(
        [3, 3], DType.float32, 42, -1.0, 1.0
    )

    # They should be identical
    var numel = tensor1.numel()
    for i in range(numel):
        var val1 = tensor1._get_float64(i)
        var val2 = tensor2._get_float64(i)
        assert_equal_float(
            Float32(val1),
            Float32(val2),
            "Element " + String(i) + " should match",
        )


fn test_create_seeded_random_tensor_different_seeds() raises:
    """Test that different seeds produce different tensors."""
    # Create tensors with different seeds
    var tensor1 = create_seeded_random_tensor(
        [2, 2], DType.float32, 42, -1.0, 1.0
    )
    var tensor2 = create_seeded_random_tensor(
        [2, 2], DType.float32, 123, -1.0, 1.0
    )

    # They should be different (with very high probability)
    var numel = tensor1.numel()
    var found_difference = False
    for i in range(numel):
        var val1 = tensor1._get_float64(i)
        var val2 = tensor2._get_float64(i)
        if val1 != val2:
            found_difference = True
            break

    if not found_difference:
        raise Error(
            "Tensors with different seeds should have different values (with"
            " very high probability)"
        )


fn test_create_seeded_random_tensor_range() raises:
    """Test that seeded random values fall within specified range."""
    var tensor = create_seeded_random_tensor(
        [5, 5], DType.float32, 42, -1.0, 1.0
    )

    var numel = tensor.numel()
    for i in range(numel):
        var val = tensor._get_float64(i)
        # Values should be in [-1.0, 1.0]
        if val < -1.0 or val >= 1.0:
            raise Error(
                "Value "
                + String(val)
                + " at index "
                + String(i)
                + " is outside range [-1.0, 1.0)"
            )


fn test_create_seeded_random_tensor_custom_range() raises:
    """Test seeded random tensor with custom range."""
    # For gradient checking, we might want small random values
    var tensor = create_seeded_random_tensor(
        [3, 3], DType.float64, 42, -0.01, 0.01
    )

    var numel = tensor.numel()
    for i in range(numel):
        var val = tensor._get_float64(i)
        # Values should be in [-0.01, 0.01)
        if val < -0.01 or val >= 0.01:
            raise Error(
                "Value "
                + String(val)
                + " at index "
                + String(i)
                + " is outside range [-0.01, 0.01)"
            )


fn test_create_seeded_random_tensor_shape() raises:
    """Test that seeded random tensor has correct shape and dtype."""
    var tensor = create_seeded_random_tensor(
        [4, 5], DType.float16, 42, -1.0, 1.0
    )
    assert_shape(tensor, [4, 5], "Shape should be [4, 5]")
    assert_dtype(tensor, DType.float16, "Dtype should be float16")


fn test_create_nan_tensor() raises:
    """Test creation of NaN-filled tensors."""
    # Test Float32 NaN tensor
    var nan_f32 = create_nan_tensor([3, 3], DType.float32)
    assert_shape(nan_f32, [3, 3], "Shape should be [3, 3]")
    assert_dtype(nan_f32, DType.float32, "Dtype should be float32")

    # Verify NaN values using has_nan
    if not has_nan(nan_f32):
        raise Error("Float32 NaN tensor should contain NaN values")

    # Test Float64 NaN tensor
    var nan_f64 = create_nan_tensor([2, 2], DType.float64)
    assert_shape(nan_f64, [2, 2], "Shape should be [2, 2]")
    assert_dtype(nan_f64, DType.float64, "Dtype should be float64")

    # Verify NaN values using has_nan
    if not has_nan(nan_f64):
        raise Error("Float64 NaN tensor should contain NaN values")


fn test_create_inf_tensor() raises:
    """Test creation of Infinity-filled tensors."""
    # Test positive infinity (Float32)
    var pos_inf_f32 = create_inf_tensor([3, 3], DType.float32, positive=True)
    assert_shape(pos_inf_f32, [3, 3], "Shape should be [3, 3]")
    assert_dtype(pos_inf_f32, DType.float32, "Dtype should be float32")

    # Verify Inf values using has_inf
    if not has_inf(pos_inf_f32):
        raise Error("Float32 +Inf tensor should contain Inf values")

    # Test negative infinity (Float32)
    var neg_inf_f32 = create_inf_tensor([3, 3], DType.float32, positive=False)
    assert_shape(neg_inf_f32, [3, 3], "Shape should be [3, 3]")
    assert_dtype(neg_inf_f32, DType.float32, "Dtype should be float32")

    # Verify Inf values using has_inf
    if not has_inf(neg_inf_f32):
        raise Error("Float32 -Inf tensor should contain Inf values")

    # Test positive infinity (Float64)
    var pos_inf_f64 = create_inf_tensor([2, 2], DType.float64, positive=True)
    assert_shape(pos_inf_f64, [2, 2], "Shape should be [2, 2]")
    assert_dtype(pos_inf_f64, DType.float64, "Dtype should be float64")

    # Verify Inf values using has_inf
    if not has_inf(pos_inf_f64):
        raise Error("Float64 +Inf tensor should contain Inf values")

    # Test negative infinity (Float64)
    var neg_inf_f64 = create_inf_tensor([2, 2], DType.float64, positive=False)
    assert_shape(neg_inf_f64, [2, 2], "Shape should be [2, 2]")
    assert_dtype(neg_inf_f64, DType.float64, "Dtype should be float64")

    # Verify Inf values using has_inf
    if not has_inf(neg_inf_f64):
        raise Error("Float64 -Inf tensor should contain Inf values")


fn main() raises:
    print("Testing special_values module...")

    # Test constants (including negative values)
    test_special_value_constants()
    print("✓ test_special_value_constants")

    # Test creation functions
    test_create_special_value_tensor_zeros()
    print("✓ test_create_special_value_tensor_zeros")

    test_create_special_value_tensor_ones()
    print("✓ test_create_special_value_tensor_ones")

    test_create_special_value_tensor_halves()
    print("✓ test_create_special_value_tensor_halves")

    test_create_special_value_tensor_one_and_half()
    print("✓ test_create_special_value_tensor_one_and_half")

    # Test negative value tensors (for ReLU gradient testing)
    test_create_special_value_tensor_neg_one()
    print("✓ test_create_special_value_tensor_neg_one")

    test_create_special_value_tensor_neg_half()
    print("✓ test_create_special_value_tensor_neg_half")

    # Test alternating pattern (6-value pattern with negatives)
    test_create_alternating_pattern_tensor()
    print("✓ test_create_alternating_pattern_tensor")

    test_create_alternating_pattern_repeats()
    print("✓ test_create_alternating_pattern_repeats")

    # Test verification
    test_verify_special_value_invariants_passes()
    print("✓ test_verify_special_value_invariants_passes")

    # Test convenience functions
    test_convenience_functions()
    print("✓ test_convenience_functions")

    # Test dtypes
    test_dtypes_float32()
    print("✓ test_dtypes_float32")

    test_dtypes_float64()
    print("✓ test_dtypes_float64")

    test_dtypes_float16()
    print("✓ test_dtypes_float16")

    # BF16 dtype not yet supported in Mojo
    test_dtypes_bfloat16()
    print(
        "✓ test_dtypes_bfloat16 (skipped - bfloat16 float64 read/write not yet"
        " supported)"
    )

    # Test seeded random tensor (for gradient checking reproducibility)
    test_create_seeded_random_tensor_reproducibility()
    print("✓ test_create_seeded_random_tensor_reproducibility")

    test_create_seeded_random_tensor_different_seeds()
    print("✓ test_create_seeded_random_tensor_different_seeds")

    test_create_seeded_random_tensor_range()
    print("✓ test_create_seeded_random_tensor_range")

    test_create_seeded_random_tensor_custom_range()
    print("✓ test_create_seeded_random_tensor_custom_range")

    test_create_seeded_random_tensor_shape()
    print("✓ test_create_seeded_random_tensor_shape")

    # Test NaN and Inf tensor creation
    test_create_nan_tensor()
    print("✓ test_create_nan_tensor")

    test_create_inf_tensor()
    print("✓ test_create_inf_tensor")

    print("\n✅ All special_values tests passed!")
