# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_special_values.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for special_values module - Part 3: Seeded random (seeds/range/shape) and NaN/Inf tensors

Tests FP-representable test value utilities:
- Seeded random tensor with different seeds
- Seeded random tensor range validation
- Seeded random tensor shape and dtype
- NaN tensor creation
- Infinity tensor creation
"""

from shared.testing.special_values import (
    create_seeded_random_tensor,
    create_nan_tensor,
    create_inf_tensor,
)
from shared.core.numerical_safety import has_nan, has_inf
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
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
        # Values should be in [-1.0, 1.0)
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
    print(
        "Testing special_values module - Part 3: Seeded random"
        " (seeds/range/shape) and NaN/Inf tensors..."
    )

    test_create_seeded_random_tensor_different_seeds()
    print("✓ test_create_seeded_random_tensor_different_seeds")

    test_create_seeded_random_tensor_range()
    print("✓ test_create_seeded_random_tensor_range")

    test_create_seeded_random_tensor_custom_range()
    print("✓ test_create_seeded_random_tensor_custom_range")

    test_create_seeded_random_tensor_shape()
    print("✓ test_create_seeded_random_tensor_shape")

    test_create_nan_tensor()
    print("✓ test_create_nan_tensor")

    test_create_inf_tensor()
    print("✓ test_create_inf_tensor")

    print("\n✅ All special_values Part 3 tests passed!")
