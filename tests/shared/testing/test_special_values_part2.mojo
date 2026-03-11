# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_special_values.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for special_values module - Part 2: Pattern repeats, verification, convenience, dtypes, seeded random

Tests FP-representable test value utilities:
- Alternating pattern repeat behavior
- Value invariant verification
- Convenience tensor creation functions
- Multi-dtype support
- Seeded random tensor reproducibility
"""

from shared.testing.special_values import (
    create_special_value_tensor,
    create_alternating_pattern_tensor,
    create_seeded_random_tensor,
    verify_special_value_invariants,
    create_zeros_tensor,
    create_ones_tensor,
    create_halves_tensor,
    create_one_and_half_tensor,
)
from shared.testing.assertions import (
    assert_equal_float,
    assert_shape,
    assert_dtype,
)


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

    NOTE: BF16 is a custom type in shared.core.types.bf16 but is not
    yet integrated with Mojo's runtime DType system. This test is skipped
    until DType.bfloat16 is added to Mojo or we implement custom dtype handling.

    Current Status:
    - BF16 struct exists in shared.core.types.bf16
    - Mojo's DType enum does not include DType.bfloat16
    - Cannot create tensors with BF16 dtype through standard ExTensor API

    Implementation Requirements:
    - Wait for Mojo to add DType.bfloat16 to the DType enum
    - OR implement custom dtype registration in ExTensor
    - OR wrap special_values functions to support struct-based dtypes

    Once Available:
    1. Uncomment the test code below
    2. Verify special values (0.5, 1.0, 1.5, -0.5, -1.0) are representable
    3. Add BF16 SIMD operations similar to FP32 paths
    4. Test mixed precision training with BF16 parameters

    Reference: shared.core.types.bf16 module for current BF16 implementation
    """
    var tensor = create_special_value_tensor([2, 2], DType.bfloat16, 1.0)
    assert_dtype(tensor, DType.bfloat16, "Should be bfloat16")
    verify_special_value_invariants(tensor, 1.0)


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


fn main() raises:
    print(
        "Testing special_values module - Part 2: Pattern repeats, verification,"
        " convenience, dtypes, seeded random..."
    )

    test_create_alternating_pattern_repeats()
    print("✓ test_create_alternating_pattern_repeats")

    test_verify_special_value_invariants_passes()
    print("✓ test_verify_special_value_invariants_passes")

    test_convenience_functions()
    print("✓ test_convenience_functions")

    test_dtypes_float32()
    print("✓ test_dtypes_float32")

    test_dtypes_float64()
    print("✓ test_dtypes_float64")

    test_dtypes_float16()
    print("✓ test_dtypes_float16")

    test_dtypes_bfloat16()
    print("✓ test_dtypes_bfloat16")

    test_create_seeded_random_tensor_reproducibility()
    print("✓ test_create_seeded_random_tensor_reproducibility")

    print("\n✅ All special_values Part 2 tests passed!")
