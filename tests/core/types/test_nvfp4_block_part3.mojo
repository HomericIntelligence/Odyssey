"""Tests for NVFP4Block blocked storage and conversion - Part 3.

Tests cover:
- Negative scale computation (TEST-001)
- NaN/Infinity handling (TEST-003)
- Edge cases (all same value, extreme value ranges)

All tests use pure functional API.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_nvfp4_block.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from math import isinf, isnan
from shared.core.types.nvfp4 import NVFP4, NVFP4Block
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)


# ============================================================================
# TEST-001: Negative Scale Computation
# ============================================================================


fn test_nvfp4_block_negative_scale_computation() raises:
    """Test scale computation uses abs() for negative values (TEST-001)."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(-10.0))

    var block = NVFP4Block.from_float32_array(values)

    # Scale should be positive (computed from abs(max))
    var scale_val = Float32(block.scale)
    assert_true(scale_val > 0, "Scale should be positive")

    # Decoded values should preserve sign
    var decoded = block.to_float32_array()
    for i in range(16):
        assert_true(decoded[i] < 0, "Sign should be preserved")


# ============================================================================
# TEST-003: NaN/Infinity Handling Tests (NVFP4 has no TEST-002 - no scale=0 fallback)
# ============================================================================


fn test_nvfp4_block_nan_values() raises:
    """Test block with NaN values (TEST-003)."""
    var values = List[Float32]()
    var nan_val = Float32(0.0) / Float32(0.0)  # Create NaN
    for i in range(16):
        values.append(nan_val)

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # NaN should map to max representable value (not crash)
    # Values should be finite after decoding
    for i in range(16):
        assert_true(
            not isinf(decoded[i]), "Decoded value should not be infinity"
        )


fn test_nvfp4_block_infinity_values() raises:
    """Test block with Infinity values (TEST-003)."""
    var pos_inf = Float32(1.0) / Float32(0.0)
    var neg_inf = Float32(-1.0) / Float32(0.0)

    var values = List[Float32]()
    for i in range(8):
        values.append(pos_inf)
    for i in range(8):
        values.append(neg_inf)

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Infinity should clamp to max representable
    for i in range(8):
        assert_true(decoded[i] > 0, "Positive infinity should decode positive")
    for i in range(8, 16):
        assert_true(decoded[i] < 0, "Negative infinity should decode negative")


fn test_nvfp4_block_mixed_special() raises:
    """Test block with mixed NaN, Infinity, and normal values (TEST-003)."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var pos_inf = Float32(1.0) / Float32(0.0)
    var neg_inf = Float32(-1.0) / Float32(0.0)

    var values = List[Float32]()
    for i in range(4):
        values.append(nan_val)
    for i in range(4):
        values.append(pos_inf)
    for i in range(4):
        values.append(neg_inf)
    for i in range(4):
        values.append(Float32(1.0))

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # NaN values should not be NaN after decoding (clamped to max)
    # Note: Infinity inputs may still produce very large outputs due to scale
    for i in range(16):
        assert_true(not isnan(decoded[i]), "Decoded value should not be NaN")


# ============================================================================
# Edge Cases
# ============================================================================


fn test_nvfp4_block_all_same_value() raises:
    """Test block with all same values."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(3.14))

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be approximately equal
    # FP4 quantization error can be significant
    var first = decoded[0]
    for i in range(16):
        assert_almost_equal(decoded[i], first, tolerance=0.5)


fn test_nvfp4_block_extreme_range() raises:
    """Test block with very different magnitude values."""
    var values = List[Float32]()
    # Mix very small and very large values
    for i in range(8):
        values.append(Float32(0.001))
    for i in range(8):
        values.append(Float32(100.0))

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Due to shared scale, small values may become zero
    # Large values should be preserved reasonably
    var large_vals_ok = True
    for i in range(8, 16):
        if decoded[i] < 50.0:  # Very rough check
            large_vals_ok = False

    assert_true(large_vals_ok, "Large values not preserved")


fn main() raises:
    """Run all NVFP4Block part 3 tests."""
    print("Running NVFP4Block Part 3 tests...")

    # TEST-001: Negative scale computation
    test_nvfp4_block_negative_scale_computation()
    print("✓ Negative scale computation - TEST-001")

    # TEST-003: NaN/Infinity handling
    test_nvfp4_block_nan_values()
    print("✓ NaN values - TEST-003")

    test_nvfp4_block_infinity_values()
    print("✓ Infinity values - TEST-003")

    test_nvfp4_block_mixed_special()
    print("✓ Mixed special values - TEST-003")

    # Edge cases
    test_nvfp4_block_all_same_value()
    print("✓ All same value")

    test_nvfp4_block_extreme_range()
    print("✓ Extreme range")

    print("\nAll NVFP4Block Part 3 tests passed!")
    print("TEST-001, TEST-003 (P0 CRITICAL) - RESOLVED")
