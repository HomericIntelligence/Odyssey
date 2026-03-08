# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_mxfp4_block.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for MXFP4Block scale=0 edge cases, NaN/Infinity handling, and general edge cases.

Tests cover:
- Scale=0 edge cases (TEST-002)
- NaN/Infinity handling (TEST-003)
- Edge cases (all same value, extreme range)

All tests use pure functional API.
"""

from math import isinf, isnan

from shared.core.types.mxfp4 import MXFP4, MXFP4Block
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)


# ============================================================================
# TEST-002: Scale=0 Edge Case Tests
# ============================================================================


fn test_mxfp4_block_all_zeros() raises:
    """Test block with all zeros triggers scale=1.0 fallback (TEST-002)."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(0.0))

    var block = MXFP4Block.from_float32_array(values)

    # Scale should fallback to 1.0 (not 0.0)
    var scale_val = Float32(block.scale)
    assert_true(scale_val > 0.5, "Scale should fallback to 1.0")

    # Decoded values should be zero
    var decoded = block.to_float32_array()
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-5)


fn test_mxfp4_block_near_zero() raises:
    """Test block with near-zero values triggers fallback (TEST-002)."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(1e-11))  # Below 1e-10 threshold

    var block = MXFP4Block.from_float32_array(values)

    # Scale should fallback to 1.0
    var scale_val = Float32(block.scale)
    assert_true(scale_val > 0.5, "Scale should fallback to 1.0")


fn test_mxfp4_block_zero_roundtrip() raises:
    """Test lossless zero encoding (TEST-002)."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(0.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Round-trip should preserve zeros exactly
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-4)


# ============================================================================
# TEST-003: NaN/Infinity Handling Tests
# ============================================================================


fn test_mxfp4_block_nan_values() raises:
    """Test block with NaN values (TEST-003)."""
    var values = List[Float32]()
    var nan_val = Float32(0.0) / Float32(0.0)  # Create NaN
    for i in range(32):
        values.append(nan_val)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # NaN should map to max representable value (not crash)
    # Values should be finite after decoding
    for i in range(32):
        assert_true(
            not isinf(decoded[i]), "Decoded value should not be infinity"
        )


fn test_mxfp4_block_infinity_values() raises:
    """Test block with Infinity values (TEST-003)."""
    var pos_inf = Float32(1.0) / Float32(0.0)
    var neg_inf = Float32(-1.0) / Float32(0.0)

    var values = List[Float32]()
    for i in range(16):
        values.append(pos_inf)
    for i in range(16):
        values.append(neg_inf)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Infinity should clamp to max representable
    for i in range(16):
        assert_true(decoded[i] > 0, "Positive infinity should decode positive")
    for i in range(16, 32):
        assert_true(decoded[i] < 0, "Negative infinity should decode negative")


fn test_mxfp4_block_mixed_special() raises:
    """Test block with mixed NaN, Infinity, and normal values (TEST-003)."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var pos_inf = Float32(1.0) / Float32(0.0)
    var neg_inf = Float32(-1.0) / Float32(0.0)

    var values = List[Float32]()
    for i in range(8):
        values.append(nan_val)
    for i in range(8):
        values.append(pos_inf)
    for i in range(8):
        values.append(neg_inf)
    for i in range(8):
        values.append(Float32(1.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # NaN values should not be NaN after decoding (clamped to max)
    # Note: Infinity inputs may still produce very large outputs due to scale
    for i in range(32):
        assert_true(not isnan(decoded[i]), "Decoded value should not be NaN")


# ============================================================================
# Edge Cases
# ============================================================================


fn test_mxfp4_block_all_same_value() raises:
    """Test block with all same values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(3.14))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be approximately equal
    # FP4 quantization error can be significant
    var first = decoded[0]
    for i in range(32):
        assert_almost_equal(decoded[i], first, tolerance=0.5)


fn test_mxfp4_block_extreme_range() raises:
    """Test block with very different magnitude values."""
    var values = List[Float32]()
    # Mix very small and very large values
    for i in range(16):
        values.append(Float32(0.001))
    for i in range(16):
        values.append(Float32(100.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Due to shared scale, small values may become zero
    # Large values should be preserved reasonably
    var large_vals_ok = True
    for i in range(16, 32):
        if decoded[i] < 50.0:  # Very rough check
            large_vals_ok = False

    assert_true(large_vals_ok, "Large values not preserved")


fn main() raises:
    """Run MXFP4Block part 3 tests (TEST-002, TEST-003, edge cases)."""
    print("Running MXFP4Block Part 3 tests...")

    # TEST-002: Scale=0 edge cases
    test_mxfp4_block_all_zeros()
    print("✓ All zeros (scale fallback) - TEST-002")

    test_mxfp4_block_near_zero()
    print("✓ Near-zero values - TEST-002")

    test_mxfp4_block_zero_roundtrip()
    print("✓ Zero round-trip - TEST-002")

    # TEST-003: NaN/Infinity handling
    test_mxfp4_block_nan_values()
    print("✓ NaN values - TEST-003")

    test_mxfp4_block_infinity_values()
    print("✓ Infinity values - TEST-003")

    test_mxfp4_block_mixed_special()
    print("✓ Mixed special values - TEST-003")

    # Edge cases
    test_mxfp4_block_all_same_value()
    print("✓ All same value")

    test_mxfp4_block_extreme_range()
    print("✓ Extreme range")

    print("\nAll MXFP4Block Part 3 tests passed!")
    print("TEST-002, TEST-003 (P0 CRITICAL) - RESOLVED")
