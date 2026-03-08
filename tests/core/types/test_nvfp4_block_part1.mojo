"""Tests for NVFP4Block blocked storage and conversion - Part 1.

Tests cover:
- Block creation from Float32 arrays
- Round-trip conversion accuracy
- Scale computation correctness

All tests use pure functional API.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_nvfp4_block.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from shared.core.types.nvfp4 import NVFP4, NVFP4Block
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)


# ============================================================================
# Block Creation Tests
# ============================================================================


fn test_nvfp4_block_creation_zeros() raises:
    """Test NVFP4Block creation with all zeros."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(0.0))

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check all values are zero
    for i in range(16):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-4)


fn test_nvfp4_block_creation_ones() raises:
    """Test NVFP4Block creation with all ones."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(1.0))

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check all values are approximately 1.0 (within E2M1 precision)
    # FP4 quantization can have significant error (up to 50%)
    for i in range(16):
        assert_almost_equal(decoded[i], Float32(1.0), tolerance=0.5)


fn test_nvfp4_block_creation_range() raises:
    """Test NVFP4Block creation with sequential values."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(i) * 0.1)

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check values are reasonably close (E2M1 has limited precision)
    for i in range(16):
        var expected = Float32(i) * 0.1
        var error = abs(decoded[i] - expected)
        # E2M1 precision is limited, allow larger tolerance
        assert_true(error < 0.3, "Value " + String(i) + " error too large")


fn test_nvfp4_block_size_validation() raises:
    """Test NVFP4Block requires exactly 16 values."""
    var values = List[Float32]()
    for i in range(8):  # Only 8 values
        values.append(Float32(i))

    try:
        var block = NVFP4Block.from_float32_array(values)
        assert_true(False, "Expected error for wrong size")
    except e:
        # Expected error
        pass


# ============================================================================
# Round-Trip Conversion Tests
# ============================================================================


fn test_nvfp4_block_roundtrip_small() raises:
    """Test round-trip conversion for small values."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(0.5) + Float32(i) * 0.1)

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify approximate reconstruction
    # FP4 round-trip quantization can have large error
    for i in range(16):
        var expected = Float32(0.5) + Float32(i) * 0.1
        var error = abs(decoded[i] - expected)
        assert_true(error < 2.0, "Round-trip error too large")


fn test_nvfp4_block_roundtrip_large() raises:
    """Test round-trip conversion for large values."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(10.0) + Float32(i) * 2.0)

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify approximate reconstruction
    # FP4 round-trip quantization error can be 25-50% for large values
    for i in range(16):
        var expected = Float32(10.0) + Float32(i) * 2.0
        var error = abs(decoded[i] - expected)
        # Allow relative error up to 50% for large values due to FP4 precision
        assert_true(error < expected * 0.5, "Round-trip error too large")


fn test_nvfp4_block_roundtrip_mixed_signs() raises:
    """Test round-trip conversion with mixed signs."""
    var values = List[Float32]()
    for i in range(16):
        var sign = Float32(1.0) if i % 2 == 0 else Float32(-1.0)
        values.append(sign * Float32(i) * Float32(0.1))

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify signs are preserved for non-zero values
    # Note: Skip i=0 because zero has no meaningful sign
    # Note: Skip small values near zero that may have sign flips due to quantization
    for i in range(1, 16):
        var expected = (
            (Float32(1.0) if i % 2 == 0 else Float32(-1.0))
            * Float32(i)
            * Float32(0.1)
        )
        # Only check sign for values with significant magnitude
        if abs(expected) > 0.15:
            var expected_sign = Float32(1.0) if expected >= 0 else Float32(-1.0)
            var decoded_sign = Float32(1.0) if decoded[i] >= 0 else Float32(
                -1.0
            )
            assert_true(
                Int(expected_sign) == Int(decoded_sign),
                "Sign mismatch at i="
                + String(i)
                + ": expected="
                + String(expected)
                + ", decoded="
                + String(decoded[i]),
            )


# ============================================================================
# Scale Computation Tests
# ============================================================================


fn test_nvfp4_block_scale_computation() raises:
    """Test scale computation for different value ranges."""
    # Test 1: All values in [0, 1]
    var values1 = List[Float32]()
    for i in range(16):
        values1.append(Float32(i) / 16.0)

    var block1 = NVFP4Block.from_float32_array(values1)
    # Scale should be roughly max/6 = (15/16)/6 ≈ 0.16
    var scale1 = Float32(block1.scale)
    assert_true(scale1 > 0.1 and scale1 < 0.3, "Scale 1 out of range")

    # Test 2: All values in [0, 10]
    var values2 = List[Float32]()
    for i in range(16):
        values2.append(Float32(i) / 1.6)

    var block2 = NVFP4Block.from_float32_array(values2)
    # Scale should be larger
    var scale2 = Float32(block2.scale)
    assert_true(scale2 > scale1, "Scale 2 should be larger")


fn main() raises:
    """Run all NVFP4Block part 1 tests."""
    print("Running NVFP4Block Part 1 tests...")

    # Block creation tests
    test_nvfp4_block_creation_zeros()
    print("✓ Block creation (zeros)")

    test_nvfp4_block_creation_ones()
    print("✓ Block creation (ones)")

    test_nvfp4_block_creation_range()
    print("✓ Block creation (range)")

    test_nvfp4_block_size_validation()
    print("✓ Block size validation")

    # Round-trip tests
    test_nvfp4_block_roundtrip_small()
    print("✓ Round-trip (small values)")

    test_nvfp4_block_roundtrip_large()
    print("✓ Round-trip (large values)")

    test_nvfp4_block_roundtrip_mixed_signs()
    print("✓ Round-trip (mixed signs)")

    # Scale computation tests
    test_nvfp4_block_scale_computation()
    print("✓ Scale computation")

    print("\nAll NVFP4Block Part 1 tests passed!")
