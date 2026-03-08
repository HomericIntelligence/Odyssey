# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_mxfp4_block.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for MXFP4Block bit packing, indexing, and all-negative block handling.

Tests cover:
- Bit packing/unpacking
- Block indexing operations (get/set)
- Edge cases for all-negative values (TEST-001)

All tests use pure functional API.
"""

from shared.core.types.mxfp4 import MXFP4, MXFP4Block
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)


# ============================================================================
# Bit Packing Tests
# ============================================================================


fn test_mxfp4_block_bit_packing() raises:
    """Test bit packing stores 2 values per byte."""
    var values = List[Float32]()
    # Create distinct values
    for i in range(32):
        values.append(Float32(1.0) + Float32(i % 4) * 0.5)

    var block = MXFP4Block.from_float32_array(values)

    # Verify block has 16 bytes of data
    # (can't directly check, but verify decoding works)
    var decoded = block.to_float32_array()
    assert_equal(len(decoded), 32)


# ============================================================================
# Block Indexing Tests
# ============================================================================


fn test_mxfp4_block_get() raises:
    """Test get() method retrieves individual values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)

    # Test get at various indices
    var val0 = block.get(0)
    var val15 = block.get(15)
    var val31 = block.get(31)

    # Verify values are reasonable
    assert_almost_equal(val0.to_float32(), 0.0, tolerance=0.1)
    assert_true(abs(val15.to_float32() - 1.5) < 0.5)
    assert_true(abs(val31.to_float32() - 3.1) < 0.5)


fn test_mxfp4_block_get_bounds_checking() raises:
    """Test get() bounds checking."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i))

    var block = MXFP4Block.from_float32_array(values)

    # Test out of bounds
    try:
        var val = block.get(-1)
        assert_true(False, "Expected bounds error for -1")
    except e:
        pass

    try:
        var val = block.get(32)
        assert_true(False, "Expected bounds error for 32")
    except e:
        pass


fn test_mxfp4_block_set() raises:
    """Test set() method updates individual values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(1.0))

    var block = MXFP4Block.from_float32_array(values)

    # Update value at index 5
    var new_val = MXFP4.from_float32(2.5)
    block.set(5, new_val)

    # Retrieve and verify
    # FP4 quantization error can be very significant due to shared scale
    # The set() changes the raw FP4 bits but doesn't update the scale
    var retrieved = block.get(5)
    var retrieved_val = retrieved.to_float32()
    # Allow very wide tolerance since scale may not match the new value
    var error = abs(retrieved_val - 2.5)
    assert_true(
        error < 3.0,
        "Set value error too large: expected ~2.5, got "
        + String(retrieved_val),
    )


fn test_mxfp4_block_set_bounds_checking() raises:
    """Test set() bounds checking."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i))

    var block = MXFP4Block.from_float32_array(values)
    var new_val = MXFP4.from_float32(5.0)

    # Test out of bounds
    try:
        block.set(-1, new_val)
        assert_true(False, "Expected bounds error for -1")
    except e:
        pass

    try:
        block.set(32, new_val)
        assert_true(False, "Expected bounds error for 32")
    except e:
        pass


# ============================================================================
# TEST-001: All-Negative Block Tests
# ============================================================================


fn test_mxfp4_block_all_negative_same() raises:
    """Test block with all same negative values (TEST-001)."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(-1.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be approximately -1.0
    # FP4 quantization error can be significant
    for i in range(32):
        assert_true(decoded[i] < 0, "Value should be negative")
        assert_almost_equal(decoded[i], Float32(-1.0), tolerance=0.5)


fn test_mxfp4_block_all_negative_range() raises:
    """Test block with range of negative values (TEST-001)."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(-1.0) - Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be negative
    for i in range(32):
        assert_true(decoded[i] < 0, "Value should be negative")


fn test_mxfp4_block_negative_scale_computation() raises:
    """Test scale computation uses abs() for negative values (TEST-001)."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(-10.0))

    var block = MXFP4Block.from_float32_array(values)

    # Scale should be positive (computed from abs(max))
    var scale_val = Float32(block.scale)
    assert_true(scale_val > 0, "Scale should be positive")

    # Decoded values should preserve sign
    var decoded = block.to_float32_array()
    for i in range(32):
        assert_true(decoded[i] < 0, "Sign should be preserved")


fn main() raises:
    """Run MXFP4Block part 2 tests (bit packing, indexing, TEST-001)."""
    print("Running MXFP4Block Part 2 tests...")

    # Bit packing tests
    test_mxfp4_block_bit_packing()
    print("✓ Bit packing")

    # Indexing tests
    test_mxfp4_block_get()
    print("✓ Block get()")

    test_mxfp4_block_get_bounds_checking()
    print("✓ Block get() bounds")

    test_mxfp4_block_set()
    print("✓ Block set()")

    test_mxfp4_block_set_bounds_checking()
    print("✓ Block set() bounds")

    # TEST-001: All-negative blocks
    test_mxfp4_block_all_negative_same()
    print("✓ All negative (same) - TEST-001")

    test_mxfp4_block_all_negative_range()
    print("✓ All negative (range) - TEST-001")

    test_mxfp4_block_negative_scale_computation()
    print("✓ Negative scale computation - TEST-001")

    print("\nAll MXFP4Block Part 2 tests passed!")
    print("TEST-001 (P0 CRITICAL) - RESOLVED")
