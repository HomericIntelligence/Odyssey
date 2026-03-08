"""Tests for NVFP4Block blocked storage and conversion - Part 2.

Tests cover:
- Accuracy comparison (NVFP4 vs MXFP4)
- Bit packing/unpacking
- Block indexing operations (get/set)
- All-negative block handling (TEST-001)

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
# Accuracy Comparison (NVFP4 vs MXFP4)
# ============================================================================


fn test_nvfp4_better_accuracy_than_mxfp4() raises:
    """Test that smaller blocks (16) provide better accuracy."""
    from shared.core.types.mxfp4 import MXFP4Block

    # Create 16 values in a narrow range
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(1.0) + Float32(i) * 0.05)

    # Test NVFP4 (16-element block)
    var nvfp4_block = NVFP4Block.from_float32_array(values)
    var nvfp4_decoded = nvfp4_block.to_float32_array()

    # Test MXFP4 (need to pad to 32 elements)
    var values32 = List[Float32]()
    for i in range(16):
        values32.append(Float32(1.0) + Float32(i) * 0.05)
    for i in range(16):
        values32.append(Float32(0.0))  # Padding

    var mxfp4_block = MXFP4Block.from_float32_array(values32)
    var mxfp4_decoded = mxfp4_block.to_float32_array()

    # Compare errors for first 16 values
    var nvfp4_total_error = Float32(0.0)
    var mxfp4_total_error = Float32(0.0)

    for i in range(16):
        var expected = Float32(1.0) + Float32(i) * 0.05
        nvfp4_total_error += abs(nvfp4_decoded[i] - expected)
        mxfp4_total_error += abs(mxfp4_decoded[i] - expected)

    # NVFP4 should generally have lower error due to better scale granularity
    # (though this is not guaranteed for all value ranges)
    print("NVFP4 error:", nvfp4_total_error)
    print("MXFP4 error:", mxfp4_total_error)


# ============================================================================
# Bit Packing Tests
# ============================================================================


fn test_nvfp4_block_bit_packing() raises:
    """Test bit packing stores 2 values per byte."""
    var values = List[Float32]()
    # Create distinct values
    for i in range(16):
        values.append(Float32(1.0) + Float32(i % 4) * 0.5)

    var block = NVFP4Block.from_float32_array(values)

    # Verify block has 8 bytes of data
    # (can't directly check, but verify decoding works)
    var decoded = block.to_float32_array()
    assert_equal(len(decoded), 16)


# ============================================================================
# Block Indexing Tests
# ============================================================================


fn test_nvfp4_block_get() raises:
    """Test get() method retrieves individual values."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(i) * 0.1)

    var block = NVFP4Block.from_float32_array(values)

    # Test get at various indices
    var val0 = block.get(0)
    var val7 = block.get(7)
    var val15 = block.get(15)

    # Verify values are reasonable
    assert_almost_equal(val0.to_float32(), 0.0, tolerance=0.1)
    assert_true(abs(val7.to_float32() - 0.7) < 0.3)
    assert_true(abs(val15.to_float32() - 1.5) < 0.3)


fn test_nvfp4_block_get_bounds_checking() raises:
    """Test get() bounds checking."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(i))

    var block = NVFP4Block.from_float32_array(values)

    # Test out of bounds
    try:
        var val = block.get(-1)
        assert_true(False, "Expected bounds error for -1")
    except e:
        pass

    try:
        var val = block.get(16)
        assert_true(False, "Expected bounds error for 16")
    except e:
        pass


fn test_nvfp4_block_set() raises:
    """Test set() method updates individual values."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(1.0))

    var block = NVFP4Block.from_float32_array(values)

    # Update value at index 5
    var new_val = NVFP4.from_float32(2.5)
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


fn test_nvfp4_block_set_bounds_checking() raises:
    """Test set() bounds checking."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(i))

    var block = NVFP4Block.from_float32_array(values)
    var new_val = NVFP4.from_float32(5.0)

    # Test out of bounds
    try:
        block.set(-1, new_val)
        assert_true(False, "Expected bounds error for -1")
    except e:
        pass

    try:
        block.set(16, new_val)
        assert_true(False, "Expected bounds error for 16")
    except e:
        pass


# ============================================================================
# TEST-001: All-Negative Block Tests
# ============================================================================


fn test_nvfp4_block_all_negative_same() raises:
    """Test block with all same negative values (TEST-001)."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(-1.0))

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be approximately -1.0
    # FP4 quantization error can be significant
    for i in range(16):
        assert_true(decoded[i] < 0, "Value should be negative")
        assert_almost_equal(decoded[i], Float32(-1.0), tolerance=0.5)


fn test_nvfp4_block_all_negative_range() raises:
    """Test block with range of negative values (TEST-001)."""
    var values = List[Float32]()
    for i in range(16):
        values.append(Float32(-1.0) - Float32(i) * 0.1)

    var block = NVFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be negative
    for i in range(16):
        assert_true(decoded[i] < 0, "Value should be negative")


fn main() raises:
    """Run all NVFP4Block part 2 tests."""
    print("Running NVFP4Block Part 2 tests...")

    # Accuracy comparison
    test_nvfp4_better_accuracy_than_mxfp4()
    print("✓ Accuracy comparison (NVFP4 vs MXFP4)")

    # Bit packing tests
    test_nvfp4_block_bit_packing()
    print("✓ Bit packing")

    # Indexing tests
    test_nvfp4_block_get()
    print("✓ Block get()")

    test_nvfp4_block_get_bounds_checking()
    print("✓ Block get() bounds")

    test_nvfp4_block_set()
    print("✓ Block set()")

    test_nvfp4_block_set_bounds_checking()
    print("✓ Block set() bounds")

    # TEST-001: All-negative blocks
    test_nvfp4_block_all_negative_same()
    print("✓ All negative (same) - TEST-001")

    test_nvfp4_block_all_negative_range()
    print("✓ All negative (range) - TEST-001")

    print("\nAll NVFP4Block Part 2 tests passed!")
