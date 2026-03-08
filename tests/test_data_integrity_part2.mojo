# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_data_integrity.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Data Integrity Tests for ExTensor Quantization Functions (Part 2).

Tests for Phase 3 data integrity fixes:
- #1912 (DATA-004): Unsafe bitcasts without bounds checks
- Quantization metadata preservation
- Backwards compatibility

Split from test_data_integrity.mojo per ADR-009 (≤10 fn test_ per file).
Run with: `mojo test_data_integrity_part2.mojo`
"""

from collections import List
from memory import UnsafePointer
from shared.core.extensor import ExTensor, zeros, ones
from tests.shared.conftest import assert_equal_int, assert_true


fn test_int_conversion_bounds() raises:
    """Test integer conversion with bounds checking."""
    print("Test: Integer conversion bounds checking...")

    var shape = List[Int]()
    shape.append(10)
    var t = zeros(shape, DType.float32)
    for i in range(10):
        t._data.bitcast[Float32]()[i] = Float32(i)

    # Test to_int8
    var i8_t = t.to_int8()
    assert_true(i8_t.numel() == 10, "Int8 tensor should have 10 elements")

    # Test to_int16
    var i16_t = t.to_int16()
    assert_true(i16_t.numel() == 10, "Int16 tensor should have 10 elements")

    # Test to_int32
    var i32_t = t.to_int32()
    assert_true(i32_t.numel() == 10, "Int32 tensor should have 10 elements")

    # Test to_uint8
    var u8_t = t.to_uint8()
    assert_true(u8_t.numel() == 10, "UInt8 tensor should have 10 elements")

    print("  ✓ Integer conversion bounds checking works")


fn test_metadata_preservation() raises:
    """Test that quantization metadata is properly set and used."""
    print("Test: Quantization metadata preservation...")

    var shape = List[Int]()
    shape.append(123)
    var t = zeros(shape, DType.float32)
    for i in range(123):
        t._data.bitcast[Float32]()[i] = Float32(i)

    # Encode to MXFP4
    var encoded = t.to_mxfp4()

    # Metadata should be set
    assert_true(
        encoded._original_numel_quantized == 123,
        "Original size should be stored in metadata",
    )

    # Decode using metadata
    var decoded = encoded.from_mxfp4()

    # Should restore exact size
    assert_true(
        decoded.numel() == 123,
        "Decoded should restore original size from metadata",
    )
    print("  ✓ Quantization metadata preserved and used correctly")


fn test_backwards_compatibility() raises:
    """Test backwards compatibility for non-quantized tensors."""
    print("Test: Backwards compatibility...")

    var shape = List[Int]()
    shape.append(10)
    var t = zeros(shape, DType.float32)

    # Non-quantized tensors should have _original_numel_quantized = -1
    assert_true(
        t._original_numel_quantized == -1,
        "Non-quantized tensor should have -1 flag",
    )

    # Regular operations should not be affected
    var t2 = t.copy()
    assert_true(t2.numel() == 10, "Copy should work normally")
    print("  ✓ Backwards compatibility maintained")


fn main() raises:
    """Run data integrity tests part 2."""
    print("=" * 60)
    print("DATA INTEGRITY TESTS PART 2 (ADR-009 split)")
    print("=" * 60)
    print()

    # DATA-004: Bounds checking
    test_int_conversion_bounds()
    print()

    # Metadata and backwards compatibility
    test_metadata_preservation()
    test_backwards_compatibility()
    print()

    print("=" * 60)
    print("DATA INTEGRITY TESTS PART 2 PASSED!")
    print("=" * 60)
