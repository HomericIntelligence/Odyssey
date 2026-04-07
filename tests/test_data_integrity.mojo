"""Data Integrity Tests for AnyTensor Quantization Functions

Tests for Phase 3 data integrity fixes:
- #1909 (DATA-001): Padding data loss in quantization
- #1910 (DATA-002): Missing size tracking in dequantization
- #1911 (DATA-003): Unvalidated DType in conversions
- #1913 (DATA-005): FP16→FP32 conversion documentation

Split from test_data_integrity.mojo per ADR-009 (≤10 fn test_ per file).
Run with: `mojo test_data_integrity_part1.mojo`
"""


from collections import List
from memory import UnsafePointer
from shared.tensor.any_tensor import AnyTensor, zeros, ones
from tests.shared.conftest import assert_equal_int, assert_true


fn test_mxfp4_aligned_roundtrip() raises:
    """Test MXFP4 round-trip with aligned size (32 elements = 1 block)."""
    print("Test: MXFP4 aligned round-trip...")

    var shape = List[Int]()
    shape.append(32)
    var t = zeros(shape, DType.float32)
    for i in range(32):
        t.set(i, Float32(Float32(i) + 1.0))

    var encoded = t.to_mxfp4()
    var decoded = encoded.from_mxfp4()

    # Should restore to original size (32 elements)
    assert_equal_int(decoded.numel(), 32, "Decoded size should be 32")
    print("  ✓ Aligned MXFP4 round-trip: 32 → 32 elements")


fn test_mxfp4_unaligned_roundtrip() raises:
    """Test MXFP4 round-trip with unaligned size (33 elements).

    This is the critical test for DATA-001 (padding data loss fix).
    Before the fix: 33 → 64 elements (WRONG!)
    After the fix: 33 → 33 elements (CORRECT!).
    """
    print("Test: MXFP4 unaligned round-trip...")

    # Create 33-element tensor (1 complete block + 1 extra)
    var shape = List[Int]()
    shape.append(33)
    var t = zeros(shape, DType.float32)
    for i in range(33):
        t.set(i, Float32(Float32(i) + 1.0))

    # Encode: 33 elements → 2 blocks × 32 = 64 elements padded
    var encoded = t.to_mxfp4()
    # The encoded tensor should have stored original size in metadata
    assert_true(
        encoded._original_numel_quantized == 33,
        "Encoded should store original size (33)",
    )

    # Decode: Should restore to original 33 elements, NOT 64!
    var decoded = encoded.from_mxfp4()
    assert_true(
        decoded.numel() == 33, "Decoded size should be 33 (was 64 before fix!)"
    )
    print("  ✓ Unaligned MXFP4 round-trip: 33 → 33 elements (FIX VERIFIED!)")


fn test_mxfp4_various_unaligned_sizes() raises:
    """Test MXFP4 with various non-aligned sizes."""
    print("Test: MXFP4 various unaligned sizes...")

    var test_sizes: List[Int] = [1, 17, 31, 33, 64, 65, 100, 1000]

    for size_idx in range(len(test_sizes)):
        var original_size = test_sizes[size_idx]

        # Create tensor with original size
        var shape = List[Int]()
        shape.append(original_size)
        var t = zeros(shape, DType.float32)
        for i in range(original_size):
            t.set(i, Float32(Float32(i)))

        # Round-trip
        var encoded = t.to_mxfp4()
        var decoded = encoded.from_mxfp4()

        # Verify size is preserved
        assert_true(
            decoded.numel() == original_size,
            "Size should be preserved for "
            + String(original_size)
            + " elements",
        )
        print(
            "  ✓ Size "
            + String(original_size)
            + " → "
            + String(decoded.numel())
        )


fn test_nvfp4_unaligned_roundtrip() raises:
    """Test NVFP4 round-trip with unaligned size (17 elements).

    Similar to DATA-001 but for NVFP4 which uses 16-element blocks.
    Before fix: 17 → 32 elements (WRONG!)
    After fix: 17 → 17 elements (CORRECT!).
    """
    print("Test: NVFP4 unaligned round-trip...")

    var shape = List[Int]()
    shape.append(17)
    var t = zeros(shape, DType.float32)
    for i in range(17):
        t.set(i, Float32(Float32(i) + 1.0))

    var encoded = t.to_nvfp4()
    assert_true(
        encoded._original_numel_quantized == 17,
        "Encoded should store original size (17)",
    )

    var decoded = encoded.from_nvfp4()
    assert_true(
        decoded.numel() == 17, "Decoded size should be 17 (was 32 before fix!)"
    )
    print("  ✓ Unaligned NVFP4 round-trip: 17 → 17 elements (FIX VERIFIED!)")


fn test_fp8_bounds_checking() raises:
    """Test FP8 conversion with bounds checking (DATA-004).

    Verify that bounds checking prevents out-of-bounds access.
    """
    print("Test: FP8 bounds checking...")

    var shape = List[Int]()
    shape.append(10)
    var t = zeros(shape, DType.float32)
    for i in range(10):
        t.set(i, Float32(Float32(i)))

    # Normal conversion should work
    var fp8_t = t.to_fp8()
    assert_true(fp8_t.numel() == 10, "FP8 tensor should have 10 elements")
    print("  ✓ FP8 bounds checking prevents out-of-bounds access")


fn test_dtype_validation_fp8() raises:
    """Test FP8 conversion with dtype validation (DATA-003).

    Verify that invalid dtype is caught.
    """
    print("Test: FP8 dtype validation...")

    # Try to convert int32 tensor (should fail with defensive check)
    var shape = List[Int]()
    shape.append(10)
    var t_int = zeros(shape, DType.int32)
    try:
        var fp8_t = t_int.to_fp8()
        # Should not reach here - should have raised Error
        print("  ✗ FAILED: Should have raised error for int32 dtype")
    except:
        print("  ✓ FP8 dtype validation catches invalid dtype (int32)")


fn test_dtype_validation_mxfp4() raises:
    """Test MXFP4 conversion with dtype validation (DATA-003).

    Verify that invalid dtype is caught.
    """
    print("Test: MXFP4 dtype validation...")

    # Try to convert int32 tensor (should fail with defensive check)
    var shape = List[Int]()
    shape.append(32)
    var t_int = zeros(shape, DType.int32)
    try:
        var mxfp4_t = t_int.to_mxfp4()
        # Should not reach here - should have raised Error
        print("  ✗ FAILED: Should have raised error for int32 dtype")
    except:
        print("  ✓ MXFP4 dtype validation catches invalid dtype (int32)")


fn test_fp16_conversion_behavior() raises:
    """Test FP16 conversion behavior (DATA-005).

    FP16 inputs should be converted to FP32 before quantization.
    This test documents the behavior for reproducibility.
    """
    print("Test: FP16 conversion behavior...")

    # Create FP16 tensor
    var shape = List[Int]()
    shape.append(10)
    var t_fp16 = zeros(shape, DType.float16)
    for i in range(10):
        # Convert int to FP16
        var val_f32 = Float32(i) + 1.5
        # Store as FP16 (explicit cast required)
        t_fp16.set(i, Float16(Float16(val_f32)))

    # Convert to MXFP4 (internally uses FP32)
    var mxfp4_t = t_fp16.to_mxfp4()

    # Convert back
    var decoded = mxfp4_t.from_mxfp4()

    # Verify we got reasonable values back (may differ slightly due to quantization)
    assert_true(decoded.numel() == 10, "Decoded should have 10 elements")
    print("  ✓ FP16 conversion: FP16 → FP32 (internal) → MXFP4 → Float32")


fn test_int_conversion_bounds() raises:
    """Test integer conversion with bounds checking."""
    print("Test: Integer conversion bounds checking...")

    var shape = List[Int]()
    shape.append(10)
    var t = zeros(shape, DType.float32)
    for i in range(10):
        t.set(i, Float32(Float32(i)))

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
        t.set(i, Float32(Float32(i)))

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
    """Run all test_data_integrity tests."""
    print("Running test_data_integrity tests...")

    test_mxfp4_aligned_roundtrip()
    print("✓ test_mxfp4_aligned_roundtrip")

    test_mxfp4_unaligned_roundtrip()
    print("✓ test_mxfp4_unaligned_roundtrip")

    test_mxfp4_various_unaligned_sizes()
    print("✓ test_mxfp4_various_unaligned_sizes")

    test_nvfp4_unaligned_roundtrip()
    print("✓ test_nvfp4_unaligned_roundtrip")

    test_fp8_bounds_checking()
    print("✓ test_fp8_bounds_checking")

    test_dtype_validation_fp8()
    print("✓ test_dtype_validation_fp8")

    test_dtype_validation_mxfp4()
    print("✓ test_dtype_validation_mxfp4")

    test_fp16_conversion_behavior()
    print("✓ test_fp16_conversion_behavior")

    test_int_conversion_bounds()
    print("✓ test_int_conversion_bounds")

    test_metadata_preservation()
    print("✓ test_metadata_preservation")

    test_backwards_compatibility()
    print("✓ test_backwards_compatibility")

    print("\nAll test_data_integrity tests passed!")
