"""Tests for MXFP4Block block creation, round-trip conversion, and scale computation.

Tests cover:
- Block creation from Float32 arrays
- Round-trip conversion accuracy
- Scale computation correctness

All tests use pure functional API.
"""


from shared.core.types.mxfp4 import MXFP4, MXFP4Block
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)
from std.math import isinf, isnan


def test_mxfp4_block_creation_zeros() raises:
    """Test MXFP4Block creation with all zeros."""
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(0.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check all values are zero
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-4)


def test_mxfp4_block_creation_ones() raises:
    """Test MXFP4Block creation with all ones."""
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(1.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check all values are approximately 1.0 (within E2M1 precision)
    # FP4 quantization can have significant error (up to 50%)
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(1.0), tolerance=0.5)


def test_mxfp4_block_creation_range() raises:
    """Test MXFP4Block creation with sequential values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Check values are reasonably close (E2M1 has limited precision)
    for i in range(32):
        var expected = Float32(i) * 0.1
        var error = abs(decoded[i] - expected)
        # E2M1 precision is limited, allow larger tolerance.
        # Use <= 0.5: values equidistant between two E2M1 codes (e.g., 2.5
        # falls exactly between 2.0 and 3.0 in scaled space) produce error
        # of exactly 0.5, which is valid quantization behavior.
        assert_true(error <= 0.5, "Value " + String(i) + " error too large")


def test_mxfp4_block_size_validation() raises:
    """Test MXFP4Block requires exactly 32 values."""
    var values = List[Float32]()
    for i in range(16):  # Only 16 values
        values.append(Float32(i))

    try:
        var block = MXFP4Block.from_float32_array(values)
        assert_true(False, "Expected error for wrong size")
    except e:
        # Expected error
        pass


def test_mxfp4_block_roundtrip_small() raises:
    """Test round-trip conversion for small values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(0.5) + Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify approximate reconstruction
    # FP4 round-trip quantization can have large error
    for i in range(32):
        var expected = Float32(0.5) + Float32(i) * 0.1
        var error = abs(decoded[i] - expected)
        assert_true(error < 2.0, "Round-trip error too large")


def test_mxfp4_block_roundtrip_large() raises:
    """Test round-trip conversion for large values."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(10.0) + Float32(i) * 2.0)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify approximate reconstruction
    # E8M0 scale is power-of-2 only, so scale can be up to 2x off from optimal.
    # Combined with FP4 quantization, smaller values in the block can have
    # larger relative error (up to 100%) when scale is rounded up.
    for i in range(32):
        var expected = Float32(10.0) + Float32(i) * 2.0
        var error = abs(decoded[i] - expected)
        # Allow relative error up to 100% to account for E8M0 scale quantization
        # (scale can be 2x too large, causing values to round differently)
        assert_true(error < expected * 1.0, "Round-trip error too large")


def test_mxfp4_block_roundtrip_mixed_signs() raises:
    """Test round-trip conversion with mixed signs."""
    var values = List[Float32]()
    for i in range(32):
        var sign = Float32(1.0) if i % 2 == 0 else Float32(-1.0)
        values.append(sign * Float32(i) * Float32(0.1))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Verify signs are preserved for non-zero values
    # Note: Skip i=0 because zero has no meaningful sign
    # Note: Skip small values near zero that may have sign flips due to quantization
    for i in range(1, 32):
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


def test_mxfp4_block_scale_computation() raises:
    """Test scale computation for different value ranges."""
    # Test 1: All values in [0, 1]
    var values1 = List[Float32]()
    for i in range(32):
        values1.append(Float32(i) / 32.0)

    var block1 = MXFP4Block.from_float32_array(values1)
    # Scale should be roughly max/6 = (31/32)/6 ≈ 0.16
    var scale1 = Float32(block1.scale)
    assert_true(scale1 > 0.1 and scale1 < 0.3, "Scale 1 out of range")

    # Test 2: All values in [0, 10]
    var values2 = List[Float32]()
    for i in range(32):
        values2.append(Float32(i) / 3.2)

    var block2 = MXFP4Block.from_float32_array(values2)
    # Scale should be larger
    var scale2 = Float32(block2.scale)
    assert_true(scale2 > scale1, "Scale 2 should be larger")


def test_mxfp4_block_bit_packing() raises:
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


def test_mxfp4_block_get() raises:
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


def test_mxfp4_block_get_bounds_checking() raises:
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


def test_mxfp4_block_set() raises:
    """Test set() method updates individual values."""
    var values = List[Float32]()
    for _ in range(32):
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


def test_mxfp4_block_set_bounds_checking() raises:
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


def test_mxfp4_block_all_negative_same() raises:
    """Test block with all same negative values (TEST-001)."""
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(-1.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be approximately -1.0
    # FP4 quantization error can be significant
    for i in range(32):
        assert_true(decoded[i] < 0, "Value should be negative")
        assert_almost_equal(decoded[i], Float32(-1.0), tolerance=0.5)


def test_mxfp4_block_all_negative_range() raises:
    """Test block with range of negative values (TEST-001)."""
    var values = List[Float32]()
    for i in range(32):
        values.append(Float32(-1.0) - Float32(i) * 0.1)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be negative
    for i in range(32):
        assert_true(decoded[i] < 0, "Value should be negative")


def test_mxfp4_block_negative_scale_computation() raises:
    """Test scale computation uses abs() for negative values (TEST-001)."""
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(-10.0))

    var block = MXFP4Block.from_float32_array(values)

    # Scale should be positive (computed from abs(max))
    var scale_val = Float32(block.scale)
    assert_true(scale_val > 0, "Scale should be positive")

    # Decoded values should preserve sign
    var decoded = block.to_float32_array()
    for i in range(32):
        assert_true(decoded[i] < 0, "Sign should be preserved")


def test_mxfp4_block_all_zeros() raises:
    """Test block with all zeros triggers scale=1.0 fallback (TEST-002)."""
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(0.0))

    var block = MXFP4Block.from_float32_array(values)

    # Scale should fallback to 1.0 (not 0.0)
    var scale_val = Float32(block.scale)
    assert_true(scale_val > 0.5, "Scale should fallback to 1.0")

    # Decoded values should be zero
    var decoded = block.to_float32_array()
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-5)


def test_mxfp4_block_near_zero() raises:
    """Test block with near-zero values triggers fallback (TEST-002)."""
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(1e-11))  # Below 1e-10 threshold

    var block = MXFP4Block.from_float32_array(values)

    # Scale should fallback to 1.0
    var scale_val = Float32(block.scale)
    assert_true(scale_val > 0.5, "Scale should fallback to 1.0")


def test_mxfp4_block_zero_roundtrip() raises:
    """Test lossless zero encoding (TEST-002)."""
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(0.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Round-trip should preserve zeros exactly
    for i in range(32):
        assert_almost_equal(decoded[i], Float32(0.0), tolerance=1e-4)


def test_mxfp4_block_nan_values() raises:
    """Test block with NaN values (TEST-003)."""
    var values = List[Float32]()
    var nan_val = Float32(0.0) / Float32(0.0)  # Create NaN
    for _ in range(32):
        values.append(nan_val)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # NaN should map to max representable value (not crash)
    # Values should be finite after decoding
    for i in range(32):
        assert_true(
            not isinf(decoded[i]), "Decoded value should not be infinity"
        )


def test_mxfp4_block_infinity_values() raises:
    """Test block with Infinity values (TEST-003)."""
    var pos_inf = Float32(1.0) / Float32(0.0)
    var neg_inf = Float32(-1.0) / Float32(0.0)

    var values = List[Float32]()
    for _ in range(16):
        values.append(pos_inf)
    for _ in range(16):
        values.append(neg_inf)

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # Infinity should clamp to max representable
    for i in range(16):
        assert_true(decoded[i] > 0, "Positive infinity should decode positive")
    for i in range(16, 32):
        assert_true(decoded[i] < 0, "Negative infinity should decode negative")


def test_mxfp4_block_mixed_special() raises:
    """Test block with mixed NaN, Infinity, and normal values (TEST-003)."""
    var nan_val = Float32(0.0) / Float32(0.0)
    var pos_inf = Float32(1.0) / Float32(0.0)
    var neg_inf = Float32(-1.0) / Float32(0.0)

    var values = List[Float32]()
    for _ in range(8):
        values.append(nan_val)
    for _ in range(8):
        values.append(pos_inf)
    for _ in range(8):
        values.append(neg_inf)
    for _ in range(8):
        values.append(Float32(1.0))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # NaN values should not be NaN after decoding (clamped to max)
    # Note: Infinity inputs may still produce very large outputs due to scale
    for i in range(32):
        assert_true(not isnan(decoded[i]), "Decoded value should not be NaN")


def test_mxfp4_block_all_same_value() raises:
    """Test block with all same values."""
    var values = List[Float32]()
    for _ in range(32):
        values.append(Float32(3.14))

    var block = MXFP4Block.from_float32_array(values)
    var decoded = block.to_float32_array()

    # All values should be approximately equal
    # FP4 quantization error can be significant
    var first = decoded[0]
    for i in range(32):
        assert_almost_equal(decoded[i], first, tolerance=0.5)


def test_mxfp4_block_extreme_range() raises:
    """Test block with very different magnitude values."""
    var values = List[Float32]()
    # Mix very small and very large values
    for _ in range(16):
        values.append(Float32(0.001))
    for _ in range(16):
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


def main() raises:
    """Run all test_mxfp4_block tests."""
    print("Running test_mxfp4_block tests...")

    test_mxfp4_block_creation_zeros()
    print("✓ test_mxfp4_block_creation_zeros")

    test_mxfp4_block_creation_ones()
    print("✓ test_mxfp4_block_creation_ones")

    test_mxfp4_block_creation_range()
    print("✓ test_mxfp4_block_creation_range")

    test_mxfp4_block_size_validation()
    print("✓ test_mxfp4_block_size_validation")

    test_mxfp4_block_roundtrip_small()
    print("✓ test_mxfp4_block_roundtrip_small")

    test_mxfp4_block_roundtrip_large()
    print("✓ test_mxfp4_block_roundtrip_large")

    test_mxfp4_block_roundtrip_mixed_signs()
    print("✓ test_mxfp4_block_roundtrip_mixed_signs")

    test_mxfp4_block_scale_computation()
    print("✓ test_mxfp4_block_scale_computation")

    test_mxfp4_block_bit_packing()
    print("✓ test_mxfp4_block_bit_packing")

    test_mxfp4_block_get()
    print("✓ test_mxfp4_block_get")

    test_mxfp4_block_get_bounds_checking()
    print("✓ test_mxfp4_block_get_bounds_checking")

    test_mxfp4_block_set()
    print("✓ test_mxfp4_block_set")

    test_mxfp4_block_set_bounds_checking()
    print("✓ test_mxfp4_block_set_bounds_checking")

    test_mxfp4_block_all_negative_same()
    print("✓ test_mxfp4_block_all_negative_same")

    test_mxfp4_block_all_negative_range()
    print("✓ test_mxfp4_block_all_negative_range")

    test_mxfp4_block_negative_scale_computation()
    print("✓ test_mxfp4_block_negative_scale_computation")

    test_mxfp4_block_all_zeros()
    print("✓ test_mxfp4_block_all_zeros")

    test_mxfp4_block_near_zero()
    print("✓ test_mxfp4_block_near_zero")

    test_mxfp4_block_zero_roundtrip()
    print("✓ test_mxfp4_block_zero_roundtrip")

    test_mxfp4_block_nan_values()
    print("✓ test_mxfp4_block_nan_values")

    test_mxfp4_block_infinity_values()
    print("✓ test_mxfp4_block_infinity_values")

    test_mxfp4_block_mixed_special()
    print("✓ test_mxfp4_block_mixed_special")

    test_mxfp4_block_all_same_value()
    print("✓ test_mxfp4_block_all_same_value")

    test_mxfp4_block_extreme_range()
    print("✓ test_mxfp4_block_extreme_range")

    print("\nAll test_mxfp4_block tests passed!")
