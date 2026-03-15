"""Tests for SIMD mixed precision FP16↔FP32 conversions.

Tests SIMD-optimized FP16→FP32 and FP32→FP16 conversions with various tensor sizes
and edge cases.
"""

from shared.core.extensor import ExTensor, full
from shared.training.mixed_precision import (
    convert_to_fp32_master,
    update_model_from_master,
)
from testing import assert_equal, assert_true


fn test_convert_fp16_to_fp32_small() raises:
    """Test FP16→FP32 conversion with small tensor."""
    print("Testing FP16→FP32 conversion (small)...")

    # Create small FP16 tensor with known values
    var fp16_params = ExTensor([8], DType.float16)
    var fp16_ptr = fp16_params._data.bitcast[Float16]()

    # Fill with test values
    for i in range(8):
        fp16_ptr[i] = Float16(Float32(i + 1))

    # Convert to FP32
    var fp32_result = convert_to_fp32_master(fp16_params)

    # Verify dtype
    assert_equal(fp32_result.dtype(), DType.float32, "Result should be float32")

    # Verify values are preserved
    var fp32_ptr = fp32_result._data.bitcast[Float32]()
    for i in range(8):
        var expected = Float32(i + 1)
        var actual = fp32_ptr[i]
        # Allow small precision loss due to FP16 -> FP32 conversion
        assert_true(
            abs(actual - expected) < 0.01,
            "Value at index " + String(i) + " mismatch",
        )

    print("✓ FP16→FP32 conversion (small) test passed")


fn test_convert_fp16_to_fp32_medium() raises:
    """Test FP16→FP32 conversion with medium tensor."""
    print("Testing FP16→FP32 conversion (medium)...")

    # Create medium FP16 tensor (256 elements)
    var fp16_params = ExTensor([16, 16], DType.float16)
    var fp16_ptr = fp16_params._data.bitcast[Float16]()

    # Fill with special values
    var test_values = [
        Float16(0.0),
        Float16(0.5),
        Float16(1.0),
        Float16(1.5),
        Float16(-0.5),
        Float16(-1.0),
        Float16(-1.5),
    ]

    for i in range(256):
        fp16_ptr[i] = test_values[i % 7]

    # Convert to FP32
    var fp32_result = convert_to_fp32_master(fp16_params)

    # Verify values
    var fp32_ptr = fp32_result._data.bitcast[Float32]()
    for i in range(256):
        var expected = Float32(test_values[i % 7])
        var actual = fp32_ptr[i]
        assert_true(
            abs(actual - expected) < 0.001,
            "Value at index " + String(i) + " mismatch",
        )

    print("✓ FP16→FP32 conversion (medium) test passed")


fn test_convert_fp16_to_fp32_large() raises:
    """Test FP16→FP32 conversion with large tensor."""
    print("Testing FP16→FP32 conversion (large)...")

    # Create large FP16 tensor (1024x1024 = 1M elements)
    var fp16_params = ExTensor([1024, 1024], DType.float16)
    var fp16_ptr = fp16_params._data.bitcast[Float16]()

    # Fill with incrementing values
    for i in range(1024 * 1024):
        fp16_ptr[i] = Float16(Float32((i % 100) / 100.0))

    # Convert to FP32
    var fp32_result = convert_to_fp32_master(fp16_params)

    # Spot check values
    var fp32_ptr = fp32_result._data.bitcast[Float32]()
    for i in [0, 100, 1000, 10000, 100000, 1000000 - 1]:
        var expected = Float32((i % 100) / 100.0)
        var actual = fp32_ptr[i]
        assert_true(
            abs(actual - expected) < 0.001,
            "Value at index " + String(i) + " mismatch",
        )

    print("✓ FP16→FP32 conversion (large) test passed")


fn test_convert_fp32_to_fp16_small() raises:
    """Test FP32→FP16 conversion with small tensor."""
    print("Testing FP32→FP16 conversion (small)...")

    # Create small FP32 tensor with known values
    var fp32_master = ExTensor([8], DType.float32)
    var fp32_ptr = fp32_master._data.bitcast[Float32]()

    # Fill with test values
    for i in range(8):
        fp32_ptr[i] = Float32(i + 1)

    # Create FP16 target
    var fp16_model = ExTensor([8], DType.float16)

    # Update from master
    update_model_from_master(fp16_model, fp32_master)

    # Verify values
    var fp16_ptr = fp16_model._data.bitcast[Float16]()
    for i in range(8):
        var expected = Float16(Float32(i + 1))
        var actual = fp16_ptr[i]
        assert_true(
            abs(Float32(actual) - Float32(expected)) < 0.01,
            "Value at index " + String(i) + " mismatch",
        )

    print("✓ FP32→FP16 conversion (small) test passed")


fn test_convert_fp32_to_fp16_medium() raises:
    """Test FP32→FP16 conversion with medium tensor."""
    print("Testing FP32→FP16 conversion (medium)...")

    # Create medium FP32 tensor
    var fp32_master = ExTensor([16, 16], DType.float32)
    var fp32_ptr = fp32_master._data.bitcast[Float32]()

    # Fill with special values
    var test_values = [
        Float32(0.0),
        Float32(0.5),
        Float32(1.0),
        Float32(1.5),
        Float32(-0.5),
        Float32(-1.0),
        Float32(-1.5),
    ]

    for i in range(256):
        fp32_ptr[i] = test_values[i % 7]

    # Create FP16 target
    var fp16_model = ExTensor([16, 16], DType.float16)

    # Update from master
    update_model_from_master(fp16_model, fp32_master)

    # Verify values
    var fp16_ptr = fp16_model._data.bitcast[Float16]()
    for i in range(256):
        var expected = Float16(test_values[i % 7])
        var actual = fp16_ptr[i]
        assert_true(
            abs(Float32(actual) - Float32(expected)) < 0.001,
            "Value at index " + String(i) + " mismatch",
        )

    print("✓ FP32→FP16 conversion (medium) test passed")


fn test_convert_fp32_to_fp16_large() raises:
    """Test FP32→FP16 conversion with large tensor."""
    print("Testing FP32→FP16 conversion (large)...")

    # Create large FP32 tensor
    var fp32_master = ExTensor([1024, 1024], DType.float32)
    var fp32_ptr = fp32_master._data.bitcast[Float32]()

    # Fill with incrementing values
    for i in range(1024 * 1024):
        fp32_ptr[i] = Float32((i % 100) / 100.0)

    # Create FP16 target
    var fp16_model = ExTensor([1024, 1024], DType.float16)

    # Update from master
    update_model_from_master(fp16_model, fp32_master)

    # Spot check values
    var fp16_ptr = fp16_model._data.bitcast[Float16]()
    for i in [0, 100, 1000, 10000, 100000, 1000000 - 1]:
        var expected = Float16(Float32((i % 100) / 100.0))
        var actual = fp16_ptr[i]
        assert_true(
            abs(Float32(actual) - Float32(expected)) < 0.001,
            "Value at index " + String(i) + " mismatch",
        )

    print("✓ FP32→FP16 conversion (large) test passed")


fn test_roundtrip_fp16_fp32_fp16() raises:
    """Test roundtrip conversion FP16→FP32→FP16."""
    print("Testing roundtrip FP16→FP32→FP16...")

    # Create initial FP16 tensor
    var fp16_original = ExTensor([64], DType.float16)
    var fp16_ptr = fp16_original._data.bitcast[Float16]()

    for i in range(64):
        fp16_ptr[i] = Float16(Float32((i % 10) / 10.0))

    # Convert to FP32
    var fp32_intermediate = convert_to_fp32_master(fp16_original)

    # Convert back to FP16
    var fp16_roundtrip = ExTensor([64], DType.float16)
    update_model_from_master(fp16_roundtrip, fp32_intermediate)

    # Verify roundtrip preserves values (with FP16 precision limits)
    var fp16_result_ptr = fp16_roundtrip._data.bitcast[Float16]()
    for i in range(64):
        var original = fp16_ptr[i]
        var result = fp16_result_ptr[i]
        assert_true(
            abs(Float32(original) - Float32(result)) < 0.001,
            "Roundtrip mismatch at index " + String(i),
        )

    print("✓ Roundtrip FP16→FP32→FP16 test passed")


fn test_convert_fp32_to_fp32() raises:
    """Test that FP32→FP32 conversion still works (SIMD path)."""
    print("Testing FP32→FP32 conversion (SIMD path)...")

    # Create FP32 tensor
    var fp32_params = ExTensor([64], DType.float32)
    var fp32_ptr = fp32_params._data.bitcast[Float32]()

    for i in range(64):
        fp32_ptr[i] = Float32(i + 1)

    # Convert to FP32 master (should use SIMD copy path)
    var fp32_master = convert_to_fp32_master(fp32_params)

    # Verify dtype and values
    assert_equal(fp32_master.dtype(), DType.float32, "Result should be float32")

    var master_ptr = fp32_master._data.bitcast[Float32]()
    for i in range(64):
        var expected = Float32(i + 1)
        var actual = master_ptr[i]
        assert_equal(
            actual,
            expected,
            "FP32→FP32 value mismatch at index " + String(i),
        )

    print("✓ FP32→FP32 conversion (SIMD path) test passed")


fn test_non_power_of_2_sizes() raises:
    """Test conversions with non-power-of-2 tensor sizes."""
    print("Testing non-power-of-2 tensor sizes...")

    # Test sizes: 7, 15, 33, 65, 127, 255, 1000
    var test_sizes = [7, 15, 33, 65, 127, 255, 1000]

    for size in test_sizes:
        # Test FP16→FP32
        var fp16_tensor = ExTensor([size], DType.float16)
        var fp16_ptr = fp16_tensor._data.bitcast[Float16]()

        for i in range(size):
            fp16_ptr[i] = Float16(Float32(i % 10))

        var fp32_result = convert_to_fp32_master(fp16_tensor)
        var fp32_ptr = fp32_result._data.bitcast[Float32]()

        for i in range(size):
            var expected = Float32(i % 10)
            var actual = fp32_ptr[i]
            assert_true(
                abs(actual - expected) < 0.01,
                "FP16→FP32 size " + String(size) + " mismatch at " + String(i),
            )

        # Test FP32→FP16
        var fp32_tensor = ExTensor([size], DType.float32)
        fp32_ptr = fp32_tensor._data.bitcast[Float32]()

        for i in range(size):
            fp32_ptr[i] = Float32(i % 10)

        var fp16_result = ExTensor([size], DType.float16)
        update_model_from_master(fp16_result, fp32_tensor)
        fp16_ptr = fp16_result._data.bitcast[Float16]()

        for i in range(size):
            var expected = Float16(Float32(i % 10))
            var actual = fp16_ptr[i]
            assert_true(
                abs(Float32(actual) - Float32(expected)) < 0.01,
                "FP32→FP16 size " + String(size) + " mismatch at " + String(i),
            )

    print("✓ Non-power-of-2 tensor sizes test passed")


fn main() raises:
    """Run all SIMD mixed precision tests."""
    print("\n=== SIMD Mixed Precision Tests ===\n")

    test_convert_fp16_to_fp32_small()
    test_convert_fp16_to_fp32_medium()
    test_convert_fp16_to_fp32_large()
    test_convert_fp32_to_fp16_small()
    test_convert_fp32_to_fp16_medium()
    test_convert_fp32_to_fp16_large()
    test_roundtrip_fp16_fp32_fp16()
    test_convert_fp32_to_fp32()
    test_non_power_of_2_sizes()

    print("\n=== All SIMD Mixed Precision Tests Passed! ===\n")
