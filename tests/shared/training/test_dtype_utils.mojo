"""Tests for dtype utilities and aliases."""

from shared.training.dtype_utils import (
    float16_dtype,
    float32_dtype,
    float64_dtype,
    bfloat16_dtype,
    is_reduced_precision,
    is_floating_point,
    get_dtype_precision_bits,
    get_dtype_exponent_bits,
    dtype_to_string,
    recommend_precision_dtype,
)
from testing import assert_equal, assert_true, assert_false


fn test_dtype_aliases() raises:
    """Test that dtype aliases point to correct types."""
    print("Testing dtype aliases...")

    assert_equal(
        float16_dtype, DType.float16, "float16_dtype should be DType.float16"
    )
    assert_equal(
        float32_dtype, DType.float32, "float32_dtype should be DType.float32"
    )
    assert_equal(
        float64_dtype, DType.float64, "float64_dtype should be DType.float64"
    )

    # BFloat16 now uses native DType.bfloat16
    assert_equal(
        bfloat16_dtype,
        DType.bfloat16,
        "bfloat16_dtype should be DType.bfloat16",
    )

    print("✓ DType aliases test passed")


fn test_is_reduced_precision() raises:
    """Test reduced precision detection."""
    print("Testing is_reduced_precision...")

    assert_true(
        is_reduced_precision(DType.float16), "FP16 should be reduced precision"
    )
    assert_false(
        is_reduced_precision(DType.float32),
        "FP32 should not be reduced precision",
    )
    assert_false(
        is_reduced_precision(DType.float64),
        "FP64 should not be reduced precision",
    )
    assert_false(
        is_reduced_precision(DType.int32),
        "Int32 should not be reduced precision",
    )

    print("✓ Reduced precision detection test passed")


fn test_is_floating_point() raises:
    """Test floating point type detection."""
    print("Testing is_floating_point...")

    assert_true(
        is_floating_point(DType.float16), "FP16 should be floating point"
    )
    assert_true(
        is_floating_point(DType.float32), "FP32 should be floating point"
    )
    assert_true(
        is_floating_point(DType.float64), "FP64 should be floating point"
    )
    assert_false(
        is_floating_point(DType.int32), "Int32 should not be floating point"
    )
    assert_false(
        is_floating_point(DType.uint8), "UInt8 should not be floating point"
    )

    print("✓ Floating point detection test passed")


fn test_get_dtype_precision_bits() raises:
    """Test precision bits retrieval."""
    print("Testing get_dtype_precision_bits...")

    assert_equal(
        get_dtype_precision_bits(DType.float16),
        10,
        "FP16 should have 10 mantissa bits",
    )
    assert_equal(
        get_dtype_precision_bits(DType.float32),
        23,
        "FP32 should have 23 mantissa bits",
    )
    assert_equal(
        get_dtype_precision_bits(DType.float64),
        52,
        "FP64 should have 52 mantissa bits",
    )
    assert_equal(
        get_dtype_precision_bits(DType.int32),
        0,
        "Int32 should return 0 mantissa bits",
    )

    print("✓ Precision bits test passed")


fn test_get_dtype_exponent_bits() raises:
    """Test exponent bits retrieval."""
    print("Testing get_dtype_exponent_bits...")

    assert_equal(
        get_dtype_exponent_bits(DType.float16),
        5,
        "FP16 should have 5 exponent bits",
    )
    assert_equal(
        get_dtype_exponent_bits(DType.float32),
        8,
        "FP32 should have 8 exponent bits",
    )
    assert_equal(
        get_dtype_exponent_bits(DType.float64),
        11,
        "FP64 should have 11 exponent bits",
    )
    assert_equal(
        get_dtype_exponent_bits(DType.int32),
        0,
        "Int32 should return 0 exponent bits",
    )

    print("✓ Exponent bits test passed")


fn test_dtype_to_string() raises:
    """Test dtype string conversion."""
    print("Testing dtype_to_string...")

    assert_equal(
        dtype_to_string(DType.float16), "float16", "FP16 should be 'float16'"
    )
    assert_equal(
        dtype_to_string(DType.float32), "float32", "FP32 should be 'float32'"
    )
    assert_equal(
        dtype_to_string(DType.float64), "float64", "FP64 should be 'float64'"
    )
    assert_equal(
        dtype_to_string(DType.int32), "int32", "Int32 should be 'int32'"
    )
    assert_equal(
        dtype_to_string(DType.uint8), "uint8", "UInt8 should be 'uint8'"
    )

    print("✓ DType to string test passed")


fn test_recommend_precision_dtype() raises:
    """Test precision recommendation logic."""
    print("Testing recommend_precision_dtype...")

    # Small model - should recommend FP32
    var small_dtype = recommend_precision_dtype(50.0, hardware_has_fp16=True)
    assert_equal(small_dtype, DType.float32, "Small models should use FP32")

    # Medium model with FP16 hardware - should recommend FP16
    var medium_dtype = recommend_precision_dtype(500.0, hardware_has_fp16=True)
    assert_equal(medium_dtype, DType.float16, "Medium models should use FP16")

    # Large model with BF16 hardware - should recommend BF16
    var large_dtype = recommend_precision_dtype(2000.0, hardware_has_fp16=True)
    assert_equal(large_dtype, DType.bfloat16, "Large models should use BF16")

    # Large model without BF16 hardware (e.g. Apple Silicon) - should fall back to FP16
    var large_no_bf16_dtype = recommend_precision_dtype(
        2000.0, hardware_has_fp16=True, hardware_has_bf16=False
    )
    assert_equal(
        large_no_bf16_dtype,
        DType.float16,
        "Large models without BF16 hardware should use FP16",
    )

    # Large model without FP16 or BF16 hardware - should recommend FP32
    var no_hw_dtype = recommend_precision_dtype(
        2000.0, hardware_has_fp16=False, hardware_has_bf16=False
    )
    assert_equal(
        no_hw_dtype, DType.float32, "Without FP16 hardware should use FP32"
    )

    print("✓ Precision recommendation test passed")


fn test_bfloat16_alias_behavior() raises:
    """Test that bfloat16 uses native DType.bfloat16."""
    print("Testing bfloat16 native dtype behavior...")

    # Verify bfloat16_dtype uses native DType.bfloat16
    from shared.core import zeros

    var tensor = zeros(List[Int](), bfloat16_dtype)
    assert_equal(
        tensor.dtype(),
        DType.bfloat16,
        "BF16 tensor should have native bfloat16 dtype",
    )

    print("✓ BFloat16 native dtype behavior test passed")


fn main() raises:
    print("\n" + "=" * 70)
    print("DTYPE UTILITIES TESTS")
    print("=" * 70)
    print()

    test_dtype_aliases()
    test_is_reduced_precision()
    test_is_floating_point()
    test_get_dtype_precision_bits()
    test_get_dtype_exponent_bits()
    test_dtype_to_string()
    test_recommend_precision_dtype()
    test_bfloat16_alias_behavior()

    print()
    print("=" * 70)
    print("ALL DTYPE UTILITIES TESTS PASSED! ✓")
    print("=" * 70)
    print()
    print("✓ bfloat16_dtype now uses native DType.bfloat16")
