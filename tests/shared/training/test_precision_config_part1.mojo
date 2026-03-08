# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_precision_config.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for PrecisionConfig module (part 1 of 2): PrecisionMode and basic config."""

from shared.training.precision_config import PrecisionConfig, PrecisionMode
from shared.core.extensor import ExTensor, zeros, ones


fn test_precision_mode_values() raises:
    """Test PrecisionMode enumeration values."""
    print("Testing PrecisionMode values...")

    # Check values are distinct
    if PrecisionMode.FP32.value != 0:
        raise Error("FP32 should have value 0")
    if PrecisionMode.FP16.value != 1:
        raise Error("FP16 should have value 1")
    if PrecisionMode.BF16.value != 2:
        raise Error("BF16 should have value 2")
    if PrecisionMode.FP8.value != 3:
        raise Error("FP8 should have value 3")

    print("✓ PrecisionMode values test passed")


fn test_precision_mode_equality() raises:
    """Test PrecisionMode equality comparison."""
    print("Testing PrecisionMode equality...")

    var fp32 = PrecisionMode.FP32
    var fp32_copy = PrecisionMode.FP32
    var fp16 = PrecisionMode.FP16

    if not (fp32 == fp32_copy):
        raise Error("FP32 should equal FP32")
    if fp32 == fp16:
        raise Error("FP32 should not equal FP16")
    if not (fp32 != fp16):
        raise Error("FP32 != FP16 should be true")

    print("✓ PrecisionMode equality test passed")


fn test_precision_mode_string() raises:
    """Test PrecisionMode string conversion."""
    print("Testing PrecisionMode string conversion...")

    if String(PrecisionMode.FP32) != "fp32":
        raise Error("FP32 should stringify to 'fp32'")
    if String(PrecisionMode.FP16) != "fp16":
        raise Error("FP16 should stringify to 'fp16'")
    if String(PrecisionMode.BF16) != "bf16":
        raise Error("BF16 should stringify to 'bf16'")
    if String(PrecisionMode.FP8) != "fp8":
        raise Error("FP8 should stringify to 'fp8'")

    print("✓ PrecisionMode string test passed")


fn test_fp32_config() raises:
    """Test FP32 PrecisionConfig."""
    print("Testing FP32 config...")

    var config = PrecisionConfig.fp32()

    if config.mode != PrecisionMode.FP32:
        raise Error("Mode should be FP32")
    if config.compute_dtype != DType.float32:
        raise Error("Compute dtype should be float32")
    if config.storage_dtype != DType.float32:
        raise Error("Storage dtype should be float32")
    if config.master_dtype != DType.float32:
        raise Error("Master dtype should be float32")
    if config.use_gradient_scaler:
        raise Error("FP32 should not use gradient scaler")

    print("✓ FP32 config test passed")


fn test_fp16_config() raises:
    """Test FP16 PrecisionConfig."""
    print("Testing FP16 config...")

    var config = PrecisionConfig.fp16()

    if config.mode != PrecisionMode.FP16:
        raise Error("Mode should be FP16")
    if config.compute_dtype != DType.float16:
        raise Error("Compute dtype should be float16")
    if config.storage_dtype != DType.float16:
        raise Error("Storage dtype should be float16")
    if config.master_dtype != DType.float32:
        raise Error("Master dtype should be float32")
    if not config.use_gradient_scaler:
        raise Error("FP16 should use gradient scaler")
    if config.get_scale() != 65536.0:
        raise Error("Initial scale should be 65536.0")

    print("✓ FP16 config test passed")


fn test_from_string() raises:
    """Test PrecisionConfig.from_string factory."""
    print("Testing from_string factory...")

    var fp32 = PrecisionConfig.from_string("fp32")
    if fp32.mode != PrecisionMode.FP32:
        raise Error("from_string('fp32') should create FP32 config")

    var fp16 = PrecisionConfig.from_string("fp16")
    if fp16.mode != PrecisionMode.FP16:
        raise Error("from_string('fp16') should create FP16 config")

    var bf16 = PrecisionConfig.from_string("bf16")
    if bf16.mode != PrecisionMode.BF16:
        raise Error("from_string('bf16') should create BF16 config")

    var fp8 = PrecisionConfig.from_string("fp8")
    if fp8.mode != PrecisionMode.FP8:
        raise Error("from_string('fp8') should create FP8 config")

    print("✓ from_string factory test passed")


fn test_from_string_invalid() raises:
    """Test from_string with invalid precision name."""
    print("Testing from_string with invalid input...")

    var caught_error = False
    try:
        var invalid = PrecisionConfig.from_string("fp64")
    except e:
        caught_error = True

    if not caught_error:
        raise Error("from_string('fp64') should raise error")

    print("✓ from_string invalid input test passed")


fn test_cast_to_compute() raises:
    """Test tensor casting to compute dtype."""
    print("Testing cast_to_compute...")

    var config = PrecisionConfig.fp16()

    # Create FP32 tensor
    var shape: List[Int] = [2, 3]
    var fp32_tensor = ones(shape, DType.float32)

    # Cast to compute dtype (FP16)
    var fp16_tensor = config.cast_to_compute(fp32_tensor)

    if fp16_tensor.dtype() != DType.float16:
        raise Error("Tensor should be cast to float16")

    # Check shape preserved
    var result_shape = fp16_tensor.shape()
    if result_shape[0] != 2 or result_shape[1] != 3:
        raise Error("Shape should be preserved after cast")

    print("✓ cast_to_compute test passed")


fn main() raises:
    """Run PrecisionConfig tests (part 1 of 2)."""
    print("=" * 60)
    print("PRECISION CONFIG TESTS (PART 1)")
    print("=" * 60)
    print()

    test_precision_mode_values()
    test_precision_mode_equality()
    test_precision_mode_string()
    test_fp32_config()
    test_fp16_config()
    test_from_string()
    test_from_string_invalid()
    test_cast_to_compute()

    print()
    print("=" * 60)
    print("ALL PRECISION CONFIG TESTS (PART 1) PASSED! ✓")
    print("=" * 60)
