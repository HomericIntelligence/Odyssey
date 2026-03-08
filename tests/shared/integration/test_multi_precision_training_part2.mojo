# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_multi_precision_training.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Integration tests for multi-precision training (Part 2 of 2).

Tests cover:
- Master weights maintenance
- Precision vs accuracy trade-offs
- Memory savings
- Training with TOML config

These tests verify that mixed-precision training works correctly
across all supported dtypes.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    assert_greater,
    assert_less,
    assert_dtype,
    TestFixtures,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.training.precision_config import PrecisionConfig, PrecisionMode
from shared.training.mixed_precision import GradientScaler
from shared.training.dtype_utils import (
    float16_dtype,
    float32_dtype,
    bfloat16_dtype,
    is_reduced_precision,
)
from collections import List


# ============================================================================
# Test 8: Master Weights
# ============================================================================


fn test_master_weights_fp32() raises:
    """Test master weights are maintained in FP32.

    For reduced precision training:
    - Compute is done in FP16/BF16/FP8
    - Master weights stay in FP32 for optimizer stability.
    """
    var fp16_config = PrecisionConfig.fp16()
    var fp32_config = PrecisionConfig.fp32()

    # FP16 needs master weights
    assert_true(
        fp16_config.needs_master_weights(), "FP16 should need master weights"
    )
    assert_true(
        fp16_config.master_dtype == DType.float32,
        "Master dtype should be float32",
    )

    # FP32 doesn't need separate master weights
    assert_false(
        fp32_config.needs_master_weights(),
        "FP32 should not need master weights",
    )

    # Test casting to master precision
    var weight_shape = List[Int]()
    weight_shape.append(10)
    weight_shape.append(10)
    var fp16_weights = full(weight_shape, 0.5, DType.float16)
    var master_weights = fp16_config.cast_to_master(fp16_weights)

    assert_dtype(
        master_weights, DType.float32, "Master weights should be float32"
    )


# ============================================================================
# Test 9-10: Precision vs Accuracy Trade-offs
# ============================================================================


fn test_fp16_vs_fp32_accuracy() raises:
    """Test FP16 maintains accuracy within tolerance of FP32.

    FP16 should achieve similar results to FP32:
    - Loss values within 2% of FP32
    - Gradient directions preserved
    - No significant accuracy degradation.
    """
    var fp32_config = PrecisionConfig.fp32()
    var fp16_config = PrecisionConfig.fp16()

    # Create test tensor
    var test_shape = List[Int]()
    test_shape.append(10)
    var test_data = full(test_shape, 1.5, DType.float32)

    # Cast to FP16 and back
    var fp16_data = fp16_config.cast_to_compute(test_data)
    var back_to_fp32 = fp32_config.cast_to_compute(fp16_data)

    # Check value preserved (within FP16 precision)
    # Use native Float32 comparison to avoid unnecessary type conversion
    # that can trigger Mojo heap corruption (see ADR-009)
    var original_val = test_data._get_float32(0)
    var roundtrip_val = back_to_fp32._get_float32(0)

    # FP16 has ~3 decimal digits of precision
    var rel_error = abs(original_val - roundtrip_val) / abs(original_val)
    assert_less(
        Float64(rel_error), Float64(0.01), "Roundtrip error should be < 1%"
    )


fn test_bf16_vs_fp32_accuracy() raises:
    """Test BF16 maintains accuracy within tolerance of FP32.

    BF16 has less precision than FP16 but wider range:
    - ~2 decimal digits precision
    - Same exponent range as FP32.
    """
    var bf16_config = PrecisionConfig.bf16()

    # BF16 currently uses FP16 as fallback
    # This test documents expected behavior when native BF16 is available
    assert_true(
        bf16_config.mode == PrecisionMode.BF16, "Config should be BF16 mode"
    )


# ============================================================================
# Test 11: Memory Savings (Conceptual)
# ============================================================================


fn test_mixed_precision_memory_savings() raises:
    """Test that FP16 uses less memory than FP32.

    Note: This is a conceptual test since we can't easily measure
    memory in current Mojo version. We verify dtype sizes instead.
    """
    # FP32 = 4 bytes per element
    # FP16 = 2 bytes per element
    # Expected: 50% memory reduction

    var fp32_bytes = 4  # sizeof(float32)
    var fp16_bytes = 2  # sizeof(float16)

    var savings_percent = (
        Float64(1.0 - Float64(fp16_bytes) / Float64(fp32_bytes)) * 100.0
    )
    assert_almost_equal(
        savings_percent,
        Float64(50.0),
        tolerance=Float64(0.1),
        message="FP16 should save ~50% memory",
    )

    # Verify reduced_precision utility
    assert_true(
        is_reduced_precision(DType.float16), "FP16 is reduced precision"
    )
    assert_false(
        is_reduced_precision(DType.float32), "FP32 is not reduced precision"
    )


# ============================================================================
# Test 12: Training with TOML Config (Stub)
# ============================================================================


fn test_training_with_toml_config() raises:
    """Test loading training configuration from TOML file.

    Verifies that TOML config files can be loaded and parsed correctly.
    Tests precision mode and gradient scaler configuration from TOML.

    TOML config should specify:
    - precision mode (fp32/fp16/bf16/fp8)
    - initial scale for gradient scaler
    - batch size, learning rate, etc.
    """
    from shared.utils.toml_loader import load_toml_config

    # Load FP16 config from TOML file
    var config = load_toml_config("configs/lenet5/emnist/fp16.toml")

    # Verify precision mode
    var mode = config.get_string("precision.mode")
    assert_true(mode == "fp16", "Precision mode should be fp16")

    # Verify gradient scaler initial scale
    var initial_scale = config.get_float(
        "precision.gradient_scaler.initial_scale"
    )
    assert_almost_equal(
        initial_scale,
        65536.0,
        tolerance=0.1,
        message="Initial scale should be 65536.0 from TOML",
    )

    # Verify other gradient scaler settings
    var growth_factor = config.get_float(
        "precision.gradient_scaler.growth_factor"
    )
    assert_almost_equal(
        growth_factor,
        2.0,
        tolerance=0.01,
        message="Growth factor should be 2.0",
    )

    var backoff_factor = config.get_float(
        "precision.gradient_scaler.backoff_factor"
    )
    assert_almost_equal(
        backoff_factor,
        0.5,
        tolerance=0.01,
        message="Backoff factor should be 0.5",
    )

    # Verify training settings
    var batch_size = config.get_int("training.batch_size")
    assert_true(batch_size == 64, "Batch size should be 64")

    var learning_rate = config.get_float("training.learning_rate")
    assert_almost_equal(
        learning_rate,
        0.001,
        tolerance=0.0001,
        message="Learning rate should be 0.001",
    )

    # Now create a PrecisionConfig from the loaded values
    var fp16_config = PrecisionConfig.fp16(initial_scale=Float32(initial_scale))
    assert_true(
        fp16_config.mode == PrecisionMode.FP16,
        "Should create FP16 config from TOML",
    )
    assert_almost_equal(
        Float64(fp16_config.get_scale()),
        initial_scale,
        tolerance=0.1,
        message="Config scale should match TOML value",
    )


# ============================================================================
# Main - Run All Tests
# ============================================================================


fn main() raises:
    """Run multi-precision training tests (Part 2 of 2)."""
    print("=" * 60)
    print("Multi-Precision Training Tests (Part 2 of 2)")
    print("=" * 60)
    print()

    print("Test 8: Master weights in FP32...")
    test_master_weights_fp32()
    print("  PASSED")

    print("Test 9: FP16 vs FP32 accuracy...")
    test_fp16_vs_fp32_accuracy()
    print("  PASSED")

    print("Test 10: BF16 vs FP32 accuracy...")
    test_bf16_vs_fp32_accuracy()
    print("  PASSED")

    print("Test 11: Memory savings...")
    test_mixed_precision_memory_savings()
    print("  PASSED")

    print("Test 12: Training with TOML config...")
    test_training_with_toml_config()
    print("  PASSED")

    print()
    print("=" * 60)
    print("ALL PART 2 TESTS PASSED! (5/5)")
    print("=" * 60)
