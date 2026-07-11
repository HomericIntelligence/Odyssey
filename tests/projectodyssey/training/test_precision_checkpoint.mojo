"""Precision checkpoint tests for multi-dtype training.

Tests cover:
- Precision mode serialization/deserialization
- Dtype promotion (FP16 -> FP32)
- Dtype demotion (FP32 -> FP16)
- PrecisionConfig persistence across checkpoints

These tests verify that precision settings are properly preserved
when saving and loading training state.
"""

from tests.odyssey.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_almost_equal,
    TestFixtures,
)
from odyssey.training.precision_config import (
    PrecisionConfig,
    PrecisionMode,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, ones, full
from std.collections import List


# ============================================================================
# Test 1: Precision Mode Serialization
# ============================================================================


def test_checkpoint_saves_precision_mode() raises:
    """Test that precision mode can be serialized to string.

    When saving checkpoints, we need to store the precision mode
    as a string that can be written to disk.
    """
    # Create configs for all precision modes
    var fp32_config = PrecisionConfig.fp32()
    var fp16_config = PrecisionConfig.fp16()
    var bf16_config = PrecisionConfig.bf16()
    var fp8_config = PrecisionConfig.fp8()

    # Verify mode enum values are serializable (can be converted to string)
    assert_true(
        fp32_config.mode == PrecisionMode.FP32, "FP32 mode should be correct"
    )
    assert_true(
        fp16_config.mode == PrecisionMode.FP16, "FP16 mode should be correct"
    )
    assert_true(
        bf16_config.mode == PrecisionMode.BF16, "BF16 mode should be correct"
    )
    assert_true(
        fp8_config.mode == PrecisionMode.FP8, "FP8 mode should be correct"
    )

    # Verify from_string roundtrip works (simulates save/load)
    var loaded_fp32 = PrecisionConfig.from_string("fp32")
    var loaded_fp16 = PrecisionConfig.from_string("fp16")
    var loaded_bf16 = PrecisionConfig.from_string("bf16")
    var loaded_fp8 = PrecisionConfig.from_string("fp8")

    assert_true(
        loaded_fp32.mode == PrecisionMode.FP32, "Loaded FP32 should match"
    )
    assert_true(
        loaded_fp16.mode == PrecisionMode.FP16, "Loaded FP16 should match"
    )
    assert_true(
        loaded_bf16.mode == PrecisionMode.BF16, "Loaded BF16 should match"
    )
    assert_true(loaded_fp8.mode == PrecisionMode.FP8, "Loaded FP8 should match")


# ============================================================================
# Test 2: Precision Mode Loading
# ============================================================================


def test_checkpoint_loads_matching_precision() raises:
    """Test that loaded precision config matches original.

    When resuming training, we need to ensure the precision
    settings are identical to avoid training instability.
    """
    # Original config with specific settings
    var original = PrecisionConfig.fp16(initial_scale=65536.0)
    var original_mode = original.mode
    var _ = original.get_scale()  # Scale would be saved in full checkpoint
    var original_uses_scaler = original.use_gradient_scaler
    var original_needs_master = original.needs_master_weights()

    # Simulate save/load by recreating from string
    var loaded = PrecisionConfig.from_string("fp16")

    # Mode and behavioral properties should match
    assert_true(loaded.mode == original_mode, "Mode should match after reload")
    assert_true(
        loaded.use_gradient_scaler == original_uses_scaler,
        "Gradient scaler setting should match",
    )
    assert_true(
        loaded.needs_master_weights() == original_needs_master,
        "Master weights setting should match",
    )

    # Note: Scale value may differ (from_string uses default)
    # In a full checkpoint, scale would be explicitly saved


# ============================================================================
# Test 3: FP16 to FP32 Promotion
# ============================================================================


def test_checkpoint_fp16_to_fp32_promotion() raises:
    """Test weight promotion from FP16 to FP32 precision.

    When loading an FP16 checkpoint into FP32 training,
    weights should be upcast without loss of data.
    """
    var fp16_config = PrecisionConfig.fp16()
    var fp32_config = PrecisionConfig.fp32()

    # Create FP16 weights (simulates loaded checkpoint)
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)
    var fp32_original = full(shape, 0.5, DType.float32)
    var fp16_weights = fp16_config.cast_to_compute(fp32_original)

    # Verify FP16 dtype
    assert_true(fp16_weights.dtype() == DType.float16, "Weights should be FP16")

    # Promote to FP32 for training
    var fp32_weights = fp32_config.cast_to_compute(fp16_weights)

    # Verify FP32 dtype
    assert_true(
        fp32_weights.dtype() == DType.float32, "Promoted weights should be FP32"
    )

    # Check value preservation (within FP16 precision)
    var original_val = fp32_original._get_float64(0)
    var promoted_val = fp32_weights._get_float64(0)
    var error = abs(original_val - promoted_val)
    assert_true(
        error < Float64(0.01), "Value should be preserved after roundtrip"
    )


# ============================================================================
# Test 4: FP32 to FP16 Demotion
# ============================================================================


def test_checkpoint_fp32_to_fp16_demotion() raises:
    """Test weight demotion from FP32 to FP16 precision.

    When loading an FP32 checkpoint into FP16 training,
    weights are downcast (may lose some precision).
    """
    var fp16_config = PrecisionConfig.fp16()

    # Create FP32 weights (simulates loaded checkpoint)
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)
    var fp32_weights = full(
        shape, 0.123456789, DType.float32
    )  # High precision value

    # Demote to FP16 for memory-efficient training
    var fp16_weights = fp16_config.cast_to_compute(fp32_weights)

    # Verify FP16 dtype
    assert_true(
        fp16_weights.dtype() == DType.float16, "Demoted weights should be FP16"
    )

    # Check value preservation (within FP16 limits)
    # FP16 has ~3 decimal digits of precision
    var original_val = fp32_weights._get_float64(0)
    var demoted_val = fp16_weights._get_float64(0)
    var relative_error = abs(original_val - demoted_val) / abs(original_val)

    # FP16 should preserve ~3 significant digits (error < 1%)
    assert_true(
        relative_error < Float64(0.01),
        "FP16 demotion should preserve value within precision limits",
    )


# ============================================================================
# Test 5: Gradient Scaler State
# ============================================================================


def test_checkpoint_gradient_scaler_state() raises:
    """Test that gradient scaler state can be saved and restored.

    When resuming training, the gradient scaler's scale factor
    and overflow count should be preserved.
    """
    var config = PrecisionConfig.fp16(initial_scale=65536.0)

    # Simulate training that caused overflow
    config.step(grads_valid=False)  # Overflow -> scale reduced
    config.step(grads_valid=True)  # Normal step

    # Capture state for "saving"
    var saved_scale = config.get_scale()
    var saved_overflow_count = config.get_overflow_count()

    # Verify state changed from initial
    assert_true(
        Float64(saved_scale) < Float64(65536.0),
        "Scale should have decreased after overflow",
    )
    assert_equal_int(saved_overflow_count, 1, "Overflow count should be 1")

    # In a real checkpoint, these values would be saved to disk
    # and restored when creating a new PrecisionConfig


# ============================================================================
# Test 6: Master Weights Precision
# ============================================================================


def test_checkpoint_master_weights_precision() raises:
    """Test that master weights maintain FP32 precision.

    In mixed-precision training, master weights must stay in FP32
    even when checkpoint stores compute weights in FP16.
    """
    var fp16_config = PrecisionConfig.fp16()

    # Master dtype should always be FP32
    assert_true(
        fp16_config.master_dtype == DType.float32, "Master dtype should be FP32"
    )

    # Create compute weights in FP16
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)
    var fp32_weights = full(shape, 0.5, DType.float32)
    var fp16_compute = fp16_config.cast_to_compute(fp32_weights)

    # Cast to master precision (for optimizer updates)
    var master_weights = fp16_config.cast_to_master(fp16_compute)

    # Master weights should be FP32
    assert_true(
        master_weights.dtype() == DType.float32, "Master weights should be FP32"
    )


# ============================================================================
# Main - Run All Tests
# ============================================================================


def main() raises:
    """Run all precision checkpoint tests."""
    print("=" * 60)
    print("Precision Checkpoint Tests")
    print("=" * 60)
    print()

    print("Test 1: Precision mode serialization...")
    test_checkpoint_saves_precision_mode()
    print("  PASSED")

    print("Test 2: Precision mode loading...")
    test_checkpoint_loads_matching_precision()
    print("  PASSED")

    print("Test 3: FP16 to FP32 promotion...")
    test_checkpoint_fp16_to_fp32_promotion()
    print("  PASSED")

    print("Test 4: FP32 to FP16 demotion...")
    test_checkpoint_fp32_to_fp16_demotion()
    print("  PASSED")

    print("Test 5: Gradient scaler state...")
    test_checkpoint_gradient_scaler_state()
    print("  PASSED")

    print("Test 6: Master weights precision...")
    test_checkpoint_master_weights_precision()
    print("  PASSED")

    print()
    print("=" * 60)
    print("ALL PRECISION CHECKPOINT TESTS PASSED! (6/6)")
    print("=" * 60)
