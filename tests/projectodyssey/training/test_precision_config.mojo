"""Tests for PrecisionConfig module.

"""


from projectodyssey.training.precision_config import PrecisionConfig, PrecisionMode
from projectodyssey.tensor.any_tensor import AnyTensor, zeros, ones


def test_precision_mode_values() raises:
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


def test_precision_mode_equality() raises:
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


def test_precision_mode_string() raises:
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


def test_fp32_config() raises:
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


def test_fp16_config() raises:
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


def test_from_string() raises:
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


def test_from_string_invalid() raises:
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


def test_cast_to_compute() raises:
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


def test_scale_unscale() raises:
    """Test loss scaling and gradient unscaling."""
    print("Testing scale/unscale operations...")

    var config = PrecisionConfig.fp16(initial_scale=1000.0)

    # Create loss tensor
    var loss_shape: List[Int] = [1]
    var loss = ones(loss_shape, DType.float32)

    # Scale loss
    var scaled_loss = config.scale_loss(loss)
    var scaled_value = scaled_loss._get_float64(0)
    if scaled_value < 999.0 or scaled_value > 1001.0:
        raise Error(
            "Scaled loss should be ~1000.0, got: " + String(scaled_value)
        )

    # Create gradient tensor
    var grad_shape: List[Int] = [10]
    var grads = ones(grad_shape, DType.float32)
    for i in range(10):
        grads._set_float64(i, 1000.0)

    # Unscale gradients
    var unscaled_grads = config.unscale_gradients(grads)
    var unscaled_value = unscaled_grads._get_float64(0)
    if unscaled_value < 0.9 or unscaled_value > 1.1:
        raise Error(
            "Unscaled gradient should be ~1.0, got: " + String(unscaled_value)
        )

    print("✓ scale/unscale test passed")


def test_gradient_checking() raises:
    """Test gradient validity checking."""
    print("Testing gradient checking...")

    var config = PrecisionConfig.fp16()

    # Create valid gradients
    var shape: List[Int] = [5]
    var valid_grads = ones(shape, DType.float32)

    if not config.check_gradients(valid_grads):
        raise Error("Valid gradients should pass check")

    print("✓ gradient checking test passed")


def test_step_tracking() raises:
    """Test step and overflow tracking."""
    print("Testing step tracking...")

    var config = PrecisionConfig.fp16()

    if config.get_step_count() != 0:
        raise Error("Initial step count should be 0")
    if config.get_overflow_count() != 0:
        raise Error("Initial overflow count should be 0")

    # Simulate successful steps
    config.step(grads_valid=True)
    config.step(grads_valid=True)

    if config.get_step_count() != 2:
        raise Error("Step count should be 2")

    # Simulate overflow
    config.step(grads_valid=False)

    if config.get_overflow_count() != 1:
        raise Error("Overflow count should be 1")
    if config.get_step_count() != 3:
        raise Error("Step count should be 3 (includes overflow)")

    print("✓ step tracking test passed")


def test_needs_master_weights() raises:
    """Test needs_master_weights check."""
    print("Testing needs_master_weights...")

    var fp32 = PrecisionConfig.fp32()
    if fp32.needs_master_weights():
        raise Error("FP32 should not need master weights")

    var fp16 = PrecisionConfig.fp16()
    if not fp16.needs_master_weights():
        raise Error("FP16 should need master weights")

    var bf16 = PrecisionConfig.bf16()
    if not bf16.needs_master_weights():
        raise Error("BF16 should need master weights")

    print("✓ needs_master_weights test passed")


def test_bf16_config() raises:
    """Test BF16 PrecisionConfig."""
    print("Testing BF16 config...")

    var config = PrecisionConfig.bf16()

    if config.mode != PrecisionMode.BF16:
        raise Error("Mode should be BF16")
    if config.compute_dtype != DType.bfloat16:
        raise Error("Compute dtype should be bfloat16")
    if config.storage_dtype != DType.bfloat16:
        raise Error("Storage dtype should be bfloat16")
    if config.master_dtype != DType.float32:
        raise Error("Master dtype should be float32")
    if not config.use_gradient_scaler:
        raise Error("BF16 should use gradient scaler")
    if config.get_scale() != 65536.0:
        raise Error("Initial scale should be 65536.0")

    print("✓ BF16 config test passed")


def test_bf16_warning_documented() raises:
    """Test that BF16 config returns the correct mode.

    Note: The Apple Silicon runtime warning is emitted via print() inside
    bf16() when compiled for Apple M1 hardware (is_apple_m1() == True).
    Mojo tests do not have a warnings capture mechanism, so the warning
    cannot be asserted here. On CI (Linux/x86), no warning is printed.
    This test verifies the non-Apple-Silicon path: config is returned
    correctly without any behavioral change.
    """
    print("Testing BF16 warning documentation...")

    var config = PrecisionConfig.bf16()

    # BF16 mode is always returned regardless of platform
    if config.mode != PrecisionMode.BF16:
        raise Error("BF16 config mode should be PrecisionMode.BF16")

    print("✓ BF16 warning documented test passed")


def test_gradient_clipping() raises:
    """Test gradient clipping."""
    print("Testing gradient clipping...")

    var config = PrecisionConfig.fp16()

    # Create gradients with large values
    var shape: List[Int] = [4]
    var grads = zeros(shape, DType.float32)
    grads._set_float64(0, 10.0)
    grads._set_float64(1, 20.0)
    grads._set_float64(2, 30.0)
    grads._set_float64(3, 40.0)

    # Clip with max_norm=2.0
    var clipped = config.clip_gradients(grads, max_norm=2.0)

    # After clipping, L2 norm should be <= 2.0
    var sum_squared = Float64(0.0)
    for i in range(4):
        var val = clipped._get_float64(i)
        sum_squared += val * val
    var norm = sum_squared**0.5

    if norm > 2.1:
        raise Error(
            "Clipped gradient norm should be <= 2.0, got: " + String(norm)
        )

    print("✓ gradient clipping test passed")


def main() raises:
    """Run all test_precision_config tests."""
    print("Running test_precision_config tests...")

    test_precision_mode_values()
    print("✓ test_precision_mode_values")

    test_precision_mode_equality()
    print("✓ test_precision_mode_equality")

    test_precision_mode_string()
    print("✓ test_precision_mode_string")

    test_fp32_config()
    print("✓ test_fp32_config")

    test_fp16_config()
    print("✓ test_fp16_config")

    test_from_string()
    print("✓ test_from_string")

    test_from_string_invalid()
    print("✓ test_from_string_invalid")

    test_cast_to_compute()
    print("✓ test_cast_to_compute")

    test_scale_unscale()
    print("✓ test_scale_unscale")

    test_gradient_checking()
    print("✓ test_gradient_checking")

    test_step_tracking()
    print("✓ test_step_tracking")

    test_needs_master_weights()
    print("✓ test_needs_master_weights")

    test_bf16_config()
    print("✓ test_bf16_config")

    test_bf16_warning_documented()
    print("✓ test_bf16_warning_documented")

    test_gradient_clipping()
    print("✓ test_gradient_clipping")

    print("\nAll test_precision_config tests passed!")
