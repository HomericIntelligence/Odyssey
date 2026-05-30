"""Tests for mixed precision training infrastructure (part 1).

Tests gradient scaler initialization, loss scaling, gradient unscaling,
scaler step updates, backoff, min/max limits, and FP32 master weight conversion.
"""


from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import full
from projectodyssey.training.mixed_precision import (
    GradientScaler,
    check_gradients_finite,
    clip_gradients_by_norm,
    clip_gradients_by_value,
    convert_to_fp32_master,
    update_model_from_master,
)
from std.testing import (
    assert_equal,
    assert_false,
    assert_true,
)
from projectodyssey.testing.special_values import (
    create_nan_tensor,
    create_inf_tensor,
)


def test_gradient_scaler_initialization() raises:
    """Test GradientScaler initializes with correct default values."""
    print("Testing GradientScaler initialization...")

    var scaler = GradientScaler()

    assert_equal(scaler.scale, 65536.0, "Default scale should be 65536.0")
    assert_equal(
        scaler.growth_factor, 2.0, "Default growth factor should be 2.0"
    )
    assert_equal(
        scaler.backoff_factor, 0.5, "Default backoff factor should be 0.5"
    )
    assert_equal(
        scaler.growth_interval, 2000, "Default growth interval should be 2000"
    )
    assert_equal(scaler.get_num_steps(), 0, "Initial steps should be 0")

    print("✓ GradientScaler initialization test passed")


def test_loss_scaling() raises:
    """Test loss scaling multiplies by scale factor."""
    print("Testing loss scaling...")

    var scaler = GradientScaler(initial_scale=1024.0)

    # Create a simple loss tensor (scalar)
    var loss = full([1], 0.5, DType.float32)

    # Scale the loss
    var scaled_loss = scaler.scale_loss(loss)

    # Check scaled value (0.5 * 1024 = 512)
    var scaled_val = scaled_loss._get_float64(0)
    assert_true(abs(scaled_val - 512.0) < 1e-5, "Scaled loss should be 512.0")

    print("✓ Loss scaling test passed")


def test_gradient_unscaling() raises:
    """Test gradient unscaling divides by scale factor."""
    print("Testing gradient unscaling...")

    var scaler = GradientScaler(initial_scale=1024.0)

    # Create scaled gradients
    var scaled_grads = full([1], 2048.0, DType.float32)

    # Unscale the gradients (2048 / 1024 = 2.0)
    var unscaled_grads = scaler.unscale_gradients(scaled_grads)

    # Check unscaled value
    var val = unscaled_grads._get_float64(0)
    assert_true(abs(val - 2.0) < 1e-5, "Unscaled gradient should be 2.0")

    print("✓ Gradient unscaling test passed")


def test_scaler_step_updates() raises:
    """Test scaler step increases scale after growth interval."""
    print("Testing scaler step updates...")

    var scaler = GradientScaler(
        initial_scale=1024.0, growth_factor=2.0, growth_interval=100
    )

    # Take 99 steps (not enough to trigger growth)
    for _ in range(99):
        scaler.step()

    assert_equal(
        scaler.get_scale(),
        1024.0,
        "Scale should not increase before growth interval",
    )

    # Take one more step (100 total - should trigger growth)
    scaler.step()

    assert_equal(
        scaler.get_scale(), 2048.0, "Scale should double after growth interval"
    )

    print("✓ Scaler step updates test passed")


def test_scaler_backoff() raises:
    """Test scaler backoff reduces scale factor."""
    print("Testing scaler backoff...")

    var scaler = GradientScaler(initial_scale=1024.0, backoff_factor=0.5)

    # Trigger backoff
    scaler.backoff()

    assert_equal(
        scaler.get_scale(), 512.0, "Scale should be halved after backoff"
    )

    print("✓ Scaler backoff test passed")


def test_scaler_min_max_limits() raises:
    """Test scaler respects min and max scale limits."""
    print("Testing scaler min/max limits...")

    var scaler = GradientScaler(
        initial_scale=1024.0,
        min_scale=512.0,
        max_scale=2048.0,
        backoff_factor=0.5,
    )

    # Try to backoff below min_scale
    scaler.backoff()  # 1024 -> 512
    assert_equal(scaler.get_scale(), 512.0, "Scale should be at min")

    scaler.backoff()  # Try to go to 256, but should stay at 512
    assert_equal(scaler.get_scale(), 512.0, "Scale should not go below min")

    # Reset to just below max
    scaler.scale = 1536.0

    # Try to grow beyond max_scale (with very small growth interval for testing)
    scaler = GradientScaler(
        initial_scale=1536.0,
        min_scale=512.0,
        max_scale=2048.0,
        growth_factor=2.0,
        growth_interval=1,
    )

    scaler.step()  # Should grow to 3072, but capped at 2048
    assert_equal(scaler.get_scale(), 2048.0, "Scale should be capped at max")

    print("✓ Scaler min/max limits test passed")


def test_fp32_master_conversion() raises:
    """Test converting FP16 parameters to FP32 master weights."""
    print("Testing FP32 master conversion...")

    # Create FP16 parameters (100 elements)
    var fp16_params = full([100], 0.5, DType.float16)

    # Convert to FP32 master weights
    var master_params = convert_to_fp32_master(fp16_params)

    assert_true(
        master_params.dtype() == DType.float32, "Master params should be FP32"
    )
    assert_equal(
        master_params._numel, 100, "Master params should have same size"
    )

    var val = master_params._get_float64(0)
    assert_true(abs(val - 0.5) < 1e-5, "Master params should have same values")

    print("✓ FP32 master conversion test passed")


def test_update_model_from_master() raises:
    """Test updating FP16 model params from FP32 master weights."""
    print("Testing model update from master...")

    # Create FP16 model params and FP32 master weights (100 elements)
    var fp16_params = full([100], 1.0, DType.float16)
    var master_params = full([100], 2.0, DType.float32)

    # Update model from master
    update_model_from_master(fp16_params, master_params)

    # Check that FP16 params now have value 2.0
    var val = fp16_params._get_float64(0)
    assert_true(abs(val - 2.0) < 1e-3, "FP16 params should be updated to 2.0")

    print("✓ Update model from master test passed")


def test_check_gradients_finite() raises:
    """Test checking for finite gradients."""
    print("Testing gradient finite check...")

    # Create finite gradients (10 elements)
    var finite_grads = full([10], 1.0, DType.float32)
    assert_true(
        check_gradients_finite(finite_grads),
        "Finite gradients should return True",
    )

    # Test with NaN gradients
    var nan_grads = create_nan_tensor([10], DType.float32)
    assert_false(
        check_gradients_finite(nan_grads),
        "NaN gradients should return False",
    )

    # Test with +Inf gradients
    var pos_inf_grads = create_inf_tensor([10], DType.float32, positive=True)
    assert_false(
        check_gradients_finite(pos_inf_grads),
        "+Inf gradients should return False",
    )

    # Test with -Inf gradients
    var neg_inf_grads = create_inf_tensor([10], DType.float32, positive=False)
    assert_false(
        check_gradients_finite(neg_inf_grads),
        "-Inf gradients should return False",
    )

    print("✓ Gradient finite check test passed")


def test_check_gradients_finite_mixed_precision() raises:
    """Test checking for finite gradients in mixed precision (FP16).

    Validates that NaN/Inf detection works correctly for lower precision types.
    This is critical for mixed precision training where gradient overflow is more
    likely due to reduced precision.
    """
    print("Testing gradient finite check with mixed precision (FP16)...")

    # Create finite FP16 gradients
    var finite_fp16_grads = full([10], 1.0, DType.float16)
    assert_true(
        check_gradients_finite(finite_fp16_grads),
        "Finite FP16 gradients should return True",
    )

    # Test NaN detection in FP16
    var nan_fp16_grads = create_nan_tensor([10], DType.float16)
    assert_false(
        check_gradients_finite(nan_fp16_grads),
        "NaN FP16 gradients should return False",
    )

    # Test +Inf detection in FP16
    var pos_inf_fp16_grads = create_inf_tensor(
        [10], DType.float16, positive=True
    )
    assert_false(
        check_gradients_finite(pos_inf_fp16_grads),
        "+Inf FP16 gradients should return False",
    )

    # Test -Inf detection in FP16
    var neg_inf_fp16_grads = create_inf_tensor(
        [10], DType.float16, positive=False
    )
    assert_false(
        check_gradients_finite(neg_inf_fp16_grads),
        "-Inf FP16 gradients should return False",
    )

    print("✓ Mixed precision gradient finite check test passed")


def test_clip_gradients_by_value() raises:
    """Test clipping gradients by value range."""
    print("Testing gradient clipping by value...")

    # Create gradients with various values (5 elements)
    var shape: List[Int] = [5]
    var grads = AnyTensor(shape, DType.float32)

    # Set some values manually
    grads._set_float64(0, -2.0)
    grads._set_float64(1, -0.5)
    grads._set_float64(2, 0.0)
    grads._set_float64(3, 0.5)
    grads._set_float64(4, 2.0)

    # Clip to [-1.0, 1.0]
    var clipped = clip_gradients_by_value(grads, -1.0, 1.0)

    # Check clipped values
    assert_equal(clipped._get_float64(0), -1.0, "Should clip to -1.0")
    assert_equal(clipped._get_float64(1), -0.5, "Should not clip -0.5")
    assert_equal(clipped._get_float64(2), 0.0, "Should not clip 0.0")
    assert_equal(clipped._get_float64(3), 0.5, "Should not clip 0.5")
    assert_equal(clipped._get_float64(4), 1.0, "Should clip to 1.0")

    print("✓ Gradient clipping by value test passed")


def test_clip_gradients_by_norm() raises:
    """Test clipping gradients by global norm."""
    print("Testing gradient clipping by norm...")

    # Create gradients with known norm (3 elements)
    var shape: List[Int] = [3]
    var grads = AnyTensor(shape, DType.float32)

    # Set values: [3.0, 4.0, 0.0] -> norm = sqrt(9 + 16) = 5.0
    grads._set_float64(0, 3.0)
    grads._set_float64(1, 4.0)
    grads._set_float64(2, 0.0)

    # Clip to max_norm = 1.0 (should scale by 1.0/5.0 = 0.2)
    var clipped = clip_gradients_by_norm(grads, 1.0)

    # Check clipped values
    var val0 = clipped._get_float64(0)
    var val1 = clipped._get_float64(1)
    var val2 = clipped._get_float64(2)

    assert_true(abs(val0 - 0.6) < 1e-5, "Should be 3.0 * 0.2 = 0.6")
    assert_true(abs(val1 - 0.8) < 1e-5, "Should be 4.0 * 0.2 = 0.8")
    assert_true(abs(val2 - 0.0) < 1e-5, "Should be 0.0")

    print("✓ Gradient clipping by norm test passed")


def test_fp16_operations() raises:
    """Test basic FP16 tensor operations."""
    print("Testing FP16 operations...")

    # Create FP16 tensors (10 elements each)
    var a_shape = List[Int]()
    a_shape.append(10)
    var a = full(a_shape, 2.0, DType.float16)
    var b_shape = List[Int]()
    b_shape.append(10)
    var b = full(b_shape, 3.0, DType.float16)

    # Test addition
    var c = a + b
    var val_add = c._get_float64(0)
    assert_true(abs(val_add - 5.0) < 1e-2, "FP16 addition: 2 + 3 = 5")

    # Test multiplication
    var d = a * b
    var val_mul = d._get_float64(0)
    assert_true(abs(val_mul - 6.0) < 1e-2, "FP16 multiplication: 2 * 3 = 6")

    # Test division
    var e = b / a
    var val_div = e._get_float64(0)
    assert_true(abs(val_div - 1.5) < 1e-2, "FP16 division: 3 / 2 = 1.5")

    print("✓ FP16 operations test passed")


def main() raises:
    """Run all test_mixed_precision tests."""
    print("Running test_mixed_precision tests...")

    test_gradient_scaler_initialization()
    print("✓ test_gradient_scaler_initialization")

    test_loss_scaling()
    print("✓ test_loss_scaling")

    test_gradient_unscaling()
    print("✓ test_gradient_unscaling")

    test_scaler_step_updates()
    print("✓ test_scaler_step_updates")

    test_scaler_backoff()
    print("✓ test_scaler_backoff")

    test_scaler_min_max_limits()
    print("✓ test_scaler_min_max_limits")

    test_fp32_master_conversion()
    print("✓ test_fp32_master_conversion")

    test_update_model_from_master()
    print("✓ test_update_model_from_master")

    test_check_gradients_finite()
    print("✓ test_check_gradients_finite")

    test_check_gradients_finite_mixed_precision()
    print("✓ test_check_gradients_finite_mixed_precision")

    test_clip_gradients_by_value()
    print("✓ test_clip_gradients_by_value")

    test_clip_gradients_by_norm()
    print("✓ test_clip_gradients_by_norm")

    test_fp16_operations()
    print("✓ test_fp16_operations")

    print("\nAll test_mixed_precision tests passed!")
