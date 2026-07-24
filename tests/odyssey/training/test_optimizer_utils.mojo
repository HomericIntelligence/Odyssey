"""Unit tests for optimizer utilities.

Tests cover:
- State initialization utilities
- Tensor scaling and normalization
- Norm computation (single and global)
- Single tensor norm clipping

These tests verify the common utilities available to all optimizer implementations.
"""


from tests.odyssey.conftest import (
    assert_true,
    assert_almost_equal,
    assert_equal,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, ones, full, zeros_like
from odyssey.training.optimizers import (
    apply_bias_correction,
    apply_weight_decay,
    clip_global_norm,
    clip_tensor_norm,
    compute_global_norm,
    compute_tensor_norm,
    compute_weight_decay_term,
    normalize_tensor_to_unit_norm,
    scale_tensor,
    scale_tensor_inplace,
    validate_optimizer_state,
)


def test_scale_tensor() raises:
    """Test tensor scaling."""
    var tensor = full([3], 2.0, DType.float32)
    var scaled = scale_tensor(tensor, scale=0.5)

    # Check values
    var expected = 1.0
    for i in range(scaled.numel()):
        var actual = Float64(scaled._get_float64(i))
        assert_almost_equal(actual, expected, tolerance=1e-6)


def test_scale_tensor_inplace() raises:
    """Test in-place tensor scaling."""
    var tensor = full([2], 4.0, DType.float32)
    scale_tensor_inplace(tensor, scale=0.25)

    # Check values after scaling
    var expected = 1.0
    for i in range(tensor.numel()):
        var actual = Float64(tensor._get_float64(i))
        assert_almost_equal(actual, expected, tolerance=1e-6)


def test_compute_tensor_norm() raises:
    """Test L2 norm computation."""
    var tensor = full([4], 3.0, DType.float32)

    # L2 norm of [3, 3, 3, 3] = sqrt(9 + 9 + 9 + 9) = sqrt(36) = 6.0
    var norm = compute_tensor_norm(tensor)
    assert_almost_equal(norm, 6.0, tolerance=1e-6)


def test_compute_tensor_norm_zero() raises:
    """Test L2 norm of zero tensor."""
    var tensor = zeros([3], DType.float32)
    var norm = compute_tensor_norm(tensor)
    assert_almost_equal(norm, 0.0, tolerance=1e-6)


def test_compute_global_norm() raises:
    """Test global L2 norm computation."""
    var t1 = full([3], 1.0, DType.float32)
    var t2 = full([4], 2.0, DType.float32)

    var tensors: List[AnyTensor] = []
    tensors.append(t1)
    tensors.append(t2)

    # Global norm = sqrt(3*1^2 + 4*2^2) = sqrt(3 + 16) = sqrt(19)
    var global_norm = compute_global_norm(tensors)
    var expected = Float64(4.358898)  # sqrt(19)
    assert_almost_equal(global_norm, expected, tolerance=1e-5)


def test_clip_tensor_norm() raises:
    """Test single tensor norm clipping."""
    var tensor = full([9], 1.0, DType.float32)
    # L2 norm = sqrt(9) = 3.0

    var original_norm = clip_tensor_norm(tensor, max_norm=1.5)

    # Check original norm was returned
    assert_almost_equal(original_norm, 3.0, tolerance=1e-6)

    # Check tensor was clipped to max_norm
    var new_norm = compute_tensor_norm(tensor)
    assert_almost_equal(new_norm, 1.5, tolerance=1e-6)


def test_clip_tensor_norm_no_clip() raises:
    """Test that clipping doesn't occur when norm is below max."""
    var tensor = full([2], 0.5, DType.float32)
    # L2 norm = sqrt(0.5) ≈ 0.707

    var original_norm = clip_tensor_norm(tensor, max_norm=2.0)

    # Check tensor is unchanged (norm still below max)
    var new_norm = compute_tensor_norm(tensor)
    assert_almost_equal(new_norm, original_norm, tolerance=1e-6)


def test_clip_global_norm() raises:
    """Test global norm clipping."""
    var t1 = full([3], 1.0, DType.float32)
    var t2 = full([4], 2.0, DType.float32)

    var tensors: List[AnyTensor] = []
    tensors.append(t1)
    tensors.append(t2)

    # Original global norm = sqrt(19) ≈ 4.359
    var original_norm = clip_global_norm(tensors, max_norm=1.0)
    assert_almost_equal(original_norm, Float64(4.358898), tolerance=1e-5)

    # Check new global norm after clipping
    var new_global_norm = compute_global_norm(tensors)
    assert_almost_equal(new_global_norm, 1.0, tolerance=1e-6)


def test_apply_weight_decay() raises:
    """Test in-place weight decay application."""
    var params = full([2], 10.0, DType.float32)

    apply_weight_decay(params, weight_decay=0.1)

    # After decay: params *= (1 - 0.1) = 0.9
    # So 10.0 becomes 9.0
    var expected = 9.0
    for i in range(params.numel()):
        var actual = Float64(params._get_float64(i))
        assert_almost_equal(actual, expected, tolerance=1e-6)


def test_compute_weight_decay_term() raises:
    """Test weight decay term computation."""
    var params = full([2], 5.0, DType.float32)

    var wd_term = compute_weight_decay_term(params, weight_decay=0.1)

    # wd_term = 0.1 * params = 0.1 * 5.0 = 0.5
    var expected = 0.5
    for i in range(wd_term.numel()):
        var actual = Float64(wd_term._get_float64(i))
        assert_almost_equal(actual, expected, tolerance=1e-6)


def test_normalize_tensor_to_unit_norm() raises:
    """Test tensor normalization to unit norm."""
    var tensor = full([4], 2.0, DType.float32)

    normalize_tensor_to_unit_norm(tensor)

    var norm = compute_tensor_norm(tensor)
    assert_almost_equal(norm, 1.0, tolerance=1e-6)


def test_apply_bias_correction() raises:
    """Test bias correction for exponential moving average."""
    # Create a simple estimate
    var estimate = ones([2], DType.float32)

    # Apply bias correction with beta=0.9, t=1
    # Correction: estimate / (1 - 0.9^1) = estimate / 0.1 = estimate * 10
    var corrected = apply_bias_correction(estimate, decay=0.9, timestep=1)

    var expected_value = 10.0  # 1.0 * 10
    for i in range(corrected.numel()):
        var actual = Float64(corrected._get_float64(i))
        assert_almost_equal(actual, expected_value, tolerance=1e-6)


def test_validate_optimizer_state_valid() raises:
    """Test validation of valid optimizer state."""
    var params: List[AnyTensor] = []
    var shape1 = List[Int]()
    shape1.append(2)
    shape1.append(3)
    params.append(ones(shape1, DType.float32))
    var shape2 = List[Int]()
    shape2.append(3)
    params.append(ones(shape2, DType.float32))

    var states = List[List[AnyTensor]]()
    states.append(List[AnyTensor]())
    states[0].append(zeros_like(params[0]))
    states[0].append(zeros_like(params[0]))

    states.append(List[AnyTensor]())
    states[1].append(zeros_like(params[1]))
    states[1].append(zeros_like(params[1]))

    # Should not raise
    validate_optimizer_state(params, states)
    assert_true(True)  # Mark as passed


def main() raises:
    """Run all test_optimizer_utils tests."""
    print("Running test_optimizer_utils tests...")

    test_scale_tensor()
    print("✓ test_scale_tensor")

    test_scale_tensor_inplace()
    print("✓ test_scale_tensor_inplace")

    test_compute_tensor_norm()
    print("✓ test_compute_tensor_norm")

    test_compute_tensor_norm_zero()
    print("✓ test_compute_tensor_norm_zero")

    test_compute_global_norm()
    print("✓ test_compute_global_norm")

    test_clip_tensor_norm()
    print("✓ test_clip_tensor_norm")

    test_main()
    print("✓ test_main")

    test_clip_tensor_norm_no_clip()
    print("✓ test_clip_tensor_norm_no_clip")

    test_clip_global_norm()
    print("✓ test_clip_global_norm")

    test_apply_weight_decay()
    print("✓ test_apply_weight_decay")

    test_compute_weight_decay_term()
    print("✓ test_compute_weight_decay_term")

    test_normalize_tensor_to_unit_norm()
    print("✓ test_normalize_tensor_to_unit_norm")

    test_apply_bias_correction()
    print("✓ test_apply_bias_correction")

    test_validate_optimizer_state_valid()
    print("✓ test_validate_optimizer_state_valid")

    print("\nAll test_optimizer_utils tests passed!")
