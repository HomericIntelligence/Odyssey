"""Unit tests for optimizer utilities (Part 2 of 2).

Tests cover:
- Tensor norm clipping (no-clip case)
- Global norm clipping
- Weight decay utilities
- Tensor normalization to unit norm
- Bias correction for exponential moving averages
- State validation

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_optimizer_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

These tests verify the common utilities available to all optimizer implementations.
"""

from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal
from shared.core.extensor import ExTensor, zeros, ones, full, zeros_like
from shared.training.optimizers import (
    compute_weight_decay_term,
    apply_weight_decay,
    compute_tensor_norm,
    compute_global_norm,
    normalize_tensor_to_unit_norm,
    clip_tensor_norm,
    clip_global_norm,
    apply_bias_correction,
    validate_optimizer_state,
)


fn test_clip_tensor_norm_no_clip() raises:
    """Test that clipping doesn't occur when norm is below max."""
    var tensor = full([2], 0.5, DType.float32)
    # L2 norm = sqrt(0.5) ≈ 0.707

    var original_norm = clip_tensor_norm(tensor, max_norm=2.0)

    # Check tensor is unchanged (norm still below max)
    var new_norm = compute_tensor_norm(tensor)
    assert_almost_equal(new_norm, original_norm, tolerance=1e-6)


fn test_clip_global_norm() raises:
    """Test global norm clipping."""
    var t1 = full([3], 1.0, DType.float32)
    var t2 = full([4], 2.0, DType.float32)

    var tensors: List[ExTensor] = []
    tensors.append(t1)
    tensors.append(t2)

    # Original global norm = sqrt(19) ≈ 4.359
    var original_norm = clip_global_norm(tensors, max_norm=1.0)
    assert_almost_equal(original_norm, Float64(4.358898), tolerance=1e-5)

    # Check new global norm after clipping
    var new_global_norm = compute_global_norm(tensors)
    assert_almost_equal(new_global_norm, 1.0, tolerance=1e-6)


fn test_apply_weight_decay() raises:
    """Test in-place weight decay application."""
    var params = full([2], 10.0, DType.float32)

    apply_weight_decay(params, weight_decay=0.1)

    # After decay: params *= (1 - 0.1) = 0.9
    # So 10.0 becomes 9.0
    var expected = 9.0
    for i in range(params.numel()):
        var actual = Float64(params._get_float64(i))
        assert_almost_equal(actual, expected, tolerance=1e-6)


fn test_compute_weight_decay_term() raises:
    """Test weight decay term computation."""
    var params = full([2], 5.0, DType.float32)

    var wd_term = compute_weight_decay_term(params, weight_decay=0.1)

    # wd_term = 0.1 * params = 0.1 * 5.0 = 0.5
    var expected = 0.5
    for i in range(wd_term.numel()):
        var actual = Float64(wd_term._get_float64(i))
        assert_almost_equal(actual, expected, tolerance=1e-6)


fn test_normalize_tensor_to_unit_norm() raises:
    """Test tensor normalization to unit norm."""
    var tensor = full([4], 2.0, DType.float32)

    normalize_tensor_to_unit_norm(tensor)

    var norm = compute_tensor_norm(tensor)
    assert_almost_equal(norm, 1.0, tolerance=1e-6)


fn test_apply_bias_correction() raises:
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


fn test_validate_optimizer_state_valid() raises:
    """Test validation of valid optimizer state."""
    var params: List[ExTensor] = []
    var shape1 = List[Int]()
    shape1.append(2)
    shape1.append(3)
    params.append(ones(shape1, DType.float32))
    var shape2 = List[Int]()
    shape2.append(3)
    params.append(ones(shape2, DType.float32))

    var states = List[List[ExTensor]]()
    states.append(List[ExTensor]())
    states[0].append(zeros_like(params[0]))
    states[0].append(zeros_like(params[0]))

    states.append(List[ExTensor]())
    states[1].append(zeros_like(params[1]))
    states[1].append(zeros_like(params[1]))

    # Should not raise
    validate_optimizer_state(params, states)
    assert_true(True)  # Mark as passed


fn test_main() raises:
    """Run all tests."""
    test_clip_tensor_norm_no_clip()
    test_clip_global_norm()
    test_apply_weight_decay()
    test_compute_weight_decay_term()
    test_normalize_tensor_to_unit_norm()
    test_apply_bias_correction()
    test_validate_optimizer_state_valid()


fn main() raises:
    """Entry point for running tests."""
    print("Running optimizer utils tests (part 2)...")
    test_main()
    print("All optimizer utils tests (part 2) passed!")
