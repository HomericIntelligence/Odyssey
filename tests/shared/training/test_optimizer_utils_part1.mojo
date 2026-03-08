"""Unit tests for optimizer utilities (Part 1 of 2).

Tests cover:
- State initialization utilities
- Tensor scaling and normalization
- Norm computation (single and global)
- Single tensor norm clipping

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_optimizer_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

These tests verify the common utilities available to all optimizer implementations.
"""

from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal
from shared.core.extensor import ExTensor, zeros, ones, full, zeros_like
from shared.training.optimizers import (
    initialize_optimizer_state,
    initialize_optimizer_state_from_params,
    scale_tensor,
    scale_tensor_inplace,
    compute_tensor_norm,
    compute_global_norm,
    clip_tensor_norm,
)


fn test_initialize_optimizer_state() raises:
    """Test basic optimizer state initialization."""
    # Create shapes for 2 parameters
    var shapes = List[List[Int]]()
    shapes.append([3, 4])
    shapes.append([4])

    # Initialize 2 states per parameter (e.g., Adam: m and v)
    var states = initialize_optimizer_state(shapes, num_states=2)

    # Check structure
    assert_true(len(states) == 2, "Should have 2 state lists")
    assert_true(len(states[0]) == 2, "First param should have 2 states")
    assert_true(len(states[1]) == 2, "Second param should have 2 states")

    # Check shapes
    assert_true(states[0][0].shape()[0] == 3)
    assert_true(states[0][0].shape()[1] == 4)
    assert_true(states[1][0].shape()[0] == 4)


fn test_initialize_optimizer_state_from_params() raises:
    """Test optimizer state initialization from parameters."""
    var params: List[ExTensor] = []
    params.append(ones([2, 3], DType.float32))
    params.append(ones([3], DType.float32))

    var states = initialize_optimizer_state_from_params(params, num_states=2)

    # Check structure
    assert_true(len(states) == 2)
    assert_true(len(states[0]) == 2)
    assert_true(len(states[1]) == 2)

    # Check shapes match
    assert_true(states[0][0].shape() == params[0].shape())
    assert_true(states[1][0].shape() == params[1].shape())


fn test_scale_tensor() raises:
    """Test tensor scaling."""
    var tensor = full([3], 2.0, DType.float32)
    var scaled = scale_tensor(tensor, scale=0.5)

    # Check values
    var expected = 1.0
    for i in range(scaled.numel()):
        var actual = Float64(scaled._get_float64(i))
        assert_almost_equal(actual, expected, tolerance=1e-6)


fn test_scale_tensor_inplace() raises:
    """Test in-place tensor scaling."""
    var tensor = full([2], 4.0, DType.float32)
    scale_tensor_inplace(tensor, scale=0.25)

    # Check values after scaling
    var expected = 1.0
    for i in range(tensor.numel()):
        var actual = Float64(tensor._get_float64(i))
        assert_almost_equal(actual, expected, tolerance=1e-6)


fn test_compute_tensor_norm() raises:
    """Test L2 norm computation."""
    var tensor = full([4], 3.0, DType.float32)

    # L2 norm of [3, 3, 3, 3] = sqrt(9 + 9 + 9 + 9) = sqrt(36) = 6.0
    var norm = compute_tensor_norm(tensor)
    assert_almost_equal(norm, 6.0, tolerance=1e-6)


fn test_compute_tensor_norm_zero() raises:
    """Test L2 norm of zero tensor."""
    var tensor = zeros([3], DType.float32)
    var norm = compute_tensor_norm(tensor)
    assert_almost_equal(norm, 0.0, tolerance=1e-6)


fn test_compute_global_norm() raises:
    """Test global L2 norm computation."""
    var t1 = full([3], 1.0, DType.float32)
    var t2 = full([4], 2.0, DType.float32)

    var tensors: List[ExTensor] = []
    tensors.append(t1)
    tensors.append(t2)

    # Global norm = sqrt(3*1^2 + 4*2^2) = sqrt(3 + 16) = sqrt(19)
    var global_norm = compute_global_norm(tensors)
    var expected = Float64(4.358898)  # sqrt(19)
    assert_almost_equal(global_norm, expected, tolerance=1e-5)


fn test_clip_tensor_norm() raises:
    """Test single tensor norm clipping."""
    var tensor = full([9], 1.0, DType.float32)
    # L2 norm = sqrt(9) = 3.0

    var original_norm = clip_tensor_norm(tensor, max_norm=1.5)

    # Check original norm was returned
    assert_almost_equal(original_norm, 3.0, tolerance=1e-6)

    # Check tensor was clipped to max_norm
    var new_norm = compute_tensor_norm(tensor)
    assert_almost_equal(new_norm, 1.5, tolerance=1e-6)


fn test_main() raises:
    """Run all tests."""
    test_initialize_optimizer_state()
    test_initialize_optimizer_state_from_params()
    test_scale_tensor()
    test_scale_tensor_inplace()
    test_compute_tensor_norm()
    test_compute_tensor_norm_zero()
    test_compute_global_norm()
    test_clip_tensor_norm()


fn main() raises:
    """Entry point for running tests."""
    print("Running optimizer utils tests (part 1)...")
    test_main()
    print("All optimizer utils tests (part 1) passed!")
