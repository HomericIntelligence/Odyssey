# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_loss_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for loss utility functions - Part 1.

Tests cover:
- clip_predictions: Clipping predictions for numerical stability
- create_epsilon_tensor: Epsilon tensor creation
- validate_tensor_shapes: Shape validation

All tests use pure functional API - no internal state.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_true,
    assert_close_float,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.loss_utils import (
    clip_predictions,
    create_epsilon_tensor,
    validate_tensor_shapes,
)


# ============================================================================
# clip_predictions Tests
# ============================================================================


fn test_clip_predictions_within_range() raises:
    """Test clip_predictions with values already in safe range."""
    var shape = List[Int]()
    shape.append(3)
    var predictions = full(shape, 0.5, DType.float32)

    var clipped = clip_predictions(predictions)

    # All values should be in [1e-7, 1.0 - 1e-7]
    var clipped_data = clipped._data.bitcast[Float32]()
    for i in range(3):
        assert_greater_or_equal(
            clipped_data[i], 1e-7, "Clipped value should be >= epsilon"
        )
        assert_less_or_equal(
            clipped_data[i], 1.0 - 1e-7, "Clipped value should be <= 1-epsilon"
        )


fn test_clip_predictions_zero_lower_bound() raises:
    """Test clip_predictions clips 0 to epsilon."""
    var shape = List[Int]()
    shape.append(1)
    var predictions = zeros(shape, DType.float32)

    var clipped = clip_predictions(predictions, epsilon=1e-5)

    var clipped_data = clipped._data.bitcast[Float32]()
    assert_almost_equal(Float64(clipped_data[0]), 1e-5, tolerance=1e-6)


fn test_clip_predictions_one_upper_bound() raises:
    """Test clip_predictions clips 1 to 1 - epsilon."""
    var shape = List[Int]()
    shape.append(1)
    var predictions = full(shape, 1.0, DType.float32)

    var clipped = clip_predictions(predictions, epsilon=1e-5)

    var clipped_data = clipped._data.bitcast[Float32]()
    assert_almost_equal(Float64(clipped_data[0]), 1.0 - 1e-5, tolerance=1e-6)


fn test_clip_predictions_custom_epsilon() raises:
    """Test clip_predictions with custom epsilon value."""
    var shape = List[Int]()
    shape.append(2)
    var predictions = full(shape, 0.0, DType.float32)

    var custom_epsilon = 1e-3
    var clipped = clip_predictions(predictions, epsilon=custom_epsilon)

    var clipped_data = clipped._data.bitcast[Float32]()
    assert_almost_equal(
        Float64(clipped_data[0]), custom_epsilon, tolerance=1e-6
    )


# ============================================================================
# create_epsilon_tensor Tests
# ============================================================================


fn test_create_epsilon_tensor_shape() raises:
    """Test create_epsilon_tensor returns correct shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var template = zeros(shape, DType.float32)

    var epsilon_tensor = create_epsilon_tensor(template, epsilon=1e-5)

    assert_true(
        epsilon_tensor.shape() == template.shape(),
        "Epsilon tensor should match template shape",
    )
    assert_true(
        epsilon_tensor.numel() == template.numel(),
        "Epsilon tensor should have same numel as template",
    )


fn test_create_epsilon_tensor_values() raises:
    """Test create_epsilon_tensor fills with correct epsilon value."""
    var shape = List[Int]()
    shape.append(4)
    var template = zeros(shape, DType.float32)

    var epsilon_val = 1e-7
    var epsilon_tensor = create_epsilon_tensor(template, epsilon=epsilon_val)

    var eps_data = epsilon_tensor._data.bitcast[Float32]()
    for i in range(4):
        assert_almost_equal(Float64(eps_data[i]), epsilon_val, tolerance=1e-8)


# ============================================================================
# validate_tensor_shapes Tests
# ============================================================================


fn test_validate_tensor_shapes_matching() raises:
    """Test validate_tensor_shapes accepts matching shapes."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var tensor1 = zeros(shape, DType.float32)
    var tensor2 = ones(shape, DType.float32)

    # Should not raise
    validate_tensor_shapes(tensor1, tensor2, "test_op")


fn test_validate_tensor_shapes_mismatch() raises:
    """Test validate_tensor_shapes raises on shape mismatch."""
    var shape1 = List[Int]()
    shape1.append(2)
    shape1.append(3)

    var shape2 = List[Int]()
    shape2.append(2)
    shape2.append(4)

    var tensor1 = zeros(shape1, DType.float32)
    var tensor2 = zeros(shape2, DType.float32)

    var error_raised = False
    try:
        validate_tensor_shapes(tensor1, tensor2, "test_op")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error on shape mismatch")


fn main() raises:
    """Run loss utility tests - Part 1."""
    print("Running loss utility tests - Part 1...")

    # clip_predictions tests
    test_clip_predictions_within_range()
    test_clip_predictions_zero_lower_bound()
    test_clip_predictions_one_upper_bound()
    test_clip_predictions_custom_epsilon()

    # create_epsilon_tensor tests
    test_create_epsilon_tensor_shape()
    test_create_epsilon_tensor_values()

    # validate_tensor_shapes tests
    test_validate_tensor_shapes_matching()
    test_validate_tensor_shapes_mismatch()

    print("All loss utility tests - Part 1 passed!")
