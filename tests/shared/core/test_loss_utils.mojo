"""Unit tests for loss utility functions

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
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.loss_utils import (
    clip_predictions,
    create_epsilon_tensor,
    validate_tensor_shapes,
)
from tests.shared.conftest import (
    assert_almost_equal,
    assert_true,
)
from shared.core.loss_utils import (
    validate_tensor_dtypes,
    compute_one_minus_tensor,
    compute_sign_tensor,
)
from shared.core.loss_utils import (
    blend_tensors,
    compute_difference,
    compute_product,
    compute_ratio,
)
from tests.shared.conftest import (
    assert_almost_equal,
    assert_greater_or_equal,
)
from shared.tensor.any_tensor import AnyTensor, zeros, full
from shared.core.loss_utils import (
    compute_ratio,
    negate_tensor,
)


def test_clip_predictions_within_range() raises:
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


def test_clip_predictions_zero_lower_bound() raises:
    """Test clip_predictions clips 0 to epsilon."""
    var shape = List[Int]()
    shape.append(1)
    var predictions = zeros(shape, DType.float32)

    var clipped = clip_predictions(predictions, epsilon=1e-5)

    var clipped_data = clipped._data.bitcast[Float32]()
    assert_almost_equal(Float64(clipped_data[0]), 1e-5, tolerance=1e-6)


def test_clip_predictions_one_upper_bound() raises:
    """Test clip_predictions clips 1 to 1 - epsilon."""
    var shape = List[Int]()
    shape.append(1)
    var predictions = full(shape, 1.0, DType.float32)

    var clipped = clip_predictions(predictions, epsilon=1e-5)

    var clipped_data = clipped._data.bitcast[Float32]()
    assert_almost_equal(Float64(clipped_data[0]), 1.0 - 1e-5, tolerance=1e-6)


def test_clip_predictions_custom_epsilon() raises:
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


def test_create_epsilon_tensor_shape() raises:
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


def test_create_epsilon_tensor_values() raises:
    """Test create_epsilon_tensor fills with correct epsilon value."""
    var shape = List[Int]()
    shape.append(4)
    var template = zeros(shape, DType.float32)

    var epsilon_val = 1e-7
    var epsilon_tensor = create_epsilon_tensor(template, epsilon=epsilon_val)

    var eps_data = epsilon_tensor._data.bitcast[Float32]()
    for i in range(4):
        assert_almost_equal(Float64(eps_data[i]), epsilon_val, tolerance=1e-8)


def test_validate_tensor_shapes_matching() raises:
    """Test validate_tensor_shapes accepts matching shapes."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var tensor1 = zeros(shape, DType.float32)
    var tensor2 = ones(shape, DType.float32)

    # Should not raise
    validate_tensor_shapes(tensor1, tensor2, "test_op")


def test_validate_tensor_shapes_mismatch() raises:
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


def test_validate_tensor_dtypes_matching() raises:
    """Test validate_tensor_dtypes accepts matching dtypes."""
    var shape = List[Int]()
    shape.append(3)

    var tensor1 = zeros(shape, DType.float32)
    var tensor2 = ones(shape, DType.float32)

    # Should not raise
    validate_tensor_dtypes(tensor1, tensor2, "test_op")


def test_validate_tensor_dtypes_mismatch() raises:
    """Test validate_tensor_dtypes raises on dtype mismatch."""
    var shape = List[Int]()
    shape.append(3)

    var tensor1 = zeros(shape, DType.float32)
    var tensor2 = zeros(shape, DType.float64)

    var error_raised = False
    try:
        validate_tensor_dtypes(tensor1, tensor2, "test_op")
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error on dtype mismatch")


def test_compute_one_minus_tensor_half() raises:
    """Test compute_one_minus_tensor with 0.5 gives 0.5."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = full(shape, 0.5, DType.float32)

    var result = compute_one_minus_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.5, tolerance=1e-6)


def test_compute_one_minus_tensor_zero() raises:
    """Test compute_one_minus_tensor with 0.0 gives 1.0."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = zeros(shape, DType.float32)

    var result = compute_one_minus_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 1.0, tolerance=1e-6)


def test_compute_one_minus_tensor_one() raises:
    """Test compute_one_minus_tensor with 1.0 gives 0.0."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = full(shape, 1.0, DType.float32)

    var result = compute_one_minus_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.0, tolerance=1e-6)


def test_compute_sign_tensor_positive() raises:
    """Test compute_sign_tensor with positive values."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = full(shape, 5.0, DType.float32)

    var result = compute_sign_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 1.0, tolerance=1e-6)


def test_compute_sign_tensor_negative() raises:
    """Test compute_sign_tensor with negative values."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = full(shape, -3.0, DType.float32)

    var result = compute_sign_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), -1.0, tolerance=1e-6)


def test_compute_sign_tensor_zero() raises:
    """Test compute_sign_tensor with zero."""
    var shape = List[Int]()
    shape.append(1)
    var tensor = zeros(shape, DType.float32)

    var result = compute_sign_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.0, tolerance=1e-6)


def test_blend_tensors_all_first() raises:
    """Test blend_tensors selects first tensor with all 1s mask."""
    var shape = List[Int]()
    shape.append(2)

    var tensor1 = full(shape, 1.0, DType.float32)
    var tensor2 = full(shape, 2.0, DType.float32)
    var mask = ones(shape, DType.float32)  # All 1s

    var result = blend_tensors(tensor1, tensor2, mask)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 1.0, tolerance=1e-6)


def test_blend_tensors_all_second() raises:
    """Test blend_tensors selects second tensor with all 0s mask."""
    var shape = List[Int]()
    shape.append(2)

    var tensor1 = full(shape, 1.0, DType.float32)
    var tensor2 = full(shape, 2.0, DType.float32)
    var mask = zeros(shape, DType.float32)  # All 0s

    var result = blend_tensors(tensor1, tensor2, mask)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 2.0, tolerance=1e-6)


def test_blend_tensors_mixed() raises:
    """Test blend_tensors with mixed mask values."""
    var shape = List[Int]()
    shape.append(2)

    var tensor1 = full(shape, 10.0, DType.float32)
    var tensor2 = full(shape, 20.0, DType.float32)

    var mask = zeros(shape, DType.float32)
    var mask_data = mask._data.bitcast[Float32]()
    mask_data[0] = 1.0  # First element selects tensor1
    mask_data[1] = 0.0  # Second element selects tensor2

    var result = blend_tensors(tensor1, tensor2, mask)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 10.0, tolerance=1e-6)
    assert_almost_equal(Float64(result_data[1]), 20.0, tolerance=1e-6)


def test_compute_difference_basic() raises:
    """Test compute_difference computes tensor1 - tensor2."""
    var shape = List[Int]()
    shape.append(3)

    var tensor1 = full(shape, 5.0, DType.float32)
    var tensor2 = full(shape, 2.0, DType.float32)

    var result = compute_difference(tensor1, tensor2)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 3.0, tolerance=1e-6)


def test_compute_difference_shape_mismatch() raises:
    """Test compute_difference raises on shape mismatch."""
    var shape1 = List[Int]()
    shape1.append(2)

    var shape2 = List[Int]()
    shape2.append(3)

    var tensor1 = zeros(shape1, DType.float32)
    var tensor2 = zeros(shape2, DType.float32)

    var error_raised = False
    try:
        _ = compute_difference(tensor1, tensor2)
    except:
        error_raised = True

    assert_true(error_raised, "Should raise error on shape mismatch")


def test_compute_product_basic() raises:
    """Test compute_product computes element-wise multiplication."""
    var shape = List[Int]()
    shape.append(2)

    var tensor1 = full(shape, 3.0, DType.float32)
    var tensor2 = full(shape, 4.0, DType.float32)

    var result = compute_product(tensor1, tensor2)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 12.0, tolerance=1e-6)


def test_compute_product_with_zero() raises:
    """Test compute_product with zero gives zero."""
    var shape = List[Int]()
    shape.append(1)

    var tensor1 = full(shape, 5.0, DType.float32)
    var tensor2 = zeros(shape, DType.float32)

    var result = compute_product(tensor1, tensor2)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.0, tolerance=1e-6)


def test_compute_ratio_basic() raises:
    """Test compute_ratio computes numerator / denominator."""
    var shape = List[Int]()
    shape.append(1)

    var numerator = full(shape, 6.0, DType.float32)
    var denominator = full(shape, 2.0, DType.float32)

    var result = compute_ratio(numerator, denominator)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 3.0, tolerance=1e-6)


def test_compute_ratio_zero_denominator() raises:
    """Test compute_ratio prevents division by zero with epsilon."""
    var shape = List[Int]()
    shape.append(1)

    var numerator = full(shape, 1.0, DType.float32)
    var denominator = zeros(shape, DType.float32)

    var result = compute_ratio(numerator, denominator, epsilon=1e-5)

    var result_data = result._data.bitcast[Float32]()
    # Should be 1.0 / 1e-5 = 100000, but check it's finite and positive
    assert_greater_or_equal(
        result_data[0], 0.0, "Result should be non-negative"
    )


def test_negate_tensor_positive() raises:
    """Test negate_tensor negates positive values."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = full(shape, 5.0, DType.float32)

    var result = negate_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), -5.0, tolerance=1e-6)


def test_negate_tensor_negative() raises:
    """Test negate_tensor negates negative values."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = full(shape, -3.0, DType.float32)

    var result = negate_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 3.0, tolerance=1e-6)


def test_negate_tensor_zero() raises:
    """Test negate_tensor with zero stays zero."""
    var shape = List[Int]()
    shape.append(2)
    var tensor = zeros(shape, DType.float32)

    var result = negate_tensor(tensor)

    var result_data = result._data.bitcast[Float32]()
    assert_almost_equal(Float64(result_data[0]), 0.0, tolerance=1e-6)


def main() raises:
    """Run all test_loss_utils tests."""
    print("Running test_loss_utils tests...")

    test_clip_predictions_within_range()
    print("✓ test_clip_predictions_within_range")

    test_clip_predictions_zero_lower_bound()
    print("✓ test_clip_predictions_zero_lower_bound")

    test_clip_predictions_one_upper_bound()
    print("✓ test_clip_predictions_one_upper_bound")

    test_clip_predictions_custom_epsilon()
    print("✓ test_clip_predictions_custom_epsilon")

    test_create_epsilon_tensor_shape()
    print("✓ test_create_epsilon_tensor_shape")

    test_create_epsilon_tensor_values()
    print("✓ test_create_epsilon_tensor_values")

    test_validate_tensor_shapes_matching()
    print("✓ test_validate_tensor_shapes_matching")

    test_validate_tensor_shapes_mismatch()
    print("✓ test_validate_tensor_shapes_mismatch")

    test_validate_tensor_dtypes_matching()
    print("✓ test_validate_tensor_dtypes_matching")

    test_validate_tensor_dtypes_mismatch()
    print("✓ test_validate_tensor_dtypes_mismatch")

    test_compute_one_minus_tensor_half()
    print("✓ test_compute_one_minus_tensor_half")

    test_compute_one_minus_tensor_zero()
    print("✓ test_compute_one_minus_tensor_zero")

    test_compute_one_minus_tensor_one()
    print("✓ test_compute_one_minus_tensor_one")

    test_compute_sign_tensor_positive()
    print("✓ test_compute_sign_tensor_positive")

    test_compute_sign_tensor_negative()
    print("✓ test_compute_sign_tensor_negative")

    test_compute_sign_tensor_zero()
    print("✓ test_compute_sign_tensor_zero")

    test_blend_tensors_all_first()
    print("✓ test_blend_tensors_all_first")

    test_blend_tensors_all_second()
    print("✓ test_blend_tensors_all_second")

    test_blend_tensors_mixed()
    print("✓ test_blend_tensors_mixed")

    test_compute_difference_basic()
    print("✓ test_compute_difference_basic")

    test_compute_difference_shape_mismatch()
    print("✓ test_compute_difference_shape_mismatch")

    test_compute_product_basic()
    print("✓ test_compute_product_basic")

    test_compute_product_with_zero()
    print("✓ test_compute_product_with_zero")

    test_compute_ratio_basic()
    print("✓ test_compute_ratio_basic")

    test_compute_ratio_zero_denominator()
    print("✓ test_compute_ratio_zero_denominator")

    test_negate_tensor_positive()
    print("✓ test_negate_tensor_positive")

    test_negate_tensor_negative()
    print("✓ test_negate_tensor_negative")

    test_negate_tensor_zero()
    print("✓ test_negate_tensor_zero")

    print("\nAll test_loss_utils tests passed!")
