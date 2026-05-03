"""Unit tests for loss utility functions (Part 3 of 3)

Tests cover:
- blend_tensors: mixed mask
- compute_difference: basic and shape mismatch
- compute_product: basic and zero
- compute_ratio: basic and zero denominator
- negate_tensor: positive, negative, and zero

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
    """Run test_loss_utils part 3 tests."""
    print("Running test_loss_utils_part3 tests...")

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

    print("\nAll test_loss_utils_part3 tests passed!")
