"""Unit tests for loss utility functions (Part 2 of 3)

Tests cover:
- validate_tensor_dtypes: dtype mismatch detection
- compute_one_minus_tensor: 1 - x computation
- compute_sign_tensor: sign function

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


def main() raises:
    """Run test_loss_utils part 2 tests."""
    print("Running test_loss_utils_part2 tests...")

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

    print("\nAll test_loss_utils_part2 tests passed!")
