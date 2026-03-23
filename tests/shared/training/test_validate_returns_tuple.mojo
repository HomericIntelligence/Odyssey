"""Tests for validate() return value.

Issue #3683: Verify that validate() returns average validation loss
as a Float64 scalar, consistent with the validation_loop API.
"""

from tests.shared.conftest import (
    assert_true,
    assert_almost_equal,
    assert_less,
    assert_greater,
)
from shared.training.loops.validation_loop import validate
from shared.training.trainer_interface import DataLoader
from shared.tensor.any_tensor import AnyTensor
from shared.tensor.any_tensor import ones, zeros


# ============================================================================
# Helper functions
# ============================================================================


fn simple_forward(data: AnyTensor) raises -> AnyTensor:
    """Simple forward: returns ones matching data shape."""
    return ones(data.shape(), data.dtype())


fn simple_loss(pred: AnyTensor, labels: AnyTensor) raises -> AnyTensor:
    """Simple loss: returns scalar ones tensor."""
    return ones([1], DType.float32)


fn create_loader(n_batches: Int = 3) raises -> DataLoader:
    """Create a DataLoader with n_batches * 4 samples, batch_size=4."""
    var n_samples = n_batches * 4
    var data = ones([n_samples, 10], DType.float32)
    var labels = zeros([n_samples, 1], DType.float32)
    return DataLoader(data^, labels^, batch_size=4)


# ============================================================================
# Return value tests
# ============================================================================


fn test_validate_returns_float64() raises:
    """Test validate() returns a Float64 average loss value."""
    var loader = create_loader(n_batches=3)
    var result = validate(simple_forward, simple_loss, loader)
    assert_greater(result, Float64(-1e-10))
    print("  test_validate_returns_float64: PASSED")


fn test_validate_loss_value_correct() raises:
    """Test validate() loss equals the expected average loss."""
    var loader = create_loader(n_batches=3)
    # simple_loss returns 1.0 per batch -> average loss = 1.0
    var avg_loss = validate(simple_forward, simple_loss, loader)
    assert_almost_equal(avg_loss, Float64(1.0), Float64(1e-5))
    print("  test_validate_loss_value_correct: PASSED")


fn test_validate_loss_nonnegative() raises:
    """Test validate() returns a non-negative loss value."""
    var loader = create_loader(n_batches=3)
    var avg_loss = validate(simple_forward, simple_loss, loader)
    assert_greater(avg_loss, Float64(-1e-10))
    assert_less(avg_loss, Float64(1e10))
    print("  test_validate_loss_nonnegative: PASSED")


fn test_validate_with_accuracy_enabled() raises:
    """Test validate() with compute_accuracy=True still returns Float64 loss."""
    var loader = create_loader(n_batches=3)
    var avg_loss = validate(
        simple_forward, simple_loss, loader, compute_accuracy=True
    )
    assert_almost_equal(avg_loss, Float64(1.0), Float64(1e-5))
    print("  test_validate_with_accuracy_enabled: PASSED")


fn test_validate_with_accuracy_disabled() raises:
    """Test validate() with compute_accuracy=False still returns Float64 loss."""
    var loader = create_loader(n_batches=3)
    var avg_loss = validate(
        simple_forward, simple_loss, loader, compute_accuracy=False
    )
    assert_almost_equal(avg_loss, Float64(1.0), Float64(1e-5))
    print("  test_validate_with_accuracy_disabled: PASSED")


fn test_validate_different_batch_counts() raises:
    """Test validate() returns correct loss with different batch counts."""
    var loader = create_loader(n_batches=2)
    var avg_loss = validate(simple_forward, simple_loss, loader)
    assert_almost_equal(avg_loss, Float64(1.0), Float64(1e-5))
    print("  test_validate_different_batch_counts: PASSED")


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all validate() return value tests."""
    print("Running validate() return value tests...")
    test_validate_returns_float64()
    test_validate_loss_value_correct()
    test_validate_loss_nonnegative()
    test_validate_with_accuracy_enabled()
    test_validate_with_accuracy_disabled()
    test_validate_different_batch_counts()
    print("\nAll validate() return value tests passed!")
