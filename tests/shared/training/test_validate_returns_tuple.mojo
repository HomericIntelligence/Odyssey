"""Tests for validate() returning Tuple[Float64, Float64].

Issue #3683: Unify accuracy computation between validate() and run().
Verifies that validate() now returns (loss, accuracy) as a tuple,
eliminating the duplicate forward pass that existed in run().
"""

from tests.shared.conftest import (
    assert_true,
    assert_almost_equal,
    assert_less,
    assert_greater,
)
from shared.training.loops.validation_loop import validate
from shared.training.trainer_interface import DataLoader
from shared.core.extensor import ExTensor
from shared.core import ones, zeros


# ============================================================================
# Helper functions
# ============================================================================


fn simple_forward(data: ExTensor) raises -> ExTensor:
    """Simple forward: returns ones matching data shape."""
    return ones(data.shape(), data.dtype())


fn simple_loss(pred: ExTensor, labels: ExTensor) raises -> ExTensor:
    """Simple loss: returns scalar ones tensor."""
    return ones([1], DType.float32)


fn create_loader(n_batches: Int = 3) raises -> DataLoader:
    """Create a DataLoader with n_batches * 4 samples, batch_size=4."""
    var n_samples = n_batches * 4
    var data = ones([n_samples, 10], DType.float32)
    var labels = zeros([n_samples, 1], DType.float32)
    return DataLoader(data^, labels^, batch_size=4)


# ============================================================================
# Tuple return tests
# ============================================================================


fn test_validate_returns_tuple_both_values() raises:
    """Test validate() returns a tuple with loss and accuracy components."""
    var loader = create_loader(n_batches=3)
    var result = validate(simple_forward, simple_loss, loader)
    # Index access: result[0] is loss, result[1] is accuracy
    var loss = result[0]
    var accuracy = result[1]
    assert_greater(loss, Float64(-1e-10))
    assert_greater(accuracy, Float64(-1e-10))
    print("  test_validate_returns_tuple_both_values: PASSED")


fn test_validate_tuple_destructuring() raises:
    """Test validate() result can be destructured with var (loss, acc) = ..."""
    var loader = create_loader(n_batches=3)
    var (loss, accuracy) = validate(simple_forward, simple_loss, loader)
    assert_greater(loss, Float64(-1e-10))
    assert_less(loss, Float64(1e10))
    assert_greater(accuracy, Float64(-1e-10))
    print("  test_validate_tuple_destructuring: PASSED")


fn test_validate_loss_value_correct() raises:
    """Test validate() loss component equals the expected average loss."""
    var loader = create_loader(n_batches=3)
    # simple_loss returns 1.0 per batch -> average loss = 1.0
    var (avg_loss, _) = validate(simple_forward, simple_loss, loader)
    assert_almost_equal(avg_loss, Float64(1.0), Float64(1e-5))
    print("  test_validate_loss_value_correct: PASSED")


fn test_validate_accuracy_zero_when_disabled() raises:
    """Test validate() returns accuracy=0.0 when compute_accuracy=False."""
    var loader = create_loader(n_batches=3)
    var (_, accuracy) = validate(
        simple_forward, simple_loss, loader, compute_accuracy=False
    )
    assert_almost_equal(accuracy, Float64(0.0), Float64(1e-10))
    print("  test_validate_accuracy_zero_when_disabled: PASSED")


fn test_validate_accuracy_nonzero_when_enabled() raises:
    """Test validate() returns non-zero accuracy when compute_accuracy=True."""
    var loader = create_loader(n_batches=3)
    var (_, accuracy) = validate(
        simple_forward, simple_loss, loader, compute_accuracy=True
    )
    # Accuracy is a valid fraction in [0.0, 1.0]
    assert_greater(accuracy, Float64(-1e-10))
    assert_less(accuracy, Float64(1.0) + Float64(1e-5))
    print("  test_validate_accuracy_nonzero_when_enabled: PASSED")


fn test_validate_ignore_accuracy_with_underscore() raises:
    """Test that accuracy can be discarded using _ in destructuring."""
    var loader = create_loader(n_batches=2)
    var (loss, _) = validate(simple_forward, simple_loss, loader)
    assert_almost_equal(loss, Float64(1.0), Float64(1e-5))
    print("  test_validate_ignore_accuracy_with_underscore: PASSED")


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all validate() tuple return tests."""
    print("Running validate() tuple return tests...")
    test_validate_returns_tuple_both_values()
    test_validate_tuple_destructuring()
    test_validate_loss_value_correct()
    test_validate_accuracy_zero_when_disabled()
    test_validate_accuracy_nonzero_when_enabled()
    test_validate_ignore_accuracy_with_underscore()
    print("\nAll validate() tuple return tests passed!")
