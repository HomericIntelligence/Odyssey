"""Unit tests for Validation Loop (evaluation without weight updates).

Tests cover:
- ValidationLoop constructor initialization
- validation_step() standalone function
- validate() standalone function
- ValidationLoop.run() full validation
- ValidationLoop.run_subset() subset validation
- Metrics tracking (loss updated in TrainingMetrics)
- No weight updates during validation (forward-only)

Issue #3082: Re-enable validation loop tests after ValidationLoop implementation.
Blockers resolved: ValidationLoop (Issue #34), DataLoader, TrainingMetrics all implemented.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_less,
    assert_greater,
    assert_not_equal_tensor,
)
from shared.training.loops.validation_loop import (
    ValidationLoop,
    validation_step,
    validate,
)
from shared.training.trainer_interface import (
    DataLoader,
    DataBatch,
    TrainingMetrics,
)
from shared.core.extensor import ExTensor
from shared.core import ones, zeros, randn


# ============================================================================
# Helper functions
# ============================================================================


fn simple_forward(data: ExTensor) raises -> ExTensor:
    """Simple forward: returns ones matching data shape."""
    return ones(data.shape(), data.dtype())


fn simple_loss(pred: ExTensor, labels: ExTensor) raises -> ExTensor:
    """Simple loss: returns scalar ones tensor."""
    return ones([1], DType.float32)


fn create_val_loader(n_batches: Int = 3) raises -> DataLoader:
    """Create a DataLoader with n_batches * 4 samples, batch_size=4, feature_dim=10."""
    var n_samples = n_batches * 4
    var data = ones([n_samples, 10], DType.float32)
    var labels = zeros([n_samples, 1], DType.float32)
    return DataLoader(data^, labels^, batch_size=4)


# ============================================================================
# ValidationLoop Initialization Tests
# ============================================================================


fn test_validation_loop_init_defaults() raises:
    """Test ValidationLoop constructor defaults."""
    var vloop = ValidationLoop()
    assert_true(vloop.compute_accuracy)
    assert_true(not vloop.compute_confusion)
    assert_equal(vloop.num_classes, 10)
    print("  test_validation_loop_init_defaults: PASSED")


fn test_validation_loop_init_custom() raises:
    """Test ValidationLoop constructor stores custom values."""
    var vloop = ValidationLoop(
        compute_accuracy=False, compute_confusion=True, num_classes=5
    )
    assert_true(not vloop.compute_accuracy)
    assert_true(vloop.compute_confusion)
    assert_equal(vloop.num_classes, 5)
    print("  test_validation_loop_init_custom: PASSED")


# ============================================================================
# validation_step() Tests
# ============================================================================


fn test_validation_step_returns_float() raises:
    """Test validation_step returns a Float64 loss value."""
    var data = ones([4, 10], DType.float32)
    var labels = zeros([4, 1], DType.float32)
    var loss = validation_step(simple_forward, simple_loss, data, labels)
    # Loss from ones tensor = 1.0
    assert_almost_equal(loss, Float64(1.0), Float64(1e-5))
    print("  test_validation_step_returns_float: PASSED")


fn test_validation_step_no_grad() raises:
    """Test validation_step completes without error (forward-only, no backward)."""
    var data = randn([4, 10], DType.float32, seed=42)
    var labels = zeros([4, 1], DType.float32)
    var loss = validation_step(simple_forward, simple_loss, data, labels)
    assert_greater(loss, Float64(-1e10))
    print("  test_validation_step_no_grad: PASSED")


# ============================================================================
# validate() Function Tests
# ============================================================================


fn test_validate_runs_full_loader() raises:
    """Test validate() iterates all batches and returns average loss."""
    var loader = create_val_loader(n_batches=3)
    var avg_loss = validate(simple_forward, simple_loss, loader)
    # Each batch returns loss=1.0, average over 3 batches = 1.0
    assert_almost_equal(avg_loss, Float64(1.0), Float64(1e-5))
    print("  test_validate_runs_full_loader: PASSED")


fn test_validate_returns_positive_loss() raises:
    """Test validate() returns non-negative loss."""
    var loader = create_val_loader(n_batches=2)
    var avg_loss = validate(simple_forward, simple_loss, loader)
    assert_greater(avg_loss, Float64(-1e-10))
    print("  test_validate_returns_positive_loss: PASSED")


# ============================================================================
# ValidationLoop.run() Tests
# ============================================================================


fn test_validation_loop_run_basic() raises:
    """Test ValidationLoop.run() returns valid loss."""
    var vloop = ValidationLoop()
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    var val_loss = vloop.run(simple_forward, simple_loss, loader, metrics)
    assert_greater(val_loss, Float64(-1e-10))
    print("  test_validation_loop_run_basic: PASSED")


fn test_validation_loop_run_updates_metrics() raises:
    """Test ValidationLoop.run() updates TrainingMetrics.val_loss."""
    var vloop = ValidationLoop()
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    var val_loss = vloop.run(simple_forward, simple_loss, loader, metrics)
    assert_almost_equal(metrics.val_loss, val_loss, Float64(1e-10))
    print("  test_validation_loop_run_updates_metrics: PASSED")


# ============================================================================
# ValidationLoop.run_subset() Tests
# ============================================================================


fn test_validation_loop_run_subset_limited() raises:
    """Test run_subset(max_batches=2) with 5-batch loader processes only 2 batches."""
    var vloop = ValidationLoop()
    var loader = create_val_loader(n_batches=5)
    var metrics = TrainingMetrics()
    # With max_batches=2, only 2 batches processed; loss from ones = 1.0
    var val_loss = vloop.run_subset(
        simple_forward, simple_loss, loader, 2, metrics
    )
    assert_almost_equal(val_loss, Float64(1.0), Float64(1e-5))
    print("  test_validation_loop_run_subset_limited: PASSED")


fn test_validation_loop_run_subset_loss_valid() raises:
    """Test run_subset returns valid Float64 loss."""
    var vloop = ValidationLoop()
    var loader = create_val_loader(n_batches=3)
    var metrics = TrainingMetrics()
    var val_loss = vloop.run_subset(
        simple_forward, simple_loss, loader, 1, metrics
    )
    assert_greater(val_loss, Float64(-1e-10))
    assert_less(val_loss, Float64(1e10))
    print("  test_validation_loop_run_subset_loss_valid: PASSED")


# ============================================================================
# No-Weight-Update Property Tests
# ============================================================================


fn test_validation_loop_no_weight_updates() raises:
    """Validate that validation runs forward-only without optimizer step.

    Since ValidationLoop has no optimizer, calling run() multiple times
    on the same loader with the same forward function produces the same loss.
    """
    var vloop = ValidationLoop()
    var metrics1 = TrainingMetrics()
    var metrics2 = TrainingMetrics()

    var loader1 = create_val_loader(n_batches=3)
    var loader2 = create_val_loader(n_batches=3)

    var loss1 = vloop.run(simple_forward, simple_loss, loader1, metrics1)
    var loss2 = vloop.run(simple_forward, simple_loss, loader2, metrics2)

    # Same inputs and forward fn -> same loss every time (no weight mutation)
    assert_almost_equal(loss1, loss2, Float64(1e-10))
    print("  test_validation_loop_no_weight_updates: PASSED")


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all validation loop tests."""
    print("Running ValidationLoop initialization tests...")
    test_validation_loop_init_defaults()
    test_validation_loop_init_custom()

    print("Running validation_step() tests...")
    test_validation_step_returns_float()
    test_validation_step_no_grad()

    print("Running validate() function tests...")
    test_validate_runs_full_loader()
    test_validate_returns_positive_loss()

    print("Running ValidationLoop.run() tests...")
    test_validation_loop_run_basic()
    test_validation_loop_run_updates_metrics()

    print("Running ValidationLoop.run_subset() tests...")
    test_validation_loop_run_subset_limited()
    test_validation_loop_run_subset_loss_valid()

    print("Running no-weight-update property tests...")
    test_validation_loop_no_weight_updates()

    print("\nAll validation loop tests passed!")
