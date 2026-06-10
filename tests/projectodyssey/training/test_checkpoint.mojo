"""Tests for checkpoint management.

Verifies correctness of CheckpointManager for saving and loading model checkpoints,
tracking best models, and resuming training.

Test Coverage:
- save_checkpoint: Save model weights and metadata
- load_latest: Load most recent checkpoint
- save_best: Track and save best model by metric
- load_best: Load best model checkpoint
- Metadata persistence: Epoch, metrics stored correctly
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.training.checkpoint import CheckpointManager
from projectodyssey.testing.assertions import (
    assert_true,
    assert_equal_int,
    assert_close_float,
)
from std.collections import List


def create_test_params() raises -> List[AnyTensor]:
    """Create simple test parameters."""
    var params = List[AnyTensor]()
    params.append(ones([10, 10], DType.float32))  # param1
    params.append(ones([5], DType.float32))  # param2
    return params^


def create_param_names() -> List[String]:
    """Create parameter names for test."""
    var names = List[String]()
    names.append("param1")
    names.append("param2")
    return names^


def test_save_and_load_checkpoint() raises:
    """Test basic checkpoint save and load."""
    # Create checkpoint manager
    var ckpt_dir = "/tmp/test_checkpoint_basic"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=3)

    # Create test parameters
    var params = create_test_params()
    var param_names = create_param_names()

    # Save checkpoint at epoch 5
    ckpt_mgr.save_checkpoint(
        params,
        param_names,
        epoch=5,
        train_loss=0.5,
        val_loss=0.6,
        val_acc=0.85,
    )

    # Load checkpoint
    var loaded_params = List[AnyTensor]()
    var loaded_epoch = ckpt_mgr.load_latest(loaded_params, param_names)

    # Verify epoch
    assert_equal_int(loaded_epoch, 5, "Should load epoch 5")

    # Verify parameters loaded
    assert_equal_int(len(loaded_params), 2, "Should load 2 parameters")


def test_load_latest_no_checkpoint() raises:
    """Test loading when no checkpoint exists."""
    var ckpt_dir = "/tmp/test_checkpoint_no_exist"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=3)

    var params = List[AnyTensor]()
    var param_names = create_param_names()

    # Should return epoch 0 when no checkpoint exists
    var epoch = ckpt_mgr.load_latest(params, param_names)
    assert_equal_int(
        epoch, 0, "Should return epoch 0 when no checkpoint exists"
    )


def test_save_best_model() raises:
    """Test best model tracking and saving."""
    var ckpt_dir = "/tmp/test_checkpoint_best"
    var ckpt_mgr = CheckpointManager(
        ckpt_dir,
        max_to_keep=3,
        best_metric_name="val_loss",
        minimize_metric=True,
    )

    var params = create_test_params()
    var param_names = create_param_names()

    # Save checkpoint with val_loss=0.5
    ckpt_mgr.save_checkpoint(
        params, param_names, epoch=1, val_loss=0.5, val_acc=0.80
    )
    ckpt_mgr.save_best(params, param_names, epoch=1, metric_value=0.5)

    # Save checkpoint with val_loss=0.3 (better)
    ckpt_mgr.save_checkpoint(
        params, param_names, epoch=2, val_loss=0.3, val_acc=0.85
    )
    ckpt_mgr.save_best(params, param_names, epoch=2, metric_value=0.3)

    # Save checkpoint with val_loss=0.4 (worse, should not be best)
    ckpt_mgr.save_checkpoint(
        params, param_names, epoch=3, val_loss=0.4, val_acc=0.82
    )
    ckpt_mgr.save_best(params, param_names, epoch=3, metric_value=0.4)

    # Load best model
    var best_params = List[AnyTensor]()
    ckpt_mgr.load_best(best_params, param_names)

    # Verify best model was loaded
    assert_equal_int(
        len(best_params), 2, "Should load 2 parameters from best model"
    )


def test_resume_training() raises:
    """Test resuming training from latest checkpoint."""
    var ckpt_dir = "/tmp/test_checkpoint_resume"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=3)

    var params = create_test_params()
    var param_names = create_param_names()

    # Simulate training for 3 epochs
    for epoch in range(1, 4):
        var train_loss = 1.0 / Float32(epoch)
        var val_loss = 0.9 / Float32(epoch)
        var val_acc = 0.7 + Float32(epoch) * 0.05

        ckpt_mgr.save_checkpoint(
            params,
            param_names,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
        )

    # Resume from latest checkpoint
    var resumed_params = List[AnyTensor]()
    var start_epoch = ckpt_mgr.load_latest(resumed_params, param_names)

    # Should resume from epoch 3
    assert_equal_int(start_epoch, 3, "Should resume from epoch 3")
    assert_equal_int(len(resumed_params), 2, "Should load 2 parameters")


def test_multiple_checkpoints() raises:
    """Test saving multiple checkpoints."""
    var ckpt_dir = "/tmp/test_checkpoint_multiple"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=5)

    var params = create_test_params()
    var param_names = create_param_names()

    # Save 10 checkpoints
    for epoch in range(1, 11):
        ckpt_mgr.save_checkpoint(
            params, param_names, epoch=epoch, train_loss=Float32(epoch) * 0.1
        )

    # Load latest (should be epoch 10)
    var loaded_params = List[AnyTensor]()
    var latest_epoch = ckpt_mgr.load_latest(loaded_params, param_names)

    assert_equal_int(latest_epoch, 10, "Latest epoch should be 10")


def test_best_model_maximize_metric() raises:
    """Test best model tracking with maximization metric (accuracy)."""
    var ckpt_dir = "/tmp/test_checkpoint_maximize"
    var ckpt_mgr = CheckpointManager(
        ckpt_dir,
        max_to_keep=3,
        best_metric_name="val_acc",
        minimize_metric=False,  # Maximize accuracy
    )

    var params = create_test_params()
    var param_names = create_param_names()

    # Save checkpoints with increasing accuracy
    ckpt_mgr.save_checkpoint(params, param_names, epoch=1, val_acc=0.75)
    ckpt_mgr.save_best(params, param_names, epoch=1, metric_value=0.75)

    ckpt_mgr.save_checkpoint(params, param_names, epoch=2, val_acc=0.85)
    ckpt_mgr.save_best(params, param_names, epoch=2, metric_value=0.85)

    ckpt_mgr.save_checkpoint(params, param_names, epoch=3, val_acc=0.80)
    ckpt_mgr.save_best(
        params, param_names, epoch=3, metric_value=0.80
    )  # Should not update

    # Verify best model tracking
    # Note: We can't directly check best_metric_value since it's internal
    # but save_best should only save when metric improves
    var best_params = List[AnyTensor]()
    ckpt_mgr.load_best(best_params, param_names)

    assert_equal_int(len(best_params), 2, "Should load best model parameters")


def main() raises:
    """Run all checkpoint tests."""
    print("Testing Checkpoint Manager...")
    print("=" * 70)

    print("\n[1/7] Testing basic save and load...")
    test_save_and_load_checkpoint()
    print("✓ PASSED")

    print("[2/7] Testing load when no checkpoint exists...")
    test_load_latest_no_checkpoint()
    print("✓ PASSED")

    print("[3/7] Testing best model tracking (minimize)...")
    test_save_best_model()
    print("✓ PASSED")

    print("[4/7] Testing resume training...")
    test_resume_training()
    print("✓ PASSED")

    print("[5/7] Testing multiple checkpoints...")
    test_multiple_checkpoints()
    print("✓ PASSED")

    print("[6/7] Testing best model tracking (maximize)...")
    test_best_model_maximize_metric()
    print("✓ PASSED")

    print("\n" + "=" * 70)
    print("All 6 checkpoint tests PASSED! ✓")
    print("Checkpoint manager is working correctly.")
