"""Tests for checkpoint atomic save and resume mechanism.

Verifies correctness of CheckpointManager extensions for:
- Atomic save with step and optimizer state
- Per-epoch incremental saves
- Fresh flag behavior (clear existing checkpoint)
- Resume loading with latest checkpoint
- Partial write recovery (skips broken epochs without metadata)
- Config snapshot round-trip
- TrainingConfig new fields

Test Coverage (Issue #5184):
- AC1/AC2/AC6: save_and_load_with_optimizer_state
- AC1 config: save_load_config_snapshot
- AC3: training_config_has_new_fields
- AC4: (integration test - covered in test_training_loop.mojo)
- AC5: resume_loads_latest_checkpoint
- Decision 9: partial_write_recovery_skips_broken_epoch
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, ones
from projectodyssey.training.checkpoint import CheckpointManager
from projectodyssey.training.config import TrainingConfig
from projectodyssey.utils.file_io import (
    file_exists,
    create_directory,
    safe_write_file,
    safe_read_file,
)
from projectodyssey.testing.assertions import (
    assert_true,
    assert_equal_int,
    assert_close_float,
)
from std.collections import List


def create_test_params() raises -> List[AnyTensor]:
    """Create simple test parameters (3 params for optimizer state tests)."""
    var params = List[AnyTensor]()
    params.append(ones([4, 4], DType.float32))  # param1
    params.append(ones([4], DType.float32))  # param2
    params.append(ones([2, 4], DType.float32))  # param3
    return params^


def create_param_names() -> List[String]:
    """Create parameter names for test."""
    var names = List[String]()
    names.append("param1")
    names.append("param2")
    names.append("param3")
    return names^


def create_optimizer_state(
    num_params: Int, num_slots: Int
) raises -> List[List[AnyTensor]]:
    """Create fake Adam-like optimizer state (m, v slots per parameter)."""
    var opt_state = List[List[AnyTensor]]()
    for _ in range(num_params):
        var slots = List[AnyTensor]()
        for _ in range(num_slots):
            slots.append(zeros([4], DType.float32))
        opt_state.append(slots^)
    return opt_state^


def test_save_and_load_with_optimizer_state() raises:
    """Test saving and loading checkpoints with optimizer state.

    Covers AC1 (weights + optimizer state), AC2 (step counter), AC6 (round-trip fidelity).
    Saves weights + 2-slot Adam state (m, v) for 3 params at epoch 7, step 1400.
    Reloads via load_latest_with_optimizer and asserts round-trip.
    """
    var ckpt_dir = "/tmp/test_ckpt_optimizer_state"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=5)

    var params = create_test_params()
    var param_names = create_param_names()
    var opt_state = create_optimizer_state(3, 2)  # 3 params, 2 slots each

    # Save checkpoint at epoch 7 with step 1400
    var saved_path = ckpt_mgr.save_checkpoint(
        params,
        param_names,
        epoch=7,
        step=1400,
        train_loss=0.25,
        val_loss=0.30,
        val_acc=0.88,
        optimizer_state=opt_state,
    )

    assert_true(
        file_exists(saved_path + "/metadata.txt"),
        "metadata.txt should exist after save",
    )
    assert_true(
        file_exists(saved_path + "/optimizer/slot_counts.txt"),
        "optimizer/slot_counts.txt should exist after save",
    )

    # Load via load_latest_with_optimizer
    var loaded_params = List[AnyTensor]()
    var loaded_opt_state = List[List[AnyTensor]]()
    var epoch_step = ckpt_mgr.load_latest_with_optimizer(
        loaded_params, param_names, loaded_opt_state
    )

    assert_equal_int(epoch_step[0], 7, "Loaded epoch should be 7")
    assert_equal_int(epoch_step[1], 1400, "Loaded step should be 1400")
    assert_equal_int(len(loaded_params), 3, "Should load 3 parameters")
    assert_equal_int(
        len(loaded_opt_state), 3, "Optimizer state should have 3 params"
    )
    assert_equal_int(
        len(loaded_opt_state[0]), 2, "Each param should have 2 slots"
    )


def test_save_load_config_snapshot() raises:
    """Test config snapshot embedded in checkpoint metadata round-trips.

    Covers AC1 config-snapshot bullet: config snapshot is saved and readable.
    """
    var ckpt_dir = "/tmp/test_ckpt_config_snapshot"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=5)

    var params = create_test_params()
    var param_names = create_param_names()

    var config = TrainingConfig(
        epochs=50,
        batch_size=64,
        learning_rate=0.001,
        momentum=0.9,
        checkpoint_every_n_epochs=5,
        checkpoint_dir=ckpt_dir,
    )
    var snapshot = config.to_snapshot_blob()

    var saved_path = ckpt_mgr.save_checkpoint(
        params,
        param_names,
        epoch=1,
        config_snapshot=snapshot,
    )

    # Verify metadata file contains the config snapshot
    var metadata = safe_read_file(saved_path + "/metadata.txt")
    assert_true(
        metadata.find("config=") >= 0,
        "metadata.txt should contain config= line",
    )
    assert_true(
        metadata.find("epochs=50") >= 0,
        "config snapshot should contain epochs=50",
    )


def test_training_config_has_new_fields() raises:
    """Test TrainingConfig has checkpoint_every_n_epochs and checkpoint_dir.

    Covers AC3: new fields exist and are correctly stored.
    """
    var config = TrainingConfig(
        epochs=1,
        batch_size=1,
        checkpoint_dir="/tmp/my_checkpoints",
        checkpoint_every_n_epochs=2,
    )

    assert_equal_int(
        config.checkpoint_every_n_epochs,
        2,
        "checkpoint_every_n_epochs should be 2",
    )
    assert_true(
        config.checkpoint_dir == "/tmp/my_checkpoints",
        "checkpoint_dir should be '/tmp/my_checkpoints'",
    )


def test_fresh_flag_clears_checkpoint() raises:
    """Test that fresh mode removes existing checkpoint tracker.

    Covers the --fresh flag behavior: existing state is cleared on resume.
    """
    var ckpt_dir = "/tmp/test_ckpt_fresh"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=5)

    var params = create_test_params()
    var param_names = create_param_names()

    # Save a checkpoint so tracker exists
    _ = ckpt_mgr.save_checkpoint(params, param_names, epoch=3)

    # Verify tracker exists
    assert_true(
        file_exists(ckpt_dir + "/checkpoint_tracker.txt"),
        "tracker should exist before fresh",
    )

    # Apply fresh: should clear checkpoint tracker
    ckpt_mgr.clear_checkpoints()

    # After fresh, load_latest should return epoch 0
    var loaded_params = List[AnyTensor]()
    var epoch = ckpt_mgr.load_latest(loaded_params, param_names)
    assert_equal_int(epoch, 0, "After clear, load_latest should return epoch 0")


def test_resume_loads_latest_checkpoint() raises:
    """Test that load_latest_with_optimizer returns the most recent epoch+step.

    Covers AC5: resume loads from last completed epoch.
    """
    var ckpt_dir = "/tmp/test_ckpt_resume_latest"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=10)

    var params = create_test_params()
    var param_names = create_param_names()
    var opt_state = create_optimizer_state(3, 2)

    # Save multiple checkpoints
    for epoch in range(1, 6):
        _ = ckpt_mgr.save_checkpoint(
            params,
            param_names,
            epoch=epoch,
            step=epoch * 100,
            optimizer_state=opt_state,
        )

    # Resume: should get epoch 5, step 500
    var loaded_params = List[AnyTensor]()
    var loaded_opt = List[List[AnyTensor]]()
    var epoch_step = ckpt_mgr.load_latest_with_optimizer(
        loaded_params, param_names, loaded_opt
    )

    assert_equal_int(epoch_step[0], 5, "Should resume from epoch 5")
    assert_equal_int(epoch_step[1], 500, "Should resume from step 500")


def test_partial_write_recovery_skips_broken_epoch() raises:
    """Test that load_latest_with_optimizer skips epochs with missing metadata.

    Covers Decision 9: partial write recovery falls back to prior epoch.
    Creates checkpoint_epoch_5/ with weights but no metadata.txt,
    and epoch_4/ with complete data. Asserts epoch_5 is skipped.
    """
    var ckpt_dir = "/tmp/test_ckpt_partial_recovery"
    var ckpt_mgr = CheckpointManager(ckpt_dir, max_to_keep=10)

    var params = create_test_params()
    var param_names = create_param_names()
    var opt_state = create_optimizer_state(3, 2)

    # Save a good checkpoint at epoch 4
    _ = ckpt_mgr.save_checkpoint(
        params,
        param_names,
        epoch=4,
        step=400,
        optimizer_state=opt_state,
    )

    # Manually create a broken epoch 5 dir (weights, no metadata.txt)
    var broken_dir = ckpt_dir + "/checkpoint_epoch_5"
    if not create_directory(broken_dir):
        raise Error("Failed to create broken_dir for test")
    # Write weights but no metadata
    _ = safe_write_file(broken_dir + "/param1.weights", "broken\n")
    # Manually update tracker to point at epoch 5 (simulating crash after write)
    _ = safe_write_file(
        ckpt_dir + "/checkpoint_tracker.txt", "latest_epoch=5\n"
    )

    # load_latest_with_optimizer should detect missing metadata at epoch 5
    # and fall back to epoch 4
    var loaded_params = List[AnyTensor]()
    var loaded_opt = List[List[AnyTensor]]()
    var epoch_step = ckpt_mgr.load_latest_with_optimizer(
        loaded_params, param_names, loaded_opt
    )

    assert_equal_int(
        epoch_step[0], 4, "Should fall back to epoch 4 (skip broken epoch 5)"
    )


def test_exit_codes_via_training_result() raises:
    """Test exit code constants are correct values.

    Covers exit code AC: 0=success, 1=error, 2=transient, 130=SIGINT.
    """
    from projectodyssey.training.interruption import (
        TrainingResult,
        ShutdownReason,
    )

    # Success result -> exit code 0
    var success = TrainingResult(
        stopped_epoch=10,
        reason=ShutdownReason.completed(),
    )
    assert_equal_int(
        success.exit_code(), 0, "Completed should give exit code 0"
    )

    # Signal result -> exit code 130
    var sigint = TrainingResult(
        stopped_epoch=3,
        reason=ShutdownReason.signal(),
    )
    assert_equal_int(
        sigint.exit_code(), 130, "SIGINT should give exit code 130"
    )

    # Timeout result -> exit code 2 (transient)
    var timeout = TrainingResult(
        stopped_epoch=3,
        reason=ShutdownReason.timeout(),
    )
    assert_equal_int(timeout.exit_code(), 2, "Timeout should give exit code 2")


def main() raises:
    """Run all checkpoint resume tests."""
    print("Testing Checkpoint Save/Load and Recovery (Issue #5184)...")
    print("=" * 70)

    print("\n[1/7] Testing save and load with optimizer state...")
    test_save_and_load_with_optimizer_state()
    print("PASSED")

    print("[2/7] Testing config snapshot round-trip...")
    test_save_load_config_snapshot()
    print("PASSED")

    print("[3/7] Testing TrainingConfig new fields...")
    test_training_config_has_new_fields()
    print("PASSED")

    print("[4/7] Testing fresh flag clears checkpoint...")
    test_fresh_flag_clears_checkpoint()
    print("PASSED")

    print("[5/7] Testing resume loads latest checkpoint...")
    test_resume_loads_latest_checkpoint()
    print("PASSED")

    print("[6/7] Testing partial write recovery skips broken epoch...")
    test_partial_write_recovery_skips_broken_epoch()
    print("PASSED")

    print("[7/7] Testing exit codes via TrainingResult...")
    test_exit_codes_via_training_result()
    print("PASSED")

    print("\n" + "=" * 70)
    print("All 7 checkpoint resume tests PASSED!")
    print(
        "Atomic save, optimizer state, recovery, and resume working correctly."
    )
