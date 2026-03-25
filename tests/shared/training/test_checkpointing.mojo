"""Unit tests for Checkpointing Callback

Tests cover:
- Saving model state during training
- Save frequency control
- Best model tracking with save_best_only
- Both minimize and maximize modes (save_best_only variants)
- Best value tracking

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_checkpointing.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

All tests use the real ModelCheckpoint implementation.
"""


from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    TestFixtures,
)
from shared.training.callbacks import ModelCheckpoint
from shared.training.base import TrainingState


fn test_checkpointing_initialization() raises:
    """Test ModelCheckpoint callback initialization with parameters."""
    var checkpoint = ModelCheckpoint(
        filepath="checkpoints/model.pt",
        monitor="val_loss",
        save_best_only=False,
        save_frequency=1,
        mode="min",
    )

    # Verify parameters
    assert_equal(checkpoint.filepath, "checkpoints/model.pt")
    assert_equal(checkpoint.monitor, "val_loss")
    assert_equal(checkpoint.save_frequency, 1)
    assert_equal(checkpoint.save_count, 0)
    assert_equal(checkpoint.mode, "min")


fn test_checkpointing_saves_at_epoch_end() raises:
    """Test ModelCheckpoint saves at the end of each epoch (by frequency)."""
    var checkpoint = ModelCheckpoint(filepath="/tmp/model.pt", save_frequency=1)
    var state = TrainingState(epoch=1, learning_rate=0.1)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6

    # Simulate epoch end
    _ = checkpoint.on_epoch_end(state)

    # Checkpoint should have incremented save count
    assert_equal(checkpoint.save_count, 1)


fn test_checkpointing_save_frequency() raises:
    """Test ModelCheckpoint respects save_frequency parameter."""
    var checkpoint = ModelCheckpoint(filepath="/tmp/model.pt", save_frequency=3)
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Epochs 1-2: Don't save
    state.epoch = 1
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 0)

    state.epoch = 2
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 0)

    # Epoch 3: Save (3 % 3 == 0)
    state.epoch = 3
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 1)

    # Epochs 4-5: Don't save
    state.epoch = 4
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 1)

    state.epoch = 5
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 1)

    # Epoch 6: Save (6 % 3 == 0)
    state.epoch = 6
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 2)


fn test_checkpointing_save_best_only_min_mode() raises:
    """Test ModelCheckpoint with save_best_only=True in min mode."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/best_model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Epoch 1: val_loss = 0.5 (initial best, save)
    state.metrics["val_loss"] = 0.5
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 1)
    assert_almost_equal(checkpoint.best_value, 0.5)

    # Epoch 2: val_loss = 0.3 (better, save)
    state.epoch = 2
    state.metrics["val_loss"] = 0.3
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 2)
    assert_almost_equal(checkpoint.best_value, 0.3)

    # Epoch 3: val_loss = 0.4 (worse, don't save)
    state.epoch = 3
    state.metrics["val_loss"] = 0.4
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 2)  # No change


fn test_checkpointing_save_best_only_max_mode() raises:
    """Test ModelCheckpoint with save_best_only=True in max mode."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/best_model.pt",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Epoch 1: val_accuracy = 0.6 (initial best, save)
    state.metrics["val_accuracy"] = 0.6
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 1)
    assert_almost_equal(checkpoint.best_value, 0.6)

    # Epoch 2: val_accuracy = 0.8 (better, save)
    state.epoch = 2
    state.metrics["val_accuracy"] = 0.8
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 2)
    assert_almost_equal(checkpoint.best_value, 0.8)

    # Epoch 3: val_accuracy = 0.7 (worse, don't save)
    state.epoch = 3
    state.metrics["val_accuracy"] = 0.7
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 2)  # No change


fn test_checkpointing_save_best_only_no_improvement() raises:
    """Test save_best_only doesn't save when no improvement."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/best_model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Epoch 1: val_loss = 0.5 (save)
    state.metrics["val_loss"] = 0.5
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.save_count, 1)

    # Epochs 2-5: No improvement
    for epoch in range(2, 6):
        state.epoch = epoch
        state.metrics["val_loss"] = 0.6  # Worse
        _ = checkpoint.on_epoch_end(state)

    # Should still be at 1 save
    assert_equal(checkpoint.save_count, 1)


fn test_checkpointing_tracks_best_value_min() raises:
    """Test ModelCheckpoint correctly tracks best value in min mode."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial best value should be very large for min mode
    assert_almost_equal(checkpoint.best_value, 1e9)

    # Update with first value
    state.metrics["val_loss"] = 0.5
    _ = checkpoint.on_epoch_end(state)
    assert_almost_equal(checkpoint.best_value, 0.5)

    # Better value
    state.epoch = 2
    state.metrics["val_loss"] = 0.3
    _ = checkpoint.on_epoch_end(state)
    assert_almost_equal(checkpoint.best_value, 0.3)

    # Worse value (best stays 0.3)
    state.epoch = 3
    state.metrics["val_loss"] = 0.6
    _ = checkpoint.on_epoch_end(state)
    assert_almost_equal(checkpoint.best_value, 0.3)


fn test_checkpointing_tracks_best_value_max() raises:
    """Test ModelCheckpoint correctly tracks best value in max mode."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/model.pt",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial best value should be very small for max mode
    assert_almost_equal(checkpoint.best_value, -1e9)

    # Update with first value
    state.metrics["val_accuracy"] = 0.6
    _ = checkpoint.on_epoch_end(state)
    assert_almost_equal(checkpoint.best_value, 0.6)

    # Better value
    state.epoch = 2
    state.metrics["val_accuracy"] = 0.8
    _ = checkpoint.on_epoch_end(state)
    assert_almost_equal(checkpoint.best_value, 0.8)

    # Worse value (best stays 0.8)
    state.epoch = 3
    state.metrics["val_accuracy"] = 0.7
    _ = checkpoint.on_epoch_end(state)
    assert_almost_equal(checkpoint.best_value, 0.8)


fn test_checkpointing_mode_min() raises:
    """Test ModelCheckpoint with mode='min' for loss minimization."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Decreasing loss should save
    state.metrics["val_loss"] = 1.0
    _ = checkpoint.on_epoch_end(state)
    var count1 = checkpoint.save_count

    state.epoch = 2
    state.metrics["val_loss"] = 0.5
    _ = checkpoint.on_epoch_end(state)
    var count2 = checkpoint.save_count

    assert_greater(count2, count1)  # Should have saved


fn test_checkpointing_mode_max() raises:
    """Test ModelCheckpoint with mode='max' for accuracy maximization."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/model.pt",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Increasing accuracy should save
    state.metrics["val_accuracy"] = 0.5
    _ = checkpoint.on_epoch_end(state)
    var count1 = checkpoint.save_count

    state.epoch = 2
    state.metrics["val_accuracy"] = 0.8
    _ = checkpoint.on_epoch_end(state)
    var count2 = checkpoint.save_count

    assert_greater(count2, count1)  # Should have saved


fn test_checkpointing_missing_monitored_metric() raises:
    """Test ModelCheckpoint handles missing monitored metric gracefully."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Metric not in state - should not save
    state.metrics["train_loss"] = 0.5  # Different metric
    _ = checkpoint.on_epoch_end(state)

    # Should not have saved (no val_loss)
    assert_equal(checkpoint.save_count, 0)


fn test_checkpointing_error_count_tracking() raises:
    """Test ModelCheckpoint tracks error count."""
    var checkpoint = ModelCheckpoint(filepath="/tmp/model.pt")

    # Error count should start at 0
    assert_equal(checkpoint.error_count, 0)

    # Note: Actual file I/O is stubbed, so error_count won't change
    # This test just verifies the attribute exists and is accessible


fn test_checkpointing_get_save_count() raises:
    """Test get_save_count returns correct count."""
    var checkpoint = ModelCheckpoint(filepath="/tmp/model.pt", save_frequency=1)
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initially 0
    assert_equal(checkpoint.get_save_count(), 0)

    # After one save
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1)

    # After two saves
    state.epoch = 2
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 2)


fn main() raises:
    """Run all test_checkpointing tests."""
    print("Running test_checkpointing tests...")

    test_checkpointing_initialization()
    print("✓ test_checkpointing_initialization")

    test_checkpointing_saves_at_epoch_end()
    print("✓ test_checkpointing_saves_at_epoch_end")

    test_checkpointing_save_frequency()
    print("✓ test_checkpointing_save_frequency")

    test_checkpointing_save_best_only_min_mode()
    print("✓ test_checkpointing_save_best_only_min_mode")

    test_checkpointing_save_best_only_max_mode()
    print("✓ test_checkpointing_save_best_only_max_mode")

    test_checkpointing_save_best_only_no_improvement()
    print("✓ test_checkpointing_save_best_only_no_improvement")

    test_checkpointing_tracks_best_value_min()
    print("✓ test_checkpointing_tracks_best_value_min")

    test_checkpointing_tracks_best_value_max()
    print("✓ test_checkpointing_tracks_best_value_max")

    test_checkpointing_mode_min()
    print("✓ test_checkpointing_mode_min")

    test_checkpointing_mode_max()
    print("✓ test_checkpointing_mode_max")

    test_checkpointing_missing_monitored_metric()
    print("✓ test_checkpointing_missing_monitored_metric")

    test_checkpointing_error_count_tracking()
    print("✓ test_checkpointing_error_count_tracking")

    test_checkpointing_get_save_count()
    print("✓ test_checkpointing_get_save_count")

    print("\nAll test_checkpointing tests passed!")
