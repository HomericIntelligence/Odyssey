"""Tests for ModelCheckpoint callback.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_callbacks.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- ModelCheckpoint callback with frequency and best-only modes
- Error tracking and signal handling
"""

from testing import assert_true, assert_equal
from collections import Dict
from shared.training.base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
)
from shared.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LoggingCallback,
)


fn test_model_checkpoint_save_frequency() raises:
    """Test checkpoint saving at specified frequency."""
    var checkpoint = ModelCheckpoint(
        filepath="checkpoint_{epoch}.pt", save_frequency=2
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Epoch 0: No save (0 % 2 == 0 but save_best_only is off)
    state.epoch = 0
    var signal = checkpoint.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE")
    assert_equal(checkpoint.get_save_count(), 1, "Should have saved at epoch 0")

    # Epoch 1: No save
    state.epoch = 1
    signal = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1, "Should not save at epoch 1")

    # Epoch 2: Save
    state.epoch = 2
    signal = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 2, "Should have saved at epoch 2")


fn test_model_checkpoint_save_best_only_min_mode() raises:
    """Test checkpoint saving only when best metric improves (min mode)."""
    var checkpoint = ModelCheckpoint(
        filepath="best_model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Epoch 0: val_loss = 1.0 (first value, should save)
    state.epoch = 0
    state.metrics["val_loss"] = 1.0
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1, "Should save on first epoch")

    # Epoch 1: val_loss = 0.9 (improved, should save)
    state.epoch = 1
    state.metrics["val_loss"] = 0.9
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 2, "Should save on improvement")

    # Epoch 2: val_loss = 0.95 (degraded, should not save)
    state.epoch = 2
    state.metrics["val_loss"] = 0.95
    checkpoint.on_epoch_end(state)
    assert_equal(
        checkpoint.get_save_count(), 2, "Should not save on degradation"
    )


fn test_model_checkpoint_save_best_only_max_mode() raises:
    """Test checkpoint saving only when best metric improves (max mode)."""
    var checkpoint = ModelCheckpoint(
        filepath="best_model.pt",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Epoch 0: accuracy = 0.8 (first value, should save)
    state.epoch = 0
    state.metrics["val_accuracy"] = 0.8
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1, "Should save on first epoch")

    # Epoch 1: accuracy = 0.85 (improved, should save)
    state.epoch = 1
    state.metrics["val_accuracy"] = 0.85
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 2, "Should save on improvement")

    # Epoch 2: accuracy = 0.83 (degraded, should not save)
    state.epoch = 2
    state.metrics["val_accuracy"] = 0.83
    checkpoint.on_epoch_end(state)
    assert_equal(
        checkpoint.get_save_count(), 2, "Should not save on degradation"
    )


fn test_model_checkpoint_epoch_placeholder() raises:
    """Test checkpoint path with {epoch} placeholder."""
    var checkpoint = ModelCheckpoint(
        filepath="checkpoints/model_epoch_{epoch}.pt", save_frequency=1
    )

    var state = TrainingState(epoch=5, batch=0, learning_rate=0.01)

    # Should replace {epoch} with 5
    checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1, "Should have saved checkpoint")


fn test_model_checkpoint_error_count() raises:
    """Test error tracking in checkpoint saving."""
    var checkpoint = ModelCheckpoint(filepath="checkpoint.pt", save_frequency=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Initially no errors
    assert_equal(
        checkpoint.get_error_count(), 0, "Error count should start at 0"
    )

    # Save a checkpoint (no error)
    checkpoint.on_epoch_end(state)
    assert_equal(
        checkpoint.get_error_count(), 0, "No errors from successful save"
    )


fn test_model_checkpoint_always_returns_continue() raises:
    """Test that checkpoint always returns CONTINUE signal."""
    var checkpoint = ModelCheckpoint(filepath="checkpoint.pt", save_frequency=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["val_loss"] = 1.0

    var signal = checkpoint.on_epoch_end(state)
    assert_equal(
        signal.value, 0, "Should always return CONTINUE to not stop training"
    )


fn main() raises:
    """Run ModelCheckpoint tests."""
    test_model_checkpoint_save_frequency()
    test_model_checkpoint_save_best_only_min_mode()
    test_model_checkpoint_save_best_only_max_mode()
    test_model_checkpoint_epoch_placeholder()
    test_model_checkpoint_error_count()
    test_model_checkpoint_always_returns_continue()

    print("All ModelCheckpoint tests passed!")
