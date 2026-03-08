"""Unit tests for Early Stopping Callback - Part 2.

Tests cover:
- Edge cases (zero patience, missing metric)
- State reset via on_train_begin
- Best value tracking
- Best epoch tracking
- Verbose/silent mode

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_early_stopping.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

All tests use the real EarlyStopping implementation.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)
from shared.training.callbacks import EarlyStopping
from shared.training.base import TrainingState


# ============================================================================
# Edge Cases
# ============================================================================


fn test_early_stopping_zero_patience() raises:
    """Test EarlyStopping with patience=0 stops after first non-improvement."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=0, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    # No improvement immediately triggers stop
    state.epoch = 2
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_true(early_stop.should_stop())


fn test_early_stopping_missing_monitored_metric() raises:
    """Test EarlyStopping handles missing monitored metric gracefully."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Metrics missing val_loss - should continue without error
    state.metrics["train_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    # Should not stop (metric doesn't exist)
    assert_false(early_stop.should_stop())


fn test_early_stopping_on_train_begin_resets() raises:
    """Test on_train_begin resets all state."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Simulate some training
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)

    # Reset with on_train_begin
    _ = early_stop.on_train_begin(state)

    # State should be reset
    assert_equal(early_stop.wait_count, 0)
    assert_false(early_stop.stopped)
    assert_almost_equal(
        early_stop.best_value, 1e9
    )  # Reset to initial for min mode


# ============================================================================
# Best Value Tracking Tests
# ============================================================================


fn test_early_stopping_tracks_best_value() raises:
    """Test EarlyStopping correctly tracks best value seen."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initial: 0.5
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_almost_equal(early_stop.best_value, 0.5)

    # Worse: 0.6 (best stays 0.5)
    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_almost_equal(early_stop.best_value, 0.5)

    # Better: 0.3 (best updates)
    state.epoch = 3
    state.metrics["val_loss"] = 0.3
    _ = early_stop.on_epoch_end(state)
    assert_almost_equal(early_stop.best_value, 0.3)


fn test_early_stopping_get_best_epoch() raises:
    """Test EarlyStopping get_best_epoch() returns correct epoch number."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Epoch 0: Initial best
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.get_best_epoch(), 0)

    # Epoch 1: No improvement (best stays at epoch 0)
    state.epoch = 1
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.get_best_epoch(), 0)

    # Epoch 2: Improvement (best updates to epoch 2)
    state.epoch = 2
    state.metrics["val_loss"] = 0.3
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.get_best_epoch(), 2)

    # Epoch 3: No improvement (best stays at epoch 2)
    state.epoch = 3
    state.metrics["val_loss"] = 0.4
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.get_best_epoch(), 2)


fn test_early_stopping_best_epoch_updates_on_improvement() raises:
    """Test EarlyStopping best_epoch updates when metric improves."""
    var early_stop = EarlyStopping(
        monitor="val_accuracy", patience=3, mode="max"
    )
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Epoch 0: Initial best
    state.metrics["val_accuracy"] = 0.6
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.best_epoch, 0)

    # Epoch 1: Improvement
    state.epoch = 1
    state.metrics["val_accuracy"] = 0.7
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.best_epoch, 1)

    # Epoch 2: Improvement again
    state.epoch = 2
    state.metrics["val_accuracy"] = 0.8
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.best_epoch, 2)

    # Epoch 3: No improvement (best_epoch stays at 2)
    state.epoch = 3
    state.metrics["val_accuracy"] = 0.75
    _ = early_stop.on_epoch_end(state)
    assert_equal(early_stop.best_epoch, 2)


fn test_early_stopping_verbose_mode() raises:
    """Test EarlyStopping with verbose=True (prints progress messages)."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, mode="min", verbose=True
    )
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Verbose mode should print messages (verified by manual inspection)
    # This test ensures verbose=True doesn't crash
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    state.epoch = 1
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)

    assert_false(early_stop.should_stop())


fn test_early_stopping_silent_mode() raises:
    """Test EarlyStopping with verbose=False (suppresses output)."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, mode="min", verbose=False
    )
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Silent mode should not print messages
    state.metrics["val_loss"] = 0.5
    _ = early_stop.on_epoch_end(state)

    state.epoch = 1
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)

    state.epoch = 2
    state.metrics["val_loss"] = 0.6
    _ = early_stop.on_epoch_end(state)

    # Should still stop correctly even in silent mode
    assert_true(early_stop.should_stop())


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run early stopping callback tests - part 2."""
    print("Running edge cases...")
    test_early_stopping_zero_patience()
    test_early_stopping_missing_monitored_metric()
    test_early_stopping_on_train_begin_resets()

    print("Running best value tracking tests...")
    test_early_stopping_tracks_best_value()
    test_early_stopping_get_best_epoch()
    test_early_stopping_best_epoch_updates_on_improvement()

    print("Running verbose mode tests...")
    test_early_stopping_verbose_mode()
    test_early_stopping_silent_mode()

    print("\nAll early stopping callback tests (part 2) passed! ✓")
