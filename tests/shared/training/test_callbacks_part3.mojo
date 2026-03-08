"""Tests for LoggingCallback and integration of multiple callbacks.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_callbacks.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- LoggingCallback with metric logging
- Callback signal handling
- Integration of multiple callbacks together
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


fn test_logging_callback_logs_at_interval() raises:
    """Test logging callback logs at specified interval."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6

    # Epoch 0: Should log (0 % 1 == 0)
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should have logged at epoch 0")

    # Epoch 1: Should log
    state.epoch = 1
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 2, "Should have logged at epoch 1")


fn test_logging_callback_skips_non_intervals() raises:
    """Test logging callback skips non-interval epochs."""
    var logger = LoggingCallback(log_interval=2)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["train_loss"] = 0.5

    # Epoch 0: Should log (0 % 2 == 0)
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should have logged at epoch 0")

    # Epoch 1: Should not log
    state.epoch = 1
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should not log at epoch 1")

    # Epoch 2: Should log (2 % 2 == 0)
    state.epoch = 2
    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 2, "Should have logged at epoch 2")


fn test_logging_callback_with_metrics() raises:
    """Test logging callback handles multiple metrics."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6
    state.metrics["accuracy"] = 0.95

    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should have logged metrics")


fn test_logging_callback_with_learning_rate() raises:
    """Test logging callback includes learning rate."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.001)
    state.metrics["train_loss"] = 0.5

    logger.on_epoch_end(state)
    assert_equal(
        logger.get_log_count(), 1, "Should have logged with learning rate"
    )


fn test_logging_callback_empty_metrics() raises:
    """Test logging callback with no metrics."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    # No metrics added

    logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1, "Should log even with no metrics")


fn test_logging_callback_returns_continue() raises:
    """Test logging callback always returns CONTINUE."""
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    var signal = logger.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should always return CONTINUE")


fn test_multiple_callbacks_together() raises:
    """Test multiple callbacks working together."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    var checkpoint = ModelCheckpoint(
        filepath="checkpoint.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var logger = LoggingCallback(log_interval=1)

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    state.metrics["val_loss"] = 1.0

    # All callbacks should work together
    early_stop.on_train_begin(state)
    early_stop.on_epoch_end(state)
    checkpoint.on_epoch_end(state)
    logger.on_epoch_end(state)

    assert_equal(early_stop.wait_count, 0, "Early stop initialized")
    assert_equal(checkpoint.get_save_count(), 1, "Checkpoint saved")
    assert_equal(logger.get_log_count(), 1, "Logger logged")


fn test_early_stopping_sets_should_stop_flag() raises:
    """Test that early stopping sets the should_stop flag."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=1, mode="min")

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # First epoch
    state.metrics["val_loss"] = 1.0
    early_stop.on_epoch_end(state)
    assert_true(not state.should_stop, "should_stop should be False initially")

    # Second epoch - no improvement, patience exhausted
    state.epoch = 1
    state.metrics["val_loss"] = 1.1  # Worse than 1.0 in min mode
    early_stop.on_epoch_end(state)
    assert_true(
        state.should_stop, "should_stop should be set by early stopping"
    )


fn main() raises:
    """Run LoggingCallback and integration tests."""
    test_logging_callback_logs_at_interval()
    test_logging_callback_skips_non_intervals()
    test_logging_callback_with_metrics()
    test_logging_callback_with_learning_rate()
    test_logging_callback_empty_metrics()
    test_logging_callback_returns_continue()
    test_multiple_callbacks_together()
    test_early_stopping_sets_should_stop_flag()

    print("All LoggingCallback and integration tests passed!")
