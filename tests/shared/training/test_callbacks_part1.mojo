"""Tests for EarlyStopping callback.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_callbacks.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- EarlyStopping callback with min/max modes
- Metric tracking and state transitions
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


fn test_early_stopping_min_mode_improves() raises:
    """Test early stopping in min mode when loss improves."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=3, min_delta=0.001, mode="min"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)

    # Start training
    var signal = early_stop.on_train_begin(state)
    assert_equal(signal.value, 0, "Should return CONTINUE at train begin")

    # Epoch 0: val_loss = 1.0 (first value, should be improvement)
    state.metrics["val_loss"] = 1.0
    signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE on first epoch")
    assert_equal(
        early_stop.wait_count, 0, "Wait count should be 0 after improvement"
    )

    # Epoch 1: val_loss = 0.998 (improved by > 0.001)
    state.epoch = 1
    state.metrics["val_loss"] = 0.998
    signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE on improvement")
    assert_equal(early_stop.wait_count, 0, "Wait count should reset to 0")


fn test_early_stopping_min_mode_no_improvement() raises:
    """Test early stopping in min mode when loss doesn't improve enough."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, min_delta=0.001, mode="min"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # Epoch 0: val_loss = 1.0
    state.metrics["val_loss"] = 1.0
    early_stop.on_epoch_end(state)

    # Epoch 1: val_loss = 0.9999 (didn't improve by min_delta)
    state.epoch = 1
    state.metrics["val_loss"] = 0.9999
    var signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE")
    assert_equal(early_stop.wait_count, 1, "Wait count should be 1")

    # Epoch 2: val_loss = 0.9998 (still no improvement)
    state.epoch = 2
    state.metrics["val_loss"] = 0.9998
    signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 1, "Should return STOP after patience exhausted")
    assert_true(early_stop.stopped, "Stopped flag should be True")


fn test_early_stopping_max_mode_improves() raises:
    """Test early stopping in max mode (accuracy) when metric improves."""
    var early_stop = EarlyStopping(
        monitor="val_accuracy", patience=2, min_delta=0.001, mode="max"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # Epoch 0: accuracy = 0.8
    state.metrics["val_accuracy"] = 0.8
    var signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE")
    assert_equal(early_stop.wait_count, 0, "Wait count should be 0")

    # Epoch 1: accuracy = 0.805 (improved by > 0.001)
    state.epoch = 1
    state.metrics["val_accuracy"] = 0.805
    signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE")
    assert_equal(early_stop.wait_count, 0, "Wait count should reset")


fn test_early_stopping_max_mode_no_improvement() raises:
    """Test early stopping in max mode when accuracy doesn't improve."""
    var early_stop = EarlyStopping(
        monitor="val_accuracy", patience=1, min_delta=0.001, mode="max"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # Epoch 0: accuracy = 0.8
    state.metrics["val_accuracy"] = 0.8
    early_stop.on_epoch_end(state)

    # Epoch 1: accuracy = 0.7999 (degraded)
    state.epoch = 1
    state.metrics["val_accuracy"] = 0.7999
    var signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 1, "Should return STOP after patience exhausted")
    assert_true(early_stop.stopped, "Stopped flag should be True")


fn test_early_stopping_missing_metric() raises:
    """Test early stopping when monitored metric is missing."""
    var early_stop = EarlyStopping(
        monitor="val_loss", patience=2, min_delta=0.001, mode="min"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    # Epoch has no val_loss metric
    state.metrics["train_loss"] = 0.5
    var signal = early_stop.on_epoch_end(state)
    assert_equal(signal.value, 0, "Should return CONTINUE when metric missing")
    assert_equal(early_stop.wait_count, 0, "Wait count should not change")


fn test_early_stopping_should_stop_method() raises:
    """Test should_stop method."""
    var early_stop = EarlyStopping(monitor="val_loss", patience=1, mode="min")

    assert_true(
        not early_stop.should_stop(), "should_stop should be False initially"
    )

    var state = TrainingState(epoch=0, batch=0, learning_rate=0.01)
    early_stop.on_train_begin(state)

    state.metrics["val_loss"] = 1.0
    early_stop.on_epoch_end(state)

    state.epoch = 1
    state.metrics[
        "val_loss"
    ] = 1.1  # Worse than 1.0 in min mode - no improvement
    early_stop.on_epoch_end(state)

    assert_true(
        early_stop.should_stop(),
        "should_stop should be True after patience exhausted",
    )


fn main() raises:
    """Run EarlyStopping tests."""
    test_early_stopping_min_mode_improves()
    test_early_stopping_min_mode_no_improvement()
    test_early_stopping_max_mode_improves()
    test_early_stopping_max_mode_no_improvement()
    test_early_stopping_missing_metric()
    test_early_stopping_should_stop_method()

    print("All EarlyStopping tests passed!")
