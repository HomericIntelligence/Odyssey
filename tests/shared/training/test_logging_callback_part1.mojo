# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_logging_callback.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for Logging Callback (Part 1).

Tests cover:
- Training progress logging
- Log interval control
- Log count tracking
- Callback interface (begin)

All tests use the real LoggingCallback implementation.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_greater,
    TestFixtures,
)
from shared.training.callbacks import LoggingCallback
from shared.training.base import TrainingState


# ============================================================================
# Logging Callback Core Tests
# ============================================================================


fn test_logging_callback_initialization() raises:
    """Test LoggingCallback initialization with parameters."""
    var logger = LoggingCallback(log_interval=1)

    # Verify parameters
    assert_equal(logger.log_interval, 1)
    assert_equal(logger.log_count, 0)


fn test_logging_callback_logs_at_epoch_end() raises:
    """Test LoggingCallback increments log count at epoch end."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=1, learning_rate=0.1)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6

    # Log epoch 1
    _ = logger.on_epoch_end(state)

    # Verify log count incremented
    assert_equal(logger.log_count, 1)


fn test_logging_callback_log_interval() raises:
    """Test LoggingCallback respects log_interval parameter."""
    var logger = LoggingCallback(log_interval=2)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Epoch 0: Should log (0 % 2 == 0)
    state.epoch = 0
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 1)

    # Epoch 1: Should not log (1 % 2 == 1)
    state.epoch = 1
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 1)  # Still 1

    # Epoch 2: Should log (2 % 2 == 0)
    state.epoch = 2
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 2)

    # Epoch 3: Should not log (3 % 2 == 1)
    state.epoch = 3
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 2)  # Still 2

    # Epoch 4: Should log (4 % 2 == 0)
    state.epoch = 4
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 3)


fn test_logging_callback_log_every_epoch() raises:
    """Test LoggingCallback with log_interval=1 logs every epoch."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Should log every epoch
    for epoch in range(10):
        state.epoch = epoch
        _ = logger.on_epoch_end(state)

    # Should have logged 10 times
    assert_equal(logger.log_count, 10)


fn test_logging_callback_log_every_5_epochs() raises:
    """Test LoggingCallback with log_interval=5."""
    var logger = LoggingCallback(log_interval=5)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Run for 25 epochs
    for epoch in range(25):
        state.epoch = epoch
        _ = logger.on_epoch_end(state)

    # Should have logged 5 times (epochs 0, 5, 10, 15, 20)
    assert_equal(logger.log_count, 5)


# ============================================================================
# Log Count Tracking Tests
# ============================================================================


fn test_logging_callback_get_log_count() raises:
    """Test get_log_count returns correct count."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initially 0
    assert_equal(logger.get_log_count(), 0)

    # After one log
    state.epoch = 1
    _ = logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 1)

    # After two logs
    state.epoch = 2
    _ = logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 2)

    # After three logs
    state.epoch = 3
    _ = logger.on_epoch_end(state)
    assert_equal(logger.get_log_count(), 3)


fn test_logging_callback_tracks_count_correctly() raises:
    """Test log count increments only when logging occurs."""
    var logger = LoggingCallback(log_interval=3)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Epochs 0-2
    state.epoch = 0
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 1)  # Logged

    state.epoch = 1
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 1)  # Not logged

    state.epoch = 2
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 1)  # Not logged

    # Epoch 3
    state.epoch = 3
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 2)  # Logged


# ============================================================================
# Callback Interface Tests
# ============================================================================


fn test_logging_callback_on_train_begin() raises:
    """Test on_train_begin does not affect log count."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Call on_train_begin
    _ = logger.on_train_begin(state)

    # Log count should still be 0
    assert_equal(logger.log_count, 0)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run logging callback tests part 1."""
    print("Running logging callback core tests...")
    test_logging_callback_initialization()
    test_logging_callback_logs_at_epoch_end()
    test_logging_callback_log_interval()
    test_logging_callback_log_every_epoch()
    test_logging_callback_log_every_5_epochs()

    print("Running log count tracking tests...")
    test_logging_callback_get_log_count()
    test_logging_callback_tracks_count_correctly()

    print("Running callback interface tests...")
    test_logging_callback_on_train_begin()

    print("\nAll logging callback part 1 tests passed! ✓")
