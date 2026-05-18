"""Unit tests for Logging Callback.

Tests cover:
- Training progress logging
- Log interval control
- Log count tracking
- Callback interface (begin)

All tests use the real LoggingCallback implementation.
"""


from tests.projectodyssey.conftest import (
    assert_true,
    assert_equal,
    assert_greater,
    TestFixtures,
)
from projectodyssey.training.callbacks import LoggingCallback
from projectodyssey.training.base import TrainingState


def test_logging_callback_initialization() raises:
    """Test LoggingCallback initialization with parameters."""
    var logger = LoggingCallback(log_interval=1)

    # Verify parameters
    assert_equal(logger.log_interval, 1)
    assert_equal(logger.log_count, 0)


def test_logging_callback_logs_at_epoch_end() raises:
    """Test LoggingCallback increments log count at epoch end."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=1, learning_rate=0.1)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6

    # Log epoch 1
    _ = logger.on_epoch_end(state)

    # Verify log count incremented
    assert_equal(logger.log_count, 1)


def test_logging_callback_log_interval() raises:
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


def test_logging_callback_log_every_epoch() raises:
    """Test LoggingCallback with log_interval=1 logs every epoch."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Should log every epoch
    for epoch in range(10):
        state.epoch = epoch
        _ = logger.on_epoch_end(state)

    # Should have logged 10 times
    assert_equal(logger.log_count, 10)


def test_logging_callback_log_every_5_epochs() raises:
    """Test LoggingCallback with log_interval=5."""
    var logger = LoggingCallback(log_interval=5)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Run for 25 epochs
    for epoch in range(25):
        state.epoch = epoch
        _ = logger.on_epoch_end(state)

    # Should have logged 5 times (epochs 0, 5, 10, 15, 20)
    assert_equal(logger.log_count, 5)


def test_logging_callback_get_log_count() raises:
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


def test_logging_callback_tracks_count_correctly() raises:
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


def test_logging_callback_on_train_begin() raises:
    """Test on_train_begin does not affect log count."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Call on_train_begin
    _ = logger.on_train_begin(state)

    # Log count should still be 0
    assert_equal(logger.log_count, 0)


def test_logging_callback_on_train_end() raises:
    """Test on_train_end does not affect log count."""
    var logger = LoggingCallback(log_interval=1)
    var state = TrainingState(epoch=10, learning_rate=0.1)

    # Set log count to some value
    for epoch in range(5):
        state.epoch = epoch
        _ = logger.on_epoch_end(state)

    var count_before = logger.log_count

    # Call on_train_end
    _ = logger.on_train_end(state)

    # Log count should be unchanged
    assert_equal(logger.log_count, count_before)


def test_logging_callback_zero_interval() raises:
    """Test LoggingCallback with log_interval=0 (undefined behavior)."""
    # Note: Division by zero may occur in modulo operation
    # This test just verifies the callback can be created
    var logger = LoggingCallback(log_interval=0)
    assert_equal(logger.log_interval, 0)


def test_logging_callback_large_interval() raises:
    """Test LoggingCallback with very large log_interval."""
    var logger = LoggingCallback(log_interval=1000)
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Run for 10 epochs
    for epoch in range(10):
        state.epoch = epoch
        _ = logger.on_epoch_end(state)

    # Should have logged only once (epoch 0)
    assert_equal(logger.log_count, 1)

    # Run until next log point
    state.epoch = 1000
    _ = logger.on_epoch_end(state)
    assert_equal(logger.log_count, 2)


def main() raises:
    """Run all test_logging_callback tests."""
    print("Running test_logging_callback tests...")

    test_logging_callback_initialization()
    print("✓ test_logging_callback_initialization")

    test_logging_callback_logs_at_epoch_end()
    print("✓ test_logging_callback_logs_at_epoch_end")

    test_logging_callback_log_interval()
    print("✓ test_logging_callback_log_interval")

    test_logging_callback_log_every_epoch()
    print("✓ test_logging_callback_log_every_epoch")

    test_logging_callback_log_every_5_epochs()
    print("✓ test_logging_callback_log_every_5_epochs")

    test_logging_callback_get_log_count()
    print("✓ test_logging_callback_get_log_count")

    test_logging_callback_tracks_count_correctly()
    print("✓ test_logging_callback_tracks_count_correctly")

    test_logging_callback_on_train_begin()
    print("✓ test_logging_callback_on_train_begin")

    test_logging_callback_on_train_end()
    print("✓ test_logging_callback_on_train_end")

    test_logging_callback_zero_interval()
    print("✓ test_logging_callback_zero_interval")

    test_logging_callback_large_interval()
    print("✓ test_logging_callback_large_interval")

    print("\nAll test_logging_callback tests passed!")
