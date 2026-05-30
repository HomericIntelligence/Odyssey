"""Unit tests for training interruption and timeout functionality.

Tests cover:
- ShutdownReason enum variants and equality
- WallClockTimer for measuring elapsed time
- Global shutdown flag state management
- TrainingResult struct initialization and formatting
"""

from projectodyssey.training.interruption import (
    ShutdownReason,
    WallClockTimer,
    TrainingResult,
    request_shutdown,
    is_shutdown_requested,
    reset_shutdown_flag,
)


def test_shutdown_reason_completed() raises:
    """Test ShutdownReason.COMPLETED variant."""
    var reason = ShutdownReason.completed()
    assert_equal(reason == ShutdownReason.completed(), True)
    assert_equal(reason == ShutdownReason.timeout(), False)


def test_shutdown_reason_timeout() raises:
    """Test ShutdownReason.TIMEOUT variant."""
    var reason = ShutdownReason.timeout()
    assert_equal(reason == ShutdownReason.timeout(), True)
    assert_equal(reason == ShutdownReason.completed(), False)


def test_shutdown_reason_signal() raises:
    """Test ShutdownReason.SIGNAL variant."""
    var reason = ShutdownReason.signal()
    assert_equal(reason == ShutdownReason.signal(), True)
    assert_equal(reason == ShutdownReason.timeout(), False)


def test_shutdown_reason_max_epochs() raises:
    """Test ShutdownReason.MAX_EPOCHS variant."""
    var reason = ShutdownReason.max_epochs()
    assert_equal(reason == ShutdownReason.max_epochs(), True)
    assert_equal(reason == ShutdownReason.completed(), False)


def test_shutdown_reason_to_string() raises:
    """Test ShutdownReason.to_string() for all variants."""
    assert_equal(ShutdownReason.completed().to_string(), "COMPLETED")
    assert_equal(ShutdownReason.timeout().to_string(), "TIMEOUT")
    assert_equal(ShutdownReason.signal().to_string(), "SIGNAL")
    assert_equal(ShutdownReason.max_epochs().to_string(), "MAX_EPOCHS")


def test_wall_clock_timer_creation() raises:
    """Test WallClockTimer initialization."""
    var timer = WallClockTimer()
    assert_true(timer.start_ns > 0)


def test_wall_clock_timer_elapsed_seconds_positive() raises:
    """Test that elapsed_seconds() returns a positive value."""
    var timer = WallClockTimer()
    var elapsed = timer.elapsed_seconds()
    assert_true(elapsed >= 0.0)


def test_wall_clock_timer_has_elapsed_no_timeout() raises:
    """Test has_elapsed() returns False when limit_seconds is 0."""
    var timer = WallClockTimer()
    assert_false(timer.has_elapsed(0))


def test_wall_clock_timer_has_elapsed_large_timeout() raises:
    """Test has_elapsed() returns False for large timeout."""
    var timer = WallClockTimer()
    assert_false(timer.has_elapsed(3600))


def test_wall_clock_timer_has_elapsed_negative() raises:
    """Test has_elapsed() returns False for negative limit."""
    var timer = WallClockTimer()
    assert_false(timer.has_elapsed(-1))


def test_shutdown_flag_default_false() raises:
    """Test that shutdown flag starts as False."""
    reset_shutdown_flag()
    assert_false(is_shutdown_requested())


def test_shutdown_flag_set() raises:
    """Test setting the shutdown flag."""
    reset_shutdown_flag()
    request_shutdown()
    assert_true(is_shutdown_requested())


def test_shutdown_flag_reset() raises:
    """Test resetting the shutdown flag."""
    reset_shutdown_flag()
    request_shutdown()
    assert_true(is_shutdown_requested())
    reset_shutdown_flag()
    assert_false(is_shutdown_requested())


def test_training_result_default_construction() raises:
    """Test TrainingResult construction with defaults."""
    var result = TrainingResult(
        stopped_epoch=5,
        reason=ShutdownReason.completed(),
    )
    assert_equal(result.stopped_epoch, 5)
    assert_equal(result.reason == ShutdownReason.completed(), True)
    assert_equal(result.checkpoint_path, "")
    assert_equal(result.elapsed_seconds, 0.0)


def test_training_result_full_construction() raises:
    """Test TrainingResult construction with all fields."""
    var result = TrainingResult(
        stopped_epoch=10,
        reason=ShutdownReason.timeout(),
        checkpoint_path="/tmp/checkpoint.pt",
        elapsed_seconds=120.5,
    )
    assert_equal(result.stopped_epoch, 10)
    assert_equal(result.reason == ShutdownReason.timeout(), True)
    assert_equal(result.checkpoint_path, "/tmp/checkpoint.pt")
    assert_equal(result.elapsed_seconds, 120.5)


def test_training_result_to_string() raises:
    """Test TrainingResult.to_string() formatting."""
    var result = TrainingResult(
        stopped_epoch=5,
        reason=ShutdownReason.signal(),
        checkpoint_path="/tmp/ckpt.pt",
        elapsed_seconds=60.0,
    )
    var result_str = result.to_string()
    assert_true("stopped_epoch: 5" in result_str)
    assert_true("SIGNAL" in result_str)
    assert_true("/tmp/ckpt.pt" in result_str)


def test_wall_clock_timer_monotonic() raises:
    """Test that elapsed_seconds increases with time."""
    var timer = WallClockTimer()
    var elapsed1 = timer.elapsed_seconds()
    var elapsed2 = timer.elapsed_seconds()
    assert_true(elapsed2 >= elapsed1)


def test_shutdown_flag_idempotent_set() raises:
    """Test that multiple request_shutdown() calls are safe."""
    reset_shutdown_flag()
    request_shutdown()
    request_shutdown()
    request_shutdown()
    assert_true(is_shutdown_requested())


def main() raises:
    test_shutdown_reason_completed()
    test_shutdown_reason_timeout()
    test_shutdown_reason_signal()
    test_shutdown_reason_max_epochs()
    test_shutdown_reason_to_string()
    test_wall_clock_timer_creation()
    test_wall_clock_timer_elapsed_seconds_positive()
    test_wall_clock_timer_has_elapsed_no_timeout()
    test_wall_clock_timer_has_elapsed_large_timeout()
    test_wall_clock_timer_has_elapsed_negative()
    test_shutdown_flag_default_false()
    test_shutdown_flag_set()
    test_shutdown_flag_reset()
    test_training_result_default_construction()
    test_training_result_full_construction()
    test_training_result_to_string()
    test_wall_clock_timer_monotonic()
    test_shutdown_flag_idempotent_set()
    print("All interruption tests passed!")


def assert_equal[T: Comparable](lhs: T, rhs: T) raises:
    """Assert that two values are equal."""
    if lhs != rhs:
        raise Error("Assertion failed: values are not equal")


def assert_true(condition: Bool) raises:
    """Assert that condition is True."""
    if not condition:
        raise Error("Assertion failed: expected True")


def assert_false(condition: Bool) raises:
    """Assert that condition is False."""
    if condition:
        raise Error("Assertion failed: expected False")
