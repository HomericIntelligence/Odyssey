"""Integration tests for timeout and graceful shutdown in training.

Tests cover:
- TrainingConfig timeout fields and has_timeout() method
- Timeout detection in training loops
- Graceful exit with proper TrainingResult
"""

from projectodyssey.training.config import TrainingConfig
from projectodyssey.training.interruption import (
    ShutdownReason,
    WallClockTimer,
    TrainingResult,
)


def test_training_config_default_no_timeout() raises:
    """Test that TrainingConfig has no timeout by default."""
    var config = TrainingConfig(
        epochs=10,
        batch_size=32,
    )
    assert_false(config.has_timeout())
    assert_equal(config.max_wall_time_seconds, 0)


def test_training_config_with_timeout() raises:
    """Test TrainingConfig with timeout configured."""
    var config = TrainingConfig(
        epochs=10,
        batch_size=32,
        max_wall_time_seconds=3600,
    )
    assert_true(config.has_timeout())
    assert_equal(config.max_wall_time_seconds, 3600)


def test_training_config_zero_timeout_means_no_timeout() raises:
    """Test that max_wall_time_seconds=0 means no timeout."""
    var config = TrainingConfig(
        epochs=10,
        batch_size=32,
        max_wall_time_seconds=0,
    )
    assert_false(config.has_timeout())


def test_training_config_checkpoint_on_interrupt_default() raises:
    """Test that checkpoint_on_interrupt defaults to True."""
    var config = TrainingConfig(
        epochs=10,
        batch_size=32,
    )
    assert_true(config.checkpoint_on_interrupt)


def test_training_config_checkpoint_on_interrupt_false() raises:
    """Test TrainingConfig with checkpoint_on_interrupt=False."""
    var config = TrainingConfig(
        epochs=10,
        batch_size=32,
        checkpoint_on_interrupt=False,
    )
    assert_false(config.checkpoint_on_interrupt)


def test_training_config_to_string_includes_timeout() raises:
    """Test that to_string() includes timeout fields."""
    var config = TrainingConfig(
        epochs=10,
        batch_size=32,
        max_wall_time_seconds=3600,
        checkpoint_on_interrupt=True,
    )
    var config_str = config.to_string()
    assert_true("max_wall_time_seconds: 3600" in config_str)
    assert_true("checkpoint_on_interrupt: True" in config_str)


def test_training_config_for_lenet5_no_timeout() raises:
    """Test that factory method for_lenet5() has no timeout by default."""
    var config = TrainingConfig.for_lenet5()
    assert_false(config.has_timeout())
    assert_equal(config.max_wall_time_seconds, 0)


def test_training_config_for_cifar10_no_timeout() raises:
    """Test that factory method for_cifar10() has no timeout by default."""
    var config = TrainingConfig.for_cifar10()
    assert_false(config.has_timeout())
    assert_equal(config.max_wall_time_seconds, 0)


def test_training_result_timeout_reason() raises:
    """Test TrainingResult with TIMEOUT reason."""
    var result = TrainingResult(
        stopped_epoch=5,
        reason=ShutdownReason.timeout(),
        elapsed_seconds=120.0,
    )
    assert_equal(result.reason == ShutdownReason.timeout(), True)
    assert_equal(result.stopped_epoch, 5)


def test_training_result_signal_reason() raises:
    """Test TrainingResult with SIGNAL reason."""
    var result = TrainingResult(
        stopped_epoch=3,
        reason=ShutdownReason.signal(),
        elapsed_seconds=60.0,
    )
    assert_equal(result.reason == ShutdownReason.signal(), True)


def test_training_result_completed_reason() raises:
    """Test TrainingResult with COMPLETED reason."""
    var result = TrainingResult(
        stopped_epoch=9,
        reason=ShutdownReason.completed(),
        elapsed_seconds=500.0,
    )
    assert_equal(result.reason == ShutdownReason.completed(), True)


def test_wall_clock_timer_elapsed_integration() raises:
    """Test WallClockTimer elapsed calculation."""
    var timer = WallClockTimer()
    var elapsed = timer.elapsed_seconds()
    assert_true(elapsed >= 0.0)


def test_config_with_different_timeout_values() raises:
    """Test various timeout values in TrainingConfig."""
    # Test 1 second timeout
    var config1 = TrainingConfig(
        epochs=10,
        batch_size=32,
        max_wall_time_seconds=1,
    )
    assert_true(config1.has_timeout())

    # Test large timeout
    var config2 = TrainingConfig(
        epochs=10,
        batch_size=32,
        max_wall_time_seconds=86400,
    )
    assert_true(config2.has_timeout())


def main() raises:
    test_training_config_default_no_timeout()
    test_training_config_with_timeout()
    test_training_config_zero_timeout_means_no_timeout()
    test_training_config_checkpoint_on_interrupt_default()
    test_training_config_checkpoint_on_interrupt_false()
    test_training_config_to_string_includes_timeout()
    test_training_config_for_lenet5_no_timeout()
    test_training_config_for_cifar10_no_timeout()
    test_training_result_timeout_reason()
    test_training_result_signal_reason()
    test_training_result_completed_reason()
    test_wall_clock_timer_elapsed_integration()
    test_config_with_different_timeout_values()
    print("All timeout integration tests passed!")


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
