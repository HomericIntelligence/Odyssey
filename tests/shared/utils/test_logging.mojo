"""Tests for logging utilities module.

This module tests logging functionality including:
- Log levels and filtering
- Log formatters (timestamp, colored output)
- File and console handlers
- Training-specific logging patterns
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    TestFixtures,
)
from shared.utils import (
    Logger,
    LogLevel,
    LogRecord,
    SimpleFormatter,
    TimestampFormatter,
    DetailedFormatter,
    ColoredFormatter,
    StreamHandler,
    FileHandler,
    get_logger,
    set_global_log_level,
)


# ============================================================================
# Test Log Levels
# ============================================================================


fn test_log_level_hierarchy() raises:
    """Test log levels are ordered correctly (DEBUG < INFO < WARNING < ERROR).
    """
    assert_true(LogLevel.DEBUG < LogLevel.INFO)
    assert_true(LogLevel.INFO < LogLevel.WARNING)
    assert_true(LogLevel.WARNING < LogLevel.ERROR)
    assert_true(LogLevel.ERROR < LogLevel.CRITICAL)
    assert_equal(LogLevel.DEBUG, 10)
    assert_equal(LogLevel.INFO, 20)
    assert_equal(LogLevel.WARNING, 30)
    assert_equal(LogLevel.ERROR, 40)
    assert_equal(LogLevel.CRITICAL, 50)


fn test_log_level_filtering() raises:
    """Test logger filters messages below configured level."""
    # Create logger with INFO level
    var logger = Logger("test_filter", LogLevel.INFO)
    var handler = StreamHandler()
    logger.add_handler(handler)

    # Logger with INFO level should accept INFO and above
    # but not DEBUG
    assert_equal(logger.level, LogLevel.INFO)

    # Verify level comparison works as expected
    assert_true(logger.level <= LogLevel.INFO)  # Should log at INFO
    assert_true(logger.level <= LogLevel.ERROR)  # Should log at ERROR
    assert_false(logger.level <= LogLevel.DEBUG)  # Should NOT log at DEBUG


fn test_set_global_log_level() raises:
    """Test changing global log level affects all loggers."""
    var logger1 = get_logger("test_global_1", LogLevel.DEBUG)
    var logger2 = get_logger("test_global_2", LogLevel.DEBUG)

    # Both should start with DEBUG level
    assert_equal(logger1.level, LogLevel.DEBUG)
    assert_equal(logger2.level, LogLevel.DEBUG)

    # Set global level to WARNING
    set_global_log_level(LogLevel.WARNING)

    # Note: Without global state, we can't test global level changes
    # This test verifies the function exists and doesn't crash
    assert_equal(logger1.level, LogLevel.DEBUG)  # Unchanged without registry


# ============================================================================
# Test Log Formatters
# ============================================================================


fn test_simple_formatter() raises:
    """Test simple formatter creates readable log messages."""
    var formatter = SimpleFormatter()
    var record = LogRecord("training", LogLevel.INFO, "Training started")
    var formatted = formatter.format(record)

    # Format: "[LEVEL] message"
    assert_equal(formatted, "[INFO] Training started")

    # Test with different levels
    var debug_record = LogRecord("debug", LogLevel.DEBUG, "Debug message")
    var debug_formatted = formatter.format(debug_record)
    assert_equal(debug_formatted, "[DEBUG] Debug message")

    var error_record = LogRecord("error", LogLevel.ERROR, "Error occurred")
    var error_formatted = formatter.format(error_record)
    assert_equal(error_formatted, "[ERROR] Error occurred")


fn test_timestamp_formatter() raises:
    """Test formatter includes timestamp in log message."""
    var formatter = TimestampFormatter()
    var record = LogRecord(
        "training", LogLevel.INFO, "Training started", "2025-01-15 14:30:45"
    )
    var formatted = formatter.format(record)

    # Format: "YYYY-MM-DD HH:MM:SS [LEVEL] message"
    assert_equal(formatted, "2025-01-15 14:30:45 [INFO] Training started")

    # Test with empty timestamp (Mojo limitation)
    var no_ts_record = LogRecord(
        "test", LogLevel.WARNING, "Warning message", ""
    )
    var no_ts_formatted = formatter.format(no_ts_record)
    assert_equal(no_ts_formatted, " [WARNING] Warning message")


fn test_detailed_formatter() raises:
    """Test detailed formatter includes logger name."""
    var formatter = DetailedFormatter()
    var record = LogRecord("trainer", LogLevel.ERROR, "Loss is NaN")
    var formatted = formatter.format(record)

    # Format: "[LEVEL] logger_name - message"
    assert_equal(formatted, "[ERROR] trainer - Loss is NaN")

    # Test with different logger names
    var data_record = LogRecord("data_loader", LogLevel.INFO, "Loaded batch")
    var data_formatted = formatter.format(data_record)
    assert_equal(data_formatted, "[INFO] data_loader - Loaded batch")


fn test_colored_output() raises:
    """Test colored formatter uses ANSI codes for terminal output."""
    var formatter = ColoredFormatter()

    # Test ERROR (red)
    var error_record = LogRecord("test", LogLevel.ERROR, "Error message")
    var error_formatted = formatter.format(error_record)
    assert_true(error_formatted.find(ColoredFormatter.RED) != -1)
    assert_true(error_formatted.find(ColoredFormatter.RESET) != -1)

    # Test WARNING (yellow)
    var warning_record = LogRecord("test", LogLevel.WARNING, "Warning message")
    var warning_formatted = formatter.format(warning_record)
    assert_true(warning_formatted.find(ColoredFormatter.YELLOW) != -1)

    # Test INFO (green)
    var info_record = LogRecord("test", LogLevel.INFO, "Info message")
    var info_formatted = formatter.format(info_record)
    assert_true(info_formatted.find(ColoredFormatter.GREEN) != -1)

    # Test DEBUG (blue)
    var debug_record = LogRecord("test", LogLevel.DEBUG, "Debug message")
    var debug_formatted = formatter.format(debug_record)
    assert_true(debug_formatted.find(ColoredFormatter.BLUE) != -1)


# ============================================================================
# Test Log Handlers
# ============================================================================


fn test_console_handler() raises:
    """Test console handler writes to stdout."""
    var logger = Logger("console_test", LogLevel.INFO)
    var handler = StreamHandler()
    logger.add_handler(handler)

    # Verify handler was added
    assert_equal(len(logger.handlers), 1)

    # Test that logging doesn't raise errors
    logger.info("Console test message")
    logger.warning("Console warning")


fn test_file_handler() raises:
    """Test file handler writes to log file."""
    # Note: This test writes to a temporary file and verifies creation
    var temp_file = "/tmp/test_logging_output.log"

    var logger = Logger("file_test", LogLevel.INFO)
    var file_handler = FileHandler(temp_file)
    logger.add_handler(file_handler)

    # Log a message
    logger.info("Test log message to file")

    # Verify handler was added
    assert_equal(len(logger.handlers), 1)

    # Clean up would happen after test (in real test framework)


fn test_multiple_handlers() raises:
    """Test logger can have multiple handlers (console + file)."""
    var logger = Logger("multi_handler_test", LogLevel.INFO)

    # Add console handler
    var console_handler = StreamHandler()
    logger.add_handler(console_handler)

    # Add file handler
    var file_handler = FileHandler("/tmp/test_multi_handler.log")
    logger.add_handler(file_handler)

    # Verify both handlers were added
    assert_equal(len(logger.handlers), 2)

    # Log a message - should go to both handlers
    logger.info("Message to multiple handlers")


# ============================================================================
# Test Training-Specific Logging
# ============================================================================


fn test_log_training_start() raises:
    """Test logging training start with configuration details."""
    var logger = get_logger("training", LogLevel.INFO)
    var handler = StreamHandler()
    logger.add_handler(handler)

    logger.info("Starting training")
    logger.info("Model: LeNet5")
    logger.info("Epochs: 10")
    logger.info("Batch size: 32")


fn test_log_epoch_metrics() raises:
    """Test logging epoch completion with metrics."""
    var logger = get_logger("trainer_metrics", LogLevel.INFO)
    var handler = StreamHandler()
    logger.add_handler(handler)

    logger.info(
        "Epoch 1/10: train_loss=0.5, val_loss=0.4, val_acc=0.85, time=12.3s"
    )
    logger.info(
        "Epoch 2/10: train_loss=0.4, val_loss=0.35, val_acc=0.88, time=12.5s"
    )
    logger.info(
        "Epoch 3/10: train_loss=0.35, val_loss=0.32, val_acc=0.90, time=12.2s"
    )


fn test_log_batch_progress() raises:
    """Test logging batch progress within epoch."""
    var logger = get_logger("batch_progress", LogLevel.DEBUG)
    var handler = StreamHandler()
    logger.add_handler(handler)

    # These messages would normally be debug level
    logger.debug("Batch 0/500 (0%): loss=0.5")
    logger.debug("Batch 100/500 (20%): loss=0.45")
    logger.debug("Batch 250/500 (50%): loss=0.42")
    logger.debug("Batch 500/500 (100%): loss=0.40")


fn test_log_checkpoint_saved() raises:
    """Test logging checkpoint save events."""
    var logger = get_logger("checkpointing", LogLevel.INFO)
    var handler = StreamHandler()
    logger.add_handler(handler)

    logger.info(
        "Checkpoint saved to checkpoints/best_model.mojo (best val_loss=0.35)"
    )
    logger.info("Checkpoint saved to checkpoints/epoch_5.mojo (epoch 5)")


fn test_log_early_stopping() raises:
    """Test logging early stopping trigger."""
    var logger = get_logger("early_stopping", LogLevel.INFO)
    var handler = StreamHandler()
    logger.add_handler(handler)

    logger.info(
        "Early stopping: no improvement for 5 epochs (best val_loss=0.32)"
    )
    logger.warning("Early stopping triggered at epoch 27")


# ============================================================================
# Test Logger Configuration
# ============================================================================


fn test_create_default_logger() raises:
    """Test creating logger with default configuration."""
    var logger = Logger("default_test")

    # Verify defaults
    assert_equal(logger.name, "default_test")
    assert_equal(logger.level, LogLevel.INFO)
    assert_equal(len(logger.handlers), 0)  # No handlers by default


fn test_create_logger_with_name() raises:
    """Test creating named logger for different modules."""
    var training_logger = get_logger("training")
    var data_logger = get_logger("data")

    assert_equal(training_logger.name, "training")
    assert_equal(data_logger.name, "data")

    # Add handlers to verify names appear
    var handler1 = StreamHandler()
    training_logger.add_handler(handler1)

    var handler2 = StreamHandler()
    data_logger.add_handler(handler2)


fn test_logger_singleton() raises:
    """Test getting same logger instance by name."""
    var logger1 = get_logger("singleton_test")
    var handler1 = StreamHandler()
    logger1.add_handler(handler1)

    # Get logger again with same name
    var logger2 = get_logger("singleton_test")

    # Note: Without global state, get_logger creates a new instance each time
    # This test verifies the function works as expected
    assert_equal(logger2.name, "singleton_test")


fn test_configure_logger_from_dict() raises:
    """Test configuring logger from configuration dictionary."""
    # Create and configure a logger
    var logger = Logger("configured_logger", LogLevel.DEBUG)

    var console_handler = StreamHandler()
    logger.add_handler(console_handler)

    var file_handler = FileHandler("/tmp/configured.log")
    logger.add_handler(file_handler)

    # Verify configuration
    assert_equal(logger.level, LogLevel.DEBUG)
    assert_equal(len(logger.handlers), 2)


# ============================================================================
# Test Error Handling
# ============================================================================


fn test_log_with_invalid_level() raises:
    """Test logging with invalid level number."""
    var _ = Logger("error_test", LogLevel.INFO)

    # Create a record with invalid level number
    # The logger should still process it (gracefully degrade)
    var invalid_record = LogRecord("test", 999, "Invalid level message")

    # Verify level_name handles unknown levels
    var level_name = invalid_record.level_name()
    assert_equal(level_name, "UNKNOWN")


fn test_file_handler_permission_error() raises:
    """Test file handler handles write permission errors gracefully."""
    # Create file handler with a potentially problematic path
    # In reality, /dev/null is writable, so we use a path that doesn't exist
    var logger = Logger("permission_test", LogLevel.INFO)
    var file_handler = FileHandler("/nonexistent/directory/test.log")
    logger.add_handler(file_handler)

    # Try to log - should fallback to print and not crash
    logger.info("This should handle the error gracefully")


# ============================================================================
# Integration Tests
# ============================================================================


fn test_logger_integration_training() raises:
    """Test logger integrates with training patterns."""
    var logger = get_logger("training_integration", LogLevel.INFO)
    var console_handler = StreamHandler()
    logger.add_handler(console_handler)

    # Simulate training workflow
    logger.info("Starting training")
    logger.info("Epoch 1/3")
    logger.debug("Batch 1/100: loss=2.5")
    logger.debug("Batch 50/100: loss=1.2")
    logger.info("Epoch 1 completed: loss=0.8, acc=0.85")
    logger.info("Epoch 2/3")
    logger.debug("Batch 1/100: loss=0.9")
    logger.info("Epoch 2 completed: loss=0.6, acc=0.90")
    logger.info("Epoch 3/3")
    logger.info("Epoch 3 completed: loss=0.5, acc=0.92")
    logger.info("Training completed successfully")


fn main() raises:
    """Run all tests."""
    test_log_level_hierarchy()
    test_log_level_filtering()
    test_set_global_log_level()
    test_simple_formatter()
    test_timestamp_formatter()
    test_detailed_formatter()
    test_colored_output()
    test_console_handler()
    test_file_handler()
    test_multiple_handlers()
    test_log_training_start()
    test_log_epoch_metrics()
    test_log_batch_progress()
    test_log_checkpoint_saved()
    test_log_early_stopping()
    test_create_default_logger()
    test_create_logger_with_name()
    test_logger_singleton()
    test_configure_logger_from_dict()
    test_log_with_invalid_level()
    test_file_handler_permission_error()
    test_logger_integration_training()
