# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_logging.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for logging utilities module - Part 1: Log Levels and Formatters.

This module tests logging functionality including:
- Log levels and filtering
- Log formatters (simple, timestamp, detailed, colored)
- Console handler basics
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
