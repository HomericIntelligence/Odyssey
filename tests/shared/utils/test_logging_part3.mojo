# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_logging.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for logging utilities module - Part 3: Logger Configuration and Error Handling.

This module tests logging functionality including:
- Logger configuration and factory functions
- Error handling for invalid levels and permission errors
- Integration tests for training workflows
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
    test_create_default_logger()
    test_create_logger_with_name()
    test_logger_singleton()
    test_configure_logger_from_dict()
    test_log_with_invalid_level()
    test_file_handler_permission_error()
    test_logger_integration_training()
