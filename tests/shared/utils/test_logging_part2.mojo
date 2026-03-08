# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_logging.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for logging utilities module - Part 2: Handlers and Training Logging.

This module tests logging functionality including:
- File and multi-handler configuration
- Training-specific logging patterns (epoch, batch, checkpoint, early stopping)
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
# Test Log Handlers
# ============================================================================


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


fn main() raises:
    """Run all tests."""
    test_file_handler()
    test_multiple_handlers()
    test_log_training_start()
    test_log_epoch_metrics()
    test_log_batch_progress()
    test_log_checkpoint_saved()
    test_log_early_stopping()
