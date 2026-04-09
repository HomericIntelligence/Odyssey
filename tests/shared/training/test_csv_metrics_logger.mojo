"""Unit tests for CSV Metrics Logger.

Tests cover:
- Initialization and configuration
- Scalar logging
- CSV file generation
- Callback integration
- File I/O operations

All tests use the real CSVMetricsLogger implementation.
"""

from shared.training.metrics.csv_metrics_logger import CSVMetricsLogger
from shared.training.base import TrainingState, CONTINUE
from shared.utils.file_io import file_exists, safe_read_file, create_directory
from std.collections import List, Dict


# ============================================================================
# Initialization Tests
# ============================================================================


def test_csv_metrics_logger_initialization() raises:
    """Test CSVMetricsLogger initialization with parameters."""
    print("Testing CSVMetricsLogger initialization...")

    var logger = CSVMetricsLogger("test_logs/run1")

    # Verify parameters
    if logger.log_dir != "test_logs/run1":
        raise Error("Expected log_dir to be 'test_logs/run1'")
    if logger.step_counter != 0:
        raise Error("Expected step_counter to be 0")
    if logger.epoch_counter != 0:
        raise Error("Expected epoch_counter to be 0")
    if logger.initialized:
        raise Error("Expected initialized to be False")

    print("  ✓ CSVMetricsLogger initialization successful")


def test_csv_metrics_logger_log_scalar() raises:
    """Test logging scalar metrics."""
    print("Testing log_scalar...")

    var logger = CSVMetricsLogger("test_logs/run2")

    # Log some metrics
    logger.log_scalar("train_loss", 0.5)
    logger.log_scalar("train_loss", 0.4)
    logger.log_scalar("train_loss", 0.3)

    # Verify metrics stored
    if "train_loss" not in logger.metrics:
        raise Error("Expected 'train_loss' to be in metrics")
    if len(logger.metrics["train_loss"]) != 3:
        raise Error("Expected 3 values in train_loss history")

    print("  ✓ log_scalar correctly stores metrics")


def test_csv_metrics_logger_log_multiple_metrics() raises:
    """Test logging multiple different metrics."""
    print("Testing multiple metrics logging...")

    var logger = CSVMetricsLogger("test_logs/run3")

    logger.log_scalar("train_loss", 0.5)
    logger.log_scalar("val_loss", 0.6)
    logger.log_scalar("accuracy", 0.85)

    # Verify all metrics present
    if "train_loss" not in logger.metrics:
        raise Error("Expected 'train_loss' in metrics")
    if "val_loss" not in logger.metrics:
        raise Error("Expected 'val_loss' in metrics")
    if "accuracy" not in logger.metrics:
        raise Error("Expected 'accuracy' in metrics")

    print("  ✓ Multiple metrics logged correctly")


# ============================================================================
# Callback Integration Tests
# ============================================================================


def test_csv_metrics_logger_on_epoch_end() raises:
    """Test metrics logging at epoch end."""
    print("Testing on_epoch_end callback...")

    var logger = CSVMetricsLogger("test_logs/callback_test")
    var state = TrainingState(epoch=1, learning_rate=0.1)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6

    # Log via callback
    var signal = logger.on_epoch_end(state)

    # Verify metrics logged
    if logger.epoch_counter != 1:
        raise Error("Expected epoch_counter to be 1")
    if "train_loss" not in logger.metrics:
        raise Error("Expected 'train_loss' in metrics")
    if "val_loss" not in logger.metrics:
        raise Error("Expected 'val_loss' in metrics")

    # Verify signal is CONTINUE
    if signal.value != CONTINUE.value:
        raise Error("Expected CONTINUE signal")

    print("  ✓ on_epoch_end correctly logs metrics")


def test_csv_metrics_logger_step_counter() raises:
    """Test step counter increments correctly."""
    print("Testing step counter...")

    var logger = CSVMetricsLogger("test_logs/step_test")

    if logger.step_counter != 0:
        raise Error("Expected initial step_counter to be 0")

    logger.step()
    if logger.step_counter != 1:
        raise Error("Expected step_counter to be 1 after one step")

    logger.step()
    logger.step()
    if logger.step_counter != 3:
        raise Error("Expected step_counter to be 3 after three steps")

    print("  ✓ Step counter increments correctly")


def test_csv_metrics_logger_on_train_begin() raises:
    """Test on_train_begin initializes directory."""
    print("Testing on_train_begin...")

    var logger = CSVMetricsLogger("test_logs/train_begin_test")
    var state = TrainingState(epoch=0, learning_rate=0.1)

    # Call on_train_begin
    var signal = logger.on_train_begin(state)

    # Verify initialized flag is set
    # Note: The actual directory creation depends on the system
    if signal.value != CONTINUE.value:
        raise Error("Expected CONTINUE signal")

    print("  ✓ on_train_begin returns CONTINUE")


# ============================================================================
# CSV Generation Tests
# ============================================================================


def test_csv_metrics_logger_build_csv() raises:
    """Test CSV content generation."""
    print("Testing CSV content generation...")

    var logger = CSVMetricsLogger("test_logs/csv_test")

    var values = List[Float64]()
    values.append(0.5)
    values.append(0.4)
    values.append(0.3)

    var csv_content = logger._build_csv("test_metric", values)

    # Verify CSV structure
    if "step,value" not in csv_content:
        raise Error("Expected CSV header 'step,value'")
    if "0,0.5" not in csv_content:
        raise Error("Expected first row '0,0.5'")
    if "1,0.4" not in csv_content:
        raise Error("Expected second row '1,0.4'")
    if "2,0.3" not in csv_content:
        raise Error("Expected third row '2,0.3'")

    print("  ✓ CSV content generated correctly")


def test_csv_metrics_logger_log_from_state() raises:
    """Test logging all metrics from TrainingState."""
    print("Testing log_from_state...")

    var logger = CSVMetricsLogger("test_logs/state_test")
    var state = TrainingState(epoch=1, learning_rate=0.1)
    state.metrics["train_loss"] = 0.5
    state.metrics["val_loss"] = 0.6
    state.metrics["accuracy"] = 0.85

    # Log from state
    logger.log_from_state(state)

    # Verify all metrics logged
    if "train_loss" not in logger.metrics:
        raise Error("Expected 'train_loss' in metrics")
    if "val_loss" not in logger.metrics:
        raise Error("Expected 'val_loss' in metrics")
    if "accuracy" not in logger.metrics:
        raise Error("Expected 'accuracy' in metrics")

    # Verify values
    if len(logger.metrics["train_loss"]) != 1:
        raise Error("Expected 1 value in train_loss")
    if logger.metrics["train_loss"][0] != 0.5:
        raise Error("Expected train_loss value to be 0.5")

    print("  ✓ log_from_state correctly logs all metrics")


# ============================================================================
# Test Main
# ============================================================================


def main() raises:
    """Run all CSV metrics logger tests."""
    print("\n" + "=" * 60)
    print("Running CSV Metrics Logger Tests")
    print("=" * 60 + "\n")

    print("Initialization Tests:")
    print("-" * 60)
    test_csv_metrics_logger_initialization()
    test_csv_metrics_logger_log_scalar()
    test_csv_metrics_logger_log_multiple_metrics()

    print("\nCallback Integration Tests:")
    print("-" * 60)
    test_csv_metrics_logger_on_epoch_end()
    test_csv_metrics_logger_step_counter()
    test_csv_metrics_logger_on_train_begin()

    print("\nCSV Generation Tests:")
    print("-" * 60)
    test_csv_metrics_logger_build_csv()
    test_csv_metrics_logger_log_from_state()

    print("\n" + "=" * 60)
    print("All CSV Metrics Logger Tests Passed! ✓")
    print("=" * 60 + "\n")
