"""Training infrastructure tests - Part 1: TrainerConfig, TrainingMetrics, DataLoader.

Split from test_training_infrastructure.mojo to comply with ADR-009 heap corruption
workaround (≤10 fn test_ functions per file).

Tests covered:
- TrainerConfig defaults and custom values
- TrainingMetrics initialization, update, reset
- DataLoader basic functionality and iteration

Training Infrastructure Tests (#303-322):
- #304: Trainer interface and configuration

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_training_infrastructure.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.core.any_tensor import AnyTensor
from shared.training.trainer_interface import (
    TrainerConfig,
    TrainingMetrics,
    DataLoader,
    DataBatch,
)


# ==================================================================
# TrainerConfig Tests
# ==================================================================


fn test_trainer_config_defaults() raises:
    """Test TrainerConfig default values."""
    print("Testing TrainerConfig defaults...")

    var config = TrainerConfig()

    assert_equal(config.num_epochs, 10, "Default num_epochs")
    assert_equal(config.batch_size, 32, "Default batch_size")
    assert_equal(config.learning_rate, 0.001, "Default learning_rate")
    assert_equal(config.log_interval, 10, "Default log_interval")
    assert_equal(config.validate_interval, 1, "Default validate_interval")
    assert_false(config.save_checkpoints, "Default save_checkpoints")
    assert_equal(config.checkpoint_interval, 5, "Default checkpoint_interval")

    print("  ✓ TrainerConfig defaults are correct")


fn test_trainer_config_custom() raises:
    """Test TrainerConfig custom values."""
    print("Testing TrainerConfig custom values...")

    var config = TrainerConfig(
        num_epochs=20,
        batch_size=64,
        learning_rate=0.01,
        log_interval=5,
        validate_interval=2,
        save_checkpoints=True,
        checkpoint_interval=10,
    )

    assert_equal(config.num_epochs, 20, "Custom num_epochs")
    assert_equal(config.batch_size, 64, "Custom batch_size")
    assert_equal(config.learning_rate, 0.01, "Custom learning_rate")
    assert_equal(config.log_interval, 5, "Custom log_interval")
    assert_equal(config.validate_interval, 2, "Custom validate_interval")
    assert_true(config.save_checkpoints, "Custom save_checkpoints")
    assert_equal(config.checkpoint_interval, 10, "Custom checkpoint_interval")

    print("  ✓ TrainerConfig custom values work correctly")


# ==================================================================
# TrainingMetrics Tests
# ==================================================================


fn test_training_metrics_initialization() raises:
    """Test TrainingMetrics initialization."""
    print("Testing TrainingMetrics initialization...")

    var metrics = TrainingMetrics()

    assert_equal(metrics.current_epoch, 0, "Initial epoch")
    assert_equal(metrics.current_batch, 0, "Initial batch")
    assert_equal(metrics.train_loss, 0.0, "Initial train_loss")
    assert_equal(metrics.train_accuracy, 0.0, "Initial train_accuracy")
    assert_equal(metrics.val_loss, 0.0, "Initial val_loss")
    assert_equal(metrics.val_accuracy, 0.0, "Initial val_accuracy")
    assert_equal(metrics.best_epoch, 0, "Initial best_epoch")

    print("  ✓ TrainingMetrics initialization correct")


fn test_training_metrics_update() raises:
    """Test TrainingMetrics update methods."""
    print("Testing TrainingMetrics update...")

    var metrics = TrainingMetrics()

    # Update train metrics
    metrics.update_train_metrics(0.5, 0.8)
    assert_equal(metrics.train_loss, 0.5, "Train loss updated")
    assert_equal(metrics.train_accuracy, 0.8, "Train accuracy updated")

    # Update val metrics
    metrics.update_val_metrics(0.3, 0.9)
    assert_equal(metrics.val_loss, 0.3, "Val loss updated")
    assert_equal(metrics.val_accuracy, 0.9, "Val accuracy updated")
    assert_equal(metrics.best_val_loss, 0.3, "Best val loss updated")
    assert_equal(metrics.best_val_accuracy, 0.9, "Best val accuracy updated")

    # Update with worse metrics - best should not change
    metrics.update_val_metrics(0.5, 0.7)
    assert_equal(metrics.val_loss, 0.5, "Val loss updated to new value")
    assert_equal(metrics.best_val_loss, 0.3, "Best val loss unchanged")

    print("  ✓ TrainingMetrics update methods work correctly")


fn test_training_metrics_reset() raises:
    """Test TrainingMetrics reset method."""
    print("Testing TrainingMetrics reset...")

    var metrics = TrainingMetrics()

    # Set some values
    metrics.update_train_metrics(0.5, 0.8)
    metrics.current_batch = 10

    # Reset epoch
    metrics.reset_epoch()

    assert_equal(metrics.current_batch, 0, "Batch reset")
    assert_equal(metrics.train_loss, 0.0, "Train loss reset")
    assert_equal(metrics.train_accuracy, 0.0, "Train accuracy reset")

    print("  ✓ TrainingMetrics reset works correctly")


# ==================================================================
# DataLoader Tests
# ==================================================================


fn test_dataloader_basic() raises:
    """Test DataLoader basic functionality."""
    print("Testing DataLoader basic...")

    var data_shape = List[Int]()
    data_shape.append(10)
    data_shape.append(5)
    var data = AnyTensor(data_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = AnyTensor(labels_shape, DType.int32)

    var loader = DataLoader(data, labels, batch_size=3)

    assert_equal(loader.num_samples, 10, "Number of samples")
    assert_equal(loader.num_batches, 4, "Number of batches (ceil(10/3))")
    assert_equal(loader.batch_size, 3, "Batch size")

    print("  ✓ DataLoader basic functionality works")


fn test_dataloader_iteration() raises:
    """Test DataLoader iteration."""
    print("Testing DataLoader iteration...")

    var data_shape = List[Int]()
    data_shape.append(10)
    data_shape.append(5)
    var data = AnyTensor(data_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = AnyTensor(labels_shape, DType.int32)

    var loader = DataLoader(data, labels, batch_size=3)

    # Check has_next before iteration
    assert_true(loader.has_next(), "Has batches initially")

    var batch_count = 0
    while loader.has_next():
        _ = loader.next()
        batch_count += 1

    assert_equal(batch_count, 4, "Iterated over all batches")
    assert_false(loader.has_next(), "No more batches after iteration")

    # Reset and iterate again
    loader.reset()
    assert_true(loader.has_next(), "Has batches after reset")

    print("  ✓ DataLoader iteration works correctly")


fn main() raises:
    """Run Part 1 training infrastructure tests."""
    print("\n" + "=" * 70)
    print("TRAINING INFRASTRUCTURE TEST SUITE - PART 1")
    print("TrainerConfig, TrainingMetrics, DataLoader (#303-322)")
    print("=" * 70 + "\n")

    print("TrainerConfig Tests (#304)")
    print("-" * 70)
    test_trainer_config_defaults()
    test_trainer_config_custom()

    print("\nTrainingMetrics Tests (#304)")
    print("-" * 70)
    test_training_metrics_initialization()
    test_training_metrics_update()
    test_training_metrics_reset()

    print("\nDataLoader Tests (#304)")
    print("-" * 70)
    test_dataloader_basic()
    test_dataloader_iteration()

    print("\n" + "=" * 70)
    print("ALL PART 1 TRAINING INFRASTRUCTURE TESTS PASSED ✓")
    print("=" * 70 + "\n")
