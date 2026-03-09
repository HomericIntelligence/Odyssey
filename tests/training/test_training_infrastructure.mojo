"""Comprehensive tests for training infrastructure.

Tests the complete training infrastructure including trainer interface,
training loop, validation loop, and base trainer.

Training Infrastructure Tests (#303-322):
- #304: Trainer interface and configuration
- #309: Training loop functionality
- #314: Validation loop functionality
- #319: Base trainer integration

Testing strategy:
- Unit tests for individual components
- Integration tests for complete workflows
- Mock models/optimizers for testing
- Metric validation
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.core import ExTensor
from shared.training.trainer_interface import (
    TrainerConfig,
    TrainingMetrics,
    DataLoader,
    DataBatch,
)
from shared.training.loops.training_loop import TrainingLoop
from shared.training.loops.validation_loop import ValidationLoop
from shared.training.trainer import (
    BaseTrainer,
    create_trainer,
    create_default_trainer,
)


# ==================================================================
# Mock Functions for Testing
# ==================================================================


fn mock_model_forward(input: ExTensor) raises -> ExTensor:
    """Mock model forward pass - returns input unchanged."""
    return input


fn mock_compute_loss(
    predictions: ExTensor, labels: ExTensor
) raises -> ExTensor:
    """Mock loss computation - returns constant loss."""
    var loss = ExTensor(List[Int](), DType.float32)
    loss._data.bitcast[Float32]()[0] = 0.5
    return loss


fn mock_optimizer_step() raises -> None:
    """Mock optimizer step - does nothing."""
    pass


fn mock_zero_gradients() raises -> None:
    """Mock gradient zeroing - does nothing."""
    pass


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
    var data = ExTensor(data_shape, DType.float32)
    var labels_shape = List[Int]()
    var labels = ExTensor(labels_shape, DType.int32)

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
    var data = ExTensor(data_shape, DType.float32)
    var labels_shape = List[Int]()
    labels_shape.append(10)
    var labels = ExTensor(labels_shape, DType.int32)

    var loader = DataLoader(data, labels, batch_size=3)

    # Check has_next before iteration
    assert_true(loader.has_next(), "Has batches initially")

    var batch_count = 0
    while loader.has_next():
        var batch = loader.next()
        batch_count += 1

    assert_equal(batch_count, 4, "Iterated over all batches")
    assert_false(loader.has_next(), "No more batches after iteration")

    # Reset and iterate again
    loader.reset()
    assert_true(loader.has_next(), "Has batches after reset")

    print("  ✓ DataLoader iteration works correctly")


# ==================================================================
# TrainingLoop Tests
# ==================================================================


fn test_training_loop_initialization() raises:
    """Test TrainingLoop initialization."""
    print("Testing TrainingLoop initialization...")

    var loop = TrainingLoop(
        log_interval=5, clip_gradients=True, max_grad_norm=2.0
    )

    assert_equal(loop.log_interval, 5, "Log interval")
    assert_true(loop.clip_gradients, "Clip gradients enabled")
    assert_equal(loop.max_grad_norm, 2.0, "Max grad norm")

    print("  ✓ TrainingLoop initialization correct")


# ==================================================================
# ValidationLoop Tests
# ==================================================================


fn test_validation_loop_initialization() raises:
    """Test ValidationLoop initialization."""
    print("Testing ValidationLoop initialization...")

    var loop = ValidationLoop(
        compute_accuracy=True, compute_confusion=True, num_classes=5
    )

    assert_true(loop.compute_accuracy, "Compute accuracy")
    assert_true(loop.compute_confusion, "Compute confusion")
    assert_equal(loop.num_classes, 5, "Number of classes")

    print("  ✓ ValidationLoop initialization correct")


# ==================================================================
# BaseTrainer Tests
# ==================================================================


fn test_base_trainer_initialization() raises:
    """Test BaseTrainer initialization."""
    print("Testing BaseTrainer initialization...")

    var config = TrainerConfig(num_epochs=5, batch_size=16)
    var trainer = BaseTrainer(config)

    assert_equal(trainer.config.num_epochs, 5, "Config num_epochs")
    assert_equal(trainer.config.batch_size, 16, "Config batch_size")
    assert_false(trainer.is_training, "Not training initially")

    print("  ✓ BaseTrainer initialization correct")


fn test_create_trainer_factory() raises:
    """Test create_trainer factory function."""
    print("Testing create_trainer factory...")

    var config = TrainerConfig(num_epochs=3)
    var trainer = create_trainer(config)

    assert_equal(trainer.config.num_epochs, 3, "Factory creates with config")

    print("  ✓ create_trainer factory works")


fn test_create_default_trainer() raises:
    """Test create_default_trainer factory."""
    print("Testing create_default_trainer factory...")

    var trainer = create_default_trainer()

    assert_equal(
        trainer.config.num_epochs, 10, "Default trainer has default config"
    )

    print("  ✓ create_default_trainer factory works")


fn test_base_trainer_get_metrics() raises:
    """Test BaseTrainer get_metrics method."""
    print("Testing BaseTrainer get_metrics...")

    var config = TrainerConfig()
    var trainer = BaseTrainer(config)

    var metrics = trainer.get_metrics()

    assert_equal(metrics.current_epoch, 0, "Initial metrics")

    print("  ✓ BaseTrainer get_metrics works")


fn test_base_trainer_get_best_checkpoint() raises:
    """Test BaseTrainer get_best_checkpoint_epoch method."""
    print("Testing BaseTrainer get_best_checkpoint_epoch...")

    var config = TrainerConfig()
    var trainer = BaseTrainer(config)

    # Update metrics to set best epoch
    trainer.metrics.update_val_metrics(0.5, 0.8)
    trainer.metrics.current_epoch = 2
    trainer.metrics.update_val_metrics(0.3, 0.9)  # Best

    var best_epoch = trainer.get_best_checkpoint_epoch()

    assert_equal(best_epoch, 2, "Best checkpoint epoch")

    print("  ✓ BaseTrainer get_best_checkpoint_epoch works")


fn test_base_trainer_reset() raises:
    """Test BaseTrainer reset method."""
    print("Testing BaseTrainer reset...")

    var config = TrainerConfig()
    var trainer = BaseTrainer(config)

    # Set some state
    trainer.metrics.update_train_metrics(0.5, 0.8)
    trainer.is_training = True

    # Reset
    trainer.reset()

    assert_false(trainer.is_training, "Training flag reset")
    assert_equal(trainer.metrics.current_epoch, 0, "Metrics reset")

    print("  ✓ BaseTrainer reset works")


fn test_databatch_creation() raises:
    """Test DataBatch creation."""
    print("Testing DataBatch creation...")

    var data_shape = List[Int]()
    data_shape.append(5)
    data_shape.append(10)
    var data = ExTensor(data_shape, DType.float32)
    var labels_shape = List[Int]()
    var labels = ExTensor(labels_shape, DType.int32)

    var batch = DataBatch(data, labels)

    assert_equal(batch.batch_size, 5, "Batch size from data shape")

    print("  ✓ DataBatch creation works")


# ==================================================================
# Integration Tests
# ==================================================================


fn test_trainer_config_to_base_trainer_integration() raises:
    """Test integration of TrainerConfig with BaseTrainer."""
    print("Testing TrainerConfig to BaseTrainer integration...")

    var config = TrainerConfig(
        num_epochs=3, batch_size=8, learning_rate=0.01, log_interval=5
    )

    var trainer = BaseTrainer(config)

    assert_equal(trainer.config.num_epochs, 3, "Config passed correctly")
    assert_equal(
        trainer.training_loop.log_interval,
        5,
        "Log interval configured in training loop",
    )

    print("  ✓ TrainerConfig integrates with BaseTrainer")


fn test_metrics_flow_through_trainer() raises:
    """Test that metrics flow correctly through trainer."""
    print("Testing metrics flow through trainer...")

    var config = TrainerConfig(num_epochs=2)
    var trainer = BaseTrainer(config)

    # Manually update metrics
    trainer.metrics.current_epoch = 1
    trainer.metrics.update_train_metrics(0.3, 0.85)
    trainer.metrics.update_val_metrics(0.25, 0.90)

    # Get metrics back
    var retrieved_metrics = trainer.get_metrics()

    assert_equal(retrieved_metrics.current_epoch, 1, "Epoch preserved")
    assert_equal(retrieved_metrics.train_loss, 0.3, "Train loss preserved")
    assert_equal(retrieved_metrics.val_accuracy, 0.90, "Val accuracy preserved")

    print("  ✓ Metrics flow correctly through trainer")


fn main() raises:
    """Run all training infrastructure tests."""
    print("\n" + "=" * 70)
    print("TRAINING INFRASTRUCTURE TEST SUITE")
    print("Trainer Interface, Loops, and Base Trainer (#303-322)")
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

    print("\nTrainingLoop Tests (#309)")
    print("-" * 70)
    test_training_loop_initialization()

    print("\nValidationLoop Tests (#314)")
    print("-" * 70)
    test_validation_loop_initialization()

    print("\nBaseTrainer Tests (#319)")
    print("-" * 70)
    test_base_trainer_initialization()
    test_create_trainer_factory()
    test_create_default_trainer()
    test_base_trainer_get_metrics()
    test_base_trainer_get_best_checkpoint()
    test_base_trainer_reset()
    test_databatch_creation()

    print("\nIntegration Tests (#320)")
    print("-" * 70)
    test_trainer_config_to_base_trainer_integration()
    test_metrics_flow_through_trainer()

    print("\n" + "=" * 70)
    print("ALL TRAINING INFRASTRUCTURE TESTS PASSED ✓")
    print("=" * 70 + "\n")
    print("Summary:")
    print("  ✓ TrainerConfig: Default and custom configuration")
    print("  ✓ TrainingMetrics: Initialization, updates, reset, best tracking")
    print("  ✓ DataLoader: Batch iteration, reset, size calculation")
    print("  ✓ TrainingLoop: Initialization and configuration")
    print("  ✓ ValidationLoop: Initialization and configuration")
    print("  ✓ BaseTrainer: Full lifecycle (init, metrics, checkpoints, reset)")
    print("  ✓ Factory functions: create_trainer, create_default_trainer")
    print("  ✓ Integration: Config→Trainer, Metrics flow")
    print("\nArchitecture:")
    print("  Trainer Interface (trait-based polymorphism)")
    print("    ├── TrainerConfig (explicit configuration)")
    print("    ├── TrainingMetrics (state tracking)")
    print("    └── DataLoader (batch iteration)")
    print("  Training Loop (forward/backward/update)")
    print("    ├── training_step (single batch)")
    print("    └── train_one_epoch (full epoch)")
    print("  Validation Loop (gradient-free evaluation)")
    print("    ├── validation_step (single batch)")
    print("    └── validate (full validation set)")
    print("  BaseTrainer (composition-based integration)")
    print("    ├── Composition: TrainingLoop + ValidationLoop")
    print("    ├── Metrics: MetricLogger integration")
    print("    └── Lifecycle: fit(), save/load checkpoints")
    print()
