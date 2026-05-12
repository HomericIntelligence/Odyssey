"""Training infrastructure tests.

workaround (≤10 fn test_ functions per file).

Tests covered:
- TrainerConfig defaults and custom values
- TrainingMetrics initialization, update, reset
- DataLoader basic functionality and iteration

Training Infrastructure Tests (#303-322):
- #304: Trainer interface and configuration

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""


from std.testing import (
    assert_true,
    assert_false,
    assert_equal,
    assert_almost_equal,
)
from shared.tensor.any_tensor import AnyTensor
from shared.training.trainer_interface import (
    DataBatch,
    DataLoader,
    TrainerConfig,
    TrainingMetrics,
)
from shared.training.loops.training_loop import TrainingLoop
from shared.training.loops.validation_loop import ValidationLoop
from shared.training.trainer import (
    BaseTrainer,
    create_default_trainer,
    create_trainer,
)

def mock_model_forward(input: AnyTensor) raises -> AnyTensor:
    """Mock model forward pass - returns input unchanged."""
    return input


def mock_compute_loss(
    predictions: AnyTensor, labels: AnyTensor
) raises -> AnyTensor:
    """Mock loss computation - returns constant loss."""
    var loss = AnyTensor(List[Int](), DType.float32)
    loss.set(0, Float32(0.5))
    return loss


def mock_optimizer_step() raises -> None:
    """Mock optimizer step - does nothing."""
    pass


def mock_zero_gradients() raises -> None:
    """Mock gradient zeroing - does nothing."""
    pass


def test_trainer_config_defaults() raises:
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


def test_trainer_config_custom() raises:
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


def test_training_metrics_initialization() raises:
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


def test_training_metrics_update() raises:
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


def test_training_metrics_reset() raises:
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


def test_dataloader_basic() raises:
    """Test DataLoader basic functionality."""
    print("Testing DataLoader basic...")

    var data_shape = List[Int]()
    data_shape.append(10)
    data_shape.append(5)
    var data = AnyTensor(data_shape, DType.float32)
    var labels_shape = List[Int]()
    var labels = AnyTensor(labels_shape, DType.int32)

    var loader = DataLoader(data, labels, batch_size=3)

    assert_equal(loader.num_samples, 10, "Number of samples")
    assert_equal(loader.num_batches, 4, "Number of batches (ceil(10/3))")
    assert_equal(loader.batch_size, 3, "Batch size")

    print("  ✓ DataLoader basic functionality works")


def test_dataloader_iteration() raises:
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
        var batch = loader.next()
        batch_count += 1

    assert_equal(batch_count, 4, "Iterated over all batches")
    assert_false(loader.has_next(), "No more batches after iteration")

    # Reset and iterate again
    loader.reset()
    assert_true(loader.has_next(), "Has batches after reset")

    print("  ✓ DataLoader iteration works correctly")


def test_training_loop_initialization() raises:
    """Test TrainingLoop initialization."""
    print("Testing TrainingLoop initialization...")

    var loop = TrainingLoop(
        log_interval=5, clip_gradients=True, max_grad_norm=2.0
    )

    assert_equal(loop.log_interval, 5, "Log interval")
    assert_true(loop.clip_gradients, "Clip gradients enabled")
    assert_equal(loop.max_grad_norm, 2.0, "Max grad norm")

    print("  ✓ TrainingLoop initialization correct")


def test_validation_loop_initialization() raises:
    """Test ValidationLoop initialization."""
    print("Testing ValidationLoop initialization...")

    var loop = ValidationLoop(
        compute_accuracy=True, compute_confusion=True, num_classes=5
    )

    assert_true(loop.compute_accuracy, "Compute accuracy")
    assert_true(loop.compute_confusion, "Compute confusion")
    assert_equal(loop.num_classes, 5, "Number of classes")

    print("  ✓ ValidationLoop initialization correct")


def test_validation_loop_init_defaults() raises:
    """Test ValidationLoop default initialization."""
    print("Testing ValidationLoop default initialization...")

    var loop = ValidationLoop()

    assert_true(loop.compute_accuracy, "compute_accuracy default is True")
    assert_false(loop.compute_confusion, "compute_confusion default is False")
    assert_equal(loop.num_classes, 10, "num_classes default is 10")

    print("  ✓ ValidationLoop default initialization correct")


def test_validation_loop_run_updates_val_accuracy() raises:
    """Test that ValidationLoop.run() updates metrics.val_accuracy when compute_accuracy=True.

    Uses deterministic data where all labels are class 0 and the mock model
    returns input unchanged (zeros → argmax selects class 0), so accuracy = 1.0.
    """
    print("Testing ValidationLoop.run() updates val_accuracy...")

    # Create data: 4 samples, 3 features (2D so argmax selects predicted class)
    var data_shape = List[Int]()
    data_shape.append(4)
    data_shape.append(3)
    var data = AnyTensor(data_shape, DType.float32)

    # Labels: shape [4], dtype int32, all zeros (class 0)
    var labels_shape = List[Int]()
    labels_shape.append(4)
    var labels = AnyTensor(labels_shape, DType.int32)

    var val_loader = DataLoader(data, labels, batch_size=4)
    var validation_loop = ValidationLoop(compute_accuracy=True)
    var metrics = TrainingMetrics()

    assert_equal(metrics.val_accuracy, 0.0, "val_accuracy starts at 0.0")

    _ = validation_loop.run(
        mock_model_forward, mock_compute_loss, val_loader, metrics
    )

    # mock_model_forward returns input (zeros, shape [4,3]) → argmax gives class 0
    # labels are all 0 → accuracy = 1.0
    assert_true(
        metrics.val_accuracy > 0.0,
        "val_accuracy updated to non-zero after run()",
    )

    print("  ✓ ValidationLoop.run() updates val_accuracy correctly")


def test_base_trainer_initialization() raises:
    """Test BaseTrainer initialization."""
    print("Testing BaseTrainer initialization...")

    var config = TrainerConfig(num_epochs=5, batch_size=16)
    var trainer = BaseTrainer(config)

    assert_equal(trainer.config.num_epochs, 5, "Config num_epochs")
    assert_equal(trainer.config.batch_size, 16, "Config batch_size")
    assert_false(trainer.is_training, "Not training initially")

    print("  ✓ BaseTrainer initialization correct")


def test_create_trainer_factory() raises:
    """Test create_trainer factory function."""
    print("Testing create_trainer factory...")

    var config = TrainerConfig(num_epochs=3)
    var trainer = create_trainer(config)

    assert_equal(trainer.config.num_epochs, 3, "Factory creates with config")

    print("  ✓ create_trainer factory works")


def test_create_default_trainer() raises:
    """Test create_default_trainer factory."""
    print("Testing create_default_trainer factory...")

    var trainer = create_default_trainer()

    assert_equal(
        trainer.config.num_epochs, 10, "Default trainer has default config"
    )

    print("  ✓ create_default_trainer factory works")


def test_base_trainer_get_metrics() raises:
    """Test BaseTrainer get_metrics method."""
    print("Testing BaseTrainer get_metrics...")

    var config = TrainerConfig()
    var trainer = BaseTrainer(config)

    var metrics = trainer.get_metrics()

    assert_equal(metrics.current_epoch, 0, "Initial metrics")

    print("  ✓ BaseTrainer get_metrics works")


def test_base_trainer_get_best_checkpoint() raises:
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


def test_base_trainer_reset() raises:
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


def test_databatch_creation() raises:
    """Test DataBatch creation."""
    print("Testing DataBatch creation...")

    var data_shape = List[Int]()
    data_shape.append(5)
    data_shape.append(10)
    var data = AnyTensor(data_shape, DType.float32)
    var labels_shape = List[Int]()
    var labels = AnyTensor(labels_shape, DType.int32)

    var batch = DataBatch(data, labels)

    assert_equal(batch.batch_size, 5, "Batch size from data shape")

    print("  ✓ DataBatch creation works")


def test_trainer_config_to_base_trainer_integration() raises:
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


def test_metrics_flow_through_trainer() raises:
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


def test_validation_loop_init_custom() raises:
    """Test ValidationLoop with custom parameters. Closes #3682."""
    print("Testing ValidationLoop init custom...")

    var loop = ValidationLoop(
        compute_accuracy=False, compute_confusion=True, num_classes=5
    )

    assert_false(
        loop.compute_accuracy, "Custom compute_accuracy should be False"
    )
    assert_true(
        loop.compute_confusion, "Custom compute_confusion should be True"
    )
    assert_equal(loop.num_classes, 5, "Custom num_classes should be 5")

    print("  ✓ ValidationLoop custom init test passed")


def main() raises:
    """Run all test_training_infrastructure tests."""
    print("Running test_training_infrastructure tests...")

    test_trainer_config_defaults()
    print("✓ test_trainer_config_defaults")

    test_trainer_config_custom()
    print("✓ test_trainer_config_custom")

    test_training_metrics_initialization()
    print("✓ test_training_metrics_initialization")

    test_training_metrics_update()
    print("✓ test_training_metrics_update")

    test_training_metrics_reset()
    print("✓ test_training_metrics_reset")

    test_dataloader_basic()
    print("✓ test_dataloader_basic")

    test_dataloader_iteration()
    print("✓ test_dataloader_iteration")

    test_training_loop_initialization()
    print("✓ test_training_loop_initialization")

    test_validation_loop_initialization()
    print("✓ test_validation_loop_initialization")

    test_validation_loop_init_defaults()
    print("✓ test_validation_loop_init_defaults")

    test_validation_loop_run_updates_val_accuracy()
    print("✓ test_validation_loop_run_updates_val_accuracy")

    test_base_trainer_initialization()
    print("✓ test_base_trainer_initialization")

    test_create_trainer_factory()
    print("✓ test_create_trainer_factory")

    test_create_default_trainer()
    print("✓ test_create_default_trainer")

    test_base_trainer_get_metrics()
    print("✓ test_base_trainer_get_metrics")

    test_base_trainer_get_best_checkpoint()
    print("✓ test_base_trainer_get_best_checkpoint")

    test_base_trainer_reset()
    print("✓ test_base_trainer_reset")

    test_databatch_creation()
    print("✓ test_databatch_creation")

    test_trainer_config_to_base_trainer_integration()
    print("✓ test_trainer_config_to_base_trainer_integration")

    test_metrics_flow_through_trainer()
    print("✓ test_metrics_flow_through_trainer")

    test_validation_loop_init_custom()
    print("✓ test_validation_loop_init_custom")

    print("\nAll test_training_infrastructure tests passed!")
