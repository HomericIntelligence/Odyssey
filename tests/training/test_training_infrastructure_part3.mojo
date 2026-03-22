"""Training infrastructure tests - Part 3: BaseTrainer lifecycle, DataBatch, Integration.

Split from test_training_infrastructure.mojo to comply with ADR-009 heap corruption
workaround (≤10 fn test_ functions per file).

Tests covered:
- BaseTrainer get_best_checkpoint_epoch
- BaseTrainer reset
- DataBatch creation
- Integration: TrainerConfig to BaseTrainer
- Integration: Metrics flow through trainer

Training Infrastructure Tests (#303-322):
- #319: Base trainer integration
- #320: Integration tests

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_training_infrastructure.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.core.any_tensor import AnyTensor
from shared.training.trainer_interface import (
    TrainerConfig,
    TrainingMetrics,
    DataBatch,
)
from shared.training.trainer import BaseTrainer
from shared.training.loops.validation_loop import ValidationLoop


# ==================================================================
# BaseTrainer Lifecycle Tests
# ==================================================================


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
    var data = AnyTensor(data_shape, DType.float32)
    var labels_shape = List[Int]()
    var labels = AnyTensor(labels_shape, DType.int32)

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


fn test_validation_loop_init_defaults() raises:
    """Test ValidationLoop default constructor values. Closes #3682."""
    print("Testing ValidationLoop init defaults...")

    var loop = ValidationLoop()

    assert_true(
        loop.compute_accuracy, "Default compute_accuracy should be True"
    )
    assert_false(
        loop.compute_confusion, "Default compute_confusion should be False"
    )
    assert_equal(loop.num_classes, 10, "Default num_classes should be 10")

    print("  ✓ ValidationLoop defaults test passed")


fn test_validation_loop_init_custom() raises:
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


fn main() raises:
    """Run Part 3 training infrastructure tests."""
    print("\n" + "=" * 70)
    print("TRAINING INFRASTRUCTURE TEST SUITE - PART 3")
    print("BaseTrainer lifecycle, DataBatch, Integration (#303-322)")
    print("=" * 70 + "\n")

    print("BaseTrainer Tests (#319)")
    print("-" * 70)
    test_base_trainer_get_best_checkpoint()
    test_base_trainer_reset()
    test_databatch_creation()

    print("\nIntegration Tests (#320)")
    print("-" * 70)
    test_trainer_config_to_base_trainer_integration()
    test_metrics_flow_through_trainer()

    print("\nValidation Loop Tests (#3681, #3682)")
    print("-" * 70)
    test_validation_loop_init_defaults()
    test_validation_loop_init_custom()

    print("\n" + "=" * 70)
    print("ALL PART 3 TRAINING INFRASTRUCTURE TESTS PASSED ✓")
    print("=" * 70 + "\n")
