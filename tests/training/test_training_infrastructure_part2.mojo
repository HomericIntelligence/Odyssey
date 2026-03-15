"""Training infrastructure tests - Part 2: TrainingLoop, ValidationLoop, BaseTrainer init.

Split from test_training_infrastructure.mojo to comply with ADR-009 heap corruption
workaround (≤10 fn test_ functions per file).

Tests covered:
- TrainingLoop initialization
- ValidationLoop initialization
- BaseTrainer initialization
- create_trainer factory function
- create_default_trainer factory
- BaseTrainer get_metrics

Training Infrastructure Tests (#303-322):
- #309: Training loop functionality
- #314: Validation loop functionality
- #319: Base trainer integration

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_training_infrastructure.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_false, assert_equal, assert_almost_equal
from shared.core.extensor import ExTensor
from shared.training.trainer_interface import TrainerConfig, TrainingMetrics
from shared.training.loops.training_loop import TrainingLoop
from shared.training.loops.validation_loop import ValidationLoop
from shared.training.trainer import (
    BaseTrainer,
    create_trainer,
    create_default_trainer,
)


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


fn main() raises:
    """Run Part 2 training infrastructure tests."""
    print("\n" + "=" * 70)
    print("TRAINING INFRASTRUCTURE TEST SUITE - PART 2")
    print("TrainingLoop, ValidationLoop, BaseTrainer init (#303-322)")
    print("=" * 70 + "\n")

    print("TrainingLoop Tests (#309)")
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

    print("\n" + "=" * 70)
    print("ALL PART 2 TRAINING INFRASTRUCTURE TESTS PASSED ✓")
    print("=" * 70 + "\n")
