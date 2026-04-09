"""Tests for TrainingConfig abstraction.

Tests the TrainingConfig struct which consolidates common training hyperparameters
across all model implementations (LeNet-5, AlexNet, VGG-16, ResNet-18, MobileNetV1,
GoogLeNet).

Test Coverage:
- Default configuration creation
- Static factory methods (for_lenet5, for_cifar10)
- Configuration validation
- Frequency helper methods (should_validate, should_checkpoint)
- String representation
"""

from shared.training.config import TrainingConfig
from shared.testing.assertions import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)


def test_default_config() raises:
    """Test creating a default TrainingConfig."""
    print("test_default_config...")

    var config = TrainingConfig(
        epochs=10,
        batch_size=32,
    )

    assert_equal(config.epochs, 10, "epochs should be 10")
    assert_equal(config.batch_size, 32, "batch_size should be 32")
    assert_almost_equal(
        config.learning_rate,
        Float32(0.01),
        message="learning_rate should be 0.01",
    )
    assert_almost_equal(
        config.momentum, Float32(0.9), message="momentum should be 0.9"
    )
    assert_almost_equal(
        config.weight_decay, Float32(0.0), message="weight_decay should be 0.0"
    )
    assert_equal(
        config.lr_schedule_type,
        String("none"),
        "lr_schedule_type should be 'none'",
    )
    assert_equal(config.lr_step_size, 60, "lr_step_size should be 60")
    assert_almost_equal(
        config.lr_gamma, Float32(0.2), message="lr_gamma should be 0.2"
    )
    assert_equal(config.checkpoint_every, 0, "checkpoint_every should be 0")
    assert_equal(config.validate_every, 1, "validate_every should be 1")
    assert_equal(config.log_interval, 10, "log_interval should be 10")

    print("  PASSED")


def test_custom_config() raises:
    """Test creating a custom TrainingConfig with all parameters."""
    print("test_custom_config...")

    var config = TrainingConfig(
        epochs=200,
        batch_size=128,
        learning_rate=0.001,
        momentum=0.95,
        weight_decay=5e-4,
        lr_schedule_type="step",
        lr_step_size=30,
        lr_gamma=0.1,
        checkpoint_every=10,
        validate_every=5,
        log_interval=50,
    )

    assert_equal(config.epochs, 200, "epochs should be 200")
    assert_equal(config.batch_size, 128, "batch_size should be 128")
    assert_almost_equal(
        config.learning_rate,
        Float32(0.001),
        message="learning_rate should be 0.001",
    )
    assert_almost_equal(
        config.momentum, Float32(0.95), message="momentum should be 0.95"
    )
    assert_almost_equal(
        config.weight_decay,
        Float32(5e-4),
        message="weight_decay should be 5e-4",
    )
    assert_equal(
        config.lr_schedule_type,
        String("step"),
        "lr_schedule_type should be 'step'",
    )
    assert_equal(config.lr_step_size, 30, "lr_step_size should be 30")
    assert_almost_equal(
        config.lr_gamma, Float32(0.1), message="lr_gamma should be 0.1"
    )
    assert_equal(config.checkpoint_every, 10, "checkpoint_every should be 10")
    assert_equal(config.validate_every, 5, "validate_every should be 5")
    assert_equal(config.log_interval, 50, "log_interval should be 50")

    print("  PASSED")


def test_for_lenet5() raises:
    """Test factory method for LeNet-5/EMNIST configuration."""
    print("test_for_lenet5...")

    var config = TrainingConfig.for_lenet5()

    # Verify EMNIST-specific defaults
    assert_equal(config.epochs, 10, "LeNet-5 should train for 10 epochs")
    assert_equal(
        config.batch_size, 32, "LeNet-5 EMNIST batch size should be 32"
    )
    assert_almost_equal(
        config.learning_rate,
        Float32(0.01),
        message="LeNet-5 learning rate should be 0.01",
    )
    assert_almost_equal(
        config.momentum, Float32(0.9), message="LeNet-5 momentum should be 0.9"
    )
    assert_almost_equal(
        config.weight_decay,
        Float32(0.0),
        message="LeNet-5 weight decay should be 0.0",
    )
    assert_equal(
        config.lr_schedule_type,
        String("none"),
        "LeNet-5 should use no LR schedule",
    )
    assert_equal(
        config.validate_every, 1, "LeNet-5 should validate every epoch"
    )
    assert_equal(config.log_interval, 10, "LeNet-5 log interval should be 10")

    print("  PASSED")


def test_for_cifar10() raises:
    """Test factory method for CIFAR-10 configuration."""
    print("test_for_cifar10...")

    var config = TrainingConfig.for_cifar10()

    # Verify CIFAR-10-specific defaults
    assert_equal(config.epochs, 200, "CIFAR-10 should train for 200 epochs")
    assert_equal(config.batch_size, 128, "CIFAR-10 batch size should be 128")
    assert_almost_equal(
        config.learning_rate,
        Float32(0.01),
        message="CIFAR-10 learning rate should be 0.01",
    )
    assert_almost_equal(
        config.momentum, Float32(0.9), message="CIFAR-10 momentum should be 0.9"
    )
    assert_almost_equal(
        config.weight_decay,
        Float32(5e-4),
        message="CIFAR-10 weight decay should be 5e-4",
    )
    assert_equal(
        config.lr_schedule_type,
        String("step"),
        "CIFAR-10 should use step schedule",
    )
    assert_equal(config.lr_step_size, 60, "CIFAR-10 LR step size should be 60")
    assert_almost_equal(
        config.lr_gamma, Float32(0.2), message="CIFAR-10 LR gamma should be 0.2"
    )
    assert_equal(
        config.checkpoint_every,
        10,
        "CIFAR-10 should checkpoint every 10 epochs",
    )
    assert_equal(
        config.validate_every, 1, "CIFAR-10 should validate every epoch"
    )
    assert_equal(
        config.log_interval, 100, "CIFAR-10 log interval should be 100"
    )

    print("  PASSED")


def test_should_validate() raises:
    """Test should_validate method for validation frequency."""
    print("test_should_validate...")

    # Every epoch (validate_every=1)
    var config1 = TrainingConfig(epochs=10, batch_size=32, validate_every=1)
    assert_true(
        config1.should_validate(0),
        "should validate epoch 0 when validate_every=1",
    )
    assert_true(
        config1.should_validate(1),
        "should validate epoch 1 when validate_every=1",
    )
    assert_true(
        config1.should_validate(9),
        "should validate epoch 9 when validate_every=1",
    )

    # Every 5 epochs (validate_every=5)
    var config5 = TrainingConfig(epochs=50, batch_size=32, validate_every=5)
    assert_true(
        not config5.should_validate(0),
        "should NOT validate epoch 0 when validate_every=5",
    )
    assert_true(
        config5.should_validate(4),
        "should validate epoch 4 (epoch 5, 1-indexed) when validate_every=5",
    )
    assert_true(
        config5.should_validate(9),
        "should validate epoch 9 (epoch 10, 1-indexed) when validate_every=5",
    )
    assert_true(
        config5.should_validate(14),
        "should validate epoch 14 (epoch 15, 1-indexed) when validate_every=5",
    )
    assert_true(
        not config5.should_validate(15),
        "should NOT validate epoch 15 when validate_every=5",
    )

    # Every 10 epochs (validate_every=10)
    var config10 = TrainingConfig(epochs=100, batch_size=32, validate_every=10)
    assert_true(
        not config10.should_validate(0),
        "should NOT validate epoch 0 when validate_every=10",
    )
    assert_true(
        config10.should_validate(9),
        "should validate epoch 9 (epoch 10, 1-indexed) when validate_every=10",
    )
    assert_true(
        config10.should_validate(19),
        "should validate epoch 19 (epoch 20, 1-indexed) when validate_every=10",
    )
    assert_true(
        config10.should_validate(99),
        (
            "should validate epoch 99 (epoch 100, 1-indexed) when"
            " validate_every=10"
        ),
    )

    print("  PASSED")


def test_should_checkpoint() raises:
    """Test should_checkpoint method for checkpoint frequency."""
    print("test_should_checkpoint...")

    # No checkpointing (checkpoint_every=0)
    var config_no_ckpt = TrainingConfig(
        epochs=10, batch_size=32, checkpoint_every=0
    )
    assert_true(
        not config_no_ckpt.should_checkpoint(0),
        "should NOT checkpoint epoch 0 when checkpoint_every=0",
    )
    assert_true(
        not config_no_ckpt.should_checkpoint(9),
        "should NOT checkpoint epoch 9 when checkpoint_every=0",
    )

    # Every 5 epochs (checkpoint_every=5)
    var config5 = TrainingConfig(epochs=50, batch_size=32, checkpoint_every=5)
    assert_true(
        not config5.should_checkpoint(0),
        "should NOT checkpoint epoch 0 when checkpoint_every=5",
    )
    assert_true(
        config5.should_checkpoint(4),
        (
            "should checkpoint epoch 4 (epoch 5, 1-indexed) when"
            " checkpoint_every=5"
        ),
    )
    assert_true(
        config5.should_checkpoint(9),
        (
            "should checkpoint epoch 9 (epoch 10, 1-indexed) when"
            " checkpoint_every=5"
        ),
    )
    assert_true(
        not config5.should_checkpoint(10),
        "should NOT checkpoint epoch 10 when checkpoint_every=5",
    )

    # Every 10 epochs (checkpoint_every=10)
    var config10 = TrainingConfig(
        epochs=100, batch_size=32, checkpoint_every=10
    )
    assert_true(
        not config10.should_checkpoint(0),
        "should NOT checkpoint epoch 0 when checkpoint_every=10",
    )
    assert_true(
        config10.should_checkpoint(9),
        (
            "should checkpoint epoch 9 (epoch 10, 1-indexed) when"
            " checkpoint_every=10"
        ),
    )
    assert_true(
        config10.should_checkpoint(19),
        (
            "should checkpoint epoch 19 (epoch 20, 1-indexed) when"
            " checkpoint_every=10"
        ),
    )
    assert_true(
        config10.should_checkpoint(99),
        (
            "should checkpoint epoch 99 (epoch 100, 1-indexed) when"
            " checkpoint_every=10"
        ),
    )

    print("  PASSED")


def test_to_string() raises:
    """Test string representation of TrainingConfig."""
    print("test_to_string...")

    var config = TrainingConfig(
        epochs=10,
        batch_size=32,
        learning_rate=0.01,
        momentum=0.9,
    )

    var config_str = config.to_string()

    # Verify string contains key configuration values
    assert_true(
        "TrainingConfig:" in config_str,
        "config string should contain 'TrainingConfig:'",
    )
    assert_true(
        "epochs: 10" in config_str, "config string should contain 'epochs: 10'"
    )
    assert_true(
        "batch_size: 32" in config_str,
        "config string should contain 'batch_size: 32'",
    )
    assert_true(
        "learning_rate:" in config_str,
        "config string should contain 'learning_rate:'",
    )
    assert_true(
        "momentum:" in config_str, "config string should contain 'momentum:'"
    )
    assert_true(
        "weight_decay:" in config_str,
        "config string should contain 'weight_decay:'",
    )
    assert_true(
        "lr_schedule_type:" in config_str,
        "config string should contain 'lr_schedule_type:'",
    )
    assert_true(
        "validate_every:" in config_str,
        "config string should contain 'validate_every:'",
    )
    assert_true(
        "checkpoint_every:" in config_str,
        "config string should contain 'checkpoint_every:'",
    )
    assert_true(
        "log_interval:" in config_str,
        "config string should contain 'log_interval:'",
    )

    print("  PASSED")


def test_learning_rate_schedules() raises:
    """Test different learning rate schedule types."""
    print("test_learning_rate_schedules...")

    # No schedule
    var config_none = TrainingConfig(
        epochs=100, batch_size=32, lr_schedule_type="none"
    )
    assert_equal(
        config_none.lr_schedule_type,
        String("none"),
        "lr_schedule_type should be 'none'",
    )

    # Step schedule
    var config_step = TrainingConfig(
        epochs=100, batch_size=32, lr_schedule_type="step"
    )
    assert_equal(
        config_step.lr_schedule_type,
        String("step"),
        "lr_schedule_type should be 'step'",
    )
    assert_true(
        config_step.lr_step_size > 0,
        "lr_step_size should be > 0 for step schedule",
    )
    assert_true(
        config_step.lr_gamma > 0.0, "lr_gamma should be > 0 for step schedule"
    )

    # Cosine schedule
    var config_cosine = TrainingConfig(
        epochs=100, batch_size=32, lr_schedule_type="cosine"
    )
    assert_equal(
        config_cosine.lr_schedule_type,
        String("cosine"),
        "lr_schedule_type should be 'cosine'",
    )

    print("  PASSED")


def test_weight_decay_values() raises:
    """Test weight decay configuration for regularization."""
    print("test_weight_decay_values...")

    # No regularization (weight_decay=0.0)
    var config_no_reg = TrainingConfig(
        epochs=100, batch_size=32, weight_decay=0.0
    )
    assert_almost_equal(
        config_no_reg.weight_decay,
        Float32(0.0),
        message="weight_decay should be 0.0 for no regularization",
    )

    # Standard L2 regularization
    var config_l2 = TrainingConfig(epochs=100, batch_size=32, weight_decay=5e-4)
    assert_almost_equal(
        config_l2.weight_decay,
        Float32(5e-4),
        message="weight_decay should be 5e-4 for L2 regularization",
    )

    # High regularization
    var config_high = TrainingConfig(
        epochs=100, batch_size=32, weight_decay=1e-3
    )
    assert_almost_equal(
        config_high.weight_decay,
        Float32(1e-3),
        message="weight_decay should be 1e-3 for high regularization",
    )

    print("  PASSED")


def main() raises:
    """Run all TrainingConfig tests."""
    print("=" * 60)
    print("TrainingConfig Test Suite")
    print("=" * 60)

    test_default_config()
    test_custom_config()
    test_for_lenet5()
    test_for_cifar10()
    test_should_validate()
    test_should_checkpoint()
    test_to_string()
    test_learning_rate_schedules()
    test_weight_decay_values()

    print("\n" + "=" * 60)
    print("All TrainingConfig tests PASSED!")
    print("=" * 60)
