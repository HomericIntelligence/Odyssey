"""Training configuration abstraction for model training.

This module provides a unified TrainingConfig struct that consolidates common
training hyperparameters and configuration patterns found across all example
implementations (LeNet-5, AlexNet, VGG-16, ResNet-18, MobileNetV1, GoogLeNet).

Common Training Patterns (Consolidation):
- Epoch count and batch size (vary by model/dataset)
- Learning rate and momentum (SGD hyperparameters)
- Weight decay (L2 regularization)
- Learning rate scheduling (step decay, cosine annealing, none)
- Validation frequency (every N epochs)
- Checkpoint frequency (periodic saving)
- Logging interval (batches between progress reports)

Design principles:
- Centralized configuration for consistency across examples
- Static factory methods for common presets (EMNIST, CIFAR-10)
- Backward compatible with existing training loops
- Extensible for additional hyperparameters
"""


struct TrainingConfig:
    """Unified training configuration for model training.

    Consolidates common training hyperparameters that vary across examples:
    - Epoch and batch size parameters (dataset/model specific)
    - Optimizer settings (learning rate, momentum, weight decay)
    - Learning rate scheduling (step, cosine, none)
    - Evaluation frequency (validation every N epochs)
    - Checkpoint frequency (save every N epochs, or only at end)
    - Logging interval (batches between progress logs)

    Attributes:
        epochs: Total number of training epochs.
        batch_size: Mini-batch size for training.
        learning_rate: Initial learning rate for optimizer.
        momentum: Momentum parameter for SGD (default 0.9).
        weight_decay: L2 regularization coefficient (default 0.0).
        lr_schedule_type: Learning rate schedule type ("none", "step", "cosine").
        lr_step_size: Number of epochs between step decay reductions.
        lr_gamma: Multiplicative factor for learning rate decay.
        checkpoint_every: Save checkpoint every N epochs (0 = only at end).
        validate_every: Run validation every N epochs (1 = every epoch).
        log_interval: Log progress every N batches.

    Example:
        ```mojo
        # EMNIST configuration for LeNet-5
        var config = TrainingConfig.for_lenet5()

        # CIFAR-10 configuration
        var config = TrainingConfig.for_cifar10()

        # Custom configuration
        var config = TrainingConfig(
            epochs=100,
            batch_size=128,
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=5e-4,
            lr_schedule_type="step",
            lr_step_size=30,
            lr_gamma=0.1,
            checkpoint_every=10,
            validate_every=1,
            log_interval=100
        )
        ```
    """

    var epochs: Int
    """Total number of training epochs."""
    var batch_size: Int
    """Mini-batch size for training."""
    var learning_rate: Float32
    """Initial learning rate for optimizer."""
    var momentum: Float32
    """Momentum parameter for SGD."""
    var weight_decay: Float32
    """L2 regularization coefficient."""
    var lr_schedule_type: String
    """Learning rate schedule type ("none", "step", "cosine")."""
    var lr_step_size: Int
    """Number of epochs between step decay reductions."""
    var lr_gamma: Float32
    """Multiplicative factor for learning rate decay."""
    var checkpoint_every: Int
    """Save checkpoint every N epochs (0 = only at end)."""
    var validate_every: Int
    """Run validation every N epochs (1 = every epoch)."""
    var log_interval: Int
    """Log progress every N batches."""

    def __init__(
        out self,
        epochs: Int,
        batch_size: Int,
        learning_rate: Float32 = 0.01,
        momentum: Float32 = 0.9,
        weight_decay: Float32 = 0.0,
        lr_schedule_type: String = "none",
        lr_step_size: Int = 60,
        lr_gamma: Float32 = 0.2,
        checkpoint_every: Int = 0,
        validate_every: Int = 1,
        log_interval: Int = 10,
    ):
        """Initialize training configuration.

        Args:
            epochs: Total number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Initial learning rate (default 0.01).
            momentum: SGD momentum (default 0.9).
            weight_decay: L2 regularization (default 0.0, no regularization).
            lr_schedule_type: Schedule type - "none", "step", or "cosine".
            lr_step_size: Epochs between step decay reductions (default 60).
            lr_gamma: LR decay multiplier (default 0.2).
            checkpoint_every: Save every N epochs (0 = only at end).
            validate_every: Validate every N epochs (default 1 = every epoch).
            log_interval: Log every N batches (default 10).
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_schedule_type = lr_schedule_type
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every
        self.log_interval = log_interval

    @staticmethod
    def for_lenet5() -> TrainingConfig:
        """Create configuration for LeNet-5 on EMNIST.

        LeNet-5 on EMNIST dataset configuration:
        - 10 epochs (quick training on EMNIST balanced dataset)
        - batch size 32 (smaller batches for 47 classes)
        - learning rate 0.01 (standard for EMNIST)
        - validate every epoch
        - log every batch for detailed progress

        Returns:
            TrainingConfig optimized for LeNet-5/EMNIST training.
        """
        return TrainingConfig(
            epochs=10,
            batch_size=32,
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0,
            lr_schedule_type="none",
            lr_step_size=3,
            lr_gamma=0.1,
            checkpoint_every=0,
            validate_every=1,
            log_interval=10,
        )

    @staticmethod
    def for_cifar10() -> TrainingConfig:
        """Create configuration for CIFAR-10 models.

        Standard CIFAR-10 training configuration (used for AlexNet, VGG-16, etc.):
        - 200 epochs (standard for CIFAR-10 convergence)
        - batch size 128 (standard CIFAR-10 batch size)
        - learning rate 0.01 (SGD baseline)
        - step decay: reduce by 0.2 every 60 epochs
        - validate every epoch
        - checkpoint every 10 epochs
        - log every 100 batches for moderate verbosity

        Returns:
            TrainingConfig optimized for CIFAR-10 model training.
        """
        return TrainingConfig(
            epochs=200,
            batch_size=128,
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=5e-4,
            lr_schedule_type="step",
            lr_step_size=60,
            lr_gamma=0.2,
            checkpoint_every=10,
            validate_every=1,
            log_interval=100,
        )

    def should_validate(self, epoch: Int) -> Bool:
        """Check if validation should run this epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            True if validation should run, False otherwise.
        """
        return (epoch + 1) % self.validate_every == 0

    def should_checkpoint(self, epoch: Int) -> Bool:
        """Check if checkpoint should be saved this epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            True if checkpoint should be saved, False otherwise.
            Note: Returns True for last epoch if checkpoint_every > 0.
        """
        if self.checkpoint_every == 0:
            return False
        return (epoch + 1) % self.checkpoint_every == 0

    def to_string(self) -> String:
        """Format configuration as human-readable string.

        Returns:
            Multi-line string representation of configuration.
        """
        var result = String()
        result += "TrainingConfig:\n"
        result += "  epochs: " + String(self.epochs) + "\n"
        result += "  batch_size: " + String(self.batch_size) + "\n"
        result += "  learning_rate: " + String(self.learning_rate) + "\n"
        result += "  momentum: " + String(self.momentum) + "\n"
        result += "  weight_decay: " + String(self.weight_decay) + "\n"
        result += "  lr_schedule_type: " + self.lr_schedule_type + "\n"
        result += "  lr_step_size: " + String(self.lr_step_size) + "\n"
        result += "  lr_gamma: " + String(self.lr_gamma) + "\n"
        result += "  checkpoint_every: " + String(self.checkpoint_every) + "\n"
        result += "  validate_every: " + String(self.validate_every) + "\n"
        result += "  log_interval: " + String(self.log_interval)
        return result
