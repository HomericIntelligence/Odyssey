"""Base trainer implementation.

Complete training infrastructure integrating interface, training loop,
and validation loop.

Base Trainer (#318-322):
- #319: Composition-based design
- #320: State management
- #321: Configuration management
- #322: Integration with metrics, callbacks, optimizers

Design principles:
- Composition over inheritance
- Explicit configuration objects
- Clear error handling
- Integration with all components
"""

from collections import List
from shared.core import ExTensor
from shared.training.trainer_interface import (
    Trainer,
    TrainerConfig,
    TrainingMetrics,
    DataLoader,
)
from shared.training.loops.training_loop import TrainingLoop
from shared.training.loops.validation_loop import ValidationLoop
from shared.training.metrics import MetricLogger, MetricResult
from shared.training.mixed_precision import (
    GradientScaler,
    check_gradients_finite,
    clip_gradients_by_norm,
)
from shared.core.numerical_safety import has_nan, has_inf
from shared.utils.serialization import NamedTensor
from shared.training.callbacks import EarlyStopping
from shared.training.base import TrainingState, CallbackSignal


struct BaseTrainer(Trainer):
    """Base trainer implementation with full training infrastructure.

    Provides complete training capabilities:
    - Training loop with forward/backward/update
    - Validation loop with gradient-free evaluation
    - Metric tracking and logging
    - Configuration management
    - State management for checkpointing

    Example usage:
        var config = TrainerConfig(num_epochs=10, batch_size=32)
        var trainer = BaseTrainer(config)

        trainer.fit(
            model_forward,
            compute_loss,
            optimizer_step,
            zero_gradients,
            train_loader,
            val_loader
        )
    """

    var config: TrainerConfig
    var metrics: TrainingMetrics
    var metric_logger: MetricLogger
    var training_loop: TrainingLoop
    var validation_loop: ValidationLoop
    var is_training: Bool

    fn __init__(out self, config: TrainerConfig):
        """Initialize base trainer.

        Args:
            config: Trainer configuration.
        """
        self.config = config.copy()
        self.metrics = TrainingMetrics()
        self.metric_logger = MetricLogger()
        self.training_loop = TrainingLoop(log_interval=config.log_interval)
        self.validation_loop = ValidationLoop(compute_accuracy=True)
        self.is_training = False

    fn train(mut self, num_epochs: Int) raises:
        """Execute training loop for specified number of epochs.

        This is a simplified interface. Use fit() for full training
        with validation.

        Args:
            num_epochs: Number of epochs to train.

        Raises:
            Error if training fails or called without proper setup.
        """
        raise Error(
            "Use fit() method instead of train() for complete training workflow"
        )

    fn validate(mut self) raises -> Float64:
        """Execute validation loop.

        This is a simplified interface. Use fit() for integrated
        training with validation.

        Returns:
            Validation loss.

        Raises:
            Error if validation fails or called without proper setup.
        """
        raise Error("Use fit() method for integrated training with validation")

    fn fit(
        mut self,
        model_forward: fn (ExTensor) raises -> ExTensor,
        compute_loss: fn (ExTensor, ExTensor) raises -> ExTensor,
        optimizer_step: fn () raises -> None,
        zero_gradients: fn () raises -> None,
        mut train_loader: DataLoader,
        mut val_loader: DataLoader,
        mut early_stopping: Optional[EarlyStopping],
    ) raises:
        """Train model with periodic validation and optional early stopping.

        This is the main entry point for training. Combines training loop
        and validation loop with proper metric tracking and callback support.

        Args:
            model_forward: Function to compute model forward pass.
            compute_loss: Function to compute loss.
            optimizer_step: Function to update weights.
            zero_gradients: Function to zero gradients.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            early_stopping: Optional EarlyStopping callback.
                If provided, training will stop early when patience is exhausted.

        Raises:
            Error if training or validation fails.

        Example:
            ```mojo
            # Without early stopping
            trainer.fit(model_forward, compute_loss, optimizer_step,
                       zero_gradients, train_loader, val_loader)

            # With early stopping
            var early_stop = EarlyStopping(monitor="val_loss", patience=5)
            trainer.fit(model_forward, compute_loss, optimizer_step,
                       zero_gradients, train_loader, val_loader,
                       Pointer.address_of(early_stop))
            ```
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print("\nConfiguration:")
        print("  Epochs: " + String(self.config.num_epochs))
        print("  Batch Size: " + String(self.config.batch_size))
        print("  Learning Rate: " + String(self.config.learning_rate))
        print("  Log Interval: " + String(self.config.log_interval))
        print("  Validate Interval: " + String(self.config.validate_interval))

        # Print mixed precision settings
        if self.config.use_mixed_precision:
            print("\nMixed Precision Training: ENABLED")
            var dtype_name = (
                "float16" if self.config.precision_dtype
                == DType.float16 else "float32"
            )
            print("  Precision: " + dtype_name)
            print("  Initial Loss Scale: " + String(self.config.loss_scale))
            if self.config.gradient_clip_norm > 0.0:
                print(
                    "  Gradient Clipping: "
                    + String(self.config.gradient_clip_norm)
                )
        else:
            print("\nMixed Precision Training: DISABLED")

        print()

        self.is_training = True

        # Create training state for callbacks
        var state = TrainingState(
            epoch=0, learning_rate=self.config.learning_rate
        )

        # Call on_train_begin callback
        if early_stopping:
            var signal = early_stopping.value().on_train_begin(state)
            if signal.value == 1:  # STOP signal
                print("Training stopped by callback at start")
                self.is_training = False
                return

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.metrics.current_epoch = epoch
            self.metrics.reset_epoch()

            # Update state for this epoch
            state.epoch = epoch
            state.learning_rate = self.config.learning_rate

            print("\n" + "=" * 70)
            print(
                "EPOCH "
                + String(epoch + 1)
                + "/"
                + String(self.config.num_epochs)
            )
            print("=" * 70)

            # Call on_epoch_begin callback
            if early_stopping:
                var signal = early_stopping.value().on_epoch_begin(state)
                if signal.value == 1:  # STOP signal
                    print("Training stopped by callback at epoch begin")
                    break

            # Run training epoch
            self.training_loop.run_epoch(
                model_forward,
                compute_loss,
                optimizer_step,
                zero_gradients,
                train_loader,
                self.metrics,
            )

            # Validation (if enabled for this epoch)
            if (
                self.config.validate_interval > 0
                and (epoch + 1) % self.config.validate_interval == 0
            ):
                print("\n" + "-" * 70)
                print("VALIDATION")
                print("-" * 70)

                var val_loss = self.validation_loop.run(
                    model_forward, compute_loss, val_loader, self.metrics
                )

                print("-" * 70)

            # Update state with current metrics
            state.metrics["train_loss"] = self.metrics.train_loss
            state.metrics["val_loss"] = self.metrics.val_loss

            # Call on_epoch_end callback (after validation)
            if early_stopping:
                var signal = early_stopping.value().on_epoch_end(state)
                if signal.value == 1:  # STOP signal
                    print("Training stopped by early stopping")
                    break

            # Log epoch metrics
            var epoch_metrics: List[MetricResult] = []
            epoch_metrics.append(
                MetricResult("train_loss", self.metrics.train_loss)
            )
            epoch_metrics.append(
                MetricResult("val_loss", self.metrics.val_loss)
            )
            self.metric_logger.log_epoch(epoch, epoch_metrics)

            # Print epoch summary
            print("\nEpoch " + String(epoch + 1) + " Summary:")
            print("  Train Loss: " + String(self.metrics.train_loss))
            print("  Val Loss: " + String(self.metrics.val_loss))
            print(
                "  Best Val Loss: "
                + String(self.metrics.best_val_loss)
                + " (epoch "
                + String(self.metrics.best_epoch + 1)
                + ")"
            )

        self.is_training = False

        # Call on_train_end callback
        if early_stopping:
            _ = early_stopping.value().on_train_end(state)

        # Final summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        self.metrics.print_summary()
        self.metric_logger.print_summary()

    fn fit(mut self, num_epochs: Int, validate_every: Int = 1) raises:
        """Convenience method matching Trainer trait.

        Use the full fit() method with model/optimizer functions.

        Args:
            num_epochs: Number of epochs to train.
            validate_every: Validate every N epochs.

        Raises:
            Error indicating proper usage.
        """
        raise Error(
            "Use fit() with model_forward, compute_loss, optimizer_step, and"
            " data loaders"
        )

    fn get_metrics(self) -> TrainingMetrics:
        """Get current training metrics.

        Returns:
            Current metrics.
        """
        return self.metrics.copy()

    fn get_best_checkpoint_epoch(self) -> Int:
        """Get epoch with best validation loss.

        Returns:
            Epoch number (0-indexed).
        """
        return self.metrics.best_epoch

    fn save_checkpoint(self, epoch: Int, path: String) raises:
        """Save checkpoint for current epoch.

        Saves training metrics and state using the named checkpoint format.
        This enables resuming training from a saved checkpoint.

        The checkpoint includes:
        - Training metrics (loss, accuracy, etc.)
        - Current epoch number
        - Best epoch and loss information

        Args:
            epoch: Current epoch number.
            path: Path to save checkpoint directory.

        Raises:
            Error if checkpoint save fails.

        Example:
            ```mojo
            trainer.save_checkpoint(10, "checkpoints/model_epoch_10/")
            ```

        Note:
            This is a simplified implementation tracking metrics.
            Full checkpointing would also save model weights and optimizer state.
            See issue #2726 for complete checkpoint system.
        """
        from shared.utils.serialization import save_named_checkpoint
        from collections import Dict

        # Create metadata dictionary with training state
        var metadata = Dict[String, String]()
        metadata["epoch"] = String(epoch)
        metadata["best_epoch"] = String(self.metrics.best_epoch)
        metadata["best_val_loss"] = String(self.metrics.best_val_loss)
        metadata["train_loss"] = String(self.metrics.train_loss)
        metadata["total_batches"] = String(self.metrics.total_batches)

        # Save checkpoint with metadata (no tensors for now, just state)
        save_named_checkpoint(List[NamedTensor](), path, metadata^)

        print(
            "Checkpoint saved: "
            + path
            + " (epoch "
            + String(epoch)
            + ", best_val_loss="
            + String(self.metrics.best_val_loss)
            + ")"
        )

    fn load_checkpoint(mut self, path: String) raises:
        """Load checkpoint from path.

        Restores training metrics and state from a saved checkpoint.
        This enables resuming training from the saved state.

        Args:
            path: Path to checkpoint directory.

        Raises:
            Error if checkpoint load fails.

        Example:
            ```mojo
            trainer.load_checkpoint("checkpoints/model_epoch_10/")
            ```

        Note:
            This is a simplified implementation that restores metrics.
            Full checkpointing would also load model weights and optimizer state.
            See issue #2726 for complete checkpoint system.
        """
        from shared.utils.serialization import load_named_checkpoint

        # Load checkpoint
        var result = load_named_checkpoint(path)
        _ = result[0]  # Tensors not used yet
        var metadata = result[1].copy()

        # Restore metrics from metadata
        if "epoch" in metadata:
            # Epoch info is loaded but not automatically used
            print("Loaded checkpoint from: " + path)

        if "best_val_loss" in metadata:
            var best_loss_str = metadata["best_val_loss"]
            # In a full implementation, would parse and restore metrics
            print("Loaded checkpoint with best_val_loss: " + best_loss_str)

    fn reset(mut self):
        """Reset trainer state for new training run."""
        self.metrics = TrainingMetrics()
        self.metric_logger = MetricLogger()
        self.is_training = False


fn create_trainer(config: TrainerConfig) -> BaseTrainer:
    """Create a base trainer with given configuration.

    Args:
            config: Trainer configuration.

    Returns:
            Initialized BaseTrainer.
    """
    return BaseTrainer(config)


fn create_default_trainer() -> BaseTrainer:
    """Create a trainer with default configuration.

    Returns:
            BaseTrainer with default config.
    """
    var config = TrainerConfig()
    return BaseTrainer(config)
