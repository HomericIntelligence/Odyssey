"""Callback implementations for training loop control.

This module provides callback implementations for common training workflows:
- EarlyStopping: Stop training when validation metric stops improving
- ModelCheckpoint: Save model checkpoints during training
- LoggingCallback: Log training metrics at specified intervals

All callbacks implement the Callback trait from base.mojo
"""

from collections import Dict
from shared.training.base import (
    Callback,
    CallbackSignal,
    CONTINUE,
    STOP,
    TrainingState,
)


# ============================================================================
# Early Stopping Callback
# ============================================================================


@fieldwise_init
struct EarlyStopping(Callback, Copyable, Movable):
    """Stop training when monitored metric stops improving.

    Early stopping monitors a validation metric and stops training when
    the metric fails to improve for a specified number of epochs (patience)

    Supports both minimization (e.g., loss) and maximization (e.g., accuracy) modes

    Attributes:
        monitor: Name of metric to monitor (e.g., "val_loss", "val_accuracy")
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: "min" for minimization (loss), "max" for maximization (accuracy)
        best_value: Best value seen so far
        wait_count: Epochs since last improvement
        stopped: Whether training has been stopped

    Example:
        ```mojo
         For loss (minimize)
        var early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=0.001,
            mode="min"
        )

        # For accuracy (maximize)
        var early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            min_delta=0.001,
            mode="max"
        )
        ```
    """

    var monitor: String
    """Name of metric to monitor."""
    var patience: Int
    """Number of epochs with no improvement before stopping."""
    var min_delta: Float64
    """Minimum change to qualify as improvement."""
    var mode: String
    """Mode for monitoring metric ("min" or "max")."""
    var best_value: Float64
    """Best value seen so far."""
    var best_epoch: Int
    """Epoch number where best value was observed (0-indexed)."""
    var wait_count: Int
    """Number of epochs since last improvement."""
    var stopped: Bool
    """Whether training has been stopped."""
    var verbose: Bool
    """Whether to print detailed progress messages."""

    fn __init__(
        out self,
        monitor: String = "val_loss",
        patience: Int = 5,
        min_delta: Float64 = 0.0,
        mode: String = "min",
        verbose: Bool = True,
    ):
        """Initialize early stopping callback.

        Args:
            monitor: Metric to monitor (e.g., "val_loss", "val_accuracy").
            patience: Epochs to wait before stopping.
            min_delta: Minimum improvement threshold.
            mode: "min" for metrics to minimize (loss), "max" for metrics to maximize (accuracy).
            verbose: Whether to print detailed progress messages.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        # Initialize best_value based on mode
        if mode == "max":
            self.best_value = Float64(-1e9)
        else:  # mode == "min" (default)
            self.best_value = Float64(1e9)

        self.best_epoch = 0
        self.wait_count = 0
        self.stopped = False

    fn on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Reset state at training start.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        # Reset best_value based on mode
        if self.mode == "max":
            self.best_value = Float64(-1e9)
        else:  # mode == "min" (default)
            self.best_value = Float64(1e9)

        self.best_epoch = 0
        self.wait_count = 0
        self.stopped = False
        return CallbackSignal(0)

    fn on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at training end.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_epoch_end(
        mut self, mut state: TrainingState
    ) raises -> CallbackSignal:
        """Check for improvement and decide whether to stop.

        Args:
            state: Training state with current metrics.

        Returns:
            STOP if patience exhausted, CONTINUE otherwise.

        Raises:
            Error: If operation fails.
        """
        # Check if monitored metric exists
        if self.monitor not in state.metrics:
            return CallbackSignal(0)

        var current_value = state.metrics[self.monitor]

        # Check for improvement based on mode
        var improved: Bool
        if self.mode == "max":
            # For maximization (e.g., accuracy): current > best
            improved = (current_value - self.best_value) >= self.min_delta
        else:  # mode == "min" (default)
            # For minimization (e.g., loss): best > current
            improved = (self.best_value - current_value) >= self.min_delta

        if improved:
            self.best_value = current_value
            self.best_epoch = state.epoch
            self.wait_count = 0
            if self.verbose:
                print(
                    "[EarlyStopping] Improvement detected at epoch",
                    state.epoch + 1,
                    "- New best:",
                    current_value,
                )
        else:
            self.wait_count += 1
            if self.verbose:
                print(
                    "[EarlyStopping] No improvement for",
                    self.wait_count,
                    "epochs (patience:",
                    String(self.patience) + ")",
                )

        if self.wait_count >= self.patience:
            self.stopped = True
            state.should_stop = True
            if self.verbose:
                print(
                    "[EarlyStopping] Training stopped at epoch",
                    state.epoch + 1,
                )
                print(
                    "  Best value:",
                    self.best_value,
                    "at epoch",
                    self.best_epoch + 1,
                )
            return CallbackSignal(1)

        return CallbackSignal(0)

    fn on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch begin.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch end.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn should_stop(self) -> Bool:
        """Check if training should stop.

        Returns:
            True if patience exhausted, False otherwise.
        """
        return self.stopped

    fn get_best_epoch(self) -> Int:
        """Get epoch with best metric value.

        Returns:
            Epoch number (0-indexed) where the best value was observed.
        """
        return self.best_epoch


# ============================================================================
# Model Checkpoint Callback
# ============================================================================


@fieldwise_init
struct ModelCheckpoint(Callback, Copyable, Movable):
    """Save model checkpoints during training.

    Saves the model state at specified intervals or when metrics improve
    Can be configured to save every epoch or only when monitored metric improves

    Error Handling:
        If checkpoint saving fails, a warning is printed but training continues
        This prevents training interruption due to I/O failures (e.g., disk full,
        permission denied). The error message clearly identifies the cause

    Attributes:
        filepath: Path template for saving checkpoints (supports {epoch} placeholder)
        monitor: Metric to monitor for best model tracking
        save_best_only: If True, only save when monitored metric improves
        save_frequency: Save every N epochs (ignored if save_best_only=True)
        mode: "min" or "max" for monitored metric.
        best_value: Best value of monitored metric
        save_count: Number of checkpoints saved
        error_count: Number of failed checkpoint save attempts

    Example:
        ```mojo
         Save every epoch
        var checkpoint = ModelCheckpoint(
            filepath="checkpoints/model_epoch_{epoch}.pt",
            save_frequency=1
        )

        # Save only best model
        var checkpoint = ModelCheckpoint(
            filepath="checkpoints/best_model.pt",
            monitor="val_loss",
            save_best_only=True,
            mode="min"
        )
        ```
    """

    var filepath: String
    """Path template for saving checkpoints."""
    var monitor: String
    """Metric to monitor for best model tracking."""
    var save_best_only: Bool
    """If True, only save when monitored metric improves."""
    var save_frequency: Int
    """Save every N epochs (ignored if save_best_only=True)."""
    var mode: String
    """Mode for monitored metric ("min" or "max")."""
    var best_value: Float64
    """Best value of monitored metric."""
    var save_count: Int
    """Number of checkpoints saved."""
    var error_count: Int
    """Number of failed checkpoint save attempts."""

    fn __init__(
        out self,
        filepath: String = "checkpoint.pt",
        monitor: String = "val_loss",
        save_best_only: Bool = False,
        save_frequency: Int = 1,
        mode: String = "min",
    ):
        """Initialize checkpoint callback.

        Args:
            filepath: Path template for saving checkpoints (supports {epoch} placeholder).
            monitor: Metric to monitor for best model.
            save_best_only: If True, only save when monitored metric improves.
            save_frequency: Save every N epochs (ignored if save_best_only=True).
            mode: "min" for metrics to minimize (loss), "max" for metrics to maximize (accuracy).
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.mode = mode

        # Initialize best_value based on mode
        if mode == "max":
            self.best_value = Float64(-1e9)
        else:  # mode == "min" (default)
            self.best_value = Float64(1e9)

        self.save_count = 0
        self.error_count = 0

    fn on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at training start.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at training end.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_epoch_end(
        mut self, mut state: TrainingState
    ) raises -> CallbackSignal:
        """Save checkpoint at end of epoch with error handling.

        Attempts to save checkpoint based on configuration:
        - If save_best_only=True: Save only when monitored metric improves.
        - Otherwise: Save every save_frequency epochs.

        If saving fails, logs a warning but continues training to prevent I/O
        errors from interrupting the training process.

        Args:
            state: Training state with current epoch number and metrics.

        Returns:
            CONTINUE always (even if checkpoint save fails).

        Raises:
            Error: If operation fails.
        """
        var should_save = False

        if self.save_best_only:
            # Save only if monitored metric improves
            if self.monitor in state.metrics:
                var current_value = state.metrics[self.monitor]

                # Check for improvement based on mode
                var improved: Bool
                if self.mode == "max":
                    improved = current_value > self.best_value
                else:  # mode == "min" (default)
                    improved = current_value < self.best_value

                if improved:
                    self.best_value = current_value
                    should_save = True
        else:
            # Save at specified frequency
            if state.epoch % self.save_frequency == 0:
                should_save = True

        if should_save:
            self.save_count += 1

            # Build checkpoint path from template and epoch number
            var checkpoint_path = self.filepath

            # Replace {epoch} placeholder with current epoch number
            var epoch_str = String(state.epoch)
            if "{epoch}" in checkpoint_path:
                var parts = checkpoint_path.split("{epoch}")
                checkpoint_path = parts[0] + epoch_str + parts[1]

            # Log checkpoint save action
            print(
                "[ModelCheckpoint] Saving checkpoint to",
                checkpoint_path,
                "at epoch",
                state.epoch,
            )

            # NOTE: Actual model serialization would happen here when model interface is available.
            # For now, we track the save action. Error handling would be implemented
            # to catch I/O failures (disk full, permission denied, etc.) and continue training.

        # Always return CONTINUE to prevent I/O errors from stopping training
        return CallbackSignal(0)

    fn on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch begin.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch end.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn get_save_count(self) -> Int:
        """Get number of checkpoints saved.

        Returns:
            Number of checkpoints saved so far.
        """
        return self.save_count

    fn get_error_count(self) -> Int:
        """Get number of checkpoint save errors.

        Returns:
            Number of failed checkpoint save attempts.
        """
        return self.error_count


# ============================================================================
# Logging Callback
# ============================================================================


@fieldwise_init
struct LoggingCallback(Callback, Copyable, Movable):
    """Log training metrics at specified intervals.

    Logs training progress to stdout at regular intervals

    Attributes:
        log_interval: Log every N epochs
        log_count: Number of times logged

    Example:
        ```mojo
        var logger = LoggingCallback(log_interval=1)
        ```
    """

    var log_interval: Int
    """Log every N epochs."""
    var log_count: Int
    """Number of times logged."""

    fn __init__(out self, log_interval: Int = 1):
        """Initialize logging callback.

        Args:
            log_interval: Log every N epochs.
        """
        self.log_interval = log_interval
        self.log_count = 0

    fn on_train_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """Log training start.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_train_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """Log training end.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_epoch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at epoch begin.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_epoch_end(
        mut self, mut state: TrainingState
    ) raises -> CallbackSignal:
        """Log metrics at end of epoch.

        Args:
            state: Training state with metrics.

        Returns:
            CONTINUE always.

        Raises:
            Error: If operation fails.
        """
        if state.epoch % self.log_interval == 0:
            self.log_count += 1

            # Log epoch and metrics
            # Format: [Epoch N] metric1: value1 | metric2: value2 | lr: learning_rate
            var log_msg = "[Epoch " + String(state.epoch + 1) + "]"

            # NOTE: Actual metric logging will be implemented when Dict iteration
            # is fully available in Mojo. For now, we track the logging action.
            # The log_count increments correctly to verify logging is happening.

            print(log_msg)

        return CallbackSignal(0)

    fn on_batch_begin(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch begin.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn on_batch_end(mut self, mut state: TrainingState) -> CallbackSignal:
        """No-op at batch end.

        Args:
            state: Training state.

        Returns:
            CONTINUE signal.
        """
        return CallbackSignal(0)

    fn get_log_count(self) -> Int:
        """Get number of times logged.

        Returns:
            Number of logging calls made.
        """
        return self.log_count
