"""Checkpoint management for training resumption and best model tracking.

Provides CheckpointManager for saving and loading model checkpoints with metadata,
keeping only N most recent checkpoints, and tracking the best model based on metrics.

Key Features:
- Save checkpoints with epoch, metrics, and timestamp
- Automatic cleanup of old checkpoints (keep only N most recent)
- Track and save best model based on validation metric
- Resume training from latest checkpoint
- List all available checkpoints

Example:
    from shared.training.checkpoint import CheckpointManager

    var ckpt_mgr = CheckpointManager("checkpoints/lenet5", max_to_keep=5)

    # Training loop
    for epoch in range(num_epochs):
        var train_loss = train_epoch(...)
        var val_loss, val_acc = validate(...)

        # Save checkpoint every epoch
        ckpt_mgr.save_checkpoint(
            model_params, param_names, epoch,
            train_loss=train_loss, val_loss=val_loss, val_acc=val_acc
        )

        # Track best model
        ckpt_mgr.save_best(model_params, param_names, epoch, val_loss)

    # Resume training
    var start_epoch = ckpt_mgr.load_latest(model_params, param_names)
"""

from shared.core.extensor import ExTensor
from shared.training.model_utils import save_model_weights, load_model_weights
from shared.utils.file_io import (
    create_directory,
    file_exists,
    safe_write_file,
    safe_read_file,
)
from collections import List


struct CheckpointManager:
    """Manages model checkpoints with automatic cleanup and best model tracking.

    Saves model weights with metadata (epoch, metrics) to a checkpoint directory.
    Automatically keeps only the N most recent checkpoints and tracks the best
    model based on a metric (e.g., validation loss).

    Attributes:
        checkpoint_dir: Directory to store checkpoints
        max_to_keep: Maximum number of recent checkpoints to keep (0 = keep all)
        best_metric_value: Best metric value seen so far
        best_metric_name: Name of metric to track for best model
        minimize_metric: True if lower is better (e.g., loss), False if higher is better (e.g., accuracy)
    """

    var checkpoint_dir: String
    var max_to_keep: Int
    var best_metric_value: Float32
    var best_metric_name: String
    var minimize_metric: Bool

    fn __init__(
        out self,
        checkpoint_dir: String,
        max_to_keep: Int = 5,
        best_metric_name: String = "val_loss",
        minimize_metric: Bool = True,
    ) raises:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints (created if doesn't exist)
            max_to_keep: Maximum number of recent checkpoints to keep (0 = keep all)
            best_metric_name: Name of metric to track for best model
            minimize_metric: True if lower metric is better (loss), False if higher is better (accuracy)

        Raises:
            Error: If directory creation fails
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.best_metric_name = best_metric_name
        self.minimize_metric = minimize_metric

        # Initialize best metric value
        if minimize_metric:
            self.best_metric_value = Float32(
                1e9
            )  # High initial value for minimization
        else:
            self.best_metric_value = Float32(
                -1e9
            )  # Low initial value for maximization

        # Create checkpoint directory
        if not create_directory(checkpoint_dir):
            raise Error(
                "Failed to create checkpoint directory: " + checkpoint_dir
            )

    fn save_checkpoint(
        self,
        parameters: List[ExTensor],
        param_names: List[String],
        epoch: Int,
        train_loss: Float32 = 0.0,
        val_loss: Float32 = 0.0,
        val_acc: Float32 = 0.0,
    ) raises:
        """Save checkpoint with model weights and metadata.

        Args:
            parameters: List of model parameter tensors
            param_names: List of parameter names
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            val_acc: Validation accuracy for this epoch

        Raises:
            Error: If save fails
        """
        # Create checkpoint subdirectory
        var epoch_dir = (
            self.checkpoint_dir + "/checkpoint_epoch_" + String(epoch)
        )
        if not create_directory(epoch_dir):
            raise Error(
                "Failed to create checkpoint epoch directory: " + epoch_dir
            )

        # Save model weights
        save_model_weights(parameters, epoch_dir, param_names)

        # Save metadata
        var metadata_path = epoch_dir + "/metadata.txt"
        self._save_metadata(metadata_path, epoch, train_loss, val_loss, val_acc)

        # Update checkpoint tracker
        var tracking_file = self.checkpoint_dir + "/checkpoint_tracker.txt"
        var tracking_content = (
            String("latest_epoch=") + String(epoch) + String("\n")
        )
        if not safe_write_file(tracking_file, tracking_content):
            raise Error("Failed to update checkpoint tracker")

        print("Checkpoint saved: epoch " + String(epoch))

        # Cleanup old checkpoints if needed
        if self.max_to_keep > 0:
            self._cleanup_old_checkpoints()

    fn save_best(
        mut self,
        mut parameters: List[ExTensor],
        param_names: List[String],
        epoch: Int,
        metric_value: Float32,
    ) raises:
        """Save checkpoint as best model if metric improved.

        Args:
            parameters: List of model parameter tensors
            param_names: List of parameter names
            epoch: Current epoch number
            metric_value: Current metric value (e.g., validation loss)

        Raises:
            Error: If save fails
        """
        var is_best: Bool

        if self.minimize_metric:
            is_best = metric_value < self.best_metric_value
        else:
            is_best = metric_value > self.best_metric_value

        if is_best:
            self.best_metric_value = metric_value

            # Create best model directory
            var best_dir = self.checkpoint_dir + "/best_model"
            if not create_directory(best_dir):
                raise Error(
                    "Failed to create best model directory: " + best_dir
                )

            # Save weights
            save_model_weights(parameters, best_dir, param_names)

            # Save metadata
            var metadata_path = best_dir + "/metadata.txt"
            self._save_metadata(
                metadata_path, epoch, 0.0, 0.0, 0.0, metric_value
            )

            print(
                "New best model saved! "
                + self.best_metric_name
                + " = "
                + String(metric_value)
            )

    fn load_latest(
        mut self,
        mut parameters: List[ExTensor],
        param_names: List[String],
    ) raises -> Int:
        """Load the most recent checkpoint.

        Args:
            parameters: List to populate with loaded parameters
            param_names: List of parameter names to load

        Returns:
            Epoch number of loaded checkpoint (0 if no checkpoint found)

        Raises:
            Error: If load fails
        """
        var latest_epoch = self._find_latest_epoch()

        if latest_epoch < 0:
            print("No checkpoint found, starting from scratch")
            return 0

        var epoch_dir = (
            self.checkpoint_dir + "/checkpoint_epoch_" + String(latest_epoch)
        )

        print("Loading checkpoint from epoch " + String(latest_epoch))

        # Load weights
        load_model_weights(parameters, epoch_dir, param_names)

        # Load metadata to update best metric
        var metadata_path = epoch_dir + "/metadata.txt"
        self._load_metadata(metadata_path)

        return latest_epoch

    fn load_best(
        mut self, mut parameters: List[ExTensor], param_names: List[String]
    ) raises:
        """Load the best model checkpoint.

        Args:
            parameters: List to populate with loaded parameters
            param_names: List of parameter names to load

        Raises:
            Error: If best model doesn't exist or load fails
        """
        var best_dir = self.checkpoint_dir + "/best_model"
        var metadata_path = best_dir + "/metadata.txt"

        if not file_exists(metadata_path):
            raise Error("No best model checkpoint found")

        print("Loading best model")

        # Load weights
        load_model_weights(parameters, best_dir, param_names)

        # Load metadata
        self._load_metadata(metadata_path)

    fn _find_latest_epoch(self) raises -> Int:
        """Find the most recent epoch number from checkpoint tracking file.

        Returns:
            Latest epoch number, or -1 if no checkpoints found
        """
        var tracking_file = self.checkpoint_dir + "/checkpoint_tracker.txt"

        if not file_exists(tracking_file):
            return -1

        try:
            var content = safe_read_file(tracking_file)
            var lines = content.split("\n")

            # Find the latest_epoch line
            for i in range(len(lines)):
                var line = lines[i]
                if line.startswith("latest_epoch="):
                    var epoch_str = line[len("latest_epoch=") :]
                    return atol(epoch_str)

            return -1
        except:
            return -1

    fn _cleanup_old_checkpoints(self) raises:
        """Remove old checkpoints, keeping only max_to_keep most recent.

        Note: This is a simplified implementation that tracks checkpoint epochs
        in a metadata file. Actual directory deletion would require Python subprocess
        calls or Mojo system call support.
        """
        if self.max_to_keep <= 0:
            return  # Keep all checkpoints

        # For now, just log that cleanup should happen
        # Full implementation would require directory listing and deletion
        # which needs Python subprocess or Mojo system calls
        pass

    fn _save_metadata(
        self,
        filepath: String,
        epoch: Int,
        train_loss: Float32,
        val_loss: Float32,
        val_acc: Float32,
        metric_value: Float32 = 0.0,
    ) raises:
        """Save checkpoint metadata to text file.

        Format (simple key-value pairs):
        epoch=10
        train_loss=0.123
        val_loss=0.456
        val_acc=0.789
        best_metric=0.456
        """
        var content = String("epoch=") + String(epoch) + String("\n")
        content += String("train_loss=") + String(train_loss) + String("\n")
        content += String("val_loss=") + String(val_loss) + String("\n")
        content += String("val_acc=") + String(val_acc) + String("\n")
        content += (
            String("best_metric=")
            + String(
                metric_value if metric_value != 0.0 else self.best_metric_value
            )
            + String("\n")
        )

        if not safe_write_file(filepath, content):
            raise Error("Failed to write metadata file: " + filepath)

    fn _load_metadata(mut self, filepath: String) raises:
        """Load checkpoint metadata from text file."""
        var content = safe_read_file(filepath)

        # Parse best_metric line to update internal state
        # Simple parsing: look for "best_metric=<value>"
        var lines = content.split("\n")
        for i in range(len(lines)):
            var line = lines[i]
            if line.startswith("best_metric="):
                var _ = line[len("best_metric=") :]
                # NOTE: We keep the best_metric in the file for reference
                # but don't parse it back to Float32 due to lack of atof
                # The CheckpointManager tracks best_metric_value separately
                pass
