"""Trainer interface and contract definition.

Defines the abstract interface that all trainers must implement,
establishing a consistent API for training, validation, and testing.

Trainer Interface (#303-307):
- #304: Trainer trait definition
- #305: Configuration structures
- #306: State management interfaces

Design principles:
- Trait-based polymorphism for flexible implementations
- Composition over inheritance
- Explicit configuration objects
- Clear separation of concerns (train, validate, test)
"""

from collections import List
from shared.core import ExTensor


struct TrainerConfig(Copyable, Movable):
    """Configuration for trainer behavior.

    Centralizes all training hyperparameters and settings

    Mixed Precision Training:
        The mixed precision infrastructure (GradientScaler, master weights, etc.)
        is fully implemented and tested. Configuration options are available but
        automatic integration with the training loop requires implementing the
        backward pass (autograd)

        Once autograd is available, setting use_mixed_precision=True will enable:
        - Automatic gradient scaling to prevent FP16 underflow
        - Dynamic loss scaling with overflow detection
        - Master weights in FP32 for optimizer precision
        - 2-3x speedup with minimal accuracy loss

        Current Status:
        - ✅ GradientScaler fully implemented and tested
        - ✅ Master weight conversion utilities available
        - ✅ Gradient clipping with validation
        - ⚠️  Automatic training loop integration pending autograd

        Example (when autograd is available):
            var config = TrainerConfig(
                use_mixed_precision=True,
                precision_dtype=DType.float16,
                loss_scale=65536.0,
                gradient_clip_norm=1.0
            )
    """

    var num_epochs: Int
    """Number of epochs to train for."""
    var batch_size: Int
    """Batch size for training."""
    var learning_rate: Float64
    """Learning rate for optimizer."""
    var log_interval: Int
    """Log metrics every N batches."""
    var validate_interval: Int
    """Validate every N epochs (0 = every epoch)."""
    var save_checkpoints: Bool
    """Whether to save model checkpoints."""
    var checkpoint_interval: Int
    """Save checkpoint every N epochs."""

    # Learning rate scheduling
    var use_scheduler: Bool
    """Whether to use learning rate scheduler."""
    var scheduler_type: String
    """Type of scheduler (step, cosine, exponential, warmup, etc.)."""

    # Mixed precision training settings
    var use_mixed_precision: Bool
    """Enable FP16/BF16 training."""
    var precision_dtype: DType
    """DType for mixed precision (float16 or float32, bfloat16 when available)."""
    var loss_scale: Float32
    """Initial loss scale for gradient scaling (default: 65536.0)."""
    var gradient_clip_norm: Float32
    """Clip gradients by norm (0 = no clipping)."""

    fn __init__(
        out self,
        num_epochs: Int = 10,
        batch_size: Int = 32,
        learning_rate: Float64 = 0.001,
        log_interval: Int = 10,
        validate_interval: Int = 1,
        save_checkpoints: Bool = False,
        checkpoint_interval: Int = 5,
        use_scheduler: Bool = False,
        scheduler_type: String = "none",
        use_mixed_precision: Bool = False,
        precision_dtype: DType = DType.float32,
        loss_scale: Float32 = 65536.0,
        gradient_clip_norm: Float32 = 0.0,
    ):
        """Initialize trainer configuration with defaults.

        Args:
            num_epochs: Number of epochs to train for.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            log_interval: Log metrics every N batches.
            validate_interval: Validate every N epochs (0 = every epoch).
            save_checkpoints: Whether to save checkpoints.
            checkpoint_interval: Save checkpoint every N epochs.
            use_scheduler: Whether to use learning rate scheduler.
            scheduler_type: Type of scheduler to use (none, step, cosine, warmup, etc).
            use_mixed_precision: Enable FP16/BF16 training.
            precision_dtype: DType for mixed precision training.
            loss_scale: Initial loss scale for gradient scaling.
            gradient_clip_norm: Clip gradients by norm (0 = no clipping).
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.validate_interval = validate_interval
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.use_mixed_precision = use_mixed_precision
        self.precision_dtype = precision_dtype
        self.loss_scale = loss_scale
        self.gradient_clip_norm = gradient_clip_norm


struct TrainingMetrics(Copyable, Movable):
    """Metrics collected during training.

    Stores current and historical metrics for analysis
    """

    var current_epoch: Int
    """Current epoch number."""
    var current_batch: Int
    """Current batch number within epoch."""
    var total_batches: Int
    """Total number of batches in epoch."""
    var train_loss: Float64
    """Training loss value."""
    var train_accuracy: Float64
    """Training accuracy value."""
    var val_loss: Float64
    """Validation loss value."""
    var val_accuracy: Float64
    """Validation accuracy value."""
    var best_val_loss: Float64
    """Best validation loss seen so far."""
    var best_val_accuracy: Float64
    """Best validation accuracy seen so far."""
    var best_epoch: Int
    """Epoch with best validation loss."""

    fn __init__(out self):
        """Initialize training metrics with defaults."""
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.train_loss = 0.0
        self.train_accuracy = 0.0
        self.val_loss = 0.0
        self.val_accuracy = 0.0
        self.best_val_loss = 1e10  # Initialize to very large value
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    fn update_train_metrics(mut self, loss: Float64, accuracy: Float64):
        """Update training metrics for current batch.

        Args:
            loss: Current batch loss.
            accuracy: Current batch accuracy.
        """
        self.train_loss = loss
        self.train_accuracy = accuracy

    fn update_val_metrics(mut self, loss: Float64, accuracy: Float64):
        """Update validation metrics and track best results.

        Args:
            loss: Validation loss.
            accuracy: Validation accuracy.
        """
        self.val_loss = loss
        self.val_accuracy = accuracy

        # Track best validation results
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_val_accuracy = accuracy
            self.best_epoch = self.current_epoch

    fn reset_epoch(mut self):
        """Reset epoch-level metrics."""
        self.current_batch = 0
        self.train_loss = 0.0
        self.train_accuracy = 0.0

    fn print_summary(self):
        """Print training metrics summary."""
        print("\nTraining Metrics Summary:")
        print("-" * 50)
        print("Current Epoch: " + String(self.current_epoch))
        print("Train Loss: " + String(self.train_loss))
        print("Train Accuracy: " + String(self.train_accuracy))
        print("Val Loss: " + String(self.val_loss))
        print("Val Accuracy: " + String(self.val_accuracy))
        print(
            "Best Val Loss: "
            + String(self.best_val_loss)
            + " (epoch "
            + String(self.best_epoch)
            + ")"
        )
        print(
            "Best Val Accuracy: "
            + String(self.best_val_accuracy)
            + " (epoch "
            + String(self.best_epoch)
            + ")"
        )
        print("-" * 50)


trait Trainer:
    """Abstract interface for all trainer implementations.

    Defines the contract that all trainers must follow, ensuring
    consistent API across different training strategies

    Key methods:
    - train(): Execute training loop
    - validate(): Evaluate on validation set
    - test(): Evaluate on test set
    - fit(): Convenience method combining train+validate

    Lifecycle hooks (via callbacks):
    - on_train_begin()
    - on_train_end()
    - on_epoch_begin()
    - on_epoch_end()
    - on_batch_begin()
    - on_batch_end()
    - on_validation_begin()
    - on_validation_end()
    """

    fn train(mut self, num_epochs: Int) raises:
        """Execute training loop for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train.

        Raises:
            Error: If training fails.
        """
        ...

    fn validate(mut self) raises -> Float64:
        """Evaluate model on validation set.

        Returns:
            Validation loss.

        Raises:
            Error: If validation fails.
        """
        ...

    fn fit(mut self, num_epochs: Int, validate_every: Int = 1) raises:
        """Train model with periodic validation.

        Args:
            num_epochs: Number of epochs to train.
            validate_every: Validate every N epochs (default=1).

        Raises:
            Error: If training or validation fails.
        """
        ...


struct DataBatch(Copyable, Movable):
    """Single batch of training/validation data.

    Represents a mini-batch with input features and labels
    """

    var data: ExTensor
    """Input features [batch_size, feature_dim]."""
    var labels: ExTensor
    """Labels [batch_size] or [batch_size, num_classes]."""
    var batch_size: Int
    """Batch size."""

    fn __init__(out self, var data: ExTensor, var labels: ExTensor):
        """Initialize data batch.

        Args:
            data: Input features tensor (ownership transferred).
            labels: Labels tensor (ownership transferred).
        """
        self.data = data^
        self.labels = labels^
        self.batch_size = self.data.shape()[0]


struct DataLoader(Copyable, Movable):
    """Simple data loader for batching.

    Provides iteration over dataset in batches
    NOTE: This is a minimal implementation for testing
    Production code should use proper data loading infrastructure
    """

    var data: ExTensor
    """Full dataset features."""
    var labels: ExTensor
    """Full dataset labels."""
    var batch_size: Int
    """Batch size for iteration."""
    var num_samples: Int
    """Total number of samples."""
    var num_batches: Int
    """Total number of batches."""
    var current_batch: Int
    """Current batch index."""

    fn __init__(
        out self, var data: ExTensor, var labels: ExTensor, batch_size: Int
    ):
        """Initialize data loader.

        Args:
            data: Full dataset features (ownership transferred).
            labels: Full dataset labels (ownership transferred).
            batch_size: Batch size.
        """
        self.data = data^
        self.labels = labels^
        self.batch_size = batch_size
        self.num_samples = self.data.shape()[0]
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.current_batch = 0

    fn reset(mut self):
        """Reset loader to beginning."""
        self.current_batch = 0

    fn has_next(self) -> Bool:
        """Check if more batches available.

        Returns:
            True if more batches available, False otherwise.
        """
        return self.current_batch < self.num_batches

    fn next(mut self) raises -> DataBatch:
        """Get next batch.

        Returns:
            Next data batch.

        Raises:
            Error: If no more batches available.
        """
        if not self.has_next():
            raise Error("No more batches available")

        var start_idx = self.current_batch * self.batch_size
        var end_idx = min(start_idx + self.batch_size, self.num_samples)
        var actual_batch_size = end_idx - start_idx

        # Extract batch slice
        var batch_data_shape = List[Int]()
        batch_data_shape.append(actual_batch_size)
        batch_data_shape.append(self.data.shape()[1])
        var batch_data = ExTensor(batch_data_shape, self.data.dtype())

        var batch_labels_shape = List[Int]()
        batch_labels_shape.append(actual_batch_size)
        var batch_labels = ExTensor(batch_labels_shape, self.labels.dtype())

        # Tensor slicing is implemented in ExTensor.slice() and __getitem__()
        # NOTE: Python data loader integration blocked by Track 4 (Python↔Mojo interop).
        # Tracked in #3076 (parent: #3059). Placeholder tensors used until Track 4 is ready.

        self.current_batch += 1

        return DataBatch(batch_data, batch_labels)


fn create_simple_dataloader(
    var data: ExTensor, var labels: ExTensor, batch_size: Int
) raises -> DataLoader:
    """Create a simple dataloader for training.

    Args:
        data: Full dataset features (ownership transferred).
        labels: Full dataset labels (ownership transferred).
        batch_size: Batch size.

    Returns:
        DataLoader instance.
    """
    return DataLoader(data^, labels^, batch_size)
