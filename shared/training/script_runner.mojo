"""Training script runner utilities.

Provides reusable components for paper training scripts including
callbacks, epoch runners, and display utilities.

Example:
    ```mojo
    from shared.training.script_runner import (
        TrainingCallbacks,
        print_training_header,
        print_dataset_info,
    )

    # Print training configuration
    print_training_header("LeNet-5", 100, 32, 0.01)

    # Create callbacks for progress reporting
    var callbacks = TrainingCallbacks(verbose=True, print_frequency=10)
    callbacks.on_epoch_start(0)
    callbacks.on_epoch_end(0, 0.5)
    ```
"""

from python import PythonObject


struct TrainingCallbacks(Copyable, Movable):
    """Callbacks for training loop events.

    Provides hooks for monitoring training progress with configurable
    verbosity and print frequency.

    Attributes:
        verbose: Whether to print progress information.
        print_frequency: Print progress every N batches.
    """

    var verbose: Bool
    var print_frequency: Int

    fn __init__(out self, verbose: Bool = True, print_frequency: Int = 10):
        """Initialize training callbacks.

        Args:
            verbose: Whether to print progress (default: True).
            print_frequency: Print every N batches (default: 10).
        """
        self.verbose = verbose
        self.print_frequency = print_frequency

    fn on_epoch_start(self, epoch: Int):
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        if self.verbose:
            print("Epoch", epoch + 1, "starting...")

    fn on_epoch_end(self, epoch: Int, loss: Float32):
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            loss: Average loss for the epoch.
        """
        if self.verbose:
            print("Epoch", epoch + 1, "completed. Loss:", loss)

    fn on_batch_end(self, epoch: Int, batch: Int, loss: Float32):
        """Called at the end of each batch.

        Args:
            epoch: Current epoch number (0-indexed).
            batch: Current batch number (0-indexed).
            loss: Loss for the batch.
        """
        if self.verbose and batch % self.print_frequency == 0:
            print("  Batch", batch, "loss:", loss)


fn run_epoch_with_batches(
    py_loader: PythonObject,
    callbacks: TrainingCallbacks,
) raises -> Float32:
    """Run one training epoch with batch processing.

    Args:
        py_loader: Python data loader providing batches.
        callbacks: Training callbacks for progress reporting.

    Returns:
        Average loss for the epoch.

    Note:
        Currently returns 0.0 as placeholder until Python/Mojo
        data loader integration is complete (Track 4 initiative). Tracked in #3076 (parent: #3059).
    """
    # Blocked: Track 4 (Python↔Mojo interop) - see #3076 (parent: #3059)
    # Full implementation requires Python data loader integration
    _ = py_loader
    _ = callbacks
    return Float32(0.0)


fn print_training_header(
    model_name: String,
    num_epochs: Int,
    batch_size: Int,
    learning_rate: Float32,
):
    """Print training configuration header.

    Displays a formatted header with training hyperparameters
    at the start of a training run.

    Args:
        model_name: Name of the model being trained.
        num_epochs: Total number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate.
    """
    print("=" * 60)
    print("Training:", model_name)
    print("=" * 60)
    print("Epochs:       ", num_epochs)
    print("Batch Size:   ", batch_size)
    print("Learning Rate:", learning_rate)
    print("=" * 60)


fn print_dataset_info(
    dataset_name: String,
    train_size: Int,
    test_size: Int,
    num_classes: Int,
):
    """Print dataset information.

    Displays a summary of dataset statistics.

    Args:
        dataset_name: Name of the dataset.
        train_size: Number of training samples.
        test_size: Number of test samples.
        num_classes: Number of output classes.
    """
    print("Dataset:", dataset_name)
    print("  Train samples:", train_size)
    print("  Test samples: ", test_size)
    print("  Classes:      ", num_classes)
