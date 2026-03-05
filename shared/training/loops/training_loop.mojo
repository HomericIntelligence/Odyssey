"""Training loop implementation.

Core iteration logic for model training with forward pass, backward pass,
and weight updates. Consolidates common training patterns from all examples.

Training Loop (#308-312):
- #309: Forward/backward/update iteration
- #310: Gradient management and zeroing
- #311: Metric tracking and callbacks

Common Patterns (Consolidation):
- Epoch iteration with progress reporting
- Batch processing with configurable batch size
- Loss computation and tracking
- Optimizer integration (SGD with momentum, Adam, etc.)
- Evaluation metrics and callbacks

Design principles:
- Clear separation: forward, backward, update steps
- Proper gradient management (zero before backward)
- Metric tracking at batch and epoch level
- Callback integration at all lifecycle points
- Support for custom batch processing functions
"""

from collections import List
from shared.core.extensor import ExTensor
from shared.training.metrics import AccuracyMetric, LossTracker
from shared.training.trainer_interface import (
    DataLoader,
    DataBatch,
    TrainingMetrics,
)


fn training_step(
    model_forward: fn (ExTensor) raises -> ExTensor,
    compute_loss: fn (ExTensor, ExTensor) raises -> ExTensor,
    optimizer_step: fn () raises -> None,
    zero_gradients: fn () raises -> None,
    data: ExTensor,
    labels: ExTensor,
) raises -> Float64:
    """Execute single training step (forward, backward, update).

    Args:
            model_forward: Function to compute model forward pass.
            compute_loss: Function to compute loss.
            optimizer_step: Function to update weights.
            zero_gradients: Function to zero gradients.
            data: Input batch data.
            labels: Target labels.

    Returns:
            Loss value for this batch.

    Raises:
            Error: If any step fails.

    Note:
        The backward pass is invoked implicitly via optimizer_step(). A full
        implementation would call loss_tensor.backward() directly once ExTensor
        supports automatic differentiation.
    """
    # Zero gradients from previous step
    zero_gradients()

    # Forward pass
    var predictions = model_forward(data)

    # Compute loss
    var loss_tensor = compute_loss(predictions, labels)

    # Extract scalar loss (assume first element)
    var loss_value = Float64(loss_tensor._data.bitcast[Float32]()[0])

    # Backward pass (implicit through optimizer_step)
    # Update weights
    optimizer_step()

    return loss_value


fn train_one_epoch(
    model_forward: fn (ExTensor) raises -> ExTensor,
    compute_loss: fn (ExTensor, ExTensor) raises -> ExTensor,
    optimizer_step: fn () raises -> None,
    zero_gradients: fn () raises -> None,
    mut train_loader: DataLoader,
    mut metrics: TrainingMetrics,
    log_interval: Int = 10,
) raises:
    """Train for one epoch.

    Args:
            model_forward: Forward pass function.
            compute_loss: Loss computation function.
            optimizer_step: Weight update function.
            zero_gradients: Gradient zeroing function.
            train_loader: Training data loader.
            metrics: Training metrics to update.
            log_interval: Log metrics every N batches.

    Raises:
            Error: If training fails.
    """
    var epoch_loss = Float64(0.0)
    var num_batches = 0

    # Setup metrics for epoch
    var loss_tracker = LossTracker(window_size=log_interval)

    # Reset dataloader
    train_loader.reset()

    # Iterate over batches
    while train_loader.has_next():
        var batch = train_loader.next()

        # Training step
        var batch_loss = training_step(
            model_forward,
            compute_loss,
            optimizer_step,
            zero_gradients,
            batch.data,
            batch.labels,
        )

        # Update metrics
        loss_tracker.update(Float32(batch_loss))
        epoch_loss += batch_loss
        num_batches += 1

        # Update batch counter in metrics
        metrics.current_batch = num_batches

        # Log progress
        if num_batches % log_interval == 0:
            var avg_loss = loss_tracker.get_average()
            print(
                "  Batch "
                + String(num_batches)
                + "/"
                + String(train_loader.num_batches)
                + " - Loss: "
                + String(avg_loss)
            )

    # Update epoch metrics
    var epoch_avg_loss = epoch_loss / Float64(num_batches)
    metrics.update_train_metrics(epoch_avg_loss, 0.0)  # Accuracy placeholder

    print(
        "Epoch "
        + String(metrics.current_epoch)
        + " complete - Avg Loss: "
        + String(epoch_avg_loss)
    )


struct TrainingLoop:
    """Training loop coordinator.

    Manages the training process including:
    - Batch iteration
    - Forward/backward passes
    - Gradient updates
    - Metric tracking
    - Progress logging

    Consolidates common training patterns from all examples:
    - Epoch iteration with configurable batch size
    - Custom batch processing via compute_batch_loss callback
    - Automatic progress reporting
    - Evaluation support with custom eval function

    Example usage (matches examples pattern):
        var loop = TrainingLoop(log_interval=100)
        loop.run_epoch(
            train_images, train_labels, batch_size=128,
            compute_batch_loss=my_batch_fn,
            total_epochs=100, current_epoch=1
        )
    """

    var log_interval: Int
    var clip_gradients: Bool
    var max_grad_norm: Float64

    fn __init__(
        out self,
        log_interval: Int = 10,
        clip_gradients: Bool = False,
        max_grad_norm: Float64 = 1.0,
    ):
        """Initialize training loop.

        Args:
            log_interval: Log metrics every N batches.
            clip_gradients: Whether to clip gradients.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        self.log_interval = log_interval
        self.clip_gradients = clip_gradients
        self.max_grad_norm = max_grad_norm

    fn run_epoch_manual(
        self,
        train_data: ExTensor,
        train_labels: ExTensor,
        batch_size: Int,
        compute_batch_loss: fn (ExTensor, ExTensor) raises -> Float32,
        epoch: Int,
        total_epochs: Int,
    ) raises -> Float32:
        """Run one epoch with manual batch processing.

        This method consolidates the common training pattern used across
        all examples (AlexNet, VGG16, LeNet, ResNet).

        Args:
            train_data: Training input data (batch_size dimension first).
            train_labels: Training labels.
            batch_size: Mini-batch size.
            compute_batch_loss: Function to process one batch and return loss.
                               Signature: fn(batch_data: ExTensor, batch_labels: ExTensor) -> Float32.
            epoch: Current epoch number (1-indexed).
            total_epochs: Total number of epochs.

        Returns:
            Average loss for the epoch.

        Raises:
            Error: If training fails.
        """
        var num_samples = train_data.shape()[0]
        var num_batches = (num_samples + batch_size - 1) // batch_size
        var total_loss = Float32(0.0)

        print("Epoch [", epoch, "/", total_epochs, "]")

        for batch_idx in range(num_batches):
            var start_idx = batch_idx * batch_size
            var end_idx = min(start_idx + batch_size, num_samples)

            # Extract batch slice using ExTensor.slice()
            var batch_data = train_data.slice(start_idx, end_idx, axis=0)
            var batch_labels = train_labels.slice(start_idx, end_idx, axis=0)

            # Compute loss on batch
            var batch_loss = compute_batch_loss(batch_data, batch_labels)
            total_loss += batch_loss

            # Print progress every log_interval batches
            if (batch_idx + 1) % self.log_interval == 0:
                var avg_loss = total_loss / Float32(batch_idx + 1)
                print(
                    "  Batch [",
                    batch_idx + 1,
                    "/",
                    num_batches,
                    "] - Loss: ",
                    avg_loss,
                )

        var avg_loss = total_loss / Float32(num_batches)
        print("  Average Loss: ", avg_loss)

        return avg_loss

    fn run_epoch(
        self,
        model_forward: fn (ExTensor) raises -> ExTensor,
        compute_loss: fn (ExTensor, ExTensor) raises -> ExTensor,
        optimizer_step: fn () raises -> None,
        zero_gradients: fn () raises -> None,
        mut train_loader: DataLoader,
        mut metrics: TrainingMetrics,
    ) raises:
        """Run one training epoch.

        Args:
            model_forward: Forward pass function.
            compute_loss: Loss computation function.
            optimizer_step: Weight update function.
            zero_gradients: Gradient zeroing function.
            train_loader: Training data loader.
            metrics: Training metrics to update.

        Raises:
            Error: If training fails.
        """
        train_one_epoch(
            model_forward,
            compute_loss,
            optimizer_step,
            zero_gradients,
            train_loader,
            metrics,
            self.log_interval,
        )

    fn run(
        self,
        model_forward: fn (ExTensor) raises -> ExTensor,
        compute_loss: fn (ExTensor, ExTensor) raises -> ExTensor,
        optimizer_step: fn () raises -> None,
        zero_gradients: fn () raises -> None,
        mut train_loader: DataLoader,
        num_epochs: Int,
        mut metrics: TrainingMetrics,
    ) raises:
        """Run complete training loop.

        Args:
            model_forward: Forward pass function.
            compute_loss: Loss computation function.
            optimizer_step: Weight update function.
            zero_gradients: Gradient zeroing function.
            train_loader: Training data loader (mutable for iteration).
            num_epochs: Number of epochs to train.
            metrics: Training metrics to update.

        Raises:
            Error: If training fails.
        """
        print("\nStarting training for " + String(num_epochs) + " epochs...")
        print("=" * 50)

        for epoch in range(num_epochs):
            metrics.current_epoch = epoch
            metrics.reset_epoch()

            print("\nEpoch " + String(epoch + 1) + "/" + String(num_epochs))
            print("-" * 50)

            self.run_epoch(
                model_forward,
                compute_loss,
                optimizer_step,
                zero_gradients,
                train_loader,
                metrics,
            )

        print("\n" + "=" * 50)
        print("Training complete!")
        metrics.print_summary()
