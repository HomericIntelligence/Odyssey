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

from std.collections import List
from std.time import perf_counter_ns
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.training.metrics import AccuracyMetric, LossTracker
from projectodyssey.training.trainer_interface import (
    DataLoader,
    DataBatch,
    TrainingMetrics,
)
from projectodyssey.training.interruption import (
    WallClockTimer,
    is_shutdown_requested,
    TrainingResult,
    ShutdownReason,
)


def training_step[
    FwdFn: def(AnyTensor) raises -> AnyTensor,
    LossFn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
    OptFn: def() raises -> None,
    ZeroFn: def() raises -> None,
](
    model_forward: FwdFn,
    compute_loss: LossFn,
    optimizer_step: OptFn,
    zero_gradients: ZeroFn,
    data: AnyTensor,
    labels: AnyTensor,
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
        implementation would call loss_tensor.backward() directly once AnyTensor
        supports automatic differentiation.
    """
    # Zero gradients from previous step
    zero_gradients()

    # Forward pass
    var predictions = model_forward(data)

    # Compute loss
    var loss_tensor = compute_loss(predictions, labels)

    # Extract scalar loss (assume first element)
    var loss_value = Float64(loss_tensor.load[DType.float32](0))

    # Backward pass (implicit through optimizer_step)
    # Update weights
    optimizer_step()

    return loss_value


def train_one_epoch[
    FwdFn: def(AnyTensor) raises -> AnyTensor,
    LossFn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
    OptFn: def() raises -> None,
    ZeroFn: def() raises -> None,
](
    model_forward: FwdFn,
    compute_loss: LossFn,
    optimizer_step: OptFn,
    zero_gradients: ZeroFn,
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
    - Timeout management and graceful shutdown

    Consolidates common training patterns from all examples:
    - Epoch iteration with configurable batch size
    - Custom batch processing via compute_batch_loss callback
    - Automatic progress reporting
    - Evaluation support with custom eval function
    - Wall-clock timeout and signal-based shutdown

    Example usage (matches examples pattern):
        var loop = TrainingLoop(log_interval=100, max_wall_time_seconds=3600)
        var result = loop.run(
            model_forward, compute_loss, optimizer_step, zero_gradients,
            train_loader, 100, metrics
        )
    """

    var log_interval: Int
    var clip_gradients: Bool
    var max_grad_norm: Float64
    var max_wall_time_seconds: Int
    var checkpoint_on_interrupt: Bool
    var timer: WallClockTimer

    def __init__(
        out self,
        log_interval: Int = 10,
        clip_gradients: Bool = False,
        max_grad_norm: Float64 = 1.0,
        max_wall_time_seconds: Int = 0,
        checkpoint_on_interrupt: Bool = True,
    ):
        """Initialize training loop.

        Args:
            log_interval: Log metrics every N batches.
            clip_gradients: Whether to clip gradients.
            max_grad_norm: Maximum gradient norm for clipping.
            max_wall_time_seconds: Max wall-clock seconds (0 = no limit).
            checkpoint_on_interrupt: Save checkpoint on interrupt (default True).
        """
        self.log_interval = log_interval
        self.clip_gradients = clip_gradients
        self.max_grad_norm = max_grad_norm
        self.max_wall_time_seconds = max_wall_time_seconds
        self.checkpoint_on_interrupt = checkpoint_on_interrupt
        self.timer = WallClockTimer()

    def run_epoch_manual[
        BatchLossFn: def(AnyTensor, AnyTensor) raises -> Float32,
    ](
        self,
        train_data: AnyTensor,
        train_labels: AnyTensor,
        batch_size: Int,
        compute_batch_loss: BatchLossFn,
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
                               Signature: def(batch_data: AnyTensor, batch_labels: AnyTensor) -> Float32.
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

            # Extract batch slice using AnyTensor.slice()
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

    def run_epoch[
        FwdFn: def(AnyTensor) raises -> AnyTensor,
        LossFn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
        OptFn: def() raises -> None,
        ZeroFn: def() raises -> None,
    ](
        self,
        model_forward: FwdFn,
        compute_loss: LossFn,
        optimizer_step: OptFn,
        zero_gradients: ZeroFn,
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

    def run[
        FwdFn: def(AnyTensor) raises -> AnyTensor,
        LossFn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
        OptFn: def() raises -> None,
        ZeroFn: def() raises -> None,
    ](
        self,
        model_forward: FwdFn,
        compute_loss: LossFn,
        optimizer_step: OptFn,
        zero_gradients: ZeroFn,
        mut train_loader: DataLoader,
        num_epochs: Int,
        mut metrics: TrainingMetrics,
    ) raises -> TrainingResult:
        """Run complete training loop with timeout support.

        Args:
            model_forward: Forward pass function.
            compute_loss: Loss computation function.
            optimizer_step: Weight update function.
            zero_gradients: Gradient zeroing function.
            train_loader: Training data loader (mutable for iteration).
            num_epochs: Number of epochs to train.
            metrics: Training metrics to update.

        Returns:
            TrainingResult with termination reason and final epoch count.

        Raises:
            Error: If training fails.
        """
        print("\nStarting training for " + String(num_epochs) + " epochs...")
        if self.max_wall_time_seconds > 0:
            print("Wall-clock timeout: " + String(self.max_wall_time_seconds) + " seconds")
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

            # Check for timeout or shutdown at epoch boundary
            if self.max_wall_time_seconds > 0 and self.timer.has_elapsed(self.max_wall_time_seconds):
                print("\nTraining timeout reached after " + String(self.timer.elapsed_seconds()) + " seconds")
                return TrainingResult(
                    stopped_epoch=epoch,
                    reason=ShutdownReason.timeout(),
                    checkpoint_path="",
                    elapsed_seconds=self.timer.elapsed_seconds(),
                )

            if is_shutdown_requested():
                print("\nTraining interrupted by shutdown signal")
                return TrainingResult(
                    stopped_epoch=epoch,
                    reason=ShutdownReason.signal(),
                    checkpoint_path="",
                    elapsed_seconds=self.timer.elapsed_seconds(),
                )

        print("\n" + "=" * 50)
        print("Training complete!")
        metrics.print_summary()

        return TrainingResult(
            stopped_epoch=num_epochs - 1,
            reason=ShutdownReason.completed(),
            checkpoint_path="",
            elapsed_seconds=self.timer.elapsed_seconds(),
        )
