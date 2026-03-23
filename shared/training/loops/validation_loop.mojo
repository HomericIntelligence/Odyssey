"""Validation loop implementation.

Gradient-free model evaluation for validation and testing

Validation Loop (#313-317):
- #314: Gradient-free evaluation
- #315: Metric aggregation
- #316: Memory-efficient implementation

Design principles:
- No gradient computation (evaluation mode)
- Metric aggregation across full validation set
- Support for full and subset validation
- Memory efficiency (no gradient storage)
"""

from collections import List
from shared.tensor.any_tensor import AnyTensor
from shared.training.metrics import AccuracyMetric, LossTracker, ConfusionMatrix
from shared.training.trainer_interface import (
    DataLoader,
    DataBatch,
    TrainingMetrics,
)


fn validation_step(
    model_forward: fn (AnyTensor) raises -> AnyTensor,
    compute_loss: fn (AnyTensor, AnyTensor) raises -> AnyTensor,
    data: AnyTensor,
    labels: AnyTensor,
) raises -> Float64:
    """Execute single validation step (forward pass only, no gradients).

    Args:
            model_forward: Function to compute model forward pass.
            compute_loss: Function to compute loss.
            data: Input batch data.
            labels: Target labels.

    Returns:
            Loss value for this batch.

    Raises:
            Error: If evaluation fails.
    """
    # Forward pass (no gradient tracking)
    var predictions = model_forward(data)

    # Compute loss
    var loss_tensor = compute_loss(predictions, labels)

    # Extract scalar loss
    var loss_value = Float64(loss_tensor._data.bitcast[Float32]()[0])

    return loss_value


fn validate(
    model_forward: fn (AnyTensor) raises -> AnyTensor,
    compute_loss: fn (AnyTensor, AnyTensor) raises -> AnyTensor,
    mut val_loader: DataLoader,
    compute_accuracy: Bool = True,
    compute_confusion: Bool = False,
    num_classes: Int = 10,
) raises -> Float64:
    """Run validation loop.

    Args:
            model_forward: Forward pass function.
            compute_loss: Loss computation function.
            val_loader: Validation data loader.
            compute_accuracy: Whether to compute accuracy.
            compute_confusion: Whether to compute confusion matrix.
            num_classes: Number of classes (for confusion matrix).

    Returns:
            Average validation loss.

    Raises:
            Error: If validation fails.
    """
    print("\nRunning validation...")

    var total_loss = Float64(0.0)
    var num_batches = 0

    # Setup metrics
    var accuracy_metric = AccuracyMetric()
    var loss_tracker = LossTracker(window_size=100)
    var confusion_matrix = ConfusionMatrix(num_classes=num_classes)

    # Reset dataloader
    val_loader.reset()

    # Iterate over validation batches
    while val_loader.has_next():
        var batch = val_loader.next()

        # Validation step (no gradients)
        var batch_loss = validation_step(
            model_forward, compute_loss, batch.data, batch.labels
        )

        # Update metrics
        loss_tracker.update(Float32(batch_loss))
        total_loss += batch_loss
        num_batches += 1

        # Compute predictions for accuracy/confusion
        if compute_accuracy or compute_confusion:
            var predictions = model_forward(batch.data)

            if compute_accuracy:
                accuracy_metric.update(predictions, batch.labels)

            if compute_confusion:
                confusion_matrix.update(predictions, batch.labels)

    # Compute aggregated metrics
    var avg_loss = total_loss / Float64(num_batches)

    print("Validation Results:")
    print("  Loss: " + String(avg_loss))

    if compute_accuracy:
        var accuracy = accuracy_metric.compute()
        print("  Accuracy: " + String(accuracy))

    if compute_confusion:
        var precision = confusion_matrix.get_precision()
        var recall = confusion_matrix.get_recall()
        var f1 = confusion_matrix.get_f1_score()
        print("  Per-class Precision/Recall/F1:")
        for i in range(num_classes):
            var p = precision._data.bitcast[Float64]()[i]
            var r = recall._data.bitcast[Float64]()[i]
            var f = f1._data.bitcast[Float64]()[i]
            print(
                "    Class "
                + String(i)
                + ": P="
                + String(p)
                + ", R="
                + String(r)
                + ", F1="
                + String(f)
            )

    return avg_loss


struct ValidationLoop:
    """Validation loop coordinator.

    Manages the validation process including:
    - Gradient-free forward passes
    - Metric aggregation
    - Subset validation support
    - Memory-efficient evaluation
    """

    var compute_accuracy: Bool
    var compute_confusion: Bool
    var num_classes: Int

    fn __init__(
        out self,
        compute_accuracy: Bool = True,
        compute_confusion: Bool = False,
        num_classes: Int = 10,
    ):
        """Initialize validation loop.

        Args:
            compute_accuracy: Whether to compute accuracy.
            compute_confusion: Whether to compute confusion matrix.
            num_classes: Number of classes (for confusion matrix).
        """
        self.compute_accuracy = compute_accuracy
        self.compute_confusion = compute_confusion
        self.num_classes = num_classes

    fn run(
        self,
        model_forward: fn (AnyTensor) raises -> AnyTensor,
        compute_loss: fn (AnyTensor, AnyTensor) raises -> AnyTensor,
        mut val_loader: DataLoader,
        mut metrics: TrainingMetrics,
    ) raises -> Float64:
        """Run validation loop.

        Args:
            model_forward: Forward pass function.
            compute_loss: Loss computation function.
            val_loader: Validation data loader.
            metrics: Training metrics to update.

        Returns:
            Validation loss.

        Raises:
            Error: If validation fails.
        """
        var val_loss = validate(
            model_forward,
            compute_loss,
            val_loader,
            self.compute_accuracy,
            self.compute_confusion,
            self.num_classes,
        )

        # Compute accuracy over the full validation set if enabled
        var val_accuracy = Float64(0.0)
        if self.compute_accuracy:
            var accuracy_metric = AccuracyMetric()
            val_loader.reset()
            while val_loader.has_next():
                var batch = val_loader.next()
                var predictions = model_forward(batch.data)
                accuracy_metric.update(predictions, batch.labels)
            val_accuracy = accuracy_metric.compute()

        # Update metrics with computed accuracy
        metrics.update_val_metrics(val_loss, val_accuracy)

        return val_loss

    fn run_subset(
        self,
        model_forward: fn (AnyTensor) raises -> AnyTensor,
        compute_loss: fn (AnyTensor, AnyTensor) raises -> AnyTensor,
        mut val_loader: DataLoader,
        max_batches: Int,
        mut metrics: TrainingMetrics,
    ) raises -> Float64:
        """Run validation on subset of data.

        Useful for quick validation checks during training.

        Args:
            model_forward: Forward pass function.
            compute_loss: Loss computation function.
            val_loader: Validation data loader (mutable for iteration).
            max_batches: Maximum number of batches to evaluate.
            metrics: Training metrics to update.

        Returns:
            Validation loss.

        Raises:
            Error: If validation fails.
        """
        print(
            "\nRunning subset validation (max "
            + String(max_batches)
            + " batches)..."
        )

        var total_loss = Float64(0.0)
        var num_batches = 0

        var loss_tracker = LossTracker(window_size=max_batches)
        var accuracy_metric = AccuracyMetric()

        val_loader.reset()

        while val_loader.has_next() and num_batches < max_batches:
            var batch = val_loader.next()

            var batch_loss = validation_step(
                model_forward, compute_loss, batch.data, batch.labels
            )

            loss_tracker.update(Float32(batch_loss))
            total_loss += batch_loss
            num_batches += 1

            # Compute predictions for accuracy if enabled
            if self.compute_accuracy:
                var predictions = model_forward(batch.data)
                accuracy_metric.update(predictions, batch.labels)

        var avg_loss = Float64(0.0)
        if num_batches > 0:
            avg_loss = total_loss / Float64(num_batches)
        else:
            print("Warning: No validation batches available")

        print("  Subset Validation Loss: " + String(avg_loss))

        # Compute accuracy over the subset if enabled
        var subset_accuracy = Float64(0.0)
        if self.compute_accuracy:
            subset_accuracy = accuracy_metric.compute()

        metrics.update_val_metrics(avg_loss, subset_accuracy)

        return avg_loss
