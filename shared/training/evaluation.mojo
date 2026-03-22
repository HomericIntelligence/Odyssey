"""Generic model evaluation utilities for classification tasks.

Consolidates duplicate evaluate_model implementations from example inference scripts
into a shared, reusable module supporting multiple evaluation patterns.

Patterns consolidated from:
- examples/lenet-emnist/inference.mojo (evaluate_test_set)
- examples/alexnet-cifar10/inference.mojo (evaluate_model with top-1/top-5)
- examples/resnet18-cifar10/inference.mojo (evaluate_model with per-class stats)
- examples/vgg16-cifar10/inference.mojo (compute_test_accuracy)

Features:
- Generic EvaluationResult struct with accuracy, per-class stats, and raw counts
- Generic evaluate_model function supporting any model with forward() method
- Support for both batched and single-sample evaluation
- Automatic per-class accuracy computation
- Top-k accuracy computation
- Detailed evaluation output with progress tracking

"""

from shared.core import AnyTensor
from shared.core.traits import Model
from shared.data.batch_utils import extract_batch_pair, compute_num_batches
from collections import List


# ============================================================================
# EvaluationResult Struct
# ============================================================================


struct EvaluationResult(Copyable, Movable):
    """Result from evaluating a model on a dataset.

    Consolidates evaluation metrics from different example patterns into a
    single, comprehensive result structure.

    Attributes:
        accuracy: Overall accuracy as a fraction in [0.0, 1.0].
        num_correct: Total number of correct predictions.
        num_total: Total number of samples evaluated.
        correct_per_class: Per-class correct prediction counts (optional).
        total_per_class: Per-class total sample counts (optional).
        top_k_accuracy: Top-k accuracy (optional, defaults to None).
    """

    var accuracy: Float32
    """Overall accuracy as a fraction in [0.0, 1.0]."""
    var num_correct: Int
    """Total number of correct predictions."""
    var num_total: Int
    """Total number of samples evaluated."""
    var correct_per_class: List[Int]
    """Per-class correct prediction counts."""
    var total_per_class: List[Int]
    """Per-class total sample counts."""

    fn __init__(
        out self,
        accuracy: Float32,
        num_correct: Int,
        num_total: Int,
        correct_per_class: List[Int] = List[Int](),
        total_per_class: List[Int] = List[Int](),
    ):
        """Initialize EvaluationResult.

        Args:
            accuracy: Overall accuracy as fraction [0.0, 1.0].
            num_correct: Total correct predictions.
            num_total: Total samples evaluated.
            correct_per_class: Per-class correct counts (optional).
            total_per_class: Per-class total counts (optional).
        """
        self.accuracy = accuracy
        self.num_correct = num_correct
        self.num_total = num_total
        self.correct_per_class = correct_per_class.copy()
        self.total_per_class = total_per_class.copy()


# ============================================================================
# Generic Evaluation Functions
# ============================================================================


fn evaluate_model[
    M: Model
](
    mut model: M,
    images: AnyTensor,
    labels: AnyTensor,
    batch_size: Int = 100,
    num_classes: Int = 10,
    verbose: Bool = True,
) raises -> EvaluationResult:
    """Generically evaluate a model on a dataset with per-class statistics.

    Consolidates the duplicate evaluate_model patterns from examples into a single
    generic function that works with any model type M that implements forward().

    Parameters:
        M: Model type that must implement forward(images: AnyTensor) -> AnyTensor.

    Args:
        model: Model to evaluate (must have forward() method).
        images: Input images of shape (num_samples, ...).
        labels: Ground truth labels of shape (num_samples,).
        batch_size: Batch size for evaluation (default: 100).
        num_classes: Number of classification classes (default: 10).
        verbose: Print progress during evaluation (default: True).

    Returns:
        EvaluationResult with accuracy, per-class stats, and total counts.

    Raises:
        Error: If batch sizes don't match or shapes are incompatible.

    Examples:
    ```
        Generic evaluation with any model

        var result = evaluate_model(model, test_images, test_labels, batch_size=100)
        print("Accuracy: ", result.accuracy)

        Access per-class stats

        for i in range(num_classes):
            var class_acc = Float32(result.correct_per_class[i]) / Float32(result.total_per_class[i])
            print("Class ", i, " accuracy: ", class_acc)
    ```
    """
    var num_samples = images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)

    var total_correct = 0
    var correct_per_class = List[Int](capacity=num_classes)
    var total_per_class = List[Int](capacity=num_classes)

    # Initialize per-class counters
    for _ in range(num_classes):
        correct_per_class.append(0)
        total_per_class.append(0)

    if verbose:
        print(
            "Evaluating on "
            + String(num_samples)
            + " samples ("
            + String(num_batches)
            + " batches)..."
        )

    # Evaluate in batches
    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size

        # Extract mini-batch
        var batch_pair = extract_batch_pair(
            images, labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]
        var current_batch_size = batch_images.shape()[0]

        # Forward pass (inference mode - no training state)
        var logits = model.forward(batch_images)

        # Compute batch accuracy by argmax
        var batch_correct = 0
        var logits_data = logits._data.bitcast[Float32]()

        for i in range(current_batch_size):
            # Find argmax (predicted class)
            var pred_class = 0
            var max_logit = logits_data[i * num_classes]

            for j in range(1, num_classes):
                var logit_val = logits_data[i * num_classes + j]
                if logit_val > max_logit:
                    max_logit = logit_val
                    pred_class = j

            # Get true label
            var true_class = Int(batch_labels[i])

            # Update counters
            total_per_class[true_class] += 1
            if pred_class == true_class:
                batch_correct += 1
                correct_per_class[true_class] += 1

        total_correct += batch_correct

        # Print progress
        if verbose and (batch_idx + 1) % 20 == 0:
            var progress = Float32(batch_idx + 1) / Float32(num_batches) * 100.0
            var current_acc = (
                Float32(total_correct)
                / Float32((batch_idx + 1) * batch_size)
                * 100.0
            )
            print(
                "  Progress: "
                + String(progress)
                + "% - Current Acc: "
                + String(current_acc)
                + "%"
            )

    var overall_accuracy = Float32(total_correct) / Float32(num_samples)

    if verbose:
        print("Evaluation complete!")
        print()

    return EvaluationResult(
        overall_accuracy,
        total_correct,
        num_samples,
        correct_per_class^,
        total_per_class^,
    )


fn evaluate_model_simple[
    M: Model
](
    mut model: M,
    images: AnyTensor,
    labels: AnyTensor,
    batch_size: Int = 100,
    num_classes: Int = 10,
    verbose: Bool = True,
) raises -> Float32:
    """Simplified evaluation returning only overall accuracy.

    Lightweight variant of evaluate_model for cases where only overall accuracy
    is needed, without per-class statistics.

    Parameters:
        M: Model type that must implement forward(images: AnyTensor) -> AnyTensor.

    Args:
        model: Model to evaluate (must have forward() method).
        images: Input images of shape (num_samples, ...).
        labels: Ground truth labels of shape (num_samples,).
        batch_size: Batch size for evaluation (default: 100).
        num_classes: Number of classification classes (default: 10).
        verbose: Print progress during evaluation (default: True).

    Returns:
        Overall accuracy as fraction in [0.0, 1.0].

    Raises:
        Error: If batch sizes don't match or shapes are incompatible.

    Examples:
    ```
        Simple overall accuracy

        var accuracy = evaluate_model_simple(model, test_images, test_labels)
        print("Test Accuracy: ", accuracy * 100.0, "%")
    ```
    """
    var num_samples = images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)

    var total_correct = 0

    if verbose:
        print("Evaluating on " + String(num_samples) + " samples...")

    # Evaluate in batches
    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size

        # Extract mini-batch
        var batch_pair = extract_batch_pair(
            images, labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]
        var current_batch_size = batch_images.shape()[0]

        # Forward pass
        var logits = model.forward(batch_images)

        # Compute batch accuracy by argmax
        var logits_data = logits._data.bitcast[Float32]()

        for i in range(current_batch_size):
            # Find argmax (predicted class)
            var pred_class = 0
            var max_logit = logits_data[i * num_classes]

            for j in range(1, num_classes):
                var logit_val = logits_data[i * num_classes + j]
                if logit_val > max_logit:
                    max_logit = logit_val
                    pred_class = j

            # Get true label
            var true_class = Int(batch_labels[i])

            # Update counter
            if pred_class == true_class:
                total_correct += 1

        # Print progress
        if verbose and (batch_idx + 1) % 20 == 0:
            var progress = Float32(batch_idx + 1) / Float32(num_batches) * 100.0
            print("  Progress: " + String(progress) + "%")

    var overall_accuracy = Float32(total_correct) / Float32(num_samples)

    if verbose:
        print("Evaluation complete!")
        print()

    return overall_accuracy


fn evaluate_topk[
    M: Model
](
    mut model: M,
    images: AnyTensor,
    labels: AnyTensor,
    k: Int = 5,
    batch_size: Int = 100,
    num_classes: Int = 10,
    verbose: Bool = True,
) raises -> Float32:
    """Evaluate model using top-k accuracy.

    Computes top-k accuracy where prediction is considered correct if the true
    label is in the top-k predictions.

    Parameters:
        M: Model type that must implement forward(images: AnyTensor) -> AnyTensor.

    Args:
        model: Model to evaluate.
        images: Input images of shape (num_samples, ...).
        labels: Ground truth labels of shape (num_samples,).
        k: Number of top predictions to consider (default: 5).
        batch_size: Batch size for evaluation (default: 100).
        num_classes: Number of classification classes (default: 10).
        verbose: Print progress during evaluation (default: True).

    Returns:
        Top-k accuracy as fraction in [0.0, 1.0].

    Raises:
        Error: If k > num_classes or shapes are incompatible.

    Examples:
    ```
        Top-5 accuracy for ImageNet-like tasks

        var top5_acc = evaluate_topk(model, test_images, test_labels, k=5)
        print("Top-5 Accuracy: ", top5_acc * 100.0, "%")
    ```
    """
    if k > num_classes:
        raise Error("evaluate_topk: k must be <= num_classes")

    var num_samples = images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)

    var total_correct = 0

    if verbose:
        print(
            "Evaluating top-"
            + String(k)
            + " accuracy on "
            + String(num_samples)
            + " samples..."
        )

    # Evaluate in batches
    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size

        # Extract mini-batch
        var batch_pair = extract_batch_pair(
            images, labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]
        var current_batch_size = batch_images.shape()[0]

        # Forward pass
        var logits = model.forward(batch_images)

        # Compute top-k accuracy
        var logits_data = logits._data.bitcast[Float32]()

        for i in range(current_batch_size):
            # Find top-k for this sample
            var true_class = Int(batch_labels[i])

            # Get logits for this sample
            var sample_logits = List[Float32]()
            for j in range(num_classes):
                sample_logits.append(logits_data[i * num_classes + j])

            # Find indices of top-k values
            var topk_found = False
            for _ in range(k):
                # Find max index and value
                var max_idx = 0
                var max_val = sample_logits[0]

                for j in range(1, num_classes):
                    if sample_logits[j] > max_val:
                        max_val = sample_logits[j]
                        max_idx = j

                # Check if this is the true class
                if max_idx == true_class:
                    topk_found = True
                    break

                # Set this value to -inf and continue
                sample_logits[max_idx] = Float32(-1e9)

            if topk_found:
                total_correct += 1

        # Print progress
        if verbose and (batch_idx + 1) % 20 == 0:
            var progress = Float32(batch_idx + 1) / Float32(num_batches) * 100.0
            print("  Progress: " + String(progress) + "%")

    var topk_accuracy = Float32(total_correct) / Float32(num_samples)

    if verbose:
        print("Top-" + String(k) + " evaluation complete!")
        print()

    return topk_accuracy
