"""Accuracy metrics for classification tasks.

Provides top-1, top-k, and per-class accuracy metrics with incremental
accumulation for efficient evaluation during training and testing.

Metric types:
- Top-1 accuracy: Exact match between prediction and target
- Top-k accuracy: Target in k-best predictions
- Per-class accuracy: Class-wise accuracy breakdown

Type support:
- All functions work with ExTensor (int32/int64 for labels, float32/float64 for logits)

Issues covered:
- #278-282: Accuracy metrics implementation
"""

from shared.core import ExTensor
from collections import List
from shared.training.metrics.base import Metric


# ============================================================================
# Top-1 Accuracy (#278-282)
# ============================================================================


fn top1_accuracy(predictions: ExTensor, labels: ExTensor) raises -> Float64:
    """Compute top-1 accuracy for a single batch.

        Top-1 accuracy is the fraction of samples where the predicted class
        (argmax of logits) exactly matches the true label.

    Args:
            predictions: Model predictions/logits of shape [batch_size, num_classes]
                        or predicted class indices of shape [batch_size].
            labels: True class labels of shape [batch_size].

    Returns:
            Accuracy as a fraction in [0, 1].

    Raises:
            Error: If shapes are incompatible or batch sizes don't match.

    Examples:
    ```
            # With logits (need argmax)
            var logits = ExTensor(List[Int](), DType.float32)  # 4 samples, 3 classes
            var labels = ExTensor(List[Int](), DType.int32)
            var acc = top1_accuracy(logits, labels)

            # With predicted class indices
            var preds = ExTensor(List[Int](), DType.int32)  # Already argmaxed
            var acc2 = top1_accuracy(preds, labels)
    ```

    """
    # Determine if predictions are logits (2D) or class indices (1D)
    var pred_classes: ExTensor
    var pred_shape = predictions.shape()

    if len(pred_shape) == 2:
        # Predictions are logits, need to compute argmax
        pred_classes = argmax(predictions, axis=1)
    elif len(pred_shape) == 1:
        # Predictions are already class indices (copy since we can't transfer from read-only ref)
        pred_classes = predictions
    else:
        raise Error(
            "top1_accuracy: predictions must be 1D (class indices) or 2D"
            " (logits)"
        )

    # Validate shapes
    if pred_classes._numel != labels._numel:
        raise Error("top1_accuracy: batch sizes must match")

    # Count correct predictions
    var correct = 0
    for i in range(pred_classes._numel):
        var pred_val: Int
        var label_val: Int

        if pred_classes._dtype == DType.int32:
            pred_val = Int(pred_classes._data.bitcast[Int32]()[i])
        else:  # int64.
            pred_val = Int(pred_classes._data.bitcast[Int64]()[i])

        if labels._dtype == DType.int32:
            label_val = Int(labels._data.bitcast[Int32]()[i])
        else:  # int64.
            label_val = Int(labels._data.bitcast[Int64]()[i])

        if pred_val == label_val:
            correct += 1

    return Float64(correct) / Float64(pred_classes._numel)


fn argmax(var tensor: ExTensor, axis: Int) raises -> ExTensor:
    """Compute argmax along specified axis.

    Args:
            tensor: Input tensor.
            axis: Axis along which to compute argmax (typically 1 for [batch, classes]).

    Returns:
            Tensor of indices with shape reduced along specified axis.

    Raises:
            Error: If axis is out of bounds.
    """
    var shape_vec = tensor.shape()
    if axis < 0 or axis >= len(shape_vec):
        raise Error("argmax: axis out of bounds")

    if axis == 1 and len(shape_vec) == 2:
        # Common case: [batch_size, num_classes] -> [batch_size]
        var batch_size = shape_vec[0]
        var num_classes = shape_vec[1]

        var result_shape = List[Int]()
        result_shape.append(batch_size)
        var result = ExTensor(result_shape, DType.int32)

        # For each sample in batch
        for b in range(batch_size):
            var max_idx = 0
            var max_val: Float64

            # Get first value
            if tensor._dtype == DType.float32:
                max_val = Float64(
                    tensor._data.bitcast[Float32]()[b * num_classes]
                )
            else:  # float64
                max_val = tensor._data.bitcast[Float64]()[b * num_classes]

            # Find max
            for c in range(1, num_classes):
                var idx = b * num_classes + c
                var val: Float64

                if tensor._dtype == DType.float32:
                    val = Float64(tensor._data.bitcast[Float32]()[idx])
                else:
                    val = tensor._data.bitcast[Float64]()[idx]

                if val > max_val:
                    max_val = val
                    max_idx = c

            result[b] = Int64(max_idx)

        return result^
    else:
        raise Error("argmax: only axis=1 for 2D tensors currently supported")


# ============================================================================
# Top-K Accuracy (#278-282)
# ============================================================================


fn topk_accuracy(
    predictions: ExTensor, labels: ExTensor, k: Int = 5
) raises -> Float64:
    """Compute top-k accuracy for a single batch.

        Top-k accuracy is the fraction of samples where the true label appears
        in the k predictions with highest logits.

    Args:
            predictions: Model logits of shape [batch_size, num_classes].
            labels: True class labels of shape [batch_size].
            k: Number of top predictions to consider (default: 5).

    Returns:
            Accuracy as a fraction in [0, 1].

    Raises:
            Error: If shapes are incompatible.
            Error: If k <= 0 or k > num_classes.

    Examples:
    ```
            var logits = ExTensor(List[Int](), DType.float32)  # 4 samples, 10 classes
            var labels = ExTensor(List[Int](), DType.int32)
            var acc = topk_accuracy(logits, labels, k=5)  # Top-5 accuracy
    ```

    """
    # Validate shapes
    var shape_vec = predictions.shape()
    if len(shape_vec) != 2:
        raise Error("topk_accuracy: predictions must be 2D logits")

    var batch_size = shape_vec[0]
    var num_classes = shape_vec[1]

    if labels._numel != batch_size:
        raise Error("topk_accuracy: batch sizes must match")

    if k <= 0 or k > num_classes:
        raise Error("topk_accuracy: k must be in range (0, num_classes]")

    # Count correct predictions
    var correct = 0

    for b in range(batch_size):
        # Get true label
        var label_val: Int
        if labels._dtype == DType.int32:
            label_val = Int(labels._data.bitcast[Int32]()[b])
        else:
            label_val = Int(labels._data.bitcast[Int64]()[b])

        # Get top-k indices for this sample
        var top_k_indices = get_topk_indices(predictions, b, k)

        # Check if true label is in top-k
        var found = False
        for i in range(k):
            if top_k_indices[i] == label_val:
                found = True
                break

        if found:
            correct += 1

    return Float64(correct) / Float64(batch_size)


fn get_topk_indices(
    predictions: ExTensor, batch_idx: Int, k: Int
) raises -> List[Int]:
    """Get indices of top-k predictions for a single sample.

        Uses selection algorithm (not full sort) for efficiency.

    Args:
            predictions: Logits tensor [batch_size, num_classes].
            batch_idx: Which sample in the batch.
            k: Number of top indices to return.

    Returns:
            Vector of k class indices with highest scores.

    Raises:
            Error: If operation fails.
    """
    var shape_vec = predictions.shape()
    var num_classes = shape_vec[1]
    var offset = batch_idx * num_classes

    # Create list of (value, index) pairs
    var values: List[Float64] = []
    var indices = List[Int]()

    for c in range(num_classes):
        var idx = offset + c
        if predictions._dtype == DType.float32:
            values.append(Float64(predictions._data.bitcast[Float32]()[idx]))
        else:
            values.append(predictions._data.bitcast[Float64]()[idx])
        indices.append(c)

    # Simple selection: repeatedly find max and swap to front
    var result = List[Int]()

    for i in range(k):
        # Find max in remaining elements
        var max_idx = i
        var max_val = values[i]

        for j in range(i + 1, num_classes):
            if values[j] > max_val:
                max_val = values[j]
                max_idx = j

        # Swap to front
        var temp_val = values[i]
        var temp_idx = indices[i]
        values[i] = values[max_idx]
        indices[i] = indices[max_idx]
        values[max_idx] = temp_val
        indices[max_idx] = temp_idx

        result.append(indices[i])

    return result^


# ============================================================================
# Per-Class Accuracy (#278-282)
# ============================================================================


fn per_class_accuracy(
    predictions: ExTensor, labels: ExTensor, num_classes: Int
) raises -> ExTensor:
    """Compute accuracy for each class separately.

        Returns a tensor where each element is the accuracy for that class,
        computed as: correct_for_class / total_samples_for_class

    Args:
            predictions: Model predictions/logits (same format as top1_accuracy).
            labels: True class labels of shape [batch_size].
            num_classes: Total number of classes.

    Returns:
            Tensor of shape [num_classes] with per-class accuracies.

    Raises:
            Error: If shapes are incompatible.

    Examples:
    ```
            var logits = ExTensor(List[Int](), DType.float32)
            var labels = ExTensor(List[Int](), DType.int32)
            var per_class_acc = per_class_accuracy(logits, labels, num_classes=10)
            # per_class_acc[0] = accuracy for class 0, etc
    ```

    """
    # Get predicted classes
    var pred_classes: ExTensor
    var pred_shape = predictions.shape()
    if len(pred_shape) == 2:
        pred_classes = argmax(predictions, axis=1)
    elif len(pred_shape) == 1:
        # Copy predictions (can't transfer from read-only reference)
        pred_classes = predictions
    else:
        raise Error("per_class_accuracy: invalid predictions shape")

    # Count correct and total per class
    var correct_counts = List[Int](length=num_classes, fill=0)
    var total_counts = List[Int](length=num_classes, fill=0)

    for i in range(labels._numel):
        var pred_val: Int
        var label_val: Int

        if pred_classes._dtype == DType.int32:
            pred_val = Int(pred_classes._data.bitcast[Int32]()[i])
        else:
            pred_val = Int(pred_classes._data.bitcast[Int64]()[i])

        if labels._dtype == DType.int32:
            label_val = Int(labels._data.bitcast[Int32]()[i])
        else:
            label_val = Int(labels._data.bitcast[Int64]()[i])

        # Increment total count for true class
        total_counts[label_val] += 1

        # Increment correct count if prediction matches
        if pred_val == label_val:
            correct_counts[label_val] += 1

    # Compute per-class accuracies
    var result_shape = List[Int]()
    result_shape.append(num_classes)
    var result = ExTensor(result_shape, DType.float64)

    for c in range(num_classes):
        var acc: Float64
        if total_counts[c] > 0:
            acc = Float64(correct_counts[c]) / Float64(total_counts[c])
        else:
            acc = 0.0  # No samples for this class

        result[c] = acc

    return result^


# ============================================================================
# Incremental Accuracy Metric (#278-282)
# ============================================================================


struct AccuracyMetric(Metric):
    """Incremental accuracy metric for efficient accumulation across batches.

    Maintains running count of correct predictions and total samples,
    allowing efficient computation over large datasets without storing
    all predictions

    Usage:
        var metric = AccuracyMetric()
        for batch in data_loader:
            metric.update(predictions, labels)
        var final_acc = metric.compute()
        metric.reset()  # For next epoch.

    """

    var correct_count: Int
    var total_count: Int

    fn __init__(out self):
        """Initialize with zero counts."""
        self.correct_count = 0
        self.total_count = 0

    fn update(mut self, predictions: ExTensor, labels: ExTensor) raises:
        """Update metric with a new batch of predictions.

        Args:
            predictions: Model predictions/logits (same format as top1_accuracy).
            labels: True class labels of shape [batch_size].

        Raises:
            Error: If shapes are incompatible.
        """
        # Get predicted classes
        var pred_classes: ExTensor
        var pred_shape = predictions.shape()
        if len(pred_shape) == 2:
            pred_classes = argmax(predictions, axis=1)
        elif len(pred_shape) == 1:
            # Copy predictions (ImplicitlyCopyable - creates shared reference)
            pred_classes = predictions
        else:
            raise Error("AccuracyMetric.update: invalid predictions shape")

        # Count correct predictions in this batch
        for i in range(labels._numel):
            var pred_val: Int
            var label_val: Int

            if pred_classes._dtype == DType.int32:
                pred_val = Int(pred_classes._data.bitcast[Int32]()[i])
            else:
                pred_val = Int(pred_classes._data.bitcast[Int64]()[i])

            if labels._dtype == DType.int32:
                label_val = Int(labels._data.bitcast[Int32]()[i])
            else:
                label_val = Int(labels._data.bitcast[Int64]()[i])

            if pred_val == label_val:
                self.correct_count += 1

            self.total_count += 1

    fn compute(self) -> Float64:
        """Compute final accuracy from accumulated counts.

        Returns:
            Accuracy as a fraction in [0, 1], or 0.0 if no samples.
        """
        if self.total_count == 0:
            return 0.0
        return Float64(self.correct_count) / Float64(self.total_count)

    fn reset(mut self):
        """Reset counts to zero for next epoch."""
        self.correct_count = 0
        self.total_count = 0
