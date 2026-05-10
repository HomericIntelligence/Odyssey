"""Confusion matrix for classification analysis.

Provides NxN confusion matrix with normalization modes and derived metrics
(precision, recall, F1-score) for detailed classification error analysis.

Features:
- Incremental updates for efficient computation
- Multiple normalization modes (row, column, total, none)
- Derived metrics (precision, recall, F1-score per class)
- Optional class names for interpretability

"""

from shared.tensor.any_tensor import AnyTensor
from std.collections import List
from std.math import sqrt
from shared.training.metrics.base import Metric


# ============================================================================
# Confusion Matrix (#288-292)
# ============================================================================


struct ConfusionMatrix(Metric):
    """Confusion matrix for classification analysis.

    Maintains an NxN matrix where matrix[i, j] represents the count of samples
    with true label i that were predicted as label j

    Rows = true labels
    Columns = predicted labels

    Features:
    - Incremental updates (efficient for large datasets)
    - Multiple normalization modes
    - Derived metrics (precision, recall, F1)
    - Optional class names

    Usage:
        var cm = ConfusionMatrix(num_classes=3)
        for batch in data_loader:
            cm.update(predictions, labels)

        var normalized = cm.normalize(mode="row")  # Per-class recall
        var precision = cm.get_precision()
        var recall = cm.get_recall()
        var f1 = cm.get_f1_score()

    """

    var num_classes: Int
    var matrix: AnyTensor  # Shape: [num_classes, num_classes], dtype=int32
    var class_names: List[String]
    var has_class_names: Bool

    def __init__(
        out self, num_classes: Int, class_names: List[String] = List[String]()
    ) raises:
        """Initialize NxN confusion matrix.

        Args:
            num_classes: Number of classes.
            class_names: Optional list of class names (default: empty).

        Raises:
            Error: If tensor size exceeds memory limits.
        """
        self.num_classes = num_classes

        # Initialize matrix with zeros
        var shape: List[Int] = [num_classes, num_classes]
        self.matrix = AnyTensor(shape, DType.int32)
        for i in range(num_classes * num_classes):
            self.matrix.set(i, Int64(0))

        # Explicit copy of class_names list
        self.class_names = List[String](class_names)
        self.has_class_names = len(class_names) > 0

    def update(mut self, predictions: AnyTensor, labels: AnyTensor) raises:
        """Update confusion matrix with new batch of predictions.

        Args:
            predictions: Predicted class indices [batch_size] or logits [batch_size, num_classes].
            labels: True class labels [batch_size].

        Raises:
            Error: If shapes are incompatible, labels out of range,
                   or labels/predictions dtype is not int32 or int64.
        """
        # Get predicted classes
        var pred_classes: AnyTensor
        var pred_shape = predictions.shape()

        if len(pred_shape) == 2:
            # Logits - need argmax
            pred_classes = argmax(predictions)
        elif len(pred_shape) == 1:
            # Already class indices - copy (ImplicitlyCopyable - creates shared reference)
            pred_classes = predictions
        else:
            raise Error("ConfusionMatrix.update: predictions must be 1D or 2D")

        if pred_classes._numel != labels._numel:
            raise Error("ConfusionMatrix.update: batch sizes must match")

        # Validate label dtype — float bits bitcast to int64 produces garbage indices
        if labels._dtype != DType.int32 and labels._dtype != DType.int64:
            raise Error(
                "ConfusionMatrix.update() requires int32 or int64 labels, got "
                + String(labels._dtype)
            )

        # Validate predictions dtype for 1D inputs (2D logits go through argmax → int32)
        if len(pred_shape) == 1:
            if (
                pred_classes._dtype != DType.int32
                and pred_classes._dtype != DType.int64
            ):
                raise Error(
                    "ConfusionMatrix.update() requires int32 or int64"
                    " predictions, got "
                    + String(pred_classes._dtype)
                )

        # Update matrix
        for i in range(labels._numel):
            var pred: Int
            var true_label: Int

            if pred_classes._dtype == DType.int32:
                pred = Int(pred_classes.load[DType.int32](i))
            else:
                pred = Int(pred_classes.load[DType.int64](i))

            if labels._dtype == DType.int32:
                true_label = Int(labels.load[DType.int32](i))
            else:
                true_label = Int(labels.load[DType.int64](i))

            # Validate indices
            if true_label < 0 or true_label >= self.num_classes:
                raise Error("ConfusionMatrix.update: true label out of range")
            if pred < 0 or pred >= self.num_classes:
                raise Error(
                    "ConfusionMatrix.update: predicted label out of range"
                )

            # Increment count at [true_label, pred]
            var idx = true_label * self.num_classes + pred
            var current_val = Int(self.matrix.load[DType.int32](idx))
            self.matrix.set(idx, Int64(current_val + 1))

    def reset(mut self):
        """Reset all counts to zero."""
        for i in range(self.num_classes * self.num_classes):
            self.matrix._set_int64(i, Int64(0))

    def normalize(self, mode: String = "none") raises -> AnyTensor:
        """Normalize confusion matrix by row, column, total, or none.

        Args:
            mode: Normalization mode:
                - "row": Normalize by row sum (recall per class).
                - "column": Normalize by column sum (precision per class).
                - "total": Normalize by total count.
                - "none": Return raw counts (default).

        Returns:
            Normalized confusion matrix as Float64 tensor.

        Raises:
            Error: If mode is invalid or operation fails.
        """
        var result_shape = List[Int]()
        result_shape.append(self.num_classes)
        result_shape.append(self.num_classes)
        var result = AnyTensor(result_shape, DType.float64)

        if mode == "none":
            # Raw counts
            for i in range(self.num_classes * self.num_classes):
                var count = Float64(self.matrix.load[DType.int32](i))
                result.set(i, count)

        elif mode == "row":
            # Normalize by row sum (divide each row by its sum)
            for row in range(self.num_classes):
                # Compute row sum
                var row_sum: Float64 = 0.0
                for col in range(self.num_classes):
                    var idx = row * self.num_classes + col
                    row_sum += Float64(self.matrix.load[DType.int32](idx))

                # Normalize row
                for col in range(self.num_classes):
                    var idx = row * self.num_classes + col
                    var count = Float64(self.matrix.load[DType.int32](idx))

                    if row_sum > 0.0:
                        result.set(idx, count / row_sum)
                    else:
                        result.set(idx, 0.0)

        elif mode == "column":
            # Normalize by column sum (divide each column by its sum)
            for col in range(self.num_classes):
                # Compute column sum
                var col_sum: Float64 = 0.0
                for row in range(self.num_classes):
                    var idx = row * self.num_classes + col
                    col_sum += Float64(self.matrix.load[DType.int32](idx))

                # Normalize column
                for row in range(self.num_classes):
                    var idx = row * self.num_classes + col
                    var count = Float64(self.matrix.load[DType.int32](idx))

                    if col_sum > 0.0:
                        result.set(idx, count / col_sum)
                    else:
                        result.set(idx, 0.0)

        elif mode == "total":
            # Normalize by total count
            var total_sum: Float64 = 0.0
            for i in range(self.num_classes * self.num_classes):
                total_sum += Float64(self.matrix.load[DType.int32](i))

            for i in range(self.num_classes * self.num_classes):
                var count = Float64(self.matrix.load[DType.int32](i))
                if total_sum > 0.0:
                    result.set(i, count / total_sum)
                else:
                    result.set(i, 0.0)

        else:
            raise Error(
                "ConfusionMatrix.normalize: invalid mode (use 'row', 'column',"
                " 'total', or 'none')"
            )

        return result^

    def get_precision(self) raises -> AnyTensor:
        """Compute per-class precision.

        Precision[i] = matrix[i, i] / sum(matrix[:, i])
        (Correct predictions for class i / Total predicted as class i).

        Returns:
            Tensor of shape [num_classes] with precision for each class.

        Raises:
            Error: If operation fails.

        Note: Returns 0.0 for classes with no predictions.
        """
        var result_shape = List[Int]()
        result_shape.append(self.num_classes)
        var result = AnyTensor(result_shape, DType.float64)

        for col in range(self.num_classes):
            # Compute column sum (total predicted as this class)
            var col_sum: Float64 = 0.0
            for row in range(self.num_classes):
                var idx = row * self.num_classes + col
                col_sum += Float64(self.matrix.load[DType.int32](idx))

            # Get diagonal (correct predictions)
            var diag_idx = col * self.num_classes + col
            var correct = Float64(self.matrix.load[DType.int32](diag_idx))

            # Compute precision
            if col_sum > 0.0:
                result.set(col, correct / col_sum)
            else:
                result.set(col, 0.0)

        return result^

    def get_recall(self) raises -> AnyTensor:
        """Compute per-class recall.

        Recall[i] = matrix[i, i] / sum(matrix[i, :])
        (Correct predictions for class i / Total samples of class i).

        Returns:
            Tensor of shape [num_classes] with recall for each class.

        Raises:
            Error: If operation fails.

        Note: Returns 0.0 for classes with no samples.
        """
        var result_shape = List[Int]()
        result_shape.append(self.num_classes)
        var result = AnyTensor(result_shape, DType.float64)

        for row in range(self.num_classes):
            # Compute row sum (total samples of this class)
            var row_sum: Float64 = 0.0
            for col in range(self.num_classes):
                var idx = row * self.num_classes + col
                row_sum += Float64(self.matrix.load[DType.int32](idx))

            # Get diagonal (correct predictions)
            var diag_idx = row * self.num_classes + row
            var correct = Float64(self.matrix.load[DType.int32](diag_idx))

            # Compute recall
            if row_sum > 0.0:
                result.set(row, correct / row_sum)
            else:
                result.set(row, 0.0)

        return result^

    def get_f1_score(self) raises -> AnyTensor:
        """Compute per-class F1-score.

        F1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]).

        Returns:
            Tensor of shape [num_classes] with F1-score for each class.

        Raises:
            Error: If operation fails.

        Note: Returns 0.0 when precision + recall = 0.
        """
        var precision = self.get_precision()
        var recall = self.get_recall()

        var result_shape = List[Int]()
        result_shape.append(self.num_classes)
        var result = AnyTensor(result_shape, DType.float64)

        for i in range(self.num_classes):
            var p = precision.load[DType.float64](i)
            var r = recall.load[DType.float64](i)

            if p + r > 0.0:
                result.set(i, 2.0 * (p * r) / (p + r))
            else:
                result.set(i, 0.0)

        return result^


# Helper function for argmax (same as in accuracy.mojo, but duplicated for independence)
def argmax(var tensor: AnyTensor) raises -> AnyTensor:
    """Compute argmax along last axis for 2D tensor.

    Args:
            tensor: Input tensor [batch_size, num_classes].

    Returns:
            Tensor of indices [batch_size].

    Raises:
            Error: If tensor is not 2D.
    """
    var shape_vec = tensor.shape()
    if len(shape_vec) != 2:
        raise Error("argmax: only 2D tensors supported")

    var batch_size = shape_vec[0]
    var num_classes = shape_vec[1]

    var result_shape = List[Int]()
    result_shape.append(batch_size)
    var result = AnyTensor(result_shape, DType.int32)

    for b in range(batch_size):
        var max_idx = 0
        var max_val: Float64

        # Get first value
        if tensor._dtype == DType.float32:
            max_val = Float64(tensor.load[DType.float32](b * num_classes))
        else:
            max_val = Float64(tensor.load[DType.float64](b * num_classes))

        # Find max
        for c in range(1, num_classes):
            var idx = b * num_classes + c
            var val: Float64

            if tensor._dtype == DType.float32:
                val = Float64(tensor.load[DType.float32](idx))
            else:
                val = Float64(tensor.load[DType.float64](idx))

            if val > max_val:
                max_val = val
                max_idx = c

        result.set(b, Int64(max_idx))

    return result^
