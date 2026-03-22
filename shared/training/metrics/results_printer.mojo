"""Results printer for formatted console output of training metrics.

Provides formatting utilities for displaying training progress, evaluation results,
per-class metrics, and confusion matrices in a human-readable format.

Features:
- Training progress with epoch, batch, and loss information
- Evaluation summary with accuracy and loss
- Per-class accuracy breakdown
- Confusion matrix visualization with formatting
- Customizable column widths and separators

"""

from collections import List
from shared.core import AnyTensor


# ============================================================================
# Training Progress Printer (#2353)
# ============================================================================


fn print_training_progress(
    epoch: Int,
    total_epochs: Int,
    batch: Int,
    total_batches: Int,
    loss: Float32,
    learning_rate: Float32 = 0.01,
) raises:
    """Print formatted training progress during epoch.

        Displays current epoch, batch number, loss value, and learning rate
        in a consistent format for monitoring training progress.

    Args:
            epoch: Current epoch number (1-indexed).
            total_epochs: Total number of epochs.
            batch: Current batch number (1-indexed).
            total_batches: Total number of batches in epoch.
            loss: Current batch loss value.
            learning_rate: Current learning rate (default: 0.01).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            rint_training_progress(1, 200, 10, 100, 0.234, 0.01)
            Output:
            Epoch [1/200] Batch [10/100] Loss: 0.234000 LR: 0.010000
            ```
    """
    var output = "Epoch [" + String(epoch) + "/" + String(total_epochs) + "] "
    output = (
        output + "Batch [" + String(batch) + "/" + String(total_batches) + "] "
    )
    output = output + "Loss: " + String(loss) + " "
    output = output + "LR: " + String(learning_rate)

    print(output)


# ============================================================================
# Evaluation Summary Printer (#2353)
# ============================================================================


fn print_evaluation_summary(
    epoch: Int,
    total_epochs: Int,
    train_loss: Float32,
    train_accuracy: Float32,
    test_loss: Float32,
    test_accuracy: Float32,
) raises:
    """Print formatted evaluation summary after epoch.

        Displays training and test metrics in a clean side-by-side format
        for easy comparison of model performance.

    Args:
            epoch: Current epoch number (1-indexed).
            total_epochs: Total number of epochs.
            train_loss: Training loss value.
            train_accuracy: Training accuracy (0.0 to 1.0).
            test_loss: Test/validation loss value.
            test_accuracy: Test/validation accuracy (0.0 to 1.0).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            rint_evaluation_summary(1, 200, 0.523, 0.923, 0.645, 0.891)
            Output:
            ============================================================
            Epoch [1/200] Results
            ============================================================
            Train Loss: 0.523000  Train Acc: 92.30%
            Test Loss:  0.645000  Test Acc:  89.10%
            ============================================================
            ```
    """
    print("=" * 60)
    print("Epoch [" + String(epoch) + "/" + String(total_epochs) + "] Results")
    print("=" * 60)

    # Convert accuracies to percentages and format
    var train_acc_pct = train_accuracy * 100.0
    var test_acc_pct = test_accuracy * 100.0

    # Format with consistent alignment
    var train_line = "Train Loss: " + String(train_loss) + "  Train Acc: "
    train_line = train_line + String(train_acc_pct) + "%"

    var test_line = "Test Loss:  " + String(test_loss) + "  Test Acc:  "
    test_line = test_line + String(test_acc_pct) + "%"

    print(train_line)
    print(test_line)
    print("=" * 60)


# ============================================================================
# Per-Class Accuracy Printer (#2353)
# ============================================================================


fn print_per_class_accuracy(per_class_accuracies: AnyTensor) raises:
    """Print per-class accuracy metrics with default numeric class labels.

        Displays accuracy for each class in a table format using numeric indices
        as class labels.

    Args:
            per_class_accuracies: Tensor of shape [num_classes] with per-class accuracy.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var per_class = AnyTensor([10], DType.float64)
            # ... populate with per-class accuracies ...
            print_per_class_accuracy(per_class)

            Output:
            ============================================================
            Per-Class Accuracy
            ============================================================
            Class      Accuracy
            ============================================================
            0          0.920000
            1          0.945000
            ...
            ```
    """
    var default_names = List[String]()
    var shape = per_class_accuracies.shape()
    var num_classes = shape[0]
    for i in range(num_classes):
        default_names.append("Class " + String(i))
    print_per_class_accuracy(per_class_accuracies, default_names)


fn print_per_class_accuracy(
    per_class_accuracies: AnyTensor,
    class_names: List[String],
    column_width: Int = 15,
) raises:
    """Print per-class accuracy metrics in formatted table.

        Displays accuracy for each class in a table format, with optional
        class names for improved interpretability.

    Args:
            per_class_accuracies: Tensor of shape [num_classes] with per-class accuracy.
            class_names: List of class name strings.
            column_width: Width of each column in characters (default: 15).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var per_class = AnyTensor([3], DType.float64)
            # ... populate with per-class accuracies ...
            var names = List[String]()
            names.append("Cat")
            names.append("Dog")
            names.append("Bird")
            print_per_class_accuracy(per_class, names)

            Output:
            ============================================================
            Per-Class Accuracy
            ============================================================
            Class      Accuracy
            ============================================================
            Cat        0.920000
            Dog        0.945000
            Bird       0.980000
            ```

    Note:
            If class_names is provided, it must have same length as per_class_accuracies.
            If class_names is empty, classes are displayed as numeric indices.
    """
    var shape = per_class_accuracies.shape()
    var num_classes = shape[0]

    print("=" * 60)
    print("Per-Class Accuracy")
    print("=" * 60)

    # Print header
    if len(class_names) > 0:
        print("Class" + " " * 10 + "Accuracy")
    else:
        print("Class" + " " * 10 + "Accuracy")

    print("-" * 60)

    # Print per-class accuracies
    for i in range(num_classes):
        var acc: Float64
        if per_class_accuracies._dtype == DType.float32:
            acc = Float64(per_class_accuracies._data.bitcast[Float32]()[i])
        else:  # float64.
            acc = per_class_accuracies._data.bitcast[Float64]()[i]

        var acc_str = String(acc)

        if len(class_names) > 0 and i < len(class_names):
            # Use provided class name
            var class_label = class_names[i]
            print(class_label + " " * (15 - len(class_label)) + acc_str)
        else:
            # Use numeric index
            var class_idx = String(i)
            print(class_idx + " " * (15 - len(class_idx)) + acc_str)

    print("=" * 60)


# ============================================================================
# Confusion Matrix Printer (#2353)
# ============================================================================


fn print_confusion_matrix(matrix: AnyTensor) raises:
    """Print confusion matrix with default numeric class labels.

        Displays confusion matrix with proper alignment using numeric indices
        as class labels.

    Args:
            matrix: Confusion matrix tensor of shape [num_classes, num_classes].

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var cm_shape = [3, 3]
            var matrix = AnyTensor(cm_shape, DType.int32)
            # ... populate matrix ...
            print_confusion_matrix(matrix)

            Output:
            ============================================================
            Confusion Matrix (Raw Counts)
            ============================================================
                  Predicted
                    0    1    2
            True 0 90    5    5
                 1  3   92    5
                 2  2    4   94
            ============================================================
            ```

    Note:
            - Rows represent true labels.
            - Columns represent predicted labels.
            - Values are right-aligned within columns.
            - Classes displayed as numeric indices.
    """
    var default_names = List[String]()
    var shape = matrix.shape()
    var num_classes = shape[0]
    for i in range(num_classes):
        default_names.append(String(i))
    print_confusion_matrix(matrix, default_names)


fn print_confusion_matrix(
    matrix: AnyTensor,
    class_names: List[String],
    normalized: Bool = False,
    column_width: Int = 10,
) raises:
    """Print confusion matrix in formatted table.

        Displays confusion matrix with proper alignment and optional class names.
        Can display raw counts or normalized values.

    Args:
            matrix: Confusion matrix tensor of shape [num_classes, num_classes].
            class_names: List of class name strings.
            normalized: If True, display as percentages (default: False).
            column_width: Width of each column in characters (default: 10).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var cm_shape = [3, 3]
            var matrix = AnyTensor(cm_shape, DType.int32)
            # ... populate matrix ...
            var names = List[String]()
            names.append("Cat")
            names.append("Dog")
            names.append("Bird")
            print_confusion_matrix(matrix, names)

            Output:
            ============================================================
            Confusion Matrix (Raw Counts)
            ============================================================
                  Predicted
                 Cat Dog Bird
            True Cat 90    5    5
                 Dog  3   92    5
                Bird  2    4   94
            ============================================================
            ```

    Note:
            - Rows represent true labels.
            - Columns represent predicted labels.
            - Values are right-aligned within columns.
            - If class_names provided, used for row/column labels.
    """
    var shape = matrix.shape()
    var num_classes = shape[0]

    print("=" * 60)
    if normalized:
        print("Confusion Matrix (Normalized)")
    else:
        print("Confusion Matrix (Raw Counts)")
    print("=" * 60)

    # Print column header
    var header = "      Predicted"
    print(header)

    # Print column class labels
    var class_header = "        "
    for c in range(num_classes):
        if len(class_names) > 0 and c < len(class_names):
            var name = class_names[c]
            # Pad to column width
            var padded = name
            while len(padded) < column_width:
                padded = padded + " "
            class_header = class_header + padded
        else:
            var idx_str = String(c)
            var padded = idx_str
            while len(padded) < column_width:
                padded = padded + " "
            class_header = class_header + padded

    print(class_header)

    # Print rows
    print("-" * 60)
    for r in range(num_classes):
        # Row label - define outside if/else for proper scope
        var row_label: String
        if len(class_names) > 0 and r < len(class_names):
            var name = class_names[r]
            row_label = "True " + name
        else:
            var idx_str = String(r)
            row_label = "True " + idx_str
        # Pad to label width (8 chars)
        while len(row_label) < 8:
            row_label = row_label + " "

        var row_str = row_label

        # Row values
        for c in range(num_classes):
            var idx = r * num_classes + c
            var value: Float64

            if matrix._dtype == DType.int32:
                value = Float64(matrix._data.bitcast[Int32]()[idx])
            elif matrix._dtype == DType.float32:
                value = Float64(matrix._data.bitcast[Float32]()[idx])
            else:  # float64
                value = matrix._data.bitcast[Float64]()[idx]

            var val_str: String
            if normalized:
                # Display as percentage
                var pct = value * 100.0
                val_str = String(pct)
            else:
                val_str = String(value)

            # Right-align value within column width
            while len(val_str) < column_width:
                val_str = " " + val_str
            row_str = row_str + val_str

        print(row_str)

    print("=" * 60)


# ============================================================================
# Training Summary Printer (#2353)
# ============================================================================


fn print_training_summary(
    total_epochs: Int,
    best_train_loss: Float32,
    best_test_loss: Float32,
    best_accuracy: Float32,
    best_epoch: Int,
) raises:
    """Print final training summary with best metrics.

        Displays consolidated summary of training including best achieved metrics
        and the epoch at which they occurred.

    Args:
            total_epochs: Total number of epochs trained.
            best_train_loss: Best training loss achieved.
            best_test_loss: Best test/validation loss achieved.
            best_accuracy: Best accuracy achieved (0.0 to 1.0).
            best_epoch: Epoch at which best accuracy was achieved.

    Raises:
            Error: If operation fails.

    Example:
        ```mojo
        rint_training_summary(200, 0.034, 0.156, 0.965, 187)
        Output:
        ============================================================
        Training Summary
        ============================================================
        Total Epochs:        200
        Best Train Loss:     0.034000 (Epoch 187)
        Best Test Loss:      0.156000
        Best Accuracy:       96.50%
        ============================================================
        ```
    """
    var best_acc_pct = best_accuracy * 100.0

    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print("Total Epochs:        " + String(total_epochs))
    print(
        "Best Train Loss:     "
        + String(best_train_loss)
        + " (Epoch "
        + String(best_epoch)
        + ")"
    )
    print("Best Test Loss:      " + String(best_test_loss))
    print("Best Accuracy:       " + String(best_acc_pct) + "%")
    print("=" * 60)
