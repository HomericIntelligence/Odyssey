"""Inference utilities for model evaluation scripts.

This module provides standardized InferenceConfig struct and helper functions
for consistent inference and evaluation across different models.

Example:
    from shared.utils.inference_utils import InferenceConfig, parse_inference_args
    from shared.utils.inference_utils import evaluate_accuracy

    def main() raises:
        var config = parse_inference_args()
        print("Weights dir:", config.weights_dir)
        print("Data dir:", config.data_dir)

        # Evaluate model predictions
        var accuracy = evaluate_accuracy(predictions, labels)
        print("Accuracy:", accuracy)
    ```
"""

from std.sys import argv

from shared.tensor.any_tensor import AnyTensor


# ============================================================================
# Inference Configuration Struct
# ============================================================================


@fieldwise_init
struct InferenceConfig(Copyable, Movable):
    """Container for common inference configuration.

    Attributes:
        weights_dir: Path to load model weights from
        data_dir: Path to dataset directory
        batch_size: Batch size for inference
        verbose: Whether to print verbose output
    """

    var weights_dir: String
    var data_dir: String
    var batch_size: Int
    var verbose: Bool

    def __init__(out self):
        """Initialize with default inference configuration."""
        self.weights_dir = "weights"
        self.data_dir = "datasets"
        self.batch_size = 32
        self.verbose = False


# ============================================================================
# Argument Parsing Functions
# ============================================================================


def parse_inference_args() raises -> InferenceConfig:
    """Parse common inference arguments from command line.

        Supported arguments:
            --weights-dir <str>: Weights directory (default: "weights")
            --data-dir <str>: Dataset directory (default: "datasets")
            --batch-size <int>: Batch size (default: 32)
            --verbose: Enable verbose output

    Returns:
            InferenceConfig struct with parsed values

        Example:
            ```mojo
             Command line: mojo run inference.mojo --weights-dir ./trained --verbose
            var config = parse_inference_args()
            # config.weights_dir == "./trained", config.verbose == True
            ```

    Raises:
            Error: If operation fails.
    """
    var result = InferenceConfig()

    var args = argv()
    var i = 1  # Skip program name
    while i < len(args):
        var arg = args[i]

        if arg == "--weights-dir" and i + 1 < len(args):
            result.weights_dir = args[i + 1]
            i += 2
        elif arg == "--data-dir" and i + 1 < len(args):
            result.data_dir = args[i + 1]
            i += 2
        elif arg == "--batch-size" and i + 1 < len(args):
            result.batch_size = Int(args[i + 1])
            i += 2
        elif arg == "--verbose":
            result.verbose = True
            i += 1
        else:
            # Skip unknown arguments (allows model-specific args)
            i += 1

    return result^


def parse_inference_args_with_defaults(
    default_weights_dir: String = "weights",
    default_data_dir: String = "datasets",
    default_batch_size: Int = 32,
) raises -> InferenceConfig:
    """Parse inference arguments with custom defaults.

        Allows each inference script to specify model-appropriate defaults
        while still using shared parsing logic.

    Args:
            default_weights_dir: Default weights directory.
            default_data_dir: Default dataset directory.
            default_batch_size: Default batch size.

    Returns:
            InferenceConfig struct with parsed values.

        Example:
            ```mojo
             AlexNet with custom defaults
            var config = parse_inference_args_with_defaults(
                default_weights_dir="alexnet_weights",
                default_data_dir="datasets/cifar10",
                default_batch_size=128
            )
            ```

    Raises:
            Error: If operation fails.
    """
    var result = InferenceConfig()
    result.weights_dir = default_weights_dir
    result.data_dir = default_data_dir
    result.batch_size = default_batch_size
    result.verbose = False

    var args = argv()
    var i = 1
    while i < len(args):
        var arg = args[i]

        if arg == "--weights-dir" and i + 1 < len(args):
            result.weights_dir = args[i + 1]
            i += 2
        elif arg == "--data-dir" and i + 1 < len(args):
            result.data_dir = args[i + 1]
            i += 2
        elif arg == "--batch-size" and i + 1 < len(args):
            result.batch_size = Int(args[i + 1])
            i += 2
        elif arg == "--verbose":
            result.verbose = True
            i += 1
        else:
            i += 1

    return result^


# ============================================================================
# Evaluation Utilities
# ============================================================================


def evaluate_accuracy(predictions: AnyTensor, labels: AnyTensor) raises -> Float32:
    """Calculate classification accuracy from predictions and labels.

        Computes the percentage of predictions that match the ground truth labels
        Predictions should be class indices (from argmax of logits).

    Args:
            predictions: Predicted class indices tensor of shape (batch,).
            labels: Ground truth class indices tensor of shape (batch,).

    Returns:
            Accuracy as a Float32 in range [0.0, 1.0].

        Example:
            ```mojo
            from shared.utils.inference_utils import evaluate_accuracy

            # predictions: [0, 1, 2, 1] (predicted classes)
            # labels: [0, 1, 2, 0] (ground truth)
            var accuracy = evaluate_accuracy(predictions, labels)
            # accuracy == 0.75 (3 out of 4 correct)
            ```

    Note:
            Both tensors must have the same shape and contain integer class indices.

    Raises:
            Error: If operation fails.
    """
    var pred_shape = predictions.shape()
    var label_shape = labels.shape()

    if len(pred_shape) != 1 or len(label_shape) != 1:
        raise Error(
            "evaluate_accuracy: predictions and labels must be 1D tensors"
        )

    var n = pred_shape[0]
    if n != label_shape[0]:
        raise Error(
            "evaluate_accuracy: predictions and labels must have same size"
        )

    if n == 0:
        return Float32(0.0)

    var correct = 0
    var pred_ptr = predictions._data
    var label_ptr = labels._data

    # Handle int32 labels (most common)
    if predictions.dtype() == DType.int32 and labels.dtype() == DType.int32:
        for i in range(n):
            if pred_ptr.bitcast[Int32]()[i] == label_ptr.bitcast[Int32]()[i]:
                correct += 1
    elif predictions.dtype() == DType.int64 and labels.dtype() == DType.int64:
        for i in range(n):
            if pred_ptr.bitcast[Int64]()[i] == label_ptr.bitcast[Int64]()[i]:
                correct += 1
    elif (
        predictions.dtype() == DType.float32 and labels.dtype() == DType.float32
    ):
        # Sometimes predictions are stored as float (argmax returns int as float)
        for i in range(n):
            var pred_val = Int(pred_ptr.bitcast[Float32]()[i])
            var label_val = Int(label_ptr.bitcast[Float32]()[i])
            if pred_val == label_val:
                correct += 1
    elif (
        predictions.dtype() == DType.float64 and labels.dtype() == DType.float64
    ):
        for i in range(n):
            var pred_val = Int(pred_ptr.bitcast[Float64]()[i])
            var label_val = Int(label_ptr.bitcast[Float64]()[i])
            if pred_val == label_val:
                correct += 1
    else:
        raise Error(
            "evaluate_accuracy: unsupported dtype combination, use int32,"
            " int64, float32, or float64"
        )

    return Float32(correct) / Float32(n)


def count_correct(predictions: AnyTensor, labels: AnyTensor) raises -> Int:
    """Count the number of correct predictions.

        Lower-level function for computing accuracy incrementally over batches.

    Args:
            predictions: Predicted class indices tensor of shape (batch,).
            labels: Ground truth class indices tensor of shape (batch,).

    Returns:
            Number of correct predictions as Int.

        Example:
            ```mojo
            from shared.utils.inference_utils import count_correct

            var total_correct = 0
            var total_samples = 0

            for batch in batches:
                total_correct += count_correct(predictions, labels)
                total_samples += batch_size

            var accuracy = Float32(total_correct) / Float32(total_samples)
            ```

    Raises:
            Error: If operation fails.
    """
    var pred_shape = predictions.shape()
    var label_shape = labels.shape()

    if len(pred_shape) != 1 or len(label_shape) != 1:
        raise Error("count_correct: predictions and labels must be 1D tensors")

    var n = pred_shape[0]
    if n != label_shape[0]:
        raise Error("count_correct: predictions and labels must have same size")

    var correct = 0
    var pred_ptr = predictions._data
    var label_ptr = labels._data

    if predictions.dtype() == DType.int32 and labels.dtype() == DType.int32:
        for i in range(n):
            if pred_ptr.bitcast[Int32]()[i] == label_ptr.bitcast[Int32]()[i]:
                correct += 1
    elif predictions.dtype() == DType.int64 and labels.dtype() == DType.int64:
        for i in range(n):
            if pred_ptr.bitcast[Int64]()[i] == label_ptr.bitcast[Int64]()[i]:
                correct += 1
    elif (
        predictions.dtype() == DType.float32 and labels.dtype() == DType.float32
    ):
        for i in range(n):
            var pred_val = Int(pred_ptr.bitcast[Float32]()[i])
            var label_val = Int(label_ptr.bitcast[Float32]()[i])
            if pred_val == label_val:
                correct += 1
    elif (
        predictions.dtype() == DType.float64 and labels.dtype() == DType.float64
    ):
        for i in range(n):
            var pred_val = Int(pred_ptr.bitcast[Float64]()[i])
            var label_val = Int(label_ptr.bitcast[Float64]()[i])
            if pred_val == label_val:
                correct += 1
    else:
        raise Error(
            "count_correct: unsupported dtype combination, use int32, int64,"
            " float32, or float64"
        )

    return correct
