"""Consolidated evaluation utilities for model assessment.

Provides generic evaluate() and evaluate_batched() functions that work with any
model type, consolidating duplicated evaluation patterns from train.mojo files
across the examples directory.

Patterns consolidated from:
- examples/lenet_emnist/train.mojo (fn evaluate)
- examples/resnet18_cifar10/train.mojo (fn compute_accuracy)
- examples/alexnet_cifar10/train.mojo (fn evaluate)
- examples/vgg16_cifar10/train.mojo (fn compute_accuracy)
- examples/googlenet_cifar10/train.mojo (fn compute_accuracy)
- examples/mobilenetv1_cifar10/train.mojo (fn compute_accuracy)

Features:
- Generic evaluate() function for any model with predict() method
- Automatic batching to avoid memory issues
- Proper tensor slicing and shape handling
- Detailed evaluation output with accuracy percentage
- Support for both int32 and int64 label types

"""

from shared.tensor.any_tensor import AnyTensor
from std.collections import List


def evaluate_with_predict(
    predictions: List[Int], labels: AnyTensor
) raises -> Float32:
    """Evaluate model using pre-computed predictions.

        Lightweight evaluation function for models where predictions have
        already been computed (e.g., from model.predict() in a loop).

    Args:
            predictions: List of predicted class indices.
            labels: Ground truth labels [batch_size].

    Returns:
            Accuracy as fraction in [0.0, 1.0].

    Raises:
            Error: If predictions and labels have different lengths.

    Example:
        ```mojo
        var predictions = List[Int]()
        for sample in test_images:
            predictions.append(model.predict(sample))
        var accuracy = evaluate_with_predict(predictions, test_labels)
        ```
    """
    if len(predictions) != labels._numel:
        raise Error(
            "evaluate_with_predict: predictions and labels must have same"
            " length"
        )

    var correct = 0
    for i in range(len(predictions)):
        var pred_val = predictions[i]
        var true_label = Int(labels[i])

        if pred_val == true_label:
            correct += 1

    return Float32(correct) / Float32(len(predictions))


def evaluate_logits_batch(
    logits: AnyTensor, labels: AnyTensor
) raises -> Float32:
    """Evaluate using logits (2D) by computing argmax per sample.

        Evaluates a batch of logits by computing argmax for each sample
        and comparing with true labels.

    Args:
            logits: Model logits of shape [batch_size, num_classes].
            labels: Ground truth labels [batch_size].

    Returns:
            Accuracy as fraction in [0.0, 1.0].

    Raises:
            Error: If shapes are incompatible.

    Example:
        ```mojo
        var logits = model.forward(test_images, training=False)
        var accuracy = evaluate_logits_batch(logits, test_labels)
        ```
    """
    var shape_vec = logits.shape()
    if len(shape_vec) != 2:
        raise Error(
            "evaluate_logits_batch: logits must be 2D [batch_size, num_classes]"
        )

    var batch_size = shape_vec[0]
    var num_classes = shape_vec[1]

    if batch_size != labels._numel:
        raise Error("evaluate_logits_batch: batch size mismatch")

    var correct = 0
    var logits_data = logits.data_ptr[DType.float32]()

    for i in range(batch_size):
        # Find argmax for this sample
        var max_idx = 0
        var max_val = logits_data[i * num_classes]

        for c in range(1, num_classes):
            var idx = i * num_classes + c
            if logits_data[idx] > max_val:
                max_val = logits_data[idx]
                max_idx = c

        # Compare with true label
        var true_label = Int(labels[i])
        if max_idx == true_label:
            correct += 1

    return Float32(correct) / Float32(batch_size)


def compute_accuracy_on_batch(
    predictions: AnyTensor, labels: AnyTensor
) raises -> Float32:
    """Compute accuracy for a single batch (simple utility).

        Lightweight function for computing accuracy on a single batch without
        batching logic. Useful for inline accuracy computation during training.

    Args:
            predictions: Model predictions/logits of shape [batch_size, num_classes]
                        or predicted class indices [batch_size].
            labels: Ground truth labels [batch_size].

    Returns:
            Accuracy as fraction in [0.0, 1.0].

    Raises:
            Error: If batch sizes don't match.

    Example:
        ```mojo
        # During training loop
        var batch_acc = compute_accuracy_on_batch(logits, batch_labels)
        print("Batch accuracy: ", batch_acc)
        ```
    """
    var pred_shape = predictions.shape()

    # Determine if predictions are logits (2D) or class indices (1D)
    var batch_size: Int
    if len(pred_shape) == 2:
        batch_size = pred_shape[0]
    elif len(pred_shape) == 1:
        batch_size = pred_shape[0]
    else:
        raise Error("compute_accuracy_on_batch: predictions must be 1D or 2D")

    # Validate batch size matches
    if batch_size != labels.shape()[0]:
        raise Error(
            "compute_accuracy_on_batch: batch size mismatch between predictions"
            " and labels"
        )

    var correct = 0

    if len(pred_shape) == 2:
        # Predictions are logits, compute argmax
        var num_classes = pred_shape[1]
        var pred_data = predictions.data_ptr[DType.float32]()

        for i in range(batch_size):
            # Find argmax for this sample
            var max_idx = 0
            var max_val = pred_data[i * num_classes]

            for c in range(1, num_classes):
                var idx = i * num_classes + c
                if pred_data[idx] > max_val:
                    max_val = pred_data[idx]
                    max_idx = c

            # Compare with true label
            var true_label = Int(labels[i])
            if max_idx == true_label:
                correct += 1
    else:
        # Predictions are already class indices
        for i in range(batch_size):
            var pred_val = Int(predictions[i])
            var true_label = Int(labels[i])

            if pred_val == true_label:
                correct += 1

    return Float32(correct) / Float32(batch_size)
