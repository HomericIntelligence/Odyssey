"""Inference Script for LeNet-5 on EMNIST

Runs inference with trained LeNet-5 model on EMNIST test set.

Usage:
    # Run on test set
    mojo run examples/lenet-emnist/inference.mojo --weights-dir lenet5_weights

Requirements:
    - Trained model weights (from train.mojo)
    - EMNIST dataset (for test set evaluation)

References:
    - LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
      Gradient-based learning applied to document recognition.
    - Reference Implementation: https://github.com/mattwang44/LeNet-from-Scratch
"""

from model import LeNet5
from shared.data import load_idx_labels, load_idx_images, normalize_images
from shared.core import ExTensor, zeros
from shared.utils.arg_parser import ArgumentParser
from shared.training.metrics import (
    evaluate_with_predict,
    top1_accuracy,
    AccuracyMetric,
)
from collections import List
from math import exp

# Default number of classes for EMNIST Balanced dataset
comptime DEFAULT_NUM_CLASSES = 47

# EMNIST Balanced class mapping (47 classes)
# 0-9: digits, 10-35: uppercase letters, 36-46: lowercase letters (select)
comptime CLASS_NAMES: List[String] = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",  # 0-9
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",  # 10-19
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",  # 20-29
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",  # 30-35
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "n",
    "q",
    "r",
    "t",  # 36-46 (select lowercase)
]


struct InferenceConfig:
    """Inference configuration from command line arguments."""

    var weights_dir: String
    var data_dir: String

    fn __init__(out self, weights_dir: String, data_dir: String):
        self.weights_dir = weights_dir
        self.data_dir = data_dir


struct PredictionResult:
    """Result from a single prediction."""

    var predicted_class: Int
    var confidence: Float32

    fn __init__(out self, predicted_class: Int, confidence: Float32):
        self.predicted_class = predicted_class
        self.confidence = confidence


struct EvaluationResult:
    """Result from evaluating on a dataset."""

    var accuracy: Float32
    var num_correct: Int
    var num_total: Int

    fn __init__(out self, accuracy: Float32, num_correct: Int, num_total: Int):
        self.accuracy = accuracy
        self.num_correct = num_correct
        self.num_total = num_total


fn parse_args() raises -> InferenceConfig:
    """Parse command line arguments using enhanced argument parser.

    Returns:
        InferenceConfig with parsed arguments.
    """
    var parser = ArgumentParser()
    parser.add_argument("weights-dir", "string", "lenet5_weights")
    parser.add_argument("data-dir", "string", "datasets/emnist")

    var args = parser.parse()

    var weights_dir = args.get_string("weights-dir", "lenet5_weights")
    var data_dir = args.get_string("data-dir", "datasets/emnist")

    return InferenceConfig(weights_dir, data_dir)


fn infer_single(mut model: LeNet5, image: ExTensor) raises -> PredictionResult:
    """Run inference on a single image.

    Args:
        model: Trained LeNet-5 model.
        image: Input image of shape (1, 1, 28, 28).

    Returns:
        PredictionResult with predicted class and confidence.
    """
    var logits = model.forward(image)

    # Find argmax and max value
    var num_classes = logits.shape()[1]
    var max_idx = 0
    var max_val_data = logits._data.bitcast[Float32]()
    var max_val = max_val_data[0]

    for i in range(1, num_classes):
        var val = max_val_data[i]
        if val > max_val:
            max_val = val
            max_idx = i

    # Compute softmax for confidence (simplified - just exp(max) / sum(exp))
    var exp_sum = Float32(0.0)
    for i in range(num_classes):
        exp_sum += exp(max_val_data[i])

    var confidence = exp(max_val) / exp_sum

    return PredictionResult(max_idx, confidence)


fn evaluate_test_set(
    mut model: LeNet5, images: ExTensor, labels: ExTensor
) raises -> EvaluationResult:
    """Evaluate model on entire test set.

    Args:
        model: Trained LeNet-5 model.
        images: Test images of shape (num_samples, 1, 28, 28).
        labels: Test labels of shape (num_samples,).

    Returns:
        EvaluationResult with accuracy and counts.
    """
    var num_samples = images.shape()[0]

    print("Evaluating on", num_samples, "test samples...")

    var predictions = List[Int]()

    # Collect predictions for all samples
    for i in range(num_samples):
        var sample = images.slice(i, i + 1, axis=0)
        var pred_class = model.predict(sample)
        predictions.append(pred_class)

        # Print progress
        if (i + 1) % 100 == 0:
            print("  Processed [", i + 1, "/", num_samples, "]")

    # Compute accuracy using shared function
    var accuracy = evaluate_with_predict(predictions, labels)
    var num_correct = Int(accuracy * Float32(num_samples))

    return EvaluationResult(accuracy, num_correct, num_samples)


fn main() raises:
    """Main inference entry point."""
    print("=" * 60)
    print("LeNet-5 Inference on EMNIST Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var weights_dir = config.weights_dir
    var data_dir = config.data_dir

    print("\nConfiguration:")
    print("  Weights Directory: ", weights_dir)
    print("  Data Directory: ", data_dir)
    print()

    # Initialize model
    print("Initializing LeNet-5 model...")
    var model = LeNet5(num_classes=DEFAULT_NUM_CLASSES)
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load weights
    print("Loading model weights...")
    model.load_weights(weights_dir)
    print("  Weights loaded from", weights_dir)
    print()

    # Load test dataset
    print("Loading EMNIST test set...")
    var test_images_path = data_dir + "/emnist-balanced-test-images-idx3-ubyte"
    var test_labels_path = data_dir + "/emnist-balanced-test-labels-idx1-ubyte"

    var test_images_raw = load_idx_images(test_images_path)
    var test_labels = load_idx_labels(test_labels_path)

    # Normalize images
    var test_images = normalize_images(test_images_raw)

    print("  Test samples: ", test_images.shape()[0])
    print()

    # Run inference on test set
    print("Running inference on test set...")
    var result = evaluate_test_set(model, test_images, test_labels)
    var accuracy = result.accuracy
    var correct = result.num_correct
    var total = result.num_total

    print()
    print("Results:")
    print("  Correct:", correct, "/", total)
    print("  Accuracy:", accuracy * 100.0, "%")
    print()

    print("Inference complete!")
    print("\nThis implementation demonstrates the inference structure.")
    print(
        "Batch processing will be more efficient when tensor slicing is"
        " optimized."
    )
