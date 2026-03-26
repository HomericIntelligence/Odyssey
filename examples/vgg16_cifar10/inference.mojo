"""Inference script for VGG-16 on CIFAR-10

Loads trained weights and evaluates on test set.

Usage:
    mojo run examples/vgg16_cifar10/inference.mojo --weights-dir vgg16_weights --data-dir datasets/cifar10
"""

from shared.tensor.any_tensor import AnyTensor, zeros
from shared.data.formats import load_cifar10_batch
from shared.data.datasets import CIFAR10Dataset
from shared.data import extract_batch_pair, DatasetInfo
from shared.training.metrics import evaluate_with_predict
from model import VGG16
from shared.utils.arg_parser import ArgumentParser
from collections import List


fn parse_args() raises -> Tuple[String, String]:
    """Parse command line arguments using enhanced argument parser.

    Returns:
        Tuple of (weights_dir, data_dir).
    """
    var parser = ArgumentParser()
    parser.add_argument("weights-dir", "string", "vgg16_weights")
    parser.add_argument("data-dir", "string", "datasets/cifar10")

    var args = parser.parse()

    var weights_dir = args.get_string("weights-dir", "vgg16_weights")
    var data_dir = args.get_string("data-dir", "datasets/cifar10")

    return Tuple[String, String](weights_dir, data_dir)


fn compute_test_accuracy(
    mut model: VGG16, test_images: AnyTensor, test_labels: AnyTensor
) raises -> Float32:
    """Compute accuracy on test set using shared metrics utilities.

    Uses evaluate_with_predict from shared.training.metrics to consolidate
    evaluation logic across all examples.

    Args:
        model: VGG-16 model with loaded weights.
        test_images: Test images of shape (num_samples, 3, 32, 32).
        test_labels: Test labels of shape (num_samples,).

    Returns:
        Accuracy as percentage (0-100).
    """
    var test_shape = test_images.shape()
    var num_test_samples = test_shape[0]

    print("Evaluating on " + String(num_test_samples) + " test samples...")

    # Collect predictions using model.predict()
    var predictions = List[Int]()

    # Process each test sample
    for i in range(num_test_samples):
        # Get single sample from test set
        var sample = test_images.slice(i, i + 1, axis=0)

        # Forward pass (inference mode)
        var pred_class = model.predict(sample)
        predictions.append(pred_class)

        # Print progress every 1000 samples
        if (i + 1) % 1000 == 0:
            print(
                "  Processed " + String(i + 1) + "/" + String(num_test_samples)
            )

    # Use shared evaluate function
    var accuracy_fraction = evaluate_with_predict(predictions, test_labels)
    var accuracy = accuracy_fraction * 100.0
    return accuracy


fn main() raises:
    """Main inference function."""
    print("=== VGG-16 Inference on CIFAR-10 ===")
    print()

    # Parse command-line arguments
    var parsed = parse_args()
    var weights_dir = parsed[0]
    var data_dir = parsed[1]

    # Load test set using shared data loading utilities
    print("Loading CIFAR-10 test set...")
    var dataset = CIFAR10Dataset(data_dir)
    var test_data = dataset.get_test_data()
    var test_images = test_data[0]
    var test_labels = test_data[1]

    print("Test set loaded: ", test_images.shape()[0], " images")
    print()

    # Initialize model
    print("Initializing VGG-16 model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = VGG16(num_classes=dataset_info.num_classes(), dropout_rate=0.5)
    print("Model initialized")
    print()

    # Load trained weights using model save/load utilities
    print("Loading weights from " + weights_dir + "...")
    model.load_weights(weights_dir)
    print("Weights loaded successfully")
    print()

    # Evaluate on test set using shared metrics utilities
    var test_accuracy = compute_test_accuracy(model, test_images, test_labels)

    print()
    print("=" * 50)
    print("Test Accuracy: " + String(test_accuracy) + "%")
    print("=" * 50)
