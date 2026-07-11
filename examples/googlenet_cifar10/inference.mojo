"""Inference Script for GoogLeNet on CIFAR-10.

This script loads a trained GoogLeNet model and evaluates it on the CIFAR-10 test set.

Usage:
    # Evaluate on test set
    mojo run examples/googlenet_cifar10/inference.mojo --weights-dir googlenet_weights

    # Evaluate on specific samples
    mojo run examples/googlenet_cifar10/inference.mojo --weights-dir googlenet_weights --samples 100

Features:
    - Loads saved model weights
    - Evaluates accuracy on CIFAR-10 test set
    - Reports per-class accuracy
    - Inference mode (no training, no batch norm running stats updates)
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.data.batch_utils import (
    compute_num_batches,
    extract_batch_pair,
)
from odyssey.data.constants import DatasetInfo
from odyssey.data.datasets import CIFAR10Dataset
from odyssey.training.metrics import evaluate_logits_batch
from model import GoogLeNet


# CIFAR-10 class names (runtime list)
def get_class_names() -> List[String]:
    """Get CIFAR-10 class names."""
    var classes: List[String] = []
    classes.append("airplane")
    classes.append("automobile")
    classes.append("bird")
    classes.append("cat")
    classes.append("deer")
    classes.append("dog")
    classes.append("frog")
    classes.append("horse")
    classes.append("ship")
    classes.append("truck")
    return classes^


def evaluate_model(
    mut model: GoogLeNet,
    images: AnyTensor,
    labels: AnyTensor,
    batch_size: Int = 100,
    verbose: Bool = True,
) raises -> Tuple[Float32, List[Int], List[Int]]:
    """Evaluate model on a dataset.

    Args:
        model: GoogLeNet model.
        images: Input images (N, 3, 32, 32).
        labels: Ground truth labels (N,).
        batch_size: Batch size for evaluation.
        verbose: Print progress during evaluation.

    Returns:
        Tuple of (accuracy, correct_per_class, total_per_class)
        - accuracy: Overall accuracy as percentage (0-100)
        - correct_per_class: Correct predictions per class (10,)
        - total_per_class: Total samples per class (10,).
    """
    var num_samples = images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)

    var total_correct = 0
    var correct_per_class = List[Int](capacity=10)
    var total_per_class = List[Int](capacity=10)

    # Initialize counters
    for _ in range(10):
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

        # Forward pass (inference mode - no BN running stats updates)
        var logits = model.forward(batch_images, training=False)

        # Compute batch accuracy using shared function
        var batch_acc_fraction = evaluate_logits_batch(logits, batch_labels)
        var batch_correct = Int(
            batch_acc_fraction * Float32(current_batch_size)
        )
        total_correct += batch_correct

        # Update per-class counters
        var logits_data = logits._data.bitcast[Float32]()
        for i in range(current_batch_size):
            # Find argmax (predicted class)
            var pred_class = 0
            var max_logit = logits_data[i * 10]
            for j in range(1, 10):
                if logits_data[i * 10 + j] > max_logit:
                    max_logit = logits_data[i * 10 + j]
                    pred_class = j

            # Get true label
            var true_class = Int(batch_labels[i])

            # Update counters
            total_per_class[true_class] += 1
            if pred_class == true_class:
                correct_per_class[true_class] += 1

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

    var overall_accuracy = Float32(total_correct) / Float32(num_samples) * 100.0

    if verbose:
        print("Evaluation complete!")
        print()

    return (overall_accuracy, correct_per_class^, total_per_class^)


def print_detailed_results(
    accuracy: Float32, correct_per_class: List[Int], total_per_class: List[Int]
):
    """Print detailed evaluation results.

    Args:
        accuracy: Overall accuracy percentage.
        correct_per_class: Correct predictions per class.
        total_per_class: Total samples per class.
    """
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()

    print("Overall Accuracy: " + String(accuracy) + "%")
    print()

    print("Per-Class Accuracy:")
    print("-" * 60)
    print(
        String("Class")
        + " "
        + String("Correct")
        + " "
        + String("Total")
        + " "
        + String("Accuracy")
    )
    print("-" * 60)

    var class_names = get_class_names()
    for i in range(10):
        var class_name = class_names[i]
        var correct = correct_per_class[i]
        var total = total_per_class[i]
        var class_acc = Float32(correct) / Float32(
            total
        ) * 100.0 if total > 0 else Float32(0.0)

        print(
            String(class_name)
            + " "
            + String(correct)
            + " "
            + String(total)
            + " "
            + String(class_acc)
            + "%"
        )

    print("-" * 60)
    print()


def main() raises:
    """Main inference entry point."""
    print("=" * 60)
    print("GoogLeNet Inference on CIFAR-10")
    print("=" * 60)
    print()

    # Configuration
    var weights_dir = "googlenet_weights"  # Default weights directory
    var batch_size = 100  # Batch size for evaluation
    var evaluate_all = True  # Evaluate on full test set

    print("Configuration:")
    print("  Weights directory: " + String(weights_dir))
    print("  Batch size: " + String(batch_size))
    print("  Evaluate full test set: " + String(evaluate_all))
    print()

    # Load CIFAR-10 test set
    print("Loading CIFAR-10 test set...")
    var dataset = CIFAR10Dataset("datasets/cifar10")
    var test_data = dataset.get_test_data()
    var test_images = test_data[0]
    var test_labels = test_data[1]

    var num_samples = test_images.shape()[0]
    print("  Test samples: " + String(num_samples))
    print("  Image shape: (3, 32, 32)")
    print("  Number of classes: 10")
    print()

    # Initialize model
    print("Initializing GoogLeNet model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = GoogLeNet(num_classes=dataset_info.num_classes())
    print("  Model architecture: GoogLeNet (Inception-v1)")
    print("  Parameters: ~6.8M")
    print("  Inception modules: 9")
    print()

    # Load weights
    print("Loading weights from " + String(weights_dir) + "/...")
    try:
        model.load_weights(weights_dir)
        print("  ✓ Weights loaded successfully")
        print()
    except e:
        print("  ✗ Failed to load weights: " + String(e))
        print()
        print("ERROR: Cannot proceed without trained weights.")
        print()
        print("To train a model, run:")
        print("  python examples/googlenet_cifar10/download_cifar10.py")
        print("  mojo run examples/googlenet_cifar10/train.mojo --epochs 200")
        print()
        return

    # Run inference
    print("Running inference on test set...")
    print()

    var results = evaluate_model(
        model, test_images, test_labels, batch_size, verbose=True
    )

    # Print detailed results (copy lists since they're small - 10 elements each)
    print_detailed_results(results[0], results[1].copy(), results[2].copy())

    # Performance context
    print("Performance Context:")
    print("-" * 60)
    print("Expected GoogLeNet accuracy on CIFAR-10:")
    print("  - Without data augmentation: 90-92%")
    print("  - With data augmentation: 92-94%")
    print("  - State-of-the-art: 94-96%")
    print()
    print("Training details:")
    print("  - 200 epochs recommended")
    print("  - SGD with momentum (0.9)")
    print("  - Learning rate: 0.01 with step decay")
    print("  - Batch size: 128")
    print("  - Dropout: 0.4 before final FC layer")
    print("-" * 60)
    print()

    print("Inference complete!")
