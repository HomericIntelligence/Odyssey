"""Inference Script for ResNet-18 on CIFAR-10

This script loads a trained ResNet-18 model and evaluates it on the CIFAR-10 test set.

Shared Modules Used:
    - shared.core: Tensor operations and ExTensor type
    - shared.data: Batch extraction and data utilities
    - shared.data.datasets: CIFAR-10 test set loading
    - shared.training.metrics: Evaluation and accuracy computation
    - shared.utils.arg_parser: Command-line argument parsing
    - shared.utils.serialization: Model weight loading

Usage:
    # Evaluate on test set
    mojo run examples/resnet18-cifar10/inference.mojo --weights-dir resnet18_weights

    # Evaluate on specific samples
    mojo run examples/resnet18-cifar10/inference.mojo --weights-dir resnet18_weights --samples 100

Features:
    - Loads saved model weights via shared.utils.serialization
    - Evaluates accuracy on CIFAR-10 test set using shared metrics
    - Reports per-class accuracy using shared.training.metrics
    - Inference mode (no training, no batch norm running stats updates)
"""

from shared.core import ExTensor, zeros
from shared.data import extract_batch_pair, compute_num_batches, DatasetInfo
from shared.data.datasets import CIFAR10Dataset
from shared.training.metrics import (
    evaluate_with_predict,
    top1_accuracy,
    per_class_accuracy,
    evaluate_logits_batch,
)
from shared.utils.arg_parser import ArgumentParser, ArgumentSpec
from shared.utils.serialization import load_tensor
from model import ResNet18


# CIFAR-10 class names
comptime CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


fn evaluate_model(
    mut model: ResNet18,
    images: ExTensor,
    labels: ExTensor,
    batch_size: Int = 100,
    verbose: Bool = True,
) raises -> Tuple[Float32, List[Int], List[Int]]:
    """Evaluate model on a dataset.

    Args:
        model: ResNet-18 model.
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


fn print_detailed_results(
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

    for i in range(10):
        # Get class name at runtime
        var class_name: String
        if i == 0:
            class_name = "airplane"
        elif i == 1:
            class_name = "automobile"
        elif i == 2:
            class_name = "bird"
        elif i == 3:
            class_name = "cat"
        elif i == 4:
            class_name = "deer"
        elif i == 5:
            class_name = "dog"
        elif i == 6:
            class_name = "frog"
        elif i == 7:
            class_name = "horse"
        elif i == 8:
            class_name = "ship"
        else:
            class_name = "truck"

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


fn main() raises:
    """Main inference entry point.

    Integrates command-line argument parsing via shared.utils.arg_parser.
    """
    print("=" * 60)
    print("ResNet-18 Inference on CIFAR-10")
    print("=" * 60)
    print()

    # Parse command-line arguments using shared.utils.arg_parser
    var parser = ArgumentParser()

    # Add inference arguments with defaults
    parser.add_argument("weights-dir", "string", "resnet18_weights")
    parser.add_argument("batch-size", "int", "100")
    parser.add_argument("data-dir", "string", "datasets/cifar10")

    # Parse provided arguments (simplified - using defaults for this demo)
    var weights_dir = "resnet18_weights"  # Default weights directory
    var batch_size = 100  # Batch size for evaluation
    var data_dir = "datasets/cifar10"
    var evaluate_all = True  # Evaluate on full test set

    print("Configuration:")
    print("  Weights directory: " + String(weights_dir))
    print("  Batch size: " + String(batch_size))
    print("  Data directory: " + String(data_dir))
    print("  Evaluate full test set: " + String(evaluate_all))
    print()

    # Load CIFAR-10 test set
    print("Loading CIFAR-10 test set...")
    var cifar10_dataset = CIFAR10Dataset(data_dir)
    var test_data = cifar10_dataset.get_test_data()
    var test_images = test_data[0]
    var test_labels = test_data[1]

    var num_samples = test_images.shape()[0]
    print("  Test samples: " + String(num_samples))
    print("  Image shape: (3, 32, 32)")
    print("  Number of classes: 10")
    print()

    # Initialize model
    print("Initializing ResNet-18 model...")
    var dataset_info = DatasetInfo("cifar10")
    var num_classes = dataset_info.num_classes()
    var model = ResNet18(num_classes=num_classes)
    print("  Model architecture: ResNet-18")
    print("  Parameters: ~11M")
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
        print("  python examples/resnet18-cifar10/download_cifar10.py")
        print("  mojo run examples/resnet18-cifar10/train.mojo --epochs 200")
        print()
        print("Training requires batch_norm2d_backward (see GAP_ANALYSIS.md).")
        return

    # Run inference
    print("Running inference on test set...")
    print()

    var eval_result = evaluate_model(
        model, test_images, test_labels, batch_size, verbose=True
    )
    var accuracy = eval_result[0]
    var correct_per_class = eval_result[1].copy()
    var total_per_class = eval_result[2].copy()

    # Print detailed results
    print_detailed_results(accuracy, correct_per_class^, total_per_class^)

    # Performance context
    print("Performance Context:")
    print("-" * 60)
    print("Expected ResNet-18 accuracy on CIFAR-10:")
    print("  - Without data augmentation: 91-93%")
    print("  - With data augmentation: 93-95%")
    print("  - State-of-the-art: 96-97%")
    print()
    print("Training details:")
    print("  - 200 epochs recommended")
    print("  - SGD with momentum (0.9)")
    print("  - Learning rate: 0.01 with step decay")
    print("  - Batch size: 128")
    print("-" * 60)
    print()

    print("Inference complete!")
