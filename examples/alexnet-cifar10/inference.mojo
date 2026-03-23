"""Inference Script for AlexNet on CIFAR-10

Loads trained weights and evaluates on CIFAR-10 test set.

Usage:
    mojo run examples/alexnet-cifar10/inference.mojo --weights-dir alexnet_weights

Requirements:
    - Trained model weights saved in weights-dir
    - CIFAR-10 test set downloaded (run: python examples/alexnet-cifar10/download_cifar10.py)

References:
    - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
      ImageNet classification with deep convolutional neural networks.
      Advances in Neural Information Processing Systems, 25, 1097-1105.
"""

from model import AlexNet
from shared.data.datasets import CIFAR10Dataset
from shared.data import DatasetInfo
from shared.tensor.any_tensor import AnyTensor
from shared.utils.arg_parser import ArgumentParser
from shared.training.metrics import evaluate_with_predict


fn parse_args() raises -> Tuple[String, String]:
    """Parse command line arguments using enhanced argument parser.

    Returns:
        Tuple of (weights_dir, data_dir).
    """
    var parser = ArgumentParser()
    parser.add_argument("weights-dir", "string", "alexnet_weights")
    parser.add_argument("data-dir", "string", "datasets/cifar10")

    var args = parser.parse()

    var weights_dir = args.get_string("weights-dir", "alexnet_weights")
    var data_dir = args.get_string("data-dir", "datasets/cifar10")

    return Tuple[String, String](weights_dir, data_dir)


fn evaluate_model(
    mut model: AlexNet, test_images: AnyTensor, test_labels: AnyTensor
) raises -> Tuple[Float32, Float32]:
    """Evaluate model on test set with Top-1 and Top-5 accuracy.

    Args:
        model: AlexNet model with loaded weights.
        test_images: Test images (10000, 3, 32, 32).
        test_labels: Test labels (10000,).

    Returns:
        Tuple of (top1_accuracy, top5_accuracy)

    Note:
        Top-5 accuracy: Model is correct if true label is in top 5 predictions
        For CIFAR-10 (10 classes), Top-5 is less meaningful but included for completeness.
    """
    var num_samples = test_images.shape()[0]

    print("Evaluating on test set...")
    print("  Total samples:", num_samples)

    # Process samples and collect predictions
    var predictions = List[Int]()
    var eval_samples = min(1000, num_samples)

    for i in range(eval_samples):
        # Extract single sample
        var sample = test_images.slice(i, i + 1, axis=0)
        var logits = model.forward(sample, training=False)

        # Compute Top-1 prediction (highest logit)
        var pred_class = _argmax(logits)
        predictions.append(pred_class)

        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            print("  Processed [", i + 1, "/", eval_samples, "]")

    # Compute Top-1 accuracy using shared function
    var eval_labels = test_labels.slice(0, eval_samples, axis=0)
    var top1_accuracy_fraction = evaluate_with_predict(predictions, eval_labels)

    # Compute Top-5 accuracy (true label in top 5 predictions)
    var correct_top5 = 0
    for i in range(eval_samples):
        var sample = test_images.slice(i, i + 1, axis=0)
        var logits = model.forward(sample, training=False)
        var true_label = Int(test_labels[i])
        var top5_indices = _top_k_indices(logits, k=5)

        for j in range(5):
            if top5_indices[j] == true_label:
                correct_top5 += 1
                break

    var top5_accuracy = Float32(correct_top5) / Float32(eval_samples)

    print()
    print("Final Results:")
    print(
        "  Top-1 Accuracy: ",
        top1_accuracy_fraction * 100.0,
        "% (",
        Int(top1_accuracy_fraction * Float32(eval_samples)),
        "/",
        eval_samples,
        ")",
    )
    print(
        "  Top-5 Accuracy: ",
        top5_accuracy * 100.0,
        "% (",
        correct_top5,
        "/",
        eval_samples,
        ")",
    )

    return (top1_accuracy_fraction, top5_accuracy)


fn _argmax(tensor: AnyTensor) raises -> Int:
    """Find index of maximum value in 1D tensor.

    Args:
        tensor: 1D tensor

    Returns:
        Index of maximum value.
    """
    var shape = tensor.shape()
    var max_idx = 0
    var max_val = tensor[0]

    for i in range(1, shape[1]):
        if tensor[i] > max_val:
            max_val = tensor[i]
            max_idx = i

    return max_idx


fn _top_k_indices(tensor: AnyTensor, k: Int) raises -> List[Int]:
    """Find indices of top-k maximum values in 1D tensor.

    Args:
        tensor: 1D tensor
        k: Number of top values to find

    Returns:
        DynamicVector of indices (length k) sorted by descending value

    Note:
        Simple implementation using repeated argmax (not optimal but clear).
    """
    var shape = tensor.shape()
    var num_classes = shape[1]
    var indices = List[Int]()

    # Create a copy of tensor values for modification
    var values = List[Float32]()
    var tensor_data = tensor._data.bitcast[Float32]()
    for i in range(num_classes):
        values.append(tensor_data[i])

    # Find top-k by repeatedly finding max and setting it to -inf
    for _ in range(k):
        var max_idx = 0
        var max_val = values[0]

        for i in range(1, num_classes):
            if values[i] > max_val:
                max_val = values[i]
                max_idx = i

        indices.append(max_idx)
        values[max_idx] = Float32(-1e9)  # Set to very negative value

    return indices^


fn print_class_names():
    """Print CIFAR-10 class names for reference."""
    print("\nCIFAR-10 Classes:")
    print("  0: airplane")
    print("  1: automobile")
    print("  2: bird")
    print("  3: cat")
    print("  4: deer")
    print("  5: dog")
    print("  6: frog")
    print("  7: horse")
    print("  8: ship")
    print("  9: truck")
    print()


fn main() raises:
    """Main inference routine."""
    print("=" * 60)
    print("AlexNet Inference on CIFAR-10 Dataset")
    print("=" * 60)

    # Parse arguments
    var config = parse_args()
    var weights_dir = config[0]
    var data_dir = config[1]

    print("\nConfiguration:")
    print("  Weights Directory: ", weights_dir)
    print("  Data Directory: ", data_dir)
    print()

    # Print class names for reference
    print_class_names()

    # Initialize model
    print("Initializing AlexNet model...")
    var dataset_info = DatasetInfo("cifar10")
    var model = AlexNet(
        num_classes=dataset_info.num_classes(), dropout_rate=0.5
    )
    print("  Model initialized with", model.num_classes, "classes")
    print()

    # Load trained weights
    print("Loading trained weights...")
    model.load_weights(weights_dir)
    print("  Weights loaded from", weights_dir)
    print()

    # Load test dataset
    print("Loading CIFAR-10 test set...")
    var cifar_dataset = CIFAR10Dataset(data_dir)
    var test_data = cifar_dataset.get_test_data()
    var test_images = test_data[0]
    var test_labels = test_data[1]
    print("  Test samples: ", test_images.shape()[0])
    print()

    # Evaluate model
    _ = evaluate_model(model, test_images, test_labels)

    print()
    print("Inference complete!")
    print(
        "\nNote: This implementation demonstrates the full inference structure."
    )
    print(
        "Batch processing will be more efficient when tensor slicing is"
        " optimized."
    )
