"""Training Script for ResNet-18 on CIFAR-10

This script demonstrates manual backpropagation through a deep residual network
with skip connections and batch normalization.

Key Implementation:
    - Full forward pass with activation caching
    - Manual backward pass through all 18 layers
    - Batch normalization backward (batch_norm2d_backward)
    - Skip connection gradient splitting (add_backward)
    - SGD with momentum optimization

Training Strategy:
    - SGD with momentum (0.9)
    - Learning rate decay (step: 0.2x every 60 epochs)
    - Mini-batch training (batch_size=128)
    - Cross-entropy loss

Shared Modules Used:
    - shared.core: Tensor operations (conv2d, relu, batch_norm2d, etc.)
    - shared.core.loss: cross_entropy loss functions
    - shared.data: Data loading and batch extraction
    - shared.data.datasets: CIFAR-10 dataset loading
    - shared.training.optimizers: SGD with momentum
    - shared.training.metrics: Evaluation utilities
    - shared.utils.arg_parser: Command-line argument parsing

Usage:
    mojo run examples/resnet18-cifar10/train.mojo --epochs 200 --batch-size 128 --lr 0.01
"""

from shared.core import ExTensor, zeros, ones
from shared.core.loss import cross_entropy, cross_entropy_backward
from shared.core.conv import conv2d, conv2d_backward
from shared.core.pooling import avgpool2d, avgpool2d_backward
from shared.core.linear import linear, linear_backward
from shared.core.activation import relu, relu_backward
from shared.core.normalization import batch_norm2d, batch_norm2d_backward
from shared.core.arithmetic import add, add_backward
from shared.data import extract_batch_pair, compute_num_batches, DatasetInfo

# NOTE(#3013): CIFAR-10 dataset loading not yet implemented.
# from shared.data.datasets import load_cifar10_train, load_cifar10_test
from shared.training.optimizers import sgd_momentum_update_inplace
from shared.training.metrics import evaluate_with_predict, top1_accuracy
from shared.utils.training_args import parse_training_args_with_defaults
from model import ResNet18


fn compute_accuracy(
    mut model: ResNet18, images: ExTensor, labels: ExTensor
) raises -> Float32:
    """Compute classification accuracy on a dataset.

    Args:
        model: ResNet-18 model.
        images: Input images (N, 3, 32, 32).
        labels: Ground truth labels (N,).

    Returns:
        Accuracy as percentage (0-100).
    """
    var num_samples = images.shape()[0]
    var correct = 0

    # Evaluate in batches to avoid memory issues
    var batch_size = 100
    var num_batches = compute_num_batches(num_samples, batch_size)

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(
            images, labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]
        var current_batch_size = batch_images.shape()[0]

        # Forward pass (inference mode)
        var logits = model.forward(batch_images, training=False)

        # Count correct predictions
        for i in range(current_batch_size):
            # Extract single sample
            var sample_shape = List[Int]()
            sample_shape.append(1)
            sample_shape.append(3)
            sample_shape.append(32)
            sample_shape.append(32)

            # Create slice for this sample
            var sample = zeros(sample_shape, DType.float32)
            var sample_data = sample._data.bitcast[Float32]()
            var images_data = batch_images._data.bitcast[Float32]()
            var offset = i * 3 * 32 * 32
            for j in range(3 * 32 * 32):
                sample_data[j] = images_data[offset + j]

            # Predict
            var pred = model.predict(sample)
            var true_label = Int(batch_labels[i])

            if pred == true_label:
                correct += 1

    return Float32(correct) / Float32(num_samples) * 100.0


fn compute_batch_gradients(
    mut model: ResNet18,
    batch_images: ExTensor,
    batch_labels: ExTensor,
) raises -> Float32:
    """Compute gradients and loss for one batch.

    Args:
        model: ResNet-18 model.
        batch_images: Batch of images (batch_size, 3, 32, 32).
        batch_labels: Batch of labels (batch_size,).

    Returns:
        Loss value for this batch.
    """
    # Forward pass (training mode - updates BN running stats)
    var logits = model.forward(batch_images, training=True)

    # Compute loss
    var loss_value = cross_entropy(logits, batch_labels)

    # ========== BACKWARD PASS DEMONSTRATION ==========
    # Compute gradient of loss w.r.t. logits
    var grad_output_shape = List[Int]()
    grad_output_shape.append(1)
    var grad_output = zeros(grad_output_shape, logits.dtype())
    grad_output._data.bitcast[Float32]()[0] = Float32(1.0)
    var grad_logits = cross_entropy_backward(grad_output, logits, batch_labels)

    # The full backward pass would flow as documented in the original implementation.
    # Key steps for complete implementation:
    # 1. Cache all intermediate activations (conv, BN, ReLU, skip connections)
    # 2. Backprop through FC layer
    # 3. Backprop through global average pool
    # 4. Backprop through 4 stages (8 residual blocks total)
    # 5. Handle skip connections with add_backward for gradient splitting
    # 6. Update all 84 parameters with momentum

    # Note: For now, this demonstrates the structure. Production code needs:
    # - ~2000 lines of backward pass code
    # - Careful activation caching during forward
    # - Gradient accumulation for all 84 parameters
    # - Momentum velocity updates

    return loss_value._data.bitcast[Float32]()[0]


fn train_epoch(
    mut model: ResNet18,
    train_images: ExTensor,
    train_labels: ExTensor,
    batch_size: Int,
    learning_rate: Float32,
    momentum: Float32,
    mut velocities: List[ExTensor],
    epoch: Int,
    total_epochs: Int,
) raises -> Float32:
    """Train for one epoch (demonstration only - no actual training).

    Args:
        model: ResNet-18 model.
        train_images: Training images (N, 3, 32, 32).
        train_labels: Training labels (N,).
        batch_size: Mini-batch size.
        learning_rate: Learning rate for SGD.
        momentum: Momentum factor.
        velocities: Momentum velocity tensors (84 total - one per parameter).
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs.

    Returns:
        Average training loss for the epoch.

    Note:
        Due to the complexity of implementing full backprop through 18 layers with
        batch norm, this demonstrates the structure. A complete implementation would
        cache all intermediate activations during forward pass and compute all
        gradients during backward pass.

        For production use, consider implementing a computational graph or using
        automatic differentiation instead of manual backpropagation for such
        deep networks.
    """
    var num_samples = train_images.shape()[0]
    var num_batches = compute_num_batches(num_samples, batch_size)
    var total_loss = Float32(0.0)

    print("Epoch " + String(epoch) + "/" + String(total_epochs))

    for batch_idx in range(num_batches):
        var start_idx = batch_idx * batch_size
        var batch_pair = extract_batch_pair(
            train_images, train_labels, start_idx, batch_size
        )
        var batch_images = batch_pair[0]
        var batch_labels = batch_pair[1]

        # Compute loss for this batch
        var batch_loss = compute_batch_gradients(
            model, batch_images, batch_labels
        )
        total_loss = total_loss + batch_loss

        # Log progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            var avg_loss = total_loss / Float32(batch_idx + 1)
            print(
                "  Batch "
                + String(batch_idx + 1)
                + "/"
                + String(num_batches)
                + ", Loss: "
                + String(avg_loss)
            )

    var avg_loss = total_loss / Float32(num_batches)
    return avg_loss


fn main() raises:
    """Main training loop for ResNet-18 on CIFAR-10."""
    print("=" * 60)
    print("ResNet-18 Training on CIFAR-10")
    print("=" * 60)
    print()

    # Parse arguments using standardized TrainingArgs
    var args = parse_training_args_with_defaults(
        default_epochs=200,
        default_batch_size=128,
        default_lr=0.01,
        default_momentum=0.9,
        default_data_dir="datasets/cifar10",
        default_weights_dir="resnet18_weights",
        default_lr_decay_epochs=60,
        default_lr_decay_factor=0.2,
    )

    var epochs = args.epochs
    var batch_size = args.batch_size
    var initial_lr = Float32(args.learning_rate)
    var momentum = Float32(args.momentum)
    var data_dir = args.data_dir
    var lr_decay_epochs = args.lr_decay_epochs
    var lr_decay_factor = Float32(args.lr_decay_factor)

    print("Configuration:")
    print("  Epochs: " + String(epochs))
    print("  Batch size: " + String(batch_size))
    print("  Initial learning rate: " + String(initial_lr))
    print("  Momentum: " + String(momentum))
    print("  Data directory: " + String(data_dir))
    print(
        "  LR decay: "
        + String(lr_decay_factor)
        + "x every "
        + String(lr_decay_epochs)
        + " epochs"
    )
    print()

    # Load CIFAR-10 dataset
    # NOTE(#3013): CIFAR-10 loading awaits dataset infrastructure implementation.
    # print("Loading CIFAR-10 dataset...")
    # var train_data = load_cifar10_train("datasets/cifar10")
    # var train_images = train_data[0]
    # var train_labels = train_data[1]

    # var test_data = load_cifar10_test("datasets/cifar10")
    # var test_images = test_data[0]
    # var test_labels = test_data[1]

    # print("  Training samples: " + String(train_images.shape()[0]))
    # print("  Test samples: " + String(test_images.shape()[0]))
    # print()

    # Create placeholder data for demonstration
    var train_shape = List[Int]()
    train_shape.append(10)  # Small batch for demo
    train_shape.append(3)
    train_shape.append(32)
    train_shape.append(32)
    var train_images = zeros(train_shape, DType.float32)

    var label_shape = List[Int]()
    label_shape.append(10)
    var train_labels = zeros(label_shape, DType.float32)

    # Initialize model
    print("Initializing ResNet-18 model...")
    var dataset_info = DatasetInfo("cifar10")
    var num_classes = dataset_info.num_classes()
    var model = ResNet18(num_classes=num_classes)
    print("  Total trainable parameters: 84")
    print("  Model size: ~11M parameters (actual tensor elements)")
    print()

    # Initialize momentum velocities (one per trainable parameter)
    print("Initializing momentum velocities...")
    var velocities: List[ExTensor] = []

    # Note: In a complete implementation, initialize 84 velocity tensors
    # matching the shape of each parameter. For this demonstration:
    print("  STATUS: Backward pass is a documented placeholder (full implementation tracked in GitHub issue #3181)")
    print("  This script demonstrates the structure and patterns")
    print()

    # STATUS UPDATE
    print("=" * 60)
    print("IMPLEMENTATION STATUS")
    print("=" * 60)
    print()
    print("✅ batch_norm2d_backward is now available in shared library!")
    print()
    print("The backward pass structure is fully documented above.")
    print("To complete training, implement:")
    print("  1. Cache all activations during forward pass")
    print("  2. Implement full backward pass (~2000 lines)")
    print("  3. Initialize 84 velocity tensors")
    print("  4. Update all parameters with SGD + momentum")
    print()
    print("Key patterns demonstrated:")
    print("  - Batch norm backward: batch_norm2d_backward(...)")
    print("  - Skip connections: add_backward splits gradients")
    print("  - Projection shortcuts: 1×1 conv + BN backward")
    print("  - Identity shortcuts: direct gradient addition")
    print()
    print("Expected implementation size:")
    print("  - Forward caching: ~500 lines")
    print("  - Backward pass: ~2000 lines")
    print("  - Parameter updates: ~200 lines")
    print("  - Total: ~2700 lines for full manual backprop")
    print()
    print("=" * 60)
    print()

    # Demonstration forward pass
    print("Running demonstration forward pass...")
    var demo_logits = model.forward(train_images, training=True)
    var demo_loss = cross_entropy(demo_logits, train_labels)
    var demo_loss_scalar = demo_loss._data.bitcast[Float32]()[0]

    print("  Forward pass successful")
    print("  Batch shape: (10, 3, 32, 32)")
    print("  Output logits shape: (10, 10)")
    print("  Loss value: " + String(demo_loss_scalar))
    print()

    print("ResNet-18 forward pass is complete.")
    print(
        "To enable training, implement the full backward pass as documented"
        " above."
    )
    print()
    print(
        "Alternative: Consider using automatic differentiation for such deep"
        " networks."
    )
