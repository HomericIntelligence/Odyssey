# Usage Examples - ML Odyssey Shared Library (WIP)

<!-- markdownlint-disable MD051 -->

This document provides comprehensive usage examples for the ML Odyssey shared library, from quick start to advanced patterns.

**Note**: Examples use commented imports for components not yet implemented. Uncomment imports as Issue #49 completes implementation.

## Table of Contents

1. [Quickstart (5 Minutes)](#quickstart-5-minutes)
1. [Basic Neural Network](#basic-neural-network)
1. [Convolutional Neural Network](#convolutional-neural-network)
1. [Training with Validation](#training-with-validation)
1. [Custom Training Loop](#custom-training-loop)
1. [Data Loading](#data-loading)
1. [Learning Rate Scheduling](#learning-rate-scheduling)
1. [Callbacks and Monitoring](#callbacks-and-monitoring)
1. [Model Checkpointing](#model-checkpointing)
1. [Multiple Metrics](#multiple-metrics)
1. [Complete Example: MNIST Classifier](#complete-example-mnist-classifier)
1. [Advanced Patterns](#advanced-patterns)

## Quickstart (5 Minutes)

Get started with the shared library in 5 minutes:

### 1. Install (30 seconds)

```bash
cd Odyssey
mojo package shared --install
mojo run scripts/verify_installation.mojo
```text

### 2. Hello World Model (2 minutes)

Create `hello_ml.mojo`:

```mojo
from shared import Linear, ReLU, Sequential, SGD

fn main():
    # Create a simple 3-layer network
    var model = Sequential([
        Linear(784, 128),  # Input layer
        ReLU(),
        Linear(128, 10),   # Output layer
    ])

    # Create optimizer
    var optimizer = SGD(learning_rate=0.01)

    print("Model created successfully!")
    print("Parameters:", model.num_parameters())
```text

Run it: `mojo run hello_ml.mojo`

### 3. Train on Dummy Data (2 minutes)

Add training to `hello_ml.mojo`:

```mojo
from shared import Linear, ReLU, Sequential, SGD, DataLoader, TensorDataset
from shared import train_epoch, Logger

fn main():
    # Setup logging
    var logger = Logger("quickstart")
    logger.info("Starting training...")

    # Create model
    var model = Sequential([
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    ])

    # Create dummy data (replace with real data)
    var train_data = create_random_tensor((1000, 784))
    var train_labels = create_random_labels(1000, 10)

    # Create dataset and loader
    var dataset = TensorDataset(train_data, train_labels)
    var loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create optimizer
    var optimizer = SGD(learning_rate=0.01, momentum=0.9)

    # Train for 5 epochs
    for epoch in range(5):
        var loss = train_epoch(model, optimizer, loader, cross_entropy_loss)
        logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")

    logger.info("Training complete!")
```text

**That's it!** You now have:

- ✅ A neural network model
- ✅ An optimizer
- ✅ A data loader
- ✅ A training loop

### Next Steps After Quickstart

- **Learn More**: Continue reading examples below
- **Real Data**: Replace dummy data with MNIST/CIFAR-10
- **Customize**: Add more layers, try different optimizers
- **Visualize**: Add `plot_training_curves()` to see progress

## Basic Neural Network

Simple feedforward network for classification:

```mojo
from shared import Linear, ReLU, Sequential, SGD, Tensor

fn basic_network_example():
    """Create and use a basic neural network."""

    # Define model architecture
    var model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ])

    # Create optimizer
    var optimizer = SGD(
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )

    # Training loop (simplified)
    for epoch in range(10):
        # Forward pass
        var outputs = model.forward(inputs)

        # Compute loss
        var loss = cross_entropy_loss(outputs, targets)

        # Backward pass
        var grads = compute_gradients(loss, model)

        # Update parameters
        optimizer.step(model.parameters(), grads)

        print("Epoch", epoch, "Loss:", loss.item())
```text

## Convolutional Neural Network

CNN for image classification:

```mojo
from shared import (
    Conv2D, ReLU, MaxPool2D, Flatten, Linear, Sequential,
    Adam, Tensor
)

fn cnn_example():
    """Create a convolutional neural network."""

    # Define CNN architecture
    var model = Sequential([
        # Convolutional layers
        Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),

        Conv2D(32, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),

        # Fully connected layers
        Flatten(),
        Linear(64 * 7 * 7, 128),
        ReLU(),
        Linear(128, 10),
    ])

    # Create Adam optimizer
    var optimizer = Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    )

    print("CNN created with", model.num_parameters(), "parameters")
```text

## Training with Validation

Training loop with validation:

```mojo
from shared import (
    Linear, ReLU, Sequential,
    Adam, CosineAnnealingLR,
    Accuracy, LossTracker,
    DataLoader,
)

fn training_with_validation():
    """Train model with validation."""

    # Create model
    var model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 10),
    ])

    # Create optimizer and scheduler
    var optimizer = Adam(learning_rate=0.001)
    var scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=100,
        eta_min=1e-6
    )

    # Create metrics
    var train_loss = LossTracker()
    var train_acc = Accuracy()
    var val_loss = LossTracker()
    var val_acc = Accuracy()

    # Training loop
    for epoch in range(100):
        # Training phase
        model.train()
        train_loss.reset()
        train_acc.reset()

        for batch in train_loader:
            # Forward pass
            var outputs = model.forward(batch.inputs)
            var loss = criterion(outputs, batch.targets)

            # Backward pass
            var grads = compute_gradients(loss, model)
            optimizer.step(model.parameters(), grads)

            # Update metrics
            train_loss.update(loss.item())
            train_acc.update(outputs, batch.targets)

        # Validation phase
        model.eval()
        val_loss.reset()
        val_acc.reset()

        for batch in val_loader:
            var outputs = model.forward(batch.inputs)
            var loss = criterion(outputs, batch.targets)

            val_loss.update(loss.item())
            val_acc.update(outputs, batch.targets)

        # Update learning rate
        scheduler.step()

        # Print metrics
        print("Epoch", epoch)
        print("  Train Loss:", train_loss.compute(), "Acc:", train_acc.compute())
        print("  Val Loss:", val_loss.compute(), "Acc:", val_acc.compute())
        print("  LR:", optimizer.get_lr())
```text

## Custom Training Loop

Custom training loop with paper-specific logic:

```mojo
from shared import SGD, Accuracy, Tensor
from projectodyssey.core.layers import Linear, ReLU

fn custom_training_loop(
    model: Model,
    train_data: DataLoader,
    val_data: DataLoader,
    epochs: Int
):
    """Custom training loop with special logic."""

    # Create optimizer with custom settings
    var optimizer = SGD(
        learning_rate=0.01,
        momentum=0.9,
        dampening=0.0,
        weight_decay=1e-4,
        nesterov=True
    )

    # Metrics
    var best_val_acc: Float32 = 0.0
    var patience_counter: Int = 0
    var patience: Int = 10

    for epoch in range(epochs):
        # Training phase
        var total_loss: Float32 = 0.0
        var train_acc = Accuracy()

        for batch in train_data:
            # Forward pass
            var outputs = model.forward(batch.inputs)
            var loss = compute_loss(outputs, batch.targets)

            # Add custom regularization (paper-specific)
            loss = loss + custom_regularization(model)

            # Backward pass
            var grads = compute_gradients(loss, model)

            # Gradient clipping (if needed)
            grads = clip_gradients(grads, max_norm=1.0)

            # Optimizer step
            optimizer.step(model.parameters(), grads)

            # Track metrics
            total_loss += loss.item()
            train_acc.update(outputs, batch.targets)

        # Validation phase
        var val_acc = Accuracy()

        for batch in val_data:
            var outputs = model.forward(batch.inputs)
            val_acc.update(outputs, batch.targets)

        var val_accuracy = val_acc.compute()

        # Early stopping logic
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            save_checkpoint(model, "best_model.mojo")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping at epoch", epoch)
            break

        # Custom learning rate adjustment
        if epoch % 30 == 0 and epoch > 0:
            optimizer.set_lr(optimizer.get_lr() * 0.1)

        print("Epoch", epoch, "Train Acc:", train_acc.compute(), "Val Acc:", val_accuracy)
```text

## Data Loading

Data loading with transforms:

```mojo
from shared import (
    TensorDataset, DataLoader,
    Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
)

fn data_loading_example():
    """Create data loaders with transforms."""

    # Training transforms (with augmentation)
    var train_transform = Compose([
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(probability=0.5),
        ToTensor(),
        Normalize(mean=0.1307, std=0.3081),
    ])

    # Validation transforms (no augmentation)
    var val_transform = Compose([
        ToTensor(),
        Normalize(mean=0.1307, std=0.3081),
    ])

    # Create datasets
    var train_dataset = TensorDataset(
        train_data,
        train_labels,
        transform=train_transform
    )

    var val_dataset = TensorDataset(
        val_data,
        val_labels,
        transform=val_transform
    )

    # Create data loaders
    var train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    var val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=2
    )

    # Iterate over batches
    for batch in train_loader:
        # batch.inputs: Tensor of shape (batch_size, channels, height, width)
        # batch.targets: Tensor of shape (batch_size,)
        print("Batch shape:", batch.inputs.shape())
```text

## Learning Rate Scheduling

Using learning rate schedulers:

```mojo
from shared import (
    SGD, Adam,
    StepLR, CosineAnnealingLR, ExponentialLR, WarmupLR
)

fn lr_scheduling_example():
    """Examples of learning rate scheduling."""

    # Example 1: Step decay
    var optimizer1 = SGD(learning_rate=0.1, momentum=0.9)
    var scheduler1 = StepLR(
        optimizer=optimizer1,
        step_size=30,    # Decay every 30 epochs
        gamma=0.1        # Multiply by 0.1
    )

    # Example 2: Cosine annealing
    var optimizer2 = Adam(learning_rate=0.001)
    var scheduler2 = CosineAnnealingLR(
        optimizer=optimizer2,
        T_max=100,       # Period of cosine cycle
        eta_min=1e-6     # Minimum learning rate
    )

    # Example 3: Exponential decay
    var optimizer3 = SGD(learning_rate=0.01)
    var scheduler3 = ExponentialLR(
        optimizer=optimizer3,
        gamma=0.95       # Decay rate per epoch
    )

    # Example 4: Warmup then decay
    var optimizer4 = Adam(learning_rate=0.001)
    var warmup = WarmupLR(
        optimizer=optimizer4,
        warmup_epochs=10,
        start_lr=1e-6
    )
    var main_scheduler = CosineAnnealingLR(
        optimizer=optimizer4,
        T_max=90
    )

    # Training loop with scheduling
    for epoch in range(100):
        # Training code...

        # Update learning rate
        if epoch < 10:
            warmup.step()
        else:
            main_scheduler.step()

        print("Epoch", epoch, "LR:", optimizer4.get_lr())
```text

## Callbacks and Monitoring

Using callbacks for training monitoring:

```mojo
from shared import (
    EarlyStopping, ModelCheckpoint, Logger,
    LRSchedulerCallback
)

fn callbacks_example():
    """Using callbacks for training monitoring."""

    # Create callbacks
    var early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",            # "min" for loss, "max" for accuracy
        min_delta=0.001,       # Minimum change to qualify as improvement
        verbose=True
    )

    var checkpoint = ModelCheckpoint(
        filepath="checkpoints/model_{epoch:03d}_{val_acc:.3f}.mojo",
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_freq=1            # Save every epoch
    )

    var logger = Logger(
        log_file="training.log",
        log_level="INFO",
        metrics=["loss", "acc", "val_loss", "val_acc"]
    )

    var lr_callback = LRSchedulerCallback(scheduler)

    # Training loop with callbacks
    for epoch in range(100):
        # On epoch start
        early_stop.on_epoch_begin(epoch)
        checkpoint.on_epoch_begin(epoch)
        logger.on_epoch_begin(epoch)

        # Training...
        train_loss = train_epoch(model, optimizer, train_loader)
        val_loss = validate_epoch(model, val_loader)

        # Create logs dictionary
        var logs = {
            "loss": train_loss,
            "acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        # On epoch end
        early_stop.on_epoch_end(epoch, logs)
        checkpoint.on_epoch_end(epoch, logs)
        logger.on_epoch_end(epoch, logs)
        lr_callback.on_epoch_end(epoch, logs)

        # Check if should stop
        if early_stop.should_stop:
            print("Early stopping triggered at epoch", epoch)
            break
```text

## Model Checkpointing

Saving and loading models:

```mojo
from shared import ModelCheckpoint, Logger

fn checkpointing_example():
    """Save and load model checkpoints."""

    # Save checkpoint manually
    fn save_checkpoint(model, optimizer, epoch, filepath):
        var checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": current_loss,
        }
        save(checkpoint, filepath)
        print("Saved checkpoint to", filepath)

    # Load checkpoint
    fn load_checkpoint(filepath) -> Checkpoint:
        var checkpoint = load(filepath)
        return checkpoint

    # Using checkpoint callback
    var checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/best_model.mojo",
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=False  # Save full model + optimizer state
    )

    # In training loop
    for epoch in range(epochs):
        # Training...

        var logs = {"val_acc": val_accuracy}
        checkpoint_callback.on_epoch_end(epoch, logs)

    # Resume from checkpoint
    fn resume_training(checkpoint_path):
        var checkpoint = load_checkpoint(checkpoint_path)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        var start_epoch = checkpoint["epoch"] + 1

        print("Resuming from epoch", start_epoch)

        # Continue training
        for epoch in range(start_epoch, total_epochs):
            # Training...
            pass
```text

## Multiple Metrics

Tracking multiple metrics:

```mojo
from shared import (
    Accuracy, LossTracker, Precision, Recall, ConfusionMatrix
)

fn multiple_metrics_example():
    """Track multiple evaluation metrics."""

    # Create metrics
    var train_loss = LossTracker()
    var train_acc = Accuracy()

    var val_loss = LossTracker()
    var val_acc = Accuracy()
    var val_precision = Precision(num_classes=10)
    var val_recall = Recall(num_classes=10)
    var val_confusion = ConfusionMatrix(num_classes=10)

    # Training loop
    for epoch in range(epochs):
        # Reset metrics
        train_loss.reset()
        train_acc.reset()

        # Training phase
        for batch in train_loader:
            var outputs = model.forward(batch.inputs)
            var loss = criterion(outputs, batch.targets)

            # Update training metrics
            train_loss.update(loss.item())
            train_acc.update(outputs, batch.targets)

            # Backprop and optimization...

        # Reset validation metrics
        val_loss.reset()
        val_acc.reset()
        val_precision.reset()
        val_recall.reset()
        val_confusion.reset()

        # Validation phase
        for batch in val_loader:
            var outputs = model.forward(batch.inputs)
            var loss = criterion(outputs, batch.targets)

            # Update validation metrics
            val_loss.update(loss.item())
            val_acc.update(outputs, batch.targets)
            val_precision.update(outputs, batch.targets)
            val_recall.update(outputs, batch.targets)
            val_confusion.update(outputs, batch.targets)

        # Print all metrics
        print("Epoch", epoch)
        print("  Train Loss:", train_loss.compute(), "Acc:", train_acc.compute())
        print("  Val Loss:", val_loss.compute(), "Acc:", val_acc.compute())
        print("  Precision:", val_precision.compute())
        print("  Recall:", val_recall.compute())

        # Print confusion matrix
        print("  Confusion Matrix:")
        print(val_confusion.compute())
```text

## Complete Example: MNIST Classifier

Full example bringing it all together:

```mojo
from shared import (
    # Core
    Linear, ReLU, Dropout, Sequential,
    # Training
    Adam, CosineAnnealingLR,
    Accuracy, LossTracker,
    EarlyStopping, ModelCheckpoint, Logger,
    # Data
    TensorDataset, DataLoader, Compose, ToTensor, Normalize,
)

fn mnist_classifier_complete():
    """Complete MNIST classifier example."""

    # ========================================================================
    # 1. Data Preparation
    # ========================================================================

    # Load MNIST data (assuming loaded from somewhere)
    var train_images = load_mnist_images("train")
    var train_labels = load_mnist_labels("train")
    var test_images = load_mnist_images("test")
    var test_labels = load_mnist_labels("test")

    # Create transforms
    var transform = Compose([
        ToTensor(),
        Normalize(mean=0.1307, std=0.3081),
    ])

    # Create datasets
    var train_dataset = TensorDataset(train_images, train_labels, transform)
    var test_dataset = TensorDataset(test_images, test_labels, transform)

    # Create data loaders
    var train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True
    )

    var test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    # ========================================================================
    # 2. Model Definition
    # ========================================================================

    var model = Sequential([
        Linear(784, 512),
        ReLU(),
        Dropout(0.2),

        Linear(512, 256),
        ReLU(),
        Dropout(0.2),

        Linear(256, 128),
        ReLU(),
        Dropout(0.1),

        Linear(128, 10),
    ])

    print("Model created with", model.num_parameters(), "parameters")

    # ========================================================================
    # 3. Training Setup
    # ========================================================================

    # Optimizer
    var optimizer = Adam(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    var scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=50,
        eta_min=1e-6
    )

    # Metrics
    var train_loss = LossTracker()
    var train_acc = Accuracy()
    var test_loss = LossTracker()
    var test_acc = Accuracy()

    # Callbacks
    var early_stop = EarlyStopping(
        monitor="test_loss",
        patience=10,
        mode="min"
    )

    var checkpoint = ModelCheckpoint(
        filepath="checkpoints/mnist_best.mojo",
        monitor="test_acc",
        mode="max",
        save_best_only=True
    )

    var logger = Logger(
        log_file="mnist_training.log",
        metrics=["loss", "acc", "test_loss", "test_acc", "lr"]
    )

    # ========================================================================
    # 4. Training Loop
    # ========================================================================

    var num_epochs = 50

    for epoch in range(num_epochs):
        print("\nEpoch", epoch + 1, "/", num_epochs)

        # ====================================================================
        # Training Phase
        # ====================================================================

        model.train()
        train_loss.reset()
        train_acc.reset()

        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            var outputs = model.forward(batch.inputs)
            var loss = cross_entropy_loss(outputs, batch.targets)

            # Backward pass
            var grads = compute_gradients(loss, model)
            optimizer.step(model.parameters(), grads)
            optimizer.zero_grad()

            # Update metrics
            train_loss.update(loss.item())
            train_acc.update(outputs, batch.targets)

            # Print progress
            if batch_idx % 100 == 0:
                print("  Train Batch", batch_idx, "/", len(train_loader),
                      "Loss:", loss.item())

        # ====================================================================
        # Test Phase
        # ====================================================================

        model.eval()
        test_loss.reset()
        test_acc.reset()

        for batch in test_loader:
            var outputs = model.forward(batch.inputs)
            var loss = cross_entropy_loss(outputs, batch.targets)

            test_loss.update(loss.item())
            test_acc.update(outputs, batch.targets)

        # ====================================================================
        # Logging and Callbacks
        # ====================================================================

        var logs = {
            "loss": train_loss.compute(),
            "acc": train_acc.compute(),
            "test_loss": test_loss.compute(),
            "test_acc": test_acc.compute(),
            "lr": optimizer.get_lr()
        }

        # Print epoch summary
        print("\nEpoch", epoch + 1, "Summary:")
        print("  Train Loss:", logs["loss"], "Acc:", logs["acc"])
        print("  Test Loss:", logs["test_loss"], "Acc:", logs["test_acc"])
        print("  Learning Rate:", logs["lr"])

        # Update callbacks
        logger.on_epoch_end(epoch, logs)
        checkpoint.on_epoch_end(epoch, logs)
        early_stop.on_epoch_end(epoch, logs)

        # Update learning rate
        scheduler.step()

        # Check early stopping
        if early_stop.should_stop:
            print("\nEarly stopping triggered!")
            print("Best test accuracy:", early_stop.best_score)
            break

    # ========================================================================
    # 5. Final Evaluation
    # ========================================================================

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    # Load best model
    model.load_state_dict(load("checkpoints/mnist_best.mojo"))

    # Final evaluation
    test_acc.reset()
    for batch in test_loader:
        var outputs = model.forward(batch.inputs)
        test_acc.update(outputs, batch.targets)

    print("Final Test Accuracy:", test_acc.compute())
```text

## Advanced Patterns

### Custom Layers

Create custom layers by extending the Module interface:

```mojo
from projectodyssey.core import Module, Linear, ReLU

struct ResidualBlock(Module):
    """Residual block with skip connection."""
    var conv1: Conv2D
    var conv2: Conv2D
    var relu: ReLU
    var use_projection: Bool
    var projection: Optional[Conv2D]

    fn __init__(out self, in_channels: Int, out_channels: Int, stride: Int = 1):
        """Initialize residual block."""
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = ReLU()

        # Add projection if dimensions change
        self.use_projection = (stride != 1) or (in_channels != out_channels)
        if self.use_projection:
            self.projection = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride)

    fn forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        var identity = x

        # Main path
        var out = self.relu(self.conv1(x))
        out = self.conv2(out)

        # Skip connection
        if self.use_projection:
            identity = self.projection.value()(x)

        # Add and activate
        out = out + identity
        return self.relu(out)

    fn parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        var params = List[Tensor]()
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        if self.use_projection:
            params.extend(self.projection.value().parameters())
        return params
```text

### Custom Optimizers

Implement custom optimization algorithms:

```mojo
from projectodyssey.training import Optimizer

struct CustomOptimizer(Optimizer):
    """Custom optimizer with adaptive learning rates."""
    var learning_rate: Float32
    var state: Dict[String, Tensor]

    fn __init__(out self, learning_rate: Float32 = 0.01):
        """Initialize optimizer."""
        self.learning_rate = learning_rate
        self.state = Dict[String, Tensor]()

    fn step(mut self, mut params: List[Tensor], grads: List[Tensor]):
        """Custom optimization step."""
        for i in range(len(params)):
            var param = params[i]
            var grad = grads[i]

            # Custom update rule
            var adaptive_lr = self.compute_adaptive_lr(param, grad)
            param -= adaptive_lr * grad

    fn compute_adaptive_lr(self, param: Tensor, grad: Tensor) -> Float32:
        """Compute adaptive learning rate based on gradient statistics."""
        var grad_norm = compute_norm(grad)
        return self.learning_rate / (1.0 + grad_norm)
```text

### Custom Loss Functions

Define problem-specific loss functions:

```mojo
fn focal_loss(
    predictions: Tensor,
    targets: Tensor,
    alpha: Float32 = 0.25,
    gamma: Float32 = 2.0
) -> Float32:
    """
    Focal loss for addressing class imbalance.

    Args:
        predictions: Model predictions (N, num_classes)
        targets: Ground truth labels (N,)
        alpha: Weighting factor
        gamma: Focusing parameter

    Returns:
        Focal loss value
    """
    # Convert to probabilities
    var probs = softmax(predictions)

    # Get probabilities for correct classes
    var target_probs = gather(probs, targets)

    # Compute focal loss
    var focal_weight = pow(1.0 - target_probs, gamma)
    var ce_loss = -log(target_probs)
    var loss = alpha * focal_weight * ce_loss

    return loss.mean()

fn contrastive_loss(
    embeddings1: Tensor,
    embeddings2: Tensor,
    labels: Tensor,
    margin: Float32 = 1.0
) -> Float32:
    """
    Contrastive loss for metric learning.

    Args:
        embeddings1: First set of embeddings (N, D)
        embeddings2: Second set of embeddings (N, D)
        labels: Binary labels (0 = different, 1 = similar)
        margin: Margin for dissimilar pairs

    Returns:
        Contrastive loss value
    """
    # Compute pairwise distances
    var distances = euclidean_distance(embeddings1, embeddings2)

    # Similar pairs: minimize distance
    var loss_similar = labels * pow(distances, 2)

    # Dissimilar pairs: maximize distance up to margin
    var loss_dissimilar = (1 - labels) * pow(max(margin - distances, 0), 2)

    return (loss_similar + loss_dissimilar).mean()
```text

### Custom Data Transforms

Create domain-specific transforms:

```mojo
from projectodyssey.data import Transform

struct MixUp(Transform):
    """MixUp data augmentation for better generalization."""
    var alpha: Float32

    fn __init__(out self, alpha: Float32 = 1.0):
        """Initialize MixUp with mixing parameter."""
        self.alpha = alpha

    fn __call__(self, batch: Batch) -> Batch:
        """Apply MixUp to batch."""
        var batch_size = batch.inputs.shape[0]

        # Sample mixing coefficient
        var lam = sample_beta(self.alpha, self.alpha)

        # Random permutation
        var indices = random_permutation(batch_size)

        # Mix inputs and targets
        var mixed_inputs = lam * batch.inputs + (1 - lam) * batch.inputs[indices]
        var mixed_targets = lam * batch.targets + (1 - lam) * batch.targets[indices]

        return Batch(mixed_inputs, mixed_targets, batch.indices)

struct CutOut(Transform):
    """CutOut augmentation - randomly mask regions."""
    var mask_size: Int

    fn __init__(out self, mask_size: Int = 16):
        """Initialize CutOut with mask size."""
        self.mask_size = mask_size

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply CutOut to image."""
        var h = x.shape[1]
        var w = x.shape[2]

        # Random mask position
        var y = random_int(0, h)
        var x_pos = random_int(0, w)

        # Calculate mask boundaries
        var y1 = max(0, y - self.mask_size // 2)
        var y2 = min(h, y + self.mask_size // 2)
        var x1 = max(0, x_pos - self.mask_size // 2)
        var x2 = min(w, x_pos + self.mask_size // 2)

        # Apply mask
        var output = x.copy()
        output[:, y1:y2, x1:x2] = 0.0
        return output
```text

### Model Ensemble

Combine multiple models for better performance:

```mojo
struct ModelEnsemble:
    """Ensemble of models for robust predictions."""
    var models: List[Module]
    var weights: List[Float32]

    fn __init__(out self, models: List[Module], weights: Optional[List[Float32]] = None):
        """Initialize ensemble with models and optional weights."""
        self.models = models

        if weights:
            self.weights = weights.value()
        else:
            # Equal weights
            var n = len(models)
            self.weights = List[Float32]([1.0 / n for _ in range(n)])

    fn predict(self, x: Tensor) -> Tensor:
        """Ensemble prediction (weighted average)."""
        var outputs = List[Tensor]()

        # Get predictions from all models
        for model in self.models:
            outputs.append(model.forward(x))

        # Weighted average
        var ensemble_output = zeros_like(outputs[0])
        for i in range(len(outputs)):
            ensemble_output += self.weights[i] * outputs[i]

        return ensemble_output

    fn predict_with_uncertainty(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return predictions with uncertainty estimates."""
        var outputs = List[Tensor]()

        for model in self.models:
            outputs.append(model.forward(x))

        # Compute mean and std
        var mean = compute_mean(outputs)
        var std = compute_std(outputs)

        return (mean, std)
```text

### Gradient Accumulation

Train with larger effective batch sizes:

```mojo
fn train_with_gradient_accumulation(
    model: Module,
    optimizer: Optimizer,
    loader: DataLoader,
    accumulation_steps: Int = 4
) -> Float32:
    """
    Train with gradient accumulation for larger effective batch size.

    Args:
        model: Model to train
        optimizer: Optimizer
        loader: Data loader
        accumulation_steps: Number of batches to accumulate

    Returns:
        Average loss
    """
    var total_loss: Float32 = 0.0
    var num_batches = 0
    var accumulated_grads = initialize_grads(model)

    for batch_idx, batch in enumerate(loader):
        # Forward pass
        var outputs = model.forward(batch.inputs)
        var loss = cross_entropy_loss(outputs, batch.targets)

        # Backward pass (accumulate gradients)
        var grads = compute_gradients(loss, model)
        for i in range(len(grads)):
            accumulated_grads[i] += grads[i]

        total_loss += loss.item()
        num_batches += 1

        # Update every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Average accumulated gradients
            for grad in accumulated_grads:
                grad /= Float32(accumulation_steps)

            # Optimizer step
            optimizer.step(model.parameters(), accumulated_grads)

            # Reset accumulated gradients
            accumulated_grads = initialize_grads(model)

    return total_loss / Float32(num_batches)
```text

### Mixed Precision Training

Use lower precision for faster training:

```mojo
struct MixedPrecisionTrainer:
    """Mixed precision training with automatic loss scaling."""
    var model: Module
    var optimizer: Optimizer
    var scaler: GradScaler
    var use_fp16: Bool

    fn __init__(
        out self,
        model: Module,
        optimizer: Optimizer,
        use_fp16: Bool = True
    ):
        """Initialize mixed precision trainer."""
        self.model = model
        self.optimizer = optimizer
        self.use_fp16 = use_fp16
        self.scaler = GradScaler() if use_fp16 else None

    fn train_step(mut self, batch: Batch) -> Float32:
        """Single training step with mixed precision."""
        # Convert inputs to FP16 if needed
        var inputs = batch.inputs.to(DType.float16) if self.use_fp16 else batch.inputs

        # Forward pass in FP16
        var outputs = self.model.forward(inputs)
        var loss = cross_entropy_loss(outputs, batch.targets)

        # Scale loss for FP16
        if self.use_fp16:
            loss = self.scaler.scale(loss)

        # Backward pass
        var grads = compute_gradients(loss, self.model)

        # Unscale gradients
        if self.use_fp16:
            grads = self.scaler.unscale(grads)

        # Optimizer step
        self.optimizer.step(self.model.parameters(), grads)

        # Update scaler
        if self.use_fp16:
            self.scaler.update()

        return loss.item()
```text

### Advanced Debugging

Debug training with detailed inspection:

```mojo
from projectodyssey.utils import Logger

struct TrainingDebugger:
    """Advanced debugging utilities for training."""
    var logger: Logger
    var log_gradients: Bool
    var log_activations: Bool

    fn __init__(
        out self,
        log_file: String = "debug.log",
        log_gradients: Bool = True,
        log_activations: Bool = False
    ):
        """Initialize debugger."""
        self.logger = Logger(log_file)
        self.log_gradients = log_gradients
        self.log_activations = log_activations

    fn debug_step(
        self,
        model: Module,
        batch: Batch,
        grads: List[Tensor],
        loss: Float32
    ):
        """Log debugging information for training step."""
        self.logger.debug(f"Loss: {loss:.6f}")

        # Check for NaN/Inf in loss
        if is_nan(loss) or is_inf(loss):
            self.logger.error("Invalid loss detected!")

        # Log gradient statistics
        if self.log_gradients:
            for i, grad in enumerate(grads):
                var grad_norm = compute_norm(grad)
                var grad_mean = grad.mean()
                var grad_max = grad.max()

                self.logger.debug(
                    f"Grad[{i}]: norm={grad_norm:.6f}, "
                    f"mean={grad_mean:.6f}, max={grad_max:.6f}"
                )

                # Check for vanishing/exploding gradients
                if grad_norm < 1e-7:
                    self.logger.warning(f"Vanishing gradient at layer {i}")
                if grad_norm > 100:
                    self.logger.warning(f"Exploding gradient at layer {i}")

        # Log parameter statistics
        for i, param in enumerate(model.parameters()):
            var param_norm = compute_norm(param)
            self.logger.debug(f"Param[{i}]: norm={param_norm:.6f}")
```text

## Best Practices Summary

### Performance Tips

1. **Use SIMD**: Leverage vectorization for element-wise operations
1. **Batch Operations**: Process data in batches for efficiency
1. **Release Builds**: Use `mojo build --release` for production
1. **Profile First**: Measure before optimizing

### Code Organization

1. **Separate Concerns**: Keep model, training, data separate
1. **Config Files**: Use YAML/JSON for hyperparameters
1. **Logging**: Log everything important
1. **Reproducibility**: Always set random seeds

### Common Pitfalls

1. **Forgetting to Call train()/eval()**: Set model modes correctly
1. **Not Resetting Metrics**: Reset between epochs
1. **Wrong Learning Rates**: Start conservative
1. **Ignoring Validation**: Always validate during training

## Next Steps

- Review [API Documentation](docs/api/) for detailed API reference
- Check [INSTALL.md](INSTALL.md) for installation instructions
- See [README.md](README.md) for library overview
- Browse tests in `tests/shared/` for more examples
- Read [MIGRATION.md](MIGRATION.md) for paper integration patterns

## Notes

- Examples marked with comments indicate components not yet implemented
- Uncomment imports as implementation progresses (Issue #49)
- All examples follow Mojo best practices (fn, struct, SIMD where applicable)
- See individual module documentation for more details
