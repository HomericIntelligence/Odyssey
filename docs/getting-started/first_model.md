# Building Your First Model

A conceptual orientation guide for working with ML Odyssey's shared library and implemented papers.

## Overview

ML Odyssey is a Mojo-based platform for reproducing classic research papers. This guide explains
what is available today and how to use it, so you can orient yourself before writing your own
Mojo code.

**What exists today**:

- A shared library (`shared/`) providing `AnyTensor` — a typed, N-dimensional tensor type
- Tensor creation, shape, and element operations in `shared/core/`
- Data loading utilities in `shared/data/`
- Training infrastructure in `shared/training/`
- The LeNet-5 paper implementation (`just train`, `just infer` recipes)

**What is still planned**:

- High-level APIs such as `Sequential`, `Trainer`, `BatchLoader`, and data augmentation transforms
- Additional paper implementations beyond LeNet-5

## Prerequisites

Before starting, ensure you have:

- Completed the [Quickstart Guide](quickstart.md) and confirmed `pixi run mojo --version` works
- Basic familiarity with neural network concepts (forward pass, loss, backpropagation)

## The Shared Library

The `shared/` directory contains the core ML components used across all paper implementations.
The entry point is `AnyTensor` in `shared/core/`.

### Creating Tensors

```mojo
from shared.data import TensorDataset, BatchLoader
from shared.utils import download_mnist, normalize_images

fn prepare_mnist() raises -> (TensorDataset, TensorDataset):
    """Load and prepare MNIST data for training."""

    # Download MNIST dataset (cached after first run)
    print("Loading MNIST dataset...")
    var train_images, train_labels = download_mnist(train=True)
    var test_images, test_labels = download_mnist(train=False)

    # Normalize images to [0, 1] range
    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)

    # Flatten images from 28x28 to 784
    train_images = train_images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)

    # Create datasets
    var train_data = TensorDataset(train_images, train_labels)
    var test_data = TensorDataset(test_images, test_labels)

    print("Data loaded: ", train_data.size(), " training examples")
    print("Data loaded: ", test_data.size(), " test examples")

    return train_data, test_data
```

## Step 3: Define the Model

Create `model.mojo` with your neural network architecture.

See `examples/getting_started/first_model_model.mojo`

Key architecture:

```mojo
# 3-layer network: 784 -> 128 -> 64 -> 10

self.model = Sequential([
    Layer("linear", input_size=784, output_size=128),
    ReLU(),
    Layer("linear", input_size=128, output_size=64),
    ReLU(),
    Layer("linear", input_size=64, output_size=10),
    Softmax(),
])
```

Full example: `examples/getting_started/first_model_model.mojo`

## Step 4: Training Script

Create `train.mojo` to train your model.

See `examples/getting_started/first_model_train.mojo`

Key training steps:

```mojo
# Configure training
var optimizer = SGD(learning_rate=0.01, momentum=0.9)
var loss_fn = CrossEntropyLoss()
var trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)

# Add callbacks
trainer.add_callback(EarlyStopping(patience=3, min_delta=0.001))
trainer.add_callback(ModelCheckpoint(filepath="best_model.mojo", save_best_only=True))

# Train
trainer.train(train_loader, val_loader, epochs=10, verbose=True)
```

Full example: `examples/getting_started/first_model_train.mojo`

## Step 5: Run Training

Execute your training script:

```bash
pixi run mojo run train.mojo
```

You should see output like:

```text


fn main() raises:
    # 1D tensor of zeros: shape [5]
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)

    # 2D tensor of ones: shape [3, 4]
    var shape2 = List[Int]()
    shape2.append(3)
    shape2.append(4)
    var m = ones(shape2, DType.float32)

    print("ndim:", t.ndim(), "numel:", t.numel())
    print("ndim:", m.ndim(), "numel:", m.numel())
```

```mojo
    # Load trained model
    var model = load_model[DigitClassifier]("best_model.mojo")

    # Evaluate
    var metrics = evaluate_model(model, test_data)

    print("\nTest Results:")
    print("  Accuracy:  {:.2f}%".format(metrics.accuracy * 100))
    print("  Precision: {:.2f}%".format(metrics.precision * 100))
    print("  Recall:    {:.2f}%".format(metrics.recall * 100))
    print("  F1 Score:  {:.2f}".format(metrics.f1_score))

    # Plot confusion matrix
    plot_confusion_matrix(
        metrics.confusion_matrix,
        class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        save_path="confusion_matrix.png"
    )

    print("\nConfusion matrix saved to confusion_matrix.png")
```

Run evaluation:

```bash
pixi run mojo run evaluate.mojo
```

Expected output:

```text
Evaluating model...

Test Results:
  Accuracy:  95.12%
  Precision: 95.08%
  Recall:    95.12%
  F1 Score:  95.10

Confusion matrix saved to confusion_matrix.png
```

## Step 7: Make Predictions

Create `predict.mojo` to classify individual images:

```mojo
from shared.utils import load_model, load_image, plot_image
from model import DigitClassifier

fn predict_digit(image_path: String) raises:
    """Predict the digit in an image."""

    # Load model
    var model = load_model[DigitClassifier]("best_model.mojo")

    # Load and preprocess image
    var image = load_image(image_path)
    image = image.resize(28, 28).grayscale()
    image = image.normalize().flatten()

    # Make prediction
    var output = model.forward(image)
    var predicted_digit = output.argmax()
    var confidence = output[predicted_digit]

    print("Predicted digit: ", predicted_digit)
    print("Confidence: {:.2f}%".format(confidence * 100))

    # Visualize
    plot_image(image.reshape(28, 28), title="Input Image")

fn main() raises:
    predict_digit("my_digit.png")
```

## Understanding the Code

### Data Preparation

- **Normalization**: Scales pixel values to [0, 1] for better training
- **Flattening**: Converts 28x28 images to 784-element vectors
- **Batching**: Groups examples for efficient GPU processing

### Model Architecture

```text
Input (784)
    ↓
Linear Layer (784 → 128)
    ↓
ReLU Activation
    ↓
Linear Layer (128 → 64)
    ↓
ReLU Activation
    ↓
Linear Layer (64 → 10)
    ↓
Softmax (Output Probabilities)
```

### Training Process

1. **Forward Pass**: Input flows through network to produce predictions
2. **Loss Calculation**: Compare predictions to true labels
3. **Backward Pass**: Compute gradients using backpropagation
4. **Parameter Update**: Adjust weights using optimizer (SGD)
5. **Validation**: Evaluate on test set to monitor progress

## Common Issues

### Low Accuracy (< 80%)

**Possible causes**:

- Data not normalized properly
- Learning rate too high or too low
- Not enough training epochs

**Solutions**:

```mojo
# Try adjusting learning rate
var optimizer = SGD(learning_rate=0.001)  # Lower LR

# Train for more epochs
trainer.train(train_loader, val_loader, epochs=20)

# Verify data normalization
print("Data range: ", train_images.min(), " to ", train_images.max())
# Should be [0.0, 1.0]
```

### Training Too Slow

### Solutions

```mojo
# Increase batch size
var train_loader = BatchLoader(train_data, batch_size=128)

# Use release build for better performance
```

```bash
pixi run mojo build --release train.mojo
./train
```

### 1. Create a paper directory

### Solutions

```mojo
# Reduce batch size
var train_loader = BatchLoader(train_data, batch_size=16)

# Use smaller model
self.model = Sequential([
    Layer("linear", input_size=784, output_size=64),  # Smaller
    ReLU(),
    Layer("linear", input_size=64, output_size=10),
])
```

### Import Errors

```bash
# Ensure you're in the right directory
cd ProjectOdyssey/examples/first_model

### 4. Build and verify

# Run from repository root
cd ../..
pixi run mojo run examples/first_model/train.mojo
```

## Understanding the Training Loop

All paper implementations share the same fundamental loop:

```mojo
self.model = Sequential([
    Layer("linear", input_size=784, output_size=256),
    ReLU(),
    Layer("linear", input_size=256, output_size=128),
    ReLU(),
    Layer("linear", input_size=128, output_size=64),
    ReLU(),
    Layer("linear", input_size=64, output_size=10),
    Softmax(),
])
```

### 2. Different Optimizer

```mojo
from shared.training import Adam

var optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
```

### 3. Learning Rate Scheduling

```mojo
from shared.training.schedulers import StepLR

var scheduler = StepLR(initial_lr=0.01, step_size=5, gamma=0.5)
trainer.add_scheduler(scheduler)
```

### 4. Data Augmentation

```mojo
from shared.data.transforms import RandomRotation, RandomShift

var train_loader = BatchLoader(
    train_data,
    batch_size=32,
    transforms=[
        RandomRotation(degrees=15),
        RandomShift(max_shift=2),
    ]
)
```

## Next Steps

- **[Repository Structure](repository-structure.md)** — understand where code lives
- **[Installation Guide](installation.md)** — detailed build and package setup
- **[Quickstart Guide](quickstart.md)** — 5-minute introduction to the environment
- **`shared/` source** — browse `shared/core/`, `shared/training/`, `shared/data/` directly
- **`papers/_template/`** — starting point for a new paper implementation
