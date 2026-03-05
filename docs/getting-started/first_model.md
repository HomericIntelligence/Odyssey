# Building Your First Model

Complete tutorial for creating, training, and evaluating your first neural network with ML Odyssey.

## Overview

This tutorial walks you through building a simple handwritten digit classifier using the MNIST dataset. You'll learn
how to use ML Odyssey's shared library to create a neural network, train it, and evaluate its performance.

**What you'll build**: A 3-layer neural network that classifies handwritten digits (0-9) with ~95% accuracy.

**Time required**: 30-45 minutes

## Prerequisites

Before starting, ensure you have:

- Completed the [Quickstart Guide](quickstart.md)
- ML Odyssey installed and working (`pixi run mojo --version`)
- Basic understanding of neural networks (helpful but not required)

## Step 1: Project Setup

Create a new directory for your first model:

```bash
```bash

cd ProjectOdyssey
mkdir -p examples/first_model
cd examples/first_model

```text

## Step 2: Prepare the Data

Create `prepare_data.mojo` to load and preprocess the MNIST dataset:

```mojo

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

```text

## Step 3: Define the Model

Create `model.mojo` with your neural network architecture.

See `examples/getting-started/first_model_model.mojo`

Key architecture:

```mojo
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

```text

Full example: `examples/getting-started/first_model_model.mojo`

## Step 4: Training Script

Create `train.mojo` to train your model.

See `examples/getting-started/first_model_train.mojo`

Key training steps:

```mojo

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

```text

Full example: `examples/getting-started/first_model_train.mojo`

## Step 5: Run Training

Execute your training script:

```bash
```bash

pixi run mojo run train.mojo

```text

You should see output like:

```text

```text

==================================================
Training Digit Classifier
==================================================
Loading MNIST dataset...
Data loaded: 60000 training examples
Data loaded: 10000 test examples

Model architecture:
Sequential(
  Linear(784 -> 128)
  ReLU()
  Linear(128 -> 64)
  ReLU()
  Linear(64 -> 10)
  Softmax()
)
Total parameters: 109,386

Starting training...
Epoch 1/10: 100%|████████| 1875/1875 [00:12<00:00, 156.25it/s, loss=0.523]
Validation: loss=0.321, accuracy=0.912
Epoch 2/10: 100%|████████| 1875/1875 [00:11<00:00, 162.50it/s, loss=0.287]
Validation: loss=0.245, accuracy=0.934
...
Epoch 8/10: 100%|████████| 1875/1875 [00:11<00:00, 165.00it/s, loss=0.142]
Validation: loss=0.189, accuracy=0.951
Model checkpoint saved: best_model.mojo

Training complete!

```text

## Step 6: Evaluate the Model

Create `evaluate.mojo` to test your trained model:

```mojo
```mojo

from shared.training import evaluate_model
from shared.utils import load_model, plot_confusion_matrix
from prepare_data import prepare_mnist
from model import DigitClassifier

fn main() raises:
    """Evaluate the trained model."""

    print("Evaluating model...")

    # Load test data
    var _, test_data = prepare_mnist()

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

```text

Run evaluation:

```bash

```bash

pixi run mojo run evaluate.mojo

```text

Expected output:

```text
```text

Evaluating model...

Test Results:
  Accuracy:  95.12%
  Precision: 95.08%
  Recall:    95.12%
  F1 Score:  95.10

Confusion matrix saved to confusion_matrix.png

```text

## Step 7: Make Predictions

Create `predict.mojo` to classify individual images:

```mojo

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

```text

## Understanding the Code

### Data Preparation

- **Normalization**: Scales pixel values to [0, 1] for better training
- **Flattening**: Converts 28x28 images to 784-element vectors
- **Batching**: Groups examples for efficient GPU processing

### Model Architecture

```text
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

```text

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

```mojo

# Try adjusting learning rate
var optimizer = SGD(learning_rate=0.001)  # Lower LR

# Train for more epochs
trainer.train(train_loader, val_loader, epochs=20)

# Verify data normalization
print("Data range: ", train_images.min(), " to ", train_images.max())
# Should be [0.0, 1.0]

```text

### Training Too Slow

### Solutions

```mojo
```mojo

# Increase batch size

var train_loader = BatchLoader(train_data, batch_size=128)

# Use release build for better performance

```text

```bash

```bash

pixi run mojo build --release train.mojo
./train

```text

### Out of Memory

### Solutions

```mojo
```mojo

# Reduce batch size

var train_loader = BatchLoader(train_data, batch_size=16)

# Use smaller model

self.model = Sequential([
    Layer("linear", input_size=784, output_size=64),  # Smaller
    ReLU(),
    Layer("linear", input_size=64, output_size=10),
])

```python

### Import Errors

```bash

```bash

# Ensure you're in the right directory
cd ProjectOdyssey/examples/first_model

# Verify shared library is accessible
ls ../../shared/

# Run from repository root
cd ../..
pixi run mojo run examples/first_model/train.mojo

```text

## Improving Your Model

### 1. Deeper Network

```mojo
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

```text

### 2. Different Optimizer

```mojo

```mojo

from shared.training import Adam

var optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

```text

### 3. Learning Rate Scheduling

```mojo
```mojo

from shared.training.schedulers import StepLR

var scheduler = StepLR(initial_lr=0.01, step_size=5, gamma=0.5)
trainer.add_scheduler(scheduler)

```text

### 4. Data Augmentation

```mojo

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

```text

## Next Steps

Congratulations! You've built, trained, and evaluated your first neural network with ML Odyssey.

### Learn More

- **[Performance Benchmarking](../advanced/benchmarking.md)** - Profile and optimize your models

### Try More Examples

- **LeNet-5**: Classic CNN for digit recognition (`papers/lenet5/`)
- **Custom Dataset**: Use your own data with `TensorDataset`
- **Different Architectures**: Experiment with convolutions, dropout, batch normalization

### Contribute

Found this tutorial helpful? Consider contributing:

- Share your experiments in GitHub Discussions
- Report bugs or suggest improvements via GitHub Issues
- Submit your own tutorial or example

## Related Documentation

- [Quickstart Guide](quickstart.md) - 5-minute introduction
- [Installation Guide](installation.md) - Detailed setup
