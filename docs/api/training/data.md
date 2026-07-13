# Data Loading

Dataset and DataLoader utilities for training.

## Overview

ML Odyssey provides data loading utilities for efficient batch processing:

```mojo
from odyssey.training.data import Dataset, DataLoader
```

## Dataset

Base class for datasets.

### Creating a Dataset

```mojo
struct MyDataset(Dataset):
    var data: AnyTensor
    var labels: AnyTensor

    fn __init__(out self, data: AnyTensor, labels: AnyTensor):
        self.data = data
        self.labels = labels

    fn __len__(self) -> Int:
        return self.data.shape()[0]

    fn __getitem__(self, index: Int) raises -> Tuple[AnyTensor, AnyTensor]:
        return (self.data[index], self.labels[index])
```

### Built-in Datasets

#### MNIST

```mojo
from odyssey.training.data import MNIST

var train_dataset = MNIST(root="./data", train=True, download=True)
var test_dataset = MNIST(root="./data", train=False)

print("Training samples:", len(train_dataset))  # 60000
print("Test samples:", len(test_dataset))       # 10000
```

#### CIFAR-10

```mojo
from odyssey.training.data import CIFAR10

var train_dataset = CIFAR10(root="./data", train=True, download=True)
var test_dataset = CIFAR10(root="./data", train=False)
```

#### FashionMNIST

```mojo
from odyssey.training.data import FashionMNIST

var train_dataset = FashionMNIST(root="./data", train=True)
```

## DataLoader

Batches and shuffles data for training.

### Constructor

```mojo
fn __init__(
    out self,
    dataset: Dataset,
    batch_size: Int = 1,
    shuffle: Bool = False,
    drop_last: Bool = False,
) raises
```

**Parameters:**

- `dataset`: Dataset to load from
- `batch_size`: Samples per batch (default: 1)
- `shuffle`: Randomize order each epoch (default: False)
- `drop_last`: Drop incomplete final batch (default: False)

### Example

```mojo
from odyssey.training.data import DataLoader, MNIST

var dataset = MNIST(root="./data", train=True)
var dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
)

# Iterate over batches
for batch in dataloader:
    var images = batch.data      # Shape: (32, 1, 28, 28)
    var labels = batch.labels    # Shape: (32,)
    # ... training step ...
```

### Batch Structure

Each batch contains:

```mojo
struct Batch:
    var data: AnyTensor    # Input features
    var labels: AnyTensor  # Target labels
```

## Data Transforms

Apply transformations to data.

### Normalize

Normalize tensor with mean and std.

```mojo
from odyssey.training.data.transforms import Normalize

var normalize = Normalize(mean=[0.1307], std=[0.3081])  # MNIST stats
var normalized = normalize(image)
```

### ToTensor

Convert to tensor and scale to [0, 1].

```mojo
from odyssey.training.data.transforms import ToTensor

var to_tensor = ToTensor()
var tensor = to_tensor(raw_data)
```

### Compose

Chain multiple transforms.

```mojo
from odyssey.training.data.transforms import Compose, ToTensor, Normalize

var transform = Compose([
    ToTensor(),
    Normalize(mean=[0.1307], std=[0.3081]),
])

var dataset = MNIST(root="./data", transform=transform)
```

### Data Augmentation

```mojo
from odyssey.training.data.transforms import (
    RandomHorizontalFlip,
    RandomCrop,
    RandomRotation,
)

var train_transform = Compose([
    ToTensor(),
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=28, padding=4),
    RandomRotation(degrees=15),
    Normalize(mean=[0.5], std=[0.5]),
])
```

## Train/Validation Split

Split dataset for training and validation.

```mojo
from odyssey.training.data import random_split

var dataset = MNIST(root="./data", train=True)

# 80% train, 20% validation
var train_size = Int(0.8 * len(dataset))
var val_size = len(dataset) - train_size
var (train_dataset, val_dataset) = random_split(
    dataset,
    [train_size, val_size],
)

var train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
var val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

## Custom Data Loading

### From NumPy Arrays

```mojo
from odyssey.training.data import TensorDataset

var X = randn[DType.float32](1000, 784)  # Features
var y = zeros[DType.int32](1000)         # Labels

var dataset = TensorDataset(X, y)
var loader = DataLoader(dataset, batch_size=32)
```

### From Files

```mojo
struct ImageFolderDataset(Dataset):
    var image_paths: List[String]
    var labels: List[Int]

    fn __init__(out self, root: String) raises:
        # Load image paths and labels from directory structure
        # root/class1/img1.png, root/class2/img2.png, ...
        ...

    fn __len__(self) -> Int:
        return len(self.image_paths)

    fn __getitem__(self, index: Int) raises -> Tuple[AnyTensor, AnyTensor]:
        var image = load_image(self.image_paths[index])
        var label = self.labels[index]
        return (image, label)
```

## Complete Training Example

```mojo
from odyssey.training.data import MNIST, DataLoader
from odyssey.training.data.transforms import Compose, ToTensor, Normalize
from odyssey.training.optimizers import Adam
from odyssey.core.layers import Linear, ReLU, Sequential

# Prepare data
var transform = Compose([
    ToTensor(),
    Normalize(mean=[0.1307], std=[0.3081]),
])

var train_dataset = MNIST(root="./data", train=True, transform=transform)
var test_dataset = MNIST(root="./data", train=False, transform=transform)

var train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
var test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create model
var model = Sequential()
model.add(Flatten())
model.add(Linear(784, 128))
model.add(ReLU())
model.add(Linear(128, 10))

var optimizer = Adam(model.parameters(), lr=0.001)
var criterion = CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.train()
    for batch in train_loader:
        var tape = Tape()
        with tape:
            var output = model.forward(batch.data)
            var loss = criterion.forward(output, batch.labels)

        optimizer.zero_grad()
        tape.backward(loss)
        optimizer.step()

    # Validation
    model.set_inference_mode()
    var correct = 0
    var total = 0
    with no_grad():
        for batch in test_loader:
            var output = model.forward(batch.data)
            var predictions = output.argmax(dim=-1)
            correct += (predictions == batch.labels).sum().item[DType.int32]()
            total += batch.labels.numel()

    print("Epoch", epoch, "Accuracy:", correct / total)
```

## Performance Tips

1. **Use appropriate batch size** - Larger batches for GPU, smaller for CPU
2. **Enable shuffle for training** - Improves convergence
3. **Use drop_last=True** - Avoids small final batches
4. **Prefetch data** - Load next batch while computing

## See Also

- [Optimizers](optimizers.md) - Training optimization
- [Autograd](../autograd/tape.md) - Gradient computation
- [AnyTensor Reference](../tensor.md) - Core tensor class
