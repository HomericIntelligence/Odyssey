# Migration Guide: Using the Shared Library (WIP)

This guide explains how to integrate the shared library into paper implementations in ML Odyssey.

## Overview

The shared library (`src/projectodyssey/`) provides reusable components for implementing research papers. This guide covers:

1. How to import and use shared components
1. When to use shared vs custom implementations
1. Common patterns and best practices
1. Migration examples from standalone to shared-based code

## Quick Start

### Basic Setup

```mojo
# In papers/lenet5/src/model.mojo
from projectodyssey.core import Linear, Conv2D, ReLU, MaxPool2D, Sequential
from projectodyssey.training import SGD, train_epoch, validate_epoch
from projectodyssey.data import DataLoader, TensorDataset

# Use shared components directly
var model = Sequential([
    Conv2D(1, 6, kernel_size=5),
    ReLU(),
    MaxPool2D(kernel_size=2),
    # ... more layers
])

var optimizer = SGD(learning_rate=0.01, momentum=0.9)
```text

### Verify Installation

```bash
# Test that shared library is importable
mojo run scripts/verify_installation.mojo
```text

## Import Patterns

### Importing from Root Package

For commonly-used components, import from root:

```mojo
from shared import Linear, Conv2D, ReLU, SGD, DataLoader, Logger
```text

**Exports from Root** (`src/projectodyssey/__init__.mojo`):

- Core layers: `Linear`, `Conv2D`, `ReLU`, `MaxPool2D`, `Dropout`
- Optimizers: `SGD`, `Adam`, `AdamW`
- Data: `DataLoader`, `TensorDataset`
- Utils: `Logger`

### Importing from Subpackages

For specialized components, import from subpackage:

```mojo
from projectodyssey.core import BatchNorm2D, LayerNorm, Softmax
from projectodyssey.training import CosineAnnealingLR, EarlyStopping, ModelCheckpoint
from projectodyssey.data import ImageDataset, RandomCrop, RandomHorizontalFlip
from projectodyssey.utils import plot_training_curves, set_seed
```text

### Importing from Sub-Subpackages

For advanced usage:

```mojo
from projectodyssey.training.optimizers import AdamW, RMSprop
from projectodyssey.training.schedulers import WarmupLR, ExponentialLR
from projectodyssey.training.metrics import Precision, Recall, ConfusionMatrix
from projectodyssey.training.callbacks import LRSchedulerCallback
```text

## When to Use Shared vs Custom

### Use Shared Library When

✅ **Component is standard across papers**:

- Standard layers (Linear, Conv2D, ReLU)
- Common optimizers (SGD, Adam)
- Basic training loops
- Data loading infrastructure

✅ **Performance matters**:

- Shared components are SIMD-optimized
- Maintained and benchmarked
- Used across multiple papers

✅ **Maintenance burden is high**:

- Complex implementations (Conv2D, Adam)
- Error-prone code (optimizers with state)
- Well-tested components

### Use Custom Implementation When

❌ **Component is paper-specific**:

- Novel architectures unique to paper
- Custom loss functions
- Paper-specific data preprocessing

❌ **Experimentation needed**:

- Testing variations of components
- Prototyping new ideas
- Research exploration

❌ **Shared version doesn't exist**:

- Component not yet in shared library
- Can contribute back to shared later

### Decision Matrix

| Component | Use Shared? | Rationale |
| --- | --- | --- |
| Linear layer | ✅ Yes | Standard, well-tested |
| Conv2D | ✅ Yes | Complex, SIMD-optimized |
| SGD/Adam | ✅ Yes | Stateful, error-prone |
| ResNet block | ⚠️ Maybe | Standard but paper-specific variants exist |
| Attention mechanism | ⚠️ Maybe | Standard pattern but many variations |
| Novel architecture | ❌ No | Paper-specific |
| Custom loss | ❌ No | Paper-specific |
| Custom data format | ❌ No | Paper-specific |

## Migration Examples

### Example 1: LeNet-5 Model

**Before** (standalone):

```mojo
# papers/lenet5/src/model.mojo

struct Conv2D:
    # Custom Conv2D implementation
    var weights: Tensor
    var bias: Tensor

    fn forward(self, x: Tensor) -> Tensor:
        # Custom convolution code
        pass

struct LeNet5:
    var conv1: Conv2D
    var conv2: Conv2D
    # ... more layers

    fn __init__(inout self):
        self.conv1 = Conv2D(1, 6, 5)
        self.conv2 = Conv2D(6, 16, 5)
```text

**After** (using shared):

```mojo
# papers/lenet5/src/model.mojo
from projectodyssey.core import Conv2D, ReLU, MaxPool2D, Linear, Flatten, Sequential

struct LeNet5:
    var model: Sequential

    fn __init__(out self):
        self.model = Sequential([
            Conv2D(1, 6, kernel_size=5),
            ReLU(),
            MaxPool2D(kernel_size=2),
            Conv2D(6, 16, kernel_size=5),
            ReLU(),
            MaxPool2D(kernel_size=2),
            Flatten(),
            Linear(16 * 4 * 4, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, 10),
        ])

    fn forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)
```text

### Benefits

- ✅ Less code to maintain
- ✅ SIMD-optimized convolution
- ✅ Tested and verified
- ✅ Consistent with other papers

### Example 2: Training Loop

**Before** (custom training):

```mojo
# papers/lenet5/src/train.mojo

fn train_one_epoch(
    model: LeNet5,
    optimizer: CustomSGD,
    data: List[Tensor]
) -> Float32:
    var total_loss: Float32 = 0.0

    for batch in data:
        # Custom training logic
        var output = model.forward(batch[0])
        var loss = compute_loss(output, batch[1])

        # Custom backward pass
        var grads = backward(loss)

        # Custom optimizer step
        for i in range(len(model.parameters())):
            optimizer.update(model.parameters()[i], grads[i])

        total_loss += loss

    return total_loss / Float32(len(data))
```text

**After** (using shared):

```mojo
# papers/lenet5/src/train.mojo
from projectodyssey.training import SGD, train_epoch, validate_epoch
from projectodyssey.data import DataLoader

fn train_lenet5(
    model: LeNet5,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: Int
):
    var optimizer = SGD(learning_rate=0.01, momentum=0.9)

    for epoch in range(num_epochs):
        var train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
        var val_loss = validate_epoch(model, val_loader, loss_fn)

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
```text

### Benefits

- ✅ Much less code
- ✅ Correct gradient computation
- ✅ Proper validation mode handling
- ✅ Consistent across papers

### Example 3: Data Loading

**Before** (manual batching):

```mojo
# papers/lenet5/data/loader.mojo

fn create_batches(
    data: Tensor,
    labels: Tensor,
    batch_size: Int
) -> List[Tuple[Tensor, Tensor]]:
    var batches = List[Tuple[Tensor, Tensor]]()

    for i in range(0, len(data), batch_size):
        var batch_data = data[i:i+batch_size]
        var batch_labels = labels[i:i+batch_size]
        batches.append((batch_data, batch_labels))

    return batches
```text

**After** (using shared):

```mojo
# papers/lenet5/data/loader.mojo
from projectodyssey.data import TensorDataset, DataLoader, Compose, ToTensor, Normalize

fn create_data_loaders(
    train_data: Tensor,
    train_labels: Tensor,
    test_data: Tensor,
    test_labels: Tensor,
    batch_size: Int
) -> Tuple[DataLoader, DataLoader]:
    # Create transforms
    var transform = Compose([
        ToTensor(),
        Normalize(mean=0.5, std=0.5),
    ])

    # Create datasets
    var train_dataset = TensorDataset(train_data, train_labels, transform)
    var test_dataset = TensorDataset(test_data, test_labels, transform)

    # Create loaders
    var train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    var test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader)
```text

### Benefits

- ✅ Built-in shuffling
- ✅ Transform pipeline
- ✅ Drop last batch support
- ✅ Consistent API

### Example 4: Adding Custom Components

When you need custom functionality, extend shared components:

```mojo
# papers/resnet/src/layers.mojo
from projectodyssey.core import Linear, Conv2D, ReLU, Module

struct ResidualBlock(Module):
    """Custom ResidualBlock - paper-specific."""
    var conv1: Conv2D
    var conv2: Conv2D
    var relu: ReLU
    var shortcut: Optional[Conv2D]

    fn __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        stride: Int = 1
    ):
        # Use shared layers as building blocks
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = ReLU()

        # Add shortcut if needed
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride)

    fn forward(self, x: Tensor) -> Tensor:
        """Forward with residual connection."""
        var identity = x

        var out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.shortcut:
            identity = self.shortcut.value()(x)

        out = out + identity
        return self.relu(out)
```text

**Pattern**: Use shared components as building blocks, add paper-specific logic on top.

## Common Patterns

### Pattern 1: Model Construction

```mojo
from projectodyssey.core import Sequential, Linear, ReLU, Dropout

# Simple sequential model
var model = Sequential([
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10),
])
```text

### Pattern 2: Training Setup

```mojo
from projectodyssey.training import Adam, CosineAnnealingLR, EarlyStopping, ModelCheckpoint
from projectodyssey.utils import Logger, set_seed

# Reproducibility
set_seed(42)

# Logging
var logger = Logger("experiment")

# Optimizer and scheduler
var optimizer = Adam(learning_rate=0.001)
var scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Callbacks
var early_stop = EarlyStopping(patience=10)
var checkpoint = ModelCheckpoint("best_model.mojo", monitor="val_loss")
```text

### Pattern 3: Data Pipeline

```mojo
from projectodyssey.data import ImageDataset, DataLoader, Compose, RandomCrop, RandomHorizontalFlip, Normalize

# Data augmentation pipeline
var train_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=0.5, std=0.5),
])

# No augmentation for validation
var val_transform = Normalize(mean=0.5, std=0.5)

# Datasets and loaders
var train_dataset = ImageDataset(train_paths, train_labels, train_transform)
var val_dataset = ImageDataset(val_paths, val_labels, val_transform)

var train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
var val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
```text

### Pattern 4: Complete Training Script

```mojo
from shared import Linear, Conv2D, ReLU, MaxPool2D
from projectodyssey.training import SGD, train_epoch, validate_epoch
from projectodyssey.data import DataLoader
from projectodyssey.utils import Logger, plot_training_curves, set_seed

fn main():
    # Setup
    set_seed(42)
    var logger = Logger("training")

    # Model
    var model = create_model()

    # Data
    var train_loader, val_loader = create_data_loaders()

    # Training
    var optimizer = SGD(learning_rate=0.01, momentum=0.9)

    var train_losses = List[Float32]()
    var val_losses = List[Float32]()

    for epoch in range(50):
        var train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
        var val_loss = validate_epoch(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Visualization
    plot_training_curves(train_losses, val_losses, save_path="results/curves.png")
```text

## Best Practices

### 1. Always Import What You Need

❌ **Don't**: Wildcard imports

```mojo
from projectodyssey.core import *  # Avoid this
```text

✅ **Do**: Explicit imports

```mojo
from projectodyssey.core import Linear, Conv2D, ReLU
```text

### 2. Use Type Hints

```mojo
from projectodyssey.core import Module
from projectodyssey.training import Optimizer

fn train(model: Module, optimizer: Optimizer):
    # Type-safe function
    pass
```text

### 3. Follow Mojo Best Practices

```mojo
# Use fn for performance
fn forward(self, x: Tensor) -> Tensor:
    return self.model.forward(x)

# Use owned/borrowed appropriately
fn train_step(inout model: Module, borrowed batch: Batch):
    pass
```text

### 4. Keep Paper-Specific Code Separate

```text
papers/lenet5/
├── src/
│   ├── model.mojo       # Model using shared layers
│   ├── train.mojo       # Training using shared infrastructure
│   └── custom.mojo      # Paper-specific code only
├── data/
│   └── mnist.mojo       # Paper-specific data handling
└── README.md
```text

### 5. Document Dependencies

```markdown
# papers/lenet5/README.md

## Dependencies

- `shared.core`: Linear, Conv2D, ReLU, MaxPool2D
- `shared.training`: SGD, train_epoch
- `shared.data`: DataLoader, TensorDataset
```text

## Troubleshooting

### Import Errors

**Problem**: `ImportError: cannot import name 'Linear' from 'shared.core'`

### Solution

1. Verify shared library is installed: `mojo run scripts/verify_installation.mojo`
1. Check that component is implemented (not just planned)
1. Use correct import path (check `src/projectodyssey/__init__.mojo`)

### API Mismatches

**Problem**: Method signature doesn't match documentation

### Solution

1. Check shared library version
1. Refer to implemented API in code, not just docs
1. File issue if documentation is incorrect

### Performance Issues

**Problem**: Shared component slower than expected

### Solution

1. Verify you're using release build: `mojo build --release`
1. Check if SIMD optimization is enabled
1. Profile to identify bottleneck
1. File performance issue with benchmark

### Missing Components

**Problem**: Need component that isn't in shared library yet

### Solution

1. Implement custom version in paper directory
1. Mark as candidate for shared library
1. Contribute back to shared library after validation

## Version Compatibility

### Checking Version

```mojo
from shared import __version__
print(__version__)  # "0.1.0"
```text

### Version Requirements

In `papers/lenet5/README.md`:

```markdown
## Requirements

- Shared library: >=0.1.0, <0.2.0
- Mojo: >=24.5
```text

## Contributing Back to Shared

When you implement something useful in a paper that could be shared:

1. **Identify candidate**: Is it used in multiple papers?
1. **Generalize**: Remove paper-specific assumptions
1. **Test thoroughly**: Add comprehensive tests
1. **Document**: Add docstrings and examples
1. **Submit PR**: Contribute to `src/projectodyssey/`

## Migration Checklist

When migrating a paper to use shared library:

- [ ] Identify components available in shared library
- [ ] Replace custom implementations with shared versions
- [ ] Update imports
- [ ] Test that behavior is unchanged
- [ ] Remove obsolete custom code
- [ ] Update documentation
- [ ] Update dependencies list
- [ ] Run full test suite
- [ ] Benchmark performance

## Getting Help

- **Documentation**: See `src/projectodyssey/README.md`, `src/projectodyssey/*/README.md`
- **Examples**: See `src/projectodyssey/EXAMPLES.md`
- **Issues**: File issues in repository
- **Questions**: Ask in discussions

## Summary

### Key Takeaways

1. Use shared library for standard components
1. Keep paper-specific code separate
1. Import explicitly, not with wildcards
1. Follow Mojo best practices
1. Contribute useful components back

The shared library makes paper implementations:

- ✅ Faster to write
- ✅ More maintainable
- ✅ Higher performance
- ✅ Consistent across papers

Happy implementing! 🚀
