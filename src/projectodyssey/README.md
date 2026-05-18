# Shared Library (WIP)

This directory contains reusable ML/AI components used across all paper implementations in the ProjectOdyssey repository.

## Purpose

The shared library provides:

- **Core utilities** and foundational classes that all papers depend on
- **Training infrastructure** including loops, optimizers, and schedulers
- **Data processing** tools for loading, preprocessing, and augmentation
- **Common utilities** for logging, visualization, and metrics

## Design Principles

### What Belongs in `src/projectodyssey/`

Code should live in `src/projectodyssey/` if it is:

- ✅ **Reusable** across multiple paper implementations
- ✅ **Generic** and not paper-specific
- ✅ **Stable** with well-defined interfaces
- ✅ **Foundational** - other code depends on it

Examples:

- Base neural network layers (Conv2D, Linear, etc.)
- Training loop framework
- Data loading infrastructure
- Common activation functions
- Metrics and evaluation tools

### What Belongs in `papers/`

Code should live in `papers/[paper-name]/` if it is:

- ❌ **Paper-specific** - unique to one paper's approach
- ❌ **Experimental** - still being validated
- ❌ **Specialized** - has unusual requirements
- ❌ **One-off** - unlikely to be reused

Examples:

- Novel architecture components unique to a paper
- Paper-specific data preprocessing
- Custom loss functions for one paper
- Experimental training techniques

### Decision Guidelines

When unsure, ask:

1. **Will ≥3 papers use this?** → `src/projectodyssey/`
1. **Is it a standard ML component?** → `src/projectodyssey/`
1. **Is it paper-specific innovation?** → `papers/`
1. **Is the interface still changing?** → `papers/` (move to `src/projectodyssey/` when stable)

## Directory Structure

```text
src/projectodyssey/
├── core/              # Fundamental utilities and base classes
│   ├── README.md      # Core components documentation
│   └── ...            # Base layers, tensor ops, memory management
├── training/          # Training loops, optimizers, schedulers
│   ├── README.md      # Training infrastructure documentation
│   └── ...            # Training loops, optimization algorithms
├── data/              # Data loading, preprocessing, augmentation
│   ├── README.md      # Data processing documentation
│   └── ...            # Loaders, transforms, datasets
├── utils/             # Helper functions, logging, visualization
│   ├── README.md      # Utilities documentation
│   └── ...            # Logging, metrics, plotting
└── README.md          # This file
```text

## Subdirectories

### `core/` - Fundamental Components

**Purpose**: Foundation that everything else builds on

### Contents

- Base neural network layers (Linear, Conv2D, MaxPool, etc.)
- Tensor operations and utilities
- Memory management helpers
- Module/Model base classes
- Parameter initialization
- Device management (CPU/GPU)

### Example Usage

```mojo
from projectodyssey.core.layers import Linear, ReLU
from projectodyssey.core.module import Module

struct MyModel(Module):
    var layer1: Linear
    var activation: ReLU

    fn forward(self, x: Tensor) -> Tensor:
        return self.activation(self.layer1(x))
```text

### `training/` - Training Infrastructure

**Purpose**: Everything needed to train models

### Contents

- Training loop framework
- Optimizers (SGD, Adam, RMSprop, etc.)
- Learning rate schedulers
- Gradient clipping utilities
- Checkpointing and model saving
- Distributed training support (future)

### Example Usage

```mojo
from projectodyssey.training.optimizer import SGD
from projectodyssey.training.scheduler import StepLR
from projectodyssey.training.trainer import Trainer

var optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
var scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
var trainer = Trainer(model, optimizer, scheduler)
trainer.train(train_loader, epochs=100)
```text

### `data/` - Data Processing

**Purpose**: Load, transform, and prepare data for training

### Contents

- Dataset base classes
- Data loaders and batching
- Common preprocessing (normalization, resizing, etc.)
- Data augmentation (random crops, flips, etc.)
- Format conversions (images, tensors, etc.)

### Example Usage

```mojo
from projectodyssey.data.dataset import ImageDataset
from projectodyssey.data.transforms import Normalize, RandomCrop, ToTensor
from projectodyssey.data.loader import DataLoader

var transform = Compose([
    RandomCrop(28),
    ToTensor(),
    Normalize(mean=0.5, std=0.5)
])
var dataset = ImageDataset("path/to/images", transform=transform)
var loader = DataLoader(dataset, batch_size=32, shuffle=True)
```text

### `utils/` - Helper Utilities

**Purpose**: Supporting tools that don't fit elsewhere

### Contents

- Logging and progress tracking
- Metrics computation (accuracy, loss, etc.)
- Visualization (plot training curves, show images, etc.)
- File I/O helpers
- Random seed management
- Timer and profiling utilities

### Example Usage

```mojo
from projectodyssey.utils.logger import Logger
from projectodyssey.utils.metrics import accuracy
from projectodyssey.utils.viz import plot_training_curves

var logger = Logger("training.log")
logger.info("Starting training...")

var acc = accuracy(predictions, targets)
logger.info("Accuracy: " + str(acc))

plot_training_curves(train_losses, val_losses)
```text

## Development Guidelines

### Adding New Components

When adding to the shared library:

1. **Design for Reusability**
   - Clear, generic interfaces
   - Minimal dependencies
   - Well-documented APIs

1. **Follow Mojo Best Practices**
   - Use `fn` for performance-critical code
   - Leverage `owned`/`borrowed` for memory safety
   - Use SIMD for vectorizable operations
   - Prefer compile-time constants

1. **Write Tests First**
   - Unit tests for all public functions
   - Integration tests for complex workflows
   - Property-based tests for invariants
   - Performance benchmarks for critical paths

1. **Document Thoroughly**
   - API documentation with examples
   - Performance characteristics
   - Usage patterns and anti-patterns
   - Migration guides for breaking changes

### Code Organization

- One file per logical component
- Group related functionality in subdirectories
- Keep public APIs minimal and stable
- Hide implementation details

### Testing Requirements

All shared library code must have:

- ✅ Unit tests with ≥90% coverage
- ✅ Integration tests for workflows
- ✅ Performance benchmarks
- ✅ Example usage in documentation

### Performance Expectations

Shared library code must be:

- **Fast**: Optimized for ML workloads
- **Memory-efficient**: Minimal allocations
- **Scalable**: Works with large datasets
- **Predictable**: Consistent performance characteristics

## Dependencies

### Internal Dependencies

The shared library has minimal internal dependencies:

```text
core/       # No dependencies (foundation)
training/   # Depends on: core/
data/       # Depends on: core/, utils/
utils/      # Depends on: core/
```text

### External Dependencies

- **Mojo Standard Library**: Core functionality
- **MAX Platform** (future): Hardware acceleration
- **Python Interop** (minimal): When necessary for existing tools

## Usage in Papers

Paper implementations import from `src/projectodyssey/`:

```mojo
# In papers/lenet5/src/model.mojo
from projectodyssey.core.layers import Conv2D, Linear, ReLU, MaxPool2D
from projectodyssey.core.module import Module

struct LeNet5(Module):
    var conv1: Conv2D
    var pool1: MaxPool2D
    # ... model definition
```text

Papers can extend shared components:

```mojo
# In papers/alexnet/src/model.mojo
from projectodyssey.core.layers import Conv2D
from projectodyssey.training.optimizer import SGD

# Use shared components + paper-specific additions
struct AlexNet(Module):
    var conv1: Conv2D  # From shared
    var custom_layer: CustomLayer  # Paper-specific
```text

## Contributing

### Proposing New Components

Before adding to `src/projectodyssey/`:

1. **Check if it exists**: Review existing components
1. **Verify reusability**: Will ≥3 papers use it?
1. **Design the API**: Create interface spec
1. **Get review**: Discuss with team
1. **Implement + Test**: Follow TDD
1. **Document**: Comprehensive docs + examples

### Moving Code from Papers to Shared

When a component proves reusable:

1. **Refactor for generality**: Remove paper-specific logic
1. **Design stable API**: Backward compatibility matters
1. **Add comprehensive tests**: Higher bar than paper code
1. **Write migration guide**: Help paper authors update
1. **Update all papers**: Fix any breaking changes

## Roadmap

### Phase 1: Core Foundation (Current)

- Directory structure
- Basic documentation
- Design guidelines

### Phase 2: Core Components

- Base layers (Linear, Conv2D, ReLU, etc.)
- Module system
- Tensor utilities

### Phase 3: Training Infrastructure

- Training loop framework
- Basic optimizers (SGD, Adam)
- Checkpointing

### Phase 4: Data Processing

- Dataset abstractions
- Data loaders
- Basic transforms

### Phase 5: Advanced Features

- Learning rate schedulers
- Advanced optimizers
- Distributed training support
- Performance profiling tools

## References

- [Mojo Language Documentation](https://docs.modular.com/mojo/manual/)
- [Papers Directory](../../../../../papers/README.md) - Paper-specific implementations
- [Project Architecture](../notes/review/agent-architecture-review.md) - Overall design
- [Contribution Guidelines](../../../../../CONTRIBUTING.md) - How to contribute

## License

See the main repository LICENSE file.

## Contact

For questions about the shared library:

- Open an issue on GitHub
- Discuss in the project's communication channels
- Review existing components for patterns
