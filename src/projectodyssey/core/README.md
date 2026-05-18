# Core Library (WIP)

The `core` library contains fundamental building blocks and core functionality used across all paper implementations
in ML Odyssey. All components are implemented in Mojo for maximum performance.

## Purpose

This library provides:

- Reusable neural network layer implementations
- Low-level mathematical operations optimized for performance
- Custom types and data structures for ML workloads
- Utility functions for initialization, debugging, and profiling

**Key Principle**: Only FUNDAMENTAL building blocks that are shared across ALL or MOST paper implementations belong
here. Paper-specific code should be in the respective paper directories.

## Directory Organization

```text
core/
├── __init__.mojo           # Package root - exports main components
├── README.md               # This file
├── layers/                 # Neural network layer implementations
│   ├── __init__.mojo       # Layer exports
│   ├── linear.mojo         # Fully connected layers
│   ├── conv.mojo           # Convolutional layers
│   ├── activation.mojo     # Activation functions (ReLU, Sigmoid, etc.)
│   ├── normalization.mojo  # Batch/Layer normalization
│   └── pooling.mojo        # Pooling layers (Max, Avg)
├── ops/                    # Mathematical operations
│   ├── __init__.mojo       # Operation exports
│   ├── matmul.mojo         # Matrix multiplication
│   ├── elementwise.mojo    # Element-wise operations
│   ├── reduction.mojo      # Reduction operations (sum, mean, max)
│   └── broadcast.mojo      # Broadcasting utilities
├── types/                  # Custom types and data structures
│   ├── __init__.mojo       # Type exports
│   ├── tensor.mojo         # Tensor type and utilities
│   ├── shape.mojo          # Shape definitions
│   └── dtype.mojo          # Data type extensions
└── utils/                  # Utility functions
    ├── __init__.mojo       # Utility exports
    ├── init.mojo           # Weight initialization (Xavier, He, etc.)
    ├── memory.mojo         # Memory management helpers
    └── debug.mojo          # Debugging and profiling tools
```text

## What Belongs in Core

### DO Include

- Neural network layers used by multiple papers (Linear, Conv2D, ReLU, etc.)
- Fundamental mathematical operations (matmul, transpose, reduction, etc.)
- Common data structures (Tensor, Shape, DType)
- Widely-used initialization strategies (Xavier, He, etc.)
- Performance-critical utilities (SIMD operations, memory management)

### DON'T Include

- Paper-specific architectures (e.g., LeNet-5 complete model)
- Domain-specific preprocessing (e.g., MNIST-specific transforms)
- Experiment-specific utilities (e.g., custom metrics for one paper)
- One-off helper functions (put in paper's utils instead)

**Rule of Thumb**: If you're implementing something for a specific paper and you're not sure if it should be in core,
it probably shouldn't. Wait until 2-3 papers need it, then refactor into core.

## Using Core Components in Papers

```mojo
from projectodyssey.core.layers import Linear, ReLU, Conv2D
from projectodyssey.core.ops import matmul
from projectodyssey.core.types import Tensor
from projectodyssey.core.utils import xavier_init

# Use core components in your paper implementation
struct LeNet5:
    var conv1: Conv2D
    var conv2: Conv2D
    var fc1: Linear
    var fc2: Linear
    var fc3: Linear

    fn __init__(out self):
        # Initialize using core components
        self.conv1 = Conv2D(1, 6, kernel_size=5)
        self.conv2 = Conv2D(6, 16, kernel_size=5)
        self.fc1 = Linear(256, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

        # Use core utilities
        xavier_init(self.fc1.weights)
        xavier_init(self.fc2.weights)
        xavier_init(self.fc3.weights)
```text

## Mojo-Specific Guidelines

### Language Features

The core library leverages Mojo's performance features:

1. **SIMD Vectorization**: Use `SIMD` types for parallel operations
1. **Memory Safety**: Use ownership and borrowing for safe memory management
1. **Type Safety**: Use strong typing with `struct` for data structures
1. **Zero-Cost Abstractions**: Use `@always_inline` for performance-critical code

### Code Style

```mojo
# Use 'fn' for performance-critical functions (type-checked, compiled)
fn matmul(a: Tensor, b: Tensor) -> Tensor:
    # Implementation with strict types

# Use 'def' only for prototyping or when flexibility is needed
def experimental_operation(a, b):
    # Less strict typing for rapid iteration

# Prefer struct over class for data structures
struct Tensor:
    var data: DTypePointer[DType.float32]
    var shape: Shape
    var dtype: DType

    fn __init__(out self, shape: Shape):
        # Initialization

# Use @always_inline for small, frequently-called functions
@always_inline
fn relu(x: Float32) -> Float32:
    return max(x, 0.0)
```text

### Performance Considerations

1. **Minimize Allocations**: Reuse buffers where possible
1. **Leverage SIMD**: Vectorize operations for parallel execution
1. **Use Stack Allocation**: Prefer stack over heap when feasible
1. **Inline Critical Paths**: Use `@always_inline` for hot paths
1. **Profile First**: Measure before optimizing

## Testing

All core components must have comprehensive tests:

- Unit tests for individual functions and methods
- Integration tests for component interactions
- Performance benchmarks for critical operations
- Edge case and error condition tests

See `tests/core/` for test organization and examples.

## Contributing

When adding new components to core:

1. **Verify Shared Need**: Ensure at least 2-3 papers will use it
1. **Write Tests First**: Follow TDD principles
1. **Document Thoroughly**: Include docstrings and examples
1. **Optimize Later**: Correctness first, performance second
1. **Review Carefully**: Core changes affect all papers

## Performance Targets

Core components should meet these performance goals:

- Matrix multiplication: Within 2x of BLAS performance
- Element-wise operations: Fully vectorized with SIMD
- Memory operations: Zero-copy where possible
- Layer forward/backward: Competitive with PyTorch/TensorFlow

## Related Documentation

- Shared Library: `/worktrees/issue-13-plan/worktrees/issue-19-plan/shared/README.md`
- Models Library: `/worktrees/issue-13-plan/worktrees/issue-19-plan/shared/models/README.md`
- Mojo Language Guide: <https://docs.modular.com/mojo/>
- Project Documentation: `/worktrees/issue-13-plan/worktrees/issue-19-plan/CLAUDE.md`

## License

See repository LICENSE file.
