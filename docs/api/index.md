# ML Odyssey API Reference

Comprehensive API documentation for ML Odyssey's tensor library and neural network components.

## Quick Navigation

- [AnyTensor](tensor.md) - Core tensor class with autograd support
- [Operations](operations/) - Tensor operations (arithmetic, reduction, linear algebra)
- [Neural Networks](nn/) - Layers, activations, and losses
- [Autograd](autograd/) - Automatic differentiation system
- [Training](training/) - Optimizers, schedulers, and data loading

## Core Concepts

### AnyTensor

The `AnyTensor` struct is the fundamental data structure in ML Odyssey. It supports:

- Multi-dimensional arrays with arbitrary shape
- Multiple data types (float32, float16, bfloat16, int8, etc.)
- Automatic gradient computation via tape-based autograd
- SIMD-optimized operations

```mojo
from projectodyssey.core import AnyTensor, zeros, ones, randn

# Create tensors
var x = zeros[DType.float32](2, 3)        # Shape: (2, 3)
var y = ones[DType.float32](2, 3)         # Shape: (2, 3)
var z = randn[DType.float32](64, 128)     # Random normal

# Basic operations
var sum_result = x + y
var product = x * y
var matmul_result = x @ y.T               # Matrix multiplication
```

### Automatic Differentiation

ML Odyssey uses tape-based automatic differentiation for computing gradients:

```mojo
from projectodyssey.autograd import Tape, no_grad

# Record operations on tape
var tape = Tape()
with tape:
    var x = randn[DType.float32](10, 5)
    var w = randn[DType.float32](5, 3)
    var y = x @ w
    var loss = y.sum()

# Compute gradients
var grads = tape.backward(loss)
var dx = grads.get(x)
var dw = grads.get(w)
```

### Layers and Models

Build neural networks using composable layers:

```mojo
from projectodyssey.core.layers import Linear, Conv2d, BatchNorm2d, ReLU

# Define layers
var linear = Linear(784, 128)
var conv = Conv2d(3, 64, kernel_size=3, padding=1)
var bn = BatchNorm2d(64)
var relu = ReLU()

# Forward pass
var x = randn[DType.float32](32, 3, 28, 28)  # NCHW format
var out = conv.forward(x)
out = bn.forward(out)
out = relu.forward(out)
```

## Module Structure

```text
src/projectodyssey/
├── core/                    # Core tensor library
│   ├── any_tensor.mojo       # AnyTensor struct
│   ├── arithmetic.mojo     # +, -, *, /, etc.
│   ├── reduction.mojo      # sum, mean, max, min
│   ├── shape.mojo          # reshape, transpose, view
│   ├── linalg.mojo         # matmul, dot, etc.
│   └── layers/             # Neural network layers
│       ├── linear.mojo
│       ├── conv2d.mojo
│       └── ...
├── autograd/               # Automatic differentiation
│   ├── tape.mojo           # Tape-based autograd
│   └── operations.mojo     # Differentiable ops
└── training/               # Training infrastructure
    ├── optimizers/         # SGD, Adam, AdamW, etc.
    ├── schedulers/         # Learning rate schedulers
    └── metrics/            # Accuracy, loss tracking
```

## Data Types

ML Odyssey supports multiple data types for tensors:

| Type | Description | Use Case |
| --- | --- | --- |
| `DType.float32` | 32-bit floating point | Default, general training |
| `DType.float16` | 16-bit floating point | Mixed precision training |
| `DType.bfloat16` | Brain floating point | TPU compatibility |
| `DType.int8` | 8-bit integer | Quantized inference |
| `DType.int32` | 32-bit integer | Indices, counters |

## Memory Layout

Tensors use row-major (C-contiguous) memory layout by default. Key points:

- **Contiguous tensors**: Optimal for SIMD operations
- **Strided tensors**: Created by slicing, transpose; may require `.contiguous()`
- **Broadcasting**: Automatic shape expansion following NumPy rules

## Error Handling

ML Odyssey uses explicit error handling:

```mojo
# Shape mismatch errors
var a = zeros[DType.float32](3, 4)
var b = zeros[DType.float32](5, 6)
var c = a + b  # Raises: "Cannot broadcast shapes [3, 4] and [5, 6]"

# Type mismatch errors
var x = zeros[DType.float32](3, 3)
var y = zeros[DType.int32](3, 3)
var z = x + y  # Raises: "DType mismatch: float32 vs int32"
```

## Performance Tips

1. **Use contiguous tensors** for SIMD optimization
2. **Batch operations** instead of iterating over elements
3. **Enable mixed precision** for memory/speed tradeoffs
4. **Profile with benchmarks** in `benchmarks/` directory

## Next Steps

- [AnyTensor Reference](tensor.md) - Complete tensor API
- [Operations Guide](operations/arithmetic.md) - Arithmetic operations
- [Building Models](nn/layers.md) - Neural network layers
- [Training Guide](training/optimizers.md) - Optimization algorithms
