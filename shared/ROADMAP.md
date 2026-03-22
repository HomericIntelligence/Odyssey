# Shared Library Roadmap

This document outlines the current capabilities of the shared library, identifies gaps,
and provides a prioritized roadmap for future development.

## Current Capabilities

### Core Operations (`shared/core/`)

| Module | Description | Status |
|--------|-------------|--------|
| `any_tensor.mojo` | AnyTensor type with memory management | Complete |
| `activation.mojo` | ReLU, sigmoid, tanh, softmax, GELU, swish, mish, ELU, hard activations | Complete |
| `arithmetic.mojo` | Element-wise add, subtract, multiply, divide | Complete |
| `arithmetic_simd.mojo` | SIMD-optimized arithmetic | Complete |
| `broadcasting.mojo` | Shape broadcasting utilities | Complete |
| `comparison.mojo` | Element-wise comparisons | Complete |
| `conv.mojo` | Conv2D with backward pass | Complete |
| `dropout.mojo` | Dropout with backward pass | Complete |
| `elementwise.mojo` | exp, log, sqrt, power, abs, clip | Complete |
| `initializers.mojo` | Xavier, He, uniform, normal initialization | Complete |
| `linear.mojo` | Linear/dense layer with backward | Complete |
| `loss.mojo` | BCE, MSE, cross-entropy with backward | Complete |
| `matrix.mojo` | Matrix multiplication, transpose | Complete |
| `normalization.mojo` | BatchNorm, LayerNorm with backward | Complete |
| `pooling.mojo` | MaxPool2D, AvgPool2D, GlobalAvgPool with backward | Complete |
| `reduction.mojo` | sum, mean, max, min with axis support | Complete |
| `shape.mojo` | Shape computation utilities | Complete |

### Training (`shared/training/`)

| Module | Description | Status |
|--------|-------------|--------|
| `optimizers/sgd.mojo` | SGD with momentum, weight decay | Complete |
| `optimizers/adam.mojo` | Adam optimizer | Complete |
| `optimizers/rmsprop.mojo` | RMSprop optimizer | Complete |
| `schedulers/` | Learning rate schedulers | Partial |
| `metrics/` | Training metrics | Partial |
| `loops/` | Training loop utilities | Partial |

### Autograd (`shared/autograd/`)

| Module | Description | Status |
|--------|-------------|--------|
| `tape.mojo` | Gradient tape for automatic differentiation | Complete |
| `variable.mojo` | Variable wrapper for autograd | Complete |
| `functional.mojo` | Functional autograd operations | Complete |
| `optimizers.mojo` | Optimizer integration | Complete |

### Data (`shared/data/`)

| Module | Description | Status |
|--------|-------------|--------|
| `datasets/` | Dataset loading utilities | Partial |
| `formats/` | Data format handlers | Partial |

### Utilities (`shared/utils/`)

| Module | Description | Status |
|--------|-------------|--------|
| `logging.mojo` | Logging infrastructure | Complete |
| `config.mojo` | Configuration management | Complete |
| `io.mojo` | File I/O utilities | Complete |
| `serialization.mojo` | Tensor serialization | Complete |
| `random.mojo` | Random number generation | Complete |
| `profiling.mojo` | Performance profiling | Complete |
| `arg_parser.mojo` | Command-line argument parsing | Complete |
| `visualization.mojo` | Plotting utilities | Partial |

## Gap Analysis

### Priority 1: Critical for Model Architectures

#### Depthwise Convolutions

**Issue:** [#2227](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2227),
[#2228](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2228)

Required for efficient architectures like MobileNet:

- `depthwise_conv2d` - Convolution with groups=channels
- `depthwise_separable_conv2d` - Depthwise + pointwise convolution
- Backward passes for both

#### Attention Mechanisms

**Issue:** [#2229](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2229),
[#2230](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2230)

Required for Transformer architectures:

- `scaled_dot_product_attention` - Core attention mechanism
- `multi_head_attention` - Multi-head wrapper
- Backward passes for both

### Priority 2: Normalization Extensions

#### Additional Normalization Layers

**Issue:** [#2232](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2232),
[#2233](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2233)

- `group_norm` - Group normalization (for small batch sizes)
- `instance_norm` - Instance normalization (for style transfer)
- Backward passes for both

### Priority 3: Stateful Layer Classes

Currently only functional APIs exist. Stateful wrappers would simplify model building:

```mojo
# Current (functional)
var out = conv2d(x, weights, bias, stride=1, padding=1)

# Desired (stateful)
var conv = Conv2D(in_channels=64, out_channels=128, kernel_size=3)
var out = conv(x)
```

**Missing layer classes:**

- `Conv2D` - Wraps conv2d with stored weights
- `Linear` - Already exists in `shared/core/layers/linear.mojo`
- `BatchNorm2D` - Wraps batch_norm with running stats
- `LayerNorm` - Wraps layer_norm
- `Dropout` - Wraps dropout
- `MaxPool2D`, `AvgPool2D` - Pool layers
- `ReLU`, `Sigmoid`, `Tanh` - Activation layers

### Priority 4: Module System

**Components needed:**

- `Module` base class with `forward()`, `parameters()`, `train()/eval()`
- `Sequential` container for chaining modules
- `ModuleList` for dynamic module collections
- Parameter registration and traversal

### Priority 5: Advanced Optimizers

**Currently implemented:** SGD, Adam, RMSprop

**Missing:**

- `AdamW` - Adam with decoupled weight decay (recommended for transformers)
- `LAMB` - Layer-wise Adaptive Moments (for large batch training)
- `Adagrad` - Adaptive gradient algorithm
- `Adadelta` - Extension of Adagrad

### Priority 6: Data Augmentation

Transform framework exists but needs more transforms:

**Image transforms needed:**

- `RandomHorizontalFlip`
- `RandomVerticalFlip`
- `RandomRotation`
- `RandomCrop`
- `CenterCrop`
- `ColorJitter`
- `Normalize`
- `Resize`
- `ToTensor`

## Implementation Order

Based on dependencies and use cases:

### Phase 1: Foundation Extensions

1. Depthwise convolutions (#2227, #2228)
2. Group/Instance normalization (#2232, #2233)

### Phase 2: Attention

1. Scaled dot-product attention (#2229)
2. Multi-head attention (#2230)

### Phase 3: Layer System

1. Module base class and Sequential
2. Stateful layer wrappers

### Phase 4: Advanced Training

1. AdamW optimizer
2. Additional learning rate schedulers
3. Data augmentation transforms

## API Design Guidelines

### Function Naming

- Forward: `operation_name(input, ...)` e.g., `conv2d`, `batch_norm`
- Backward: `operation_name_backward(grad, ...)` e.g., `conv2d_backward`

### Parameter Order

1. Input tensor(s)
2. Weight tensor(s)
3. Bias tensor (optional)
4. Configuration parameters (stride, padding, etc.)

### Return Values

- Forward: Output tensor (or tuple for multiple outputs)
- Backward: Gradient tensor(s) matching input shapes

### Error Handling

- Validate shapes before computation
- Provide clear error messages with expected vs actual shapes
- Use `raises` for recoverable errors

## Contributing

When adding new functionality:

1. Create an issue describing the feature
2. Follow existing API patterns
3. Include comprehensive docstrings
4. Add backward pass if gradient is needed
5. Support float16, float32, float64 dtypes
6. Add tests for forward and backward passes
