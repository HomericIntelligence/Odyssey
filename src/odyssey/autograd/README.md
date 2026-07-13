# Autograd - Automatic Differentiation for ML Odyssey

This module provides automatic differentiation capabilities through a tape-based autograd system,
enabling neural network training without manual gradient computation.

## Overview

The autograd system consists of three main components:

1. **Variable**: Tensor wrapper that tracks operations for gradient computation
2. **GradientTape**: Records operations during forward pass for backward propagation
3. **Optimizers**: Update model parameters using computed gradients

## Quick Start

```mojo
from odyssey.autograd import Variable, GradientTape, SGD
from odyssey.core import zeros, ones

# Create variables with gradient tracking
var x = Variable(zeros(shape, dtype), requires_grad=True)
var y = Variable(ones(shape, dtype), requires_grad=True)

# Enable gradient tape
var tape = GradientTape()
tape.enable()

# Forward pass (operations are recorded)
var z = x * y
var loss = z.sum()

# Backward pass (compute gradients)
tape.backward()

# Access gradients
print(x.grad)  # ∂loss/∂x
print(y.grad)  # ∂loss/∂y
```text

## Components

### Variable

`Variable` wraps an `AnyTensor` and adds gradient tracking:

```mojo
from odyssey.autograd import Variable
from odyssey.core import zeros

# Create Variable from AnyTensor
var data = zeros(DynamicVector[Int](3, 4), DType.float32)
var x = Variable(data, requires_grad=True)

# Access underlying tensor
print(x.data)

# Access gradients (None until backward() is called)
print(x.grad)

# Reset gradients
x.zero_grad()

# Detach from computation graph
var y = x.detach()  # y shares data but doesn't track gradients
```text

**Key Methods:**

- `__init__(data, requires_grad=False)`: Create Variable from AnyTensor
- `zero_grad()`: Reset gradients to None
- `backward()`: Trigger backward pass (compute gradients)
- `detach()`: Create new Variable without gradient tracking

### GradientTape

`GradientTape` records operations for automatic differentiation:

```mojo
from odyssey.autograd import GradientTape

# Create tape
var tape = GradientTape()

# Enable recording
tape.enable()

# Operations are now recorded
var y = x + 2
var z = y * 3

# Compute gradients
tape.backward()

# Disable recording
tape.disable()

# Clear tape (free memory)
tape.clear()
```text

**Key Methods:**

- `enable()`: Start recording operations
- `disable()`: Stop recording operations
- `backward()`: Compute gradients via chain rule
- `clear()`: Remove all recorded operations

### Optimizers

#### SGD (Stochastic Gradient Descent)

Basic gradient descent with optional momentum:

```mojo
from odyssey.autograd import SGD

# Create optimizer
var optimizer = SGD(learning_rate=0.01)

# With momentum
var optimizer_momentum = SGD(learning_rate=0.01, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    var predictions = model(inputs)
    var loss = loss_fn(predictions, targets)

    # Backward pass
    tape.backward()

    # Update parameters
    optimizer.step(model_parameters)

    # Reset gradients
    optimizer.zero_grad(model_parameters)
```text

**Key Methods:**

- `__init__(learning_rate, momentum=0.0)`: Create optimizer
- `step(parameters)`: Update parameters using gradients
- `zero_grad(parameters)`: Reset all parameter gradients

## Integration with Existing Code

The autograd system integrates with the 27 existing backward pass functions in `src/odyssey/core/`:

### Arithmetic Operations

- `add_backward`
- `subtract_backward`
- `multiply_backward`
- `divide_backward`
- `power_backward`

### Matrix Operations

- `matmul_backward`
- `transpose_backward`

### Activation Functions

- `relu_backward`
- `sigmoid_backward`
- `tanh_backward`
- `softmax_backward`
- `gelu_backward`
- `leaky_relu_backward`
- `prelu_backward`

### Reduction Operations

- `sum_backward`
- `mean_backward`
- `max_backward`
- `min_backward`

### Elementwise Operations

- `exp_backward`
- `log_backward`
- `sqrt_backward`
- `abs_backward`
- `clip_backward`
- `square_backward`
- `neg_backward`

### Loss Functions

The autograd system works with existing loss functions in `src/odyssey/core/loss.mojo`:

```mojo
from odyssey.core.loss import binary_cross_entropy, mean_squared_error, cross_entropy

# Binary classification
var loss = binary_cross_entropy(predictions, targets)

# Regression
var loss = mean_squared_error(predictions, targets)

# Multi-class classification
var loss = cross_entropy(logits, targets_onehot)
```text

Each loss function has a corresponding `_backward` function for gradient computation.

## Training Example

Complete training loop example:

```mojo
from odyssey.autograd import Variable, GradientTape, SGD
from odyssey.core import zeros, ones
from odyssey.core.loss import mean_squared_error, mean_squared_error_backward

# Hyperparameters
var learning_rate = 0.01
var num_epochs = 100

# Create parameters (simple linear model: y = w*x + b)
var w = Variable(zeros(DynamicVector[Int](1), DType.float32), requires_grad=True)
var b = Variable(zeros(DynamicVector[Int](1), DType.float32), requires_grad=True)

# Create optimizer
var optimizer = SGD(learning_rate)

# Training data
var X = ...  # Input data
var Y = ...  # Target data

# Create gradient tape
var tape = GradientTape()
tape.enable()

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    var predictions = w.data * X + b.data
    var mse = mean_squared_error(predictions, Y)
    var loss = mean(mse)

    # Backward pass (manual for now, automatic in future)
    # TODO: Replace with tape.backward() once fully implemented
    var grad_loss = ones_like(loss)
    var grad_mse = mean_backward(grad_loss, mse.shape())
    var grad_pred = mean_squared_error_backward(grad_mse, predictions, Y)

    # Update gradients manually
    # TODO: This will be handled automatically by Variable operations
    w.grad = grad_pred * X  # ∂loss/∂w = ∂loss/∂pred * ∂pred/∂w
    b.grad = grad_pred      # ∂loss/∂b = ∂loss/∂pred * ∂pred/∂b

    # Update parameters
    var params = DynamicVector[Variable]()
    params.push_back(w)
    params.push_back(b)
    optimizer.step(params)

    # Reset gradients
    optimizer.zero_grad(params)

    # Print progress
    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss)
```text

## Current Status

### ✅ Implemented

- Variable wrapper with gradient tracking
- GradientTape for operation recording
- SGD optimizer
- Integration points with existing backward passes
- **Automatic operation recording in Variable**
- **Backward pass dispatch (18 op types: add/sub/mul/div/matmul/sum/mean/relu/sigmoid/tanh/flatten/linear/conv2d/depthwise_conv2d/maxpool2d/cross_entropy/batch_norm2d/concat)**
- **Automatic gradient computation in tape.backward()** — forward execution
  appends nodes in topological order, so reverse iteration is correct
  reverse-topological order (no DAG sort needed)
- **Gradient accumulation** for shared parameters (VariableRegistry.set_grad
  auto-accumulates)

### Phase 2 substrate (complete)

`variable_flatten`, `variable_linear`, `variable_conv2d`,
`variable_depthwise_conv2d`, `variable_maxpool2d`, `variable_cross_entropy`,
`variable_batch_norm`, and `variable_concat` enable end-to-end convnet training
(LeNet/AlexNet/VGG/ResNet, plus depthwise-separable MobileNet and Inception
depth-concat) forward+backward via `loss.backward(tape)`. See
`tests/odyssey/autograd/test_variable_layers.mojo`,
`test_variable_batch_norm.mojo`, `test_variable_depthwise_conv2d.mojo`, and
`test_variable_concat.mojo` for FD-validated gradient checks.

Also fixed in Phase 2: `SavedTensors.add_tensor` and `VariableRegistry.set_grad`
previously hardcoded `bitcast[Float32]`, silently corrupting fp16/bf16/fp64
gradients. Both now use dtype-agnostic `_get_float64`/`_set_float64` accessors.

### 📋 TODO (Future Work)

- Adam optimizer
- RMSprop optimizer
- Higher-order gradients (grad of grad)
- Gradient checkpointing for memory efficiency
- Graph optimization and operation fusion
- GPU/distributed autograd

## Design Decisions

### Why Tape-Based?

We chose a tape-based approach (like TensorFlow's GradientTape and JAX) rather than building
the graph directly into tensors (like PyTorch) because:

1. **Simpler initial implementation** - Explicit tape is easier to understand and debug
2. **Clear lifecycle** - Tape is created/enabled/disabled explicitly
3. **Memory efficiency** - Tape can be cleared after backward pass
4. **YAGNI principle** - We can add more sophisticated graph tracking later if needed

### Integration with AnyTensor

The autograd system **wraps** AnyTensor rather than modifying it because:

1. **Separation of concerns** - AnyTensor handles tensor ops, autograd handles differentiation
2. **Backward compatibility** - Existing code using AnyTensor continues to work
3. **Optional gradients** - Not all tensors need gradient tracking
4. **KISS principle** - Simpler architecture with clear boundaries

## References

- [PyTorch Autograd Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Micrograd](https://github.com/karpathy/micrograd) - Minimal autograd implementation
- [Existing Backward Passes](../../docs/backward-passes/README.md) - Comprehensive gradient guide
- [ADR-002](../../docs/adr/ADR-002-gradient-struct-return-types.md) - Gradient return type design

## Contributing

When adding new operations to autograd:

1. Implement the forward function in `src/odyssey/core/`
2. Implement the backward function in the same file
3. Add operation recording in Variable (TODO: not yet implemented)
4. Add dispatch case in tape.backward() (TODO: not yet implemented)
5. Write tests comparing autograd vs manual gradients

See existing backward passes in `src/odyssey/core/` for examples.
