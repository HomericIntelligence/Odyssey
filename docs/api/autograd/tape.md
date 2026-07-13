# Automatic Differentiation

Tape-based automatic differentiation for gradient computation.

## Overview

ML Odyssey uses tape-based (dynamic) automatic differentiation:

1. Operations are recorded on a "tape" during the forward pass
2. Gradients are computed by replaying the tape in reverse

## Tape

The `Tape` class records operations for gradient computation.

```mojo
from odyssey.autograd import Tape
```

### Creating a Tape

```mojo
var tape = Tape()
```

### Recording Operations

Use a context manager to record operations:

```mojo
var tape = Tape()
with tape:
    var x = randn[DType.float32](10, 5)
    var w = randn[DType.float32](5, 3)
    var y = x @ w
    var loss = y.sum()
```

Or explicitly start/stop recording:

```mojo
var tape = Tape()
tape.start_recording()

var x = randn[DType.float32](10, 5)
var w = randn[DType.float32](5, 3)
var y = x @ w
var loss = y.sum()

tape.stop_recording()
```

### Computing Gradients

```mojo
# Compute gradients with respect to all recorded tensors
var grads = tape.backward(loss)

# Access individual gradients
var dx = grads.get(x)  # Gradient w.r.t x
var dw = grads.get(w)  # Gradient w.r.t w
```

### Complete Example

```mojo
from odyssey.autograd import Tape
from odyssey.core import randn

fn train_step():
    # Create input and weights
    var x = randn[DType.float32](32, 784)
    var w = randn[DType.float32](784, 10)
    var b = randn[DType.float32](10)

    # Record forward pass
    var tape = Tape()
    with tape:
        var logits = x @ w + b
        var probs = softmax(logits, dim=-1)
        var loss = cross_entropy_loss(probs, targets)

    # Compute gradients
    var grads = tape.backward(loss)

    # Update weights
    var dw = grads.get(w)
    var db = grads.get(b)
    w = w - learning_rate * dw
    b = b - learning_rate * db
```

## Gradient Context

### no_grad

Disable gradient computation for efficiency.

```mojo
from odyssey.autograd import no_grad
```

**Context Manager:**

```mojo
with no_grad():
    # No gradients computed here
    var output = model.forward(input)
    var predictions = output.argmax(dim=-1)
```

**Use Cases:**

- Inference (no training)
- Validation during training
- Computing metrics
- Preprocessing

### is_grad_enabled

Check if gradients are currently enabled.

```mojo
from odyssey.autograd import is_grad_enabled

if is_grad_enabled():
    print("Gradients are being tracked")
```

### set_grad_enabled

Programmatically enable/disable gradients.

```mojo
from odyssey.autograd import set_grad_enabled

set_grad_enabled(False)  # Disable
# ... operations without gradients ...
set_grad_enabled(True)   # Re-enable
```

## Gradient Accumulation

For large batch training with limited memory:

```mojo
var tape = Tape()
var accumulation_steps = 4

for i in range(accumulation_steps):
    var batch = get_batch(i)

    with tape:
        var output = model.forward(batch.input)
        var loss = criterion.forward(output, batch.target)
        loss = loss / accumulation_steps  # Scale for accumulation

    var grads = tape.backward(loss)
    accumulate_gradients(grads)  # Add to running total

# Apply accumulated gradients
optimizer.step()
optimizer.zero_grad()
```

## Retain Graph

By default, the computation graph is freed after `backward()`.
Use `retain_graph=True` for multiple backward passes:

```mojo
var grads1 = tape.backward(loss1, retain_graph=True)
var grads2 = tape.backward(loss2)  # Reuses same graph
```

**Use Cases:**

- Multiple losses
- Higher-order gradients
- Gradient penalty computation

## Supported Operations

The following operations are differentiable:

### Arithmetic

- Addition (+)
- Subtraction (-)
- Multiplication (*)
- Division (/)
- Power (**)
- Negation (-)

### Matrix Operations

- Matrix multiplication (@, matmul)
- Transpose (T)
- Dot product (dot)

### Reductions

- Sum (sum)
- Mean (mean)
- Max (max)

### Activations

- ReLU, LeakyReLU, ELU, GELU
- Sigmoid, Tanh
- Softmax, LogSoftmax

### Layers

- Linear, Conv2d
- BatchNorm2d
- MaxPool2d, AvgPool2d
- Dropout

### Losses

- CrossEntropyLoss
- MSELoss
- BCELoss

## Gradient Checkpointing

Save memory by recomputing forward pass during backward:

```mojo
from odyssey.autograd import checkpoint

fn expensive_forward(x: AnyTensor) -> AnyTensor:
    # Memory-intensive operations
    ...

# Checkpointed call - forward recomputed during backward
var output = checkpoint(expensive_forward, input)
```

**Trade-off:**

- Lower memory usage
- Higher computation time

## Debugging Gradients

### Check for NaN/Inf

```mojo
var grads = tape.backward(loss)
for tensor_id, grad in grads.items():
    if grad.has_nan():
        print("NaN gradient detected!")
    if grad.has_inf():
        print("Inf gradient detected!")
```

### Gradient Clipping

Prevent exploding gradients:

```mojo
from odyssey.training import clip_grad_norm

# Clip by global norm
clip_grad_norm(model.parameters(), max_norm=1.0)

# Or clip by value
clip_grad_value(model.parameters(), clip_value=0.5)
```

## See Also

- [AnyTensor Reference](../tensor.md) - Core tensor class
- [Optimizers](../training/optimizers.md) - Apply gradients
- [Layers](../nn/layers.md) - Neural network layers
