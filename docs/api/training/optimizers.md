# Optimizers

Optimization algorithms for training neural networks.

## Overview

Optimizers update model parameters based on computed gradients:

```mojo
from shared.training.optimizers import SGD, Adam, AdamW
```

All optimizers share a common interface:

```mojo
trait Optimizer:
    fn step(mut self) raises -> None
    fn zero_grad(mut self) raises -> None
```

## SGD

Stochastic Gradient Descent with optional momentum.

```mojo
from shared.training.optimizers import SGD
```

**Constructor:**

```mojo
fn __init__(
    out self,
    parameters: List[AnyTensor],
    lr: Float,
    momentum: Float = 0.0,
    weight_decay: Float = 0.0,
    nesterov: Bool = False,
) raises
```

**Parameters:**

- `parameters`: Model parameters to optimize
- `lr`: Learning rate
- `momentum`: Momentum factor (default: 0.0)
- `weight_decay`: L2 regularization (default: 0.0)
- `nesterov`: Use Nesterov momentum (default: False)

**Example:**

```mojo
from shared.training.optimizers import SGD

var model = create_model()
var optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        var output = model.forward(batch.input)
        var loss = criterion.forward(output, batch.target)
        var grads = tape.backward(loss)
        optimizer.step()
```

## Adam

Adaptive Moment Estimation optimizer.

```mojo
from shared.training.optimizers import Adam
```

**Constructor:**

```mojo
fn __init__(
    out self,
    parameters: List[AnyTensor],
    lr: Float = 0.001,
    betas: Tuple[Float, Float] = (0.9, 0.999),
    eps: Float = 1e-8,
    weight_decay: Float = 0.0,
) raises
```

**Parameters:**

- `parameters`: Model parameters to optimize
- `lr`: Learning rate (default: 0.001)
- `betas`: Coefficients for running averages (default: (0.9, 0.999))
- `eps`: Numerical stability term (default: 1e-8)
- `weight_decay`: L2 regularization (default: 0.0)

**Example:**

```mojo
from shared.training.optimizers import Adam

var optimizer = Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
)
```

**Properties:**

- Adaptive learning rates per parameter
- Works well with default hyperparameters
- Good for sparse gradients

## AdamW

Adam with decoupled weight decay (recommended over Adam).

```mojo
from shared.training.optimizers import AdamW
```

**Constructor:**

```mojo
fn __init__(
    out self,
    parameters: List[AnyTensor],
    lr: Float = 0.001,
    betas: Tuple[Float, Float] = (0.9, 0.999),
    eps: Float = 1e-8,
    weight_decay: Float = 0.01,
) raises
```

**Example:**

```mojo
from shared.training.optimizers import AdamW

var optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
)
```

**Difference from Adam:**

- Weight decay applied directly to weights, not gradients
- Better generalization, especially with large models

## RMSprop

Root Mean Square Propagation.

```mojo
from shared.training.optimizers import RMSprop
```

**Constructor:**

```mojo
fn __init__(
    out self,
    parameters: List[AnyTensor],
    lr: Float = 0.01,
    alpha: Float = 0.99,
    eps: Float = 1e-8,
    momentum: Float = 0.0,
    weight_decay: Float = 0.0,
) raises
```

**Example:**

```mojo
from shared.training.optimizers import RMSprop

var optimizer = RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
)
```

## LARS

Layer-wise Adaptive Rate Scaling (for large batch training).

```mojo
from shared.training.optimizers import LARS
```

**Constructor:**

```mojo
fn __init__(
    out self,
    parameters: List[AnyTensor],
    lr: Float,
    momentum: Float = 0.9,
    weight_decay: Float = 0.0,
    trust_coefficient: Float = 0.001,
) raises
```

**Example:**

```mojo
from shared.training.optimizers import LARS

# For large batch sizes (e.g., 4096+)
var optimizer = LARS(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    trust_coefficient=0.001,
)
```

## Optimizer Methods

### step

Apply gradients to parameters.

```mojo
optimizer.step()
```

### zero_grad

Reset gradients to zero before next backward pass.

```mojo
optimizer.zero_grad()
```

**Important:** Always call `zero_grad()` before computing new gradients.

### state_dict / load_state_dict

Save and restore optimizer state.

```mojo
# Save state
var state = optimizer.state_dict()

# Load state
optimizer.load_state_dict(state)
```

## Learning Rate Scheduling

Adjust learning rate during training.

### StepLR

Decay LR by factor every N steps.

```mojo
from shared.training.schedulers import StepLR

var scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train_epoch()
    scheduler.step()  # LR *= 0.1 every 30 epochs
```

### CosineAnnealingLR

Cosine annealing schedule.

```mojo
from shared.training.schedulers import CosineAnnealingLR

var scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

### OneCycleLR

One-cycle learning rate policy.

```mojo
from shared.training.schedulers import OneCycleLR

var scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=1000,
)

for step in range(1000):
    train_step()
    scheduler.step()
```

## Gradient Clipping

Prevent exploding gradients.

```mojo
from shared.training import clip_grad_norm, clip_grad_value

# Clip by global L2 norm
clip_grad_norm(model.parameters(), max_norm=1.0)

# Clip by value
clip_grad_value(model.parameters(), clip_value=0.5)
```

## Complete Training Loop

```mojo
from shared.training.optimizers import AdamW
from shared.training.schedulers import CosineAnnealingLR
from shared.training import clip_grad_norm
from shared.autograd import Tape

var model = create_model()
var optimizer = AdamW(model.parameters(), lr=0.001)
var scheduler = CosineAnnealingLR(optimizer, T_max=100)
var criterion = CrossEntropyLoss()

for epoch in range(100):
    model.train()
    for batch in train_loader:
        # Forward pass
        var tape = Tape()
        with tape:
            var output = model.forward(batch.input)
            var loss = criterion.forward(output, batch.target)

        # Backward pass
        optimizer.zero_grad()
        var grads = tape.backward(loss)

        # Gradient clipping
        clip_grad_norm(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

    # Update learning rate
    scheduler.step()
```

## Comparison Table

| Optimizer | Best For | Default LR |
|-----------|----------|------------|
| SGD | ConvNets, when tuned | 0.01-0.1 |
| SGD+Momentum | Most vision tasks | 0.01-0.1 |
| Adam | Default choice | 0.001 |
| AdamW | Transformers, large models | 0.001 |
| RMSprop | RNNs | 0.01 |
| LARS | Large batch training | 0.1+ |

## See Also

- [Autograd](../autograd/tape.md) - Gradient computation
- [Layers](../nn/layers.md) - Neural network layers
- [Data Loading](data.md) - Dataset and DataLoader
