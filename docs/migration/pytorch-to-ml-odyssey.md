# PyTorch to ML Odyssey Migration Guide

A comprehensive guide for migrating PyTorch models and workflows to ML Odyssey.

## Overview

ML Odyssey provides a familiar API for PyTorch users while leveraging Mojo's performance
benefits. This guide covers the key differences and provides practical migration patterns.

## Quick Reference

| Concept | PyTorch | ML Odyssey |
| --- | --- | --- |
| Tensor | `torch.Tensor` | `AnyTensor` |
| Module | `nn.Module` | `Module` trait |
| Autograd | Automatic | `Tape` context |
| Device | `.to('cuda')` | CPU (GPU coming) |
| DType | Inferred | Explicit |

## Tensor Operations

### Creating Tensors

**PyTorch:**

```python
import torch

# From list
x = torch.tensor([1, 2, 3])

# Zeros/ones
y = torch.zeros(10, 10)
z = torch.ones(5, 5)

# Random
r = torch.randn(3, 4)
u = torch.rand(3, 4)

# Range
a = torch.arange(0, 10, 1)

# Identity
I = torch.eye(3)
```

**ML Odyssey:**

```mojo
from shared.core import zeros, ones, randn, arange, eye

# From values (use full or creation functions)
var x = arange(1.0, 4.0, 1.0, DType.float32)

# Zeros/ones - note explicit dtype
var y = zeros[DType.float32](10, 10)
var z = ones[DType.float32](5, 5)

# Random
var r = randn[DType.float32](3, 4)

# Range
var a = arange(0.0, 10.0, 1.0, DType.float32)

# Identity
var I = eye(3, DType.float32)
```

**Key difference:** ML Odyssey requires explicit dtype specification.

### Basic Arithmetic

**PyTorch:**

```python
c = a + b
d = a - b
e = a * b  # Element-wise
f = a / b
g = a ** 2
```

**ML Odyssey:**

```mojo
var c = a + b
var d = a - b
var e = a * b  # Element-wise
var f = a / b
var g = a ** 2.0
```

Operations are nearly identical.

### Matrix Operations

**PyTorch:**

```python
# Matrix multiplication
c = a @ b
c = torch.matmul(a, b)
c = torch.mm(a, b)

# Transpose
t = a.T
t = a.transpose(0, 1)

# Dot product
d = torch.dot(a, b)
```

**ML Odyssey:**

```mojo
# Matrix multiplication
var c = a @ b
var c = matmul(a, b)

# Transpose
var t = a.T
var t = a.transpose()

# Dot product
var d = dot(a, b)
```

### Shape Operations

**PyTorch:**

```python
# Reshape
y = x.reshape(2, 3)
y = x.view(2, 3)

# Squeeze/unsqueeze
y = x.squeeze()
y = x.unsqueeze(0)

# Flatten
y = x.flatten()
```

**ML Odyssey:**

```mojo
# Reshape - uses List[Int] for shape
var y = x.reshape(List[Int](2, 3))

# Squeeze/unsqueeze
var y = x.squeeze()
var y = x.unsqueeze(0)

# Flatten
var y = x.flatten()
```

### Indexing and Slicing

**PyTorch:**

```python
# Single element
x[0, 1]

# Row/column
x[0]        # First row
x[:, 1]     # Second column

# Slice
x[0:5, 2:7]
```

**ML Odyssey:**

```mojo
# Single element - uses List[Int]
x[List[Int](0, 1)]

# Row
x[List[Int](0)]

# Slice
var starts = List[Int](0, 2)
var ends = List[Int](5, 7)
x.slice(starts, ends)

# Or use slice_along_axis
x.slice_along_axis(axis=1, start=2, end=7)
```

**Key difference:** ML Odyssey uses explicit slice functions instead of Python's slice syntax.

### Reductions

**PyTorch:**

```python
# Sum
s = x.sum()
s = x.sum(dim=0)
s = x.sum(dim=1, keepdim=True)

# Mean, max, min
m = x.mean()
mx = x.max()
mn = x.min()
```

**ML Odyssey:**

```mojo
# Sum
var s = x.sum()
var s = x.sum(axis=0)
var s = x.sum(axis=1, keepdim=True)

# Mean, max, min
var m = x.mean()
var mx = x.max()
var mn = x.min()
```

Nearly identical, but uses `axis` instead of `dim`.

## Neural Network Layers

### Linear Layer

**PyTorch:**

```python
import torch.nn as nn

layer = nn.Linear(784, 128)
output = layer(input)
```

**ML Odyssey:**

```mojo
from shared.core.layers import Linear

var layer = Linear(784, 128)
var output = layer.forward(input)
```

**Key difference:** ML Odyssey uses explicit `.forward()` call.

### Convolutional Layer

**PyTorch:**

```python
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
output = conv(input)  # NCHW format
```

**ML Odyssey:**

```mojo
from shared.core.layers import Conv2d

var conv = Conv2d(3, 64, kernel_size=3, padding=1)
var output = conv.forward(input)  # NCHW format
```

Same format conventions (NCHW).

### Batch Normalization

**PyTorch:**

```python
bn = nn.BatchNorm2d(64)
bn.train()  # Training mode
# bn in inference mode for testing
```

**ML Odyssey:**

```mojo
from shared.core.layers import BatchNorm2d

var bn = BatchNorm2d(64)
bn.train()  # Training mode
bn.set_inference_mode()  # Inference mode
```

### Activation Functions

**PyTorch:**

```python
import torch.nn.functional as F

y = F.relu(x)
y = F.sigmoid(x)
y = F.softmax(x, dim=-1)
y = torch.tanh(x)
```

**ML Odyssey:**

```mojo
from shared.core import relu, sigmoid, softmax, tanh

var y = relu(x)
var y = sigmoid(x)
var y = softmax(x, dim=-1)
var y = tanh(x)
```

## Model Definition

### PyTorch Model

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel()
```

### ML Odyssey Model

```mojo
from shared.core.layers import Linear, ReLU

struct SimpleModel:
    var fc1: Linear
    var relu: ReLU
    var fc2: Linear

    fn __init__(out self) raises:
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    fn forward(mut self, x: AnyTensor) raises -> AnyTensor:
        var out = self.fc1.forward(x)
        out = self.relu.forward(out)
        out = self.fc2.forward(out)
        return out

    fn parameters(self) -> List[AnyTensor]:
        var params = List[AnyTensor]()
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params^

var model = SimpleModel()
```

**Key differences:**

1. Use `struct` instead of `class`
2. Use `fn __init__(out self)` constructor
3. Explicit `.forward()` calls
4. Manual `parameters()` collection

## Automatic Differentiation

### PyTorch (Implicit)

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2 + 1
loss = y.sum()

loss.backward()  # Compute gradients
print(x.grad)    # Access gradient
```

### ML Odyssey (Explicit Tape)

```mojo
from shared.autograd import Tape

var tape = Tape()
with tape:
    var x = randn[DType.float32](1)
    var y = x * 2.0 + 1.0
    var loss = y.sum()

var grads = tape.backward(loss)
var dx = grads.get(x)  # Access gradient
```

**Key difference:** ML Odyssey uses explicit tape recording.

### Disabling Gradients

**PyTorch:**

```python
with torch.no_grad():
    output = model(input)
```

**ML Odyssey:**

```mojo
from shared.autograd import no_grad

with no_grad():
    var output = model.forward(input)
```

## Training Loop

### PyTorch

```python
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()

        output = model(batch['input'])
        loss = criterion(output, batch['target'])

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### ML Odyssey

```mojo
from shared.training.optimizers import Adam
from shared.core.layers import CrossEntropyLoss
from shared.autograd import Tape

var model = SimpleModel()
var optimizer = Adam(model.parameters(), lr=0.001)
var criterion = CrossEntropyLoss()

for epoch in range(10):
    for batch in dataloader:
        var tape = Tape()
        with tape:
            var output = model.forward(batch.input)
            var loss = criterion.forward(output, batch.target)

        optimizer.zero_grad()
        var grads = tape.backward(loss)
        optimizer.step()

    print("Epoch", epoch, "Loss:", loss.item[DType.float32]())
```

## Common Gotchas

### 1. Explicit DTypes

```python
# PyTorch - dtype inferred
x = torch.tensor([1.0])  # float32 by default
```

```mojo
# ML Odyssey - dtype required
var x = full(List[Int](1), 1.0, DType.float32)
```

### 2. List Shapes

```python
# PyTorch - tuple shapes
x = torch.zeros(3, 4)
x = x.reshape(2, 6)
```

```mojo
# ML Odyssey - List[Int] shapes
var x = zeros[DType.float32](3, 4)  # Variadic OK for creation
var x = x.reshape(List[Int](2, 6))  # List for reshape
```

### 3. Parameter Collection

```python
# PyTorch - automatic registration
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)  # Auto-registered

model.parameters()  # Works automatically
```

```mojo
# ML Odyssey - manual collection
struct Model:
    var fc: Linear

    fn parameters(self) -> List[AnyTensor]:
        return self.fc.parameters()  # Must implement
```

### 4. In-Place Operations

```python
# PyTorch - common, uses underscore suffix
x.add_(1)
x.relu_()
```

```mojo
# ML Odyssey - explicit mutation
x.add_inplace(1)  # Or create new tensor
var y = relu(x)
```

### 5. Device Management

```python
# PyTorch
x = x.to('cuda')
model = model.cuda()
```

```mojo
# ML Odyssey - CPU only (for now)
# No device management needed
```

## Weight Conversion

### Export from PyTorch

```python
import torch
import numpy as np

# Save model weights
model = SimpleModel()
state = model.state_dict()

# Convert to numpy and save
weights = {}
for name, param in state.items():
    weights[name] = param.cpu().numpy()

np.savez('weights.npz', **weights)
```

### Load in ML Odyssey

```mojo
# Load weights (when npz loading is implemented)
var weights = load_npz("weights.npz")
model.load_state_dict(weights)
```

## Feature Comparison

| Feature | PyTorch | ML Odyssey |
| --- | :---: | :---: |
| Tensor operations | Full | Most |
| Autograd | Automatic | Tape-based |
| GPU support | CUDA/ROCm | Coming soon |
| Distributed | DDP/FSDP | Planned |
| JIT compilation | TorchScript | Native Mojo |
| Mixed precision | AMP | Supported |
| Quantization | Full | Basic |
| ONNX export | Built-in | Planned |

## Migration Checklist

- [ ] Replace `torch.Tensor` with `AnyTensor`
- [ ] Add explicit dtype to tensor creation
- [ ] Replace `class` with `struct` for models
- [ ] Change `__call__` to explicit `.forward()` calls
- [ ] Implement `parameters()` method manually
- [ ] Wrap forward pass in `Tape` context
- [ ] Replace `dim=` with `axis=` in reductions
- [ ] Convert slice syntax to `.slice()` methods
- [ ] Remove `.to(device)` calls (CPU only for now)

## Resources

- [Examples](../../examples/)
