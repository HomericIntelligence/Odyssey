# Neural Network Layers

Building blocks for constructing neural network models.

## Overview

All layers follow a consistent interface:

```mojo
trait Layer:
    fn forward(mut self, x: AnyTensor) raises -> AnyTensor
    fn backward(mut self, grad_output: AnyTensor) raises -> AnyTensor
    fn parameters(self) -> List[AnyTensor]
```

## Linear Layers

### Linear

Fully connected layer: `y = xW^T + b`

```mojo
from projectodyssey.core.layers import Linear
```

**Constructor:**

```mojo
fn __init__(out self, in_features: Int, out_features: Int, bias: Bool = True) raises
```

**Parameters:**

- `in_features`: Size of input features
- `out_features`: Size of output features
- `bias`: Include bias term (default: True)

**Example:**

```mojo
from projectodyssey.core.layers import Linear
from projectodyssey.core import randn

var linear = Linear(784, 128)  # 784 -> 128
var x = randn[DType.float32](32, 784)  # Batch of 32
var y = linear.forward(x)  # Shape: (32, 128)
```

## Convolutional Layers

### Conv2d

2D convolution layer for image processing.

```mojo
from projectodyssey.core.layers import Conv2d
```

**Constructor:**

```mojo
fn __init__(
    out self,
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = True,
) raises
```

**Parameters:**

- `in_channels`: Number of input channels
- `out_channels`: Number of output channels (filters)
- `kernel_size`: Size of the convolving kernel
- `stride`: Stride of the convolution (default: 1)
- `padding`: Zero-padding added to input (default: 0)
- `dilation`: Spacing between kernel elements (default: 1)
- `groups`: Number of blocked connections (default: 1)
- `bias`: Include bias term (default: True)

**Example:**

```mojo
from projectodyssey.core.layers import Conv2d
from projectodyssey.core import randn

# 3 input channels, 64 output channels, 3x3 kernel
var conv = Conv2d(3, 64, kernel_size=3, padding=1)

# Input: NCHW format (batch, channels, height, width)
var x = randn[DType.float32](32, 3, 28, 28)
var y = conv.forward(x)  # Shape: (32, 64, 28, 28)
```

**Output Size Formula:**

```text
H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

## Normalization Layers

### BatchNorm2d

2D batch normalization for convolutional layers.

```mojo
from projectodyssey.core.layers import BatchNorm2d
```

**Constructor:**

```mojo
fn __init__(
    out self,
    num_features: Int,
    eps: Float = 1e-5,
    momentum: Float = 0.1,
    affine: Bool = True,
) raises
```

**Parameters:**

- `num_features`: Number of channels
- `eps`: Small value for numerical stability (default: 1e-5)
- `momentum`: Running stats momentum (default: 0.1)
- `affine`: Learnable scale and shift (default: True)

**Example:**

```mojo
from projectodyssey.core.layers import BatchNorm2d, Conv2d
from projectodyssey.core import randn

var conv = Conv2d(3, 64, kernel_size=3, padding=1)
var bn = BatchNorm2d(64)

var x = randn[DType.float32](32, 3, 28, 28)
var y = conv.forward(x)
y = bn.forward(y)  # Normalized output
```

**Training vs Inference:**

```mojo
# Training mode (default) - updates running stats
bn.train()
y = bn.forward(x)

# Inference mode - uses running stats
bn.set_inference_mode()
y = bn.forward(x)
```

## Pooling Layers

### MaxPool2d

2D max pooling layer.

```mojo
from projectodyssey.core.layers import MaxPool2d
```

**Constructor:**

```mojo
fn __init__(
    out self,
    kernel_size: Int,
    stride: Optional[Int] = None,
    padding: Int = 0,
) raises
```

**Parameters:**

- `kernel_size`: Size of the pooling window
- `stride`: Stride of pooling (default: same as kernel_size)
- `padding`: Zero-padding (default: 0)

**Example:**

```mojo
from projectodyssey.core.layers import MaxPool2d
from projectodyssey.core import randn

var pool = MaxPool2d(kernel_size=2, stride=2)
var x = randn[DType.float32](32, 64, 28, 28)
var y = pool.forward(x)  # Shape: (32, 64, 14, 14)
```

### AvgPool2d

2D average pooling layer.

```mojo
from projectodyssey.core.layers import AvgPool2d

var pool = AvgPool2d(kernel_size=2, stride=2)
var x = randn[DType.float32](32, 64, 28, 28)
var y = pool.forward(x)  # Shape: (32, 64, 14, 14)
```

### AdaptiveAvgPool2d

Adaptive average pooling to target output size.

```mojo
from projectodyssey.core.layers import AdaptiveAvgPool2d

var pool = AdaptiveAvgPool2d(output_size=(1, 1))  # Global average pooling
var x = randn[DType.float32](32, 512, 7, 7)
var y = pool.forward(x)  # Shape: (32, 512, 1, 1)
```

## Dropout Layers

### Dropout

Randomly zero elements during training.

```mojo
from projectodyssey.core.layers import Dropout
```

**Constructor:**

```mojo
fn __init__(out self, p: Float = 0.5) raises
```

**Parameters:**

- `p`: Probability of dropping an element (default: 0.5)

**Example:**

```mojo
from projectodyssey.core.layers import Dropout
from projectodyssey.core import randn

var dropout = Dropout(p=0.5)

# Training mode - applies dropout
dropout.train()
var x = randn[DType.float32](32, 128)
var y = dropout.forward(x)  # ~50% of values are zero, rest scaled by 2

# Inference mode - no dropout
dropout.set_inference_mode()
y = dropout.forward(x)  # Identity function
```

## Embedding Layers

### Embedding

Lookup table for discrete tokens.

```mojo
from projectodyssey.core.layers import Embedding
```

**Constructor:**

```mojo
fn __init__(out self, num_embeddings: Int, embedding_dim: Int) raises
```

**Parameters:**

- `num_embeddings`: Size of vocabulary
- `embedding_dim`: Dimension of embedding vectors

**Example:**

```mojo
from projectodyssey.core.layers import Embedding
from projectodyssey.core import zeros

var embed = Embedding(10000, 256)  # 10K vocab, 256-dim embeddings

# Input: token indices
var tokens = zeros[DType.int32](32, 100)  # 32 sequences, 100 tokens each
var embeddings = embed.forward(tokens)  # Shape: (32, 100, 256)
```

## Flatten

### Flatten

Flatten all dimensions except batch.

```mojo
from projectodyssey.core.layers import Flatten

var flatten = Flatten()
var x = randn[DType.float32](32, 64, 7, 7)
var y = flatten.forward(x)  # Shape: (32, 3136)
```

## Sequential Container

Combine layers into a sequential model.

```mojo
from projectodyssey.core.layers import Sequential, Linear, ReLU, Dropout

var model = Sequential()
model.add(Linear(784, 256))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(Linear(256, 10))

var x = randn[DType.float32](32, 784)
var y = model.forward(x)  # Shape: (32, 10)
```

## Layer Utilities

### Getting Parameters

```mojo
# Get all trainable parameters
var params = layer.parameters()

# Count parameters
var total = 0
for p in params:
    total += p.numel()
print("Total parameters:", total)
```

### Training Mode

```mojo
# Set to training mode
layer.train()

# Set to inference mode
layer.set_inference_mode()
```

## See Also

- [Activations](activations.md) - Activation functions
- [Losses](losses.md) - Loss functions
- [Optimizers](../training/optimizers.md) - Parameter optimization
