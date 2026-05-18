# Activation Functions

Non-linear activation functions for neural networks.

## Overview

Activation functions introduce non-linearity into neural networks, enabling them
to learn complex patterns. All activations support both functional and layer forms.

## ReLU Family

### ReLU

Rectified Linear Unit: `max(0, x)`

```mojo
from projectodyssey.core import relu
from projectodyssey.core.layers import ReLU
```

**Functional:**

```mojo
var y = relu(x)
```

**Layer:**

```mojo
var activation = ReLU()
var y = activation.forward(x)
```

**Properties:**

- Fast to compute
- Sparse activations (zeros for negative inputs)
- Can cause "dying ReLU" problem

### LeakyReLU

Leaky ReLU: `max(alpha*x, x)` where alpha is small (default 0.01)

```mojo
from projectodyssey.core import leaky_relu
from projectodyssey.core.layers import LeakyReLU

# Functional
var y = leaky_relu(x, alpha=0.01)

# Layer
var activation = LeakyReLU(negative_slope=0.01)
var y = activation.forward(x)
```

**Parameters:**

- `negative_slope` / `alpha`: Slope for negative values (default: 0.01)

### ELU

Exponential Linear Unit.

```mojo
from projectodyssey.core import elu
from projectodyssey.core.layers import ELU

# Functional
var y = elu(x, alpha=1.0)

# Layer
var activation = ELU(alpha=1.0)
var y = activation.forward(x)
```

**Formula:**

- `x` if `x > 0`
- `alpha * (exp(x) - 1)` if `x <= 0`

### GELU

Gaussian Error Linear Unit (used in transformers).

```mojo
from projectodyssey.core import gelu
from projectodyssey.core.layers import GELU

var y = gelu(x)
```

**Formula:** `x * Phi(x)` where Phi is the Gaussian CDF

**Properties:**

- Smooth, differentiable everywhere
- Default activation in BERT, GPT models

### SiLU / Swish

Sigmoid Linear Unit: `x * sigmoid(x)`

```mojo
from projectodyssey.core import silu
from projectodyssey.core.layers import SiLU

var y = silu(x)
```

**Properties:**

- Self-gated activation
- Smooth and non-monotonic

## Sigmoid and Tanh

### Sigmoid

Logistic sigmoid: `1 / (1 + exp(-x))`

```mojo
from projectodyssey.core import sigmoid
from projectodyssey.core.layers import Sigmoid

# Functional
var y = sigmoid(x)

# Layer
var activation = Sigmoid()
var y = activation.forward(x)
```

**Properties:**

- Output range: (0, 1)
- Used for binary classification outputs
- Can suffer from vanishing gradients

### Tanh

Hyperbolic tangent: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`

```mojo
from projectodyssey.core import tanh
from projectodyssey.core.layers import Tanh

var y = tanh(x)
```

**Properties:**

- Output range: (-1, 1)
- Zero-centered outputs
- Used in RNNs and LSTMs

## Softmax

### Softmax

Normalize to probability distribution: `exp(x_i) / sum(exp(x_j))`

```mojo
from projectodyssey.core import softmax
from projectodyssey.core.layers import Softmax

# Along last dimension (default)
var probs = softmax(logits, dim=-1)

# Layer form
var activation = Softmax(dim=-1)
var probs = activation.forward(logits)
```

**Parameters:**

- `dim`: Dimension to normalize over (default: -1)

**Properties:**

- Outputs sum to 1.0 along specified dimension
- Used for multi-class classification
- Numerically stable implementation (log-sum-exp trick)

### LogSoftmax

Log of softmax: `log(softmax(x))`

```mojo
from projectodyssey.core import log_softmax
from projectodyssey.core.layers import LogSoftmax

var log_probs = log_softmax(logits, dim=-1)
```

**Properties:**

- More numerically stable than `log(softmax(x))`
- Often used with NLLLoss

## Comparison Table

| Activation | Range | Properties | Use Case |
| --- | --- | --- | --- |
| ReLU | [0, inf) | Fast, sparse | Default for hidden layers |
| LeakyReLU | (-inf, inf) | Avoids dying ReLU | Alternative to ReLU |
| ELU | (-alpha, inf) | Smooth negative region | Alternative to ReLU |
| GELU | (-0.17, inf) | Smooth, probabilistic | Transformers |
| SiLU | (-0.28, inf) | Self-gated | Modern architectures |
| Sigmoid | (0, 1) | Bounded, smooth | Binary outputs |
| Tanh | (-1, 1) | Zero-centered | RNNs, LSTMs |
| Softmax | (0, 1) | Sums to 1 | Classification output |

## Gradient Properties

All activations support backward pass for gradient computation:

```mojo
# Forward pass
var y = relu(x)

# Backward pass (during training)
var grad_x = relu_backward(grad_y, x)
```

**ReLU Gradient:**

- 1 if x > 0
- 0 if x <= 0

**Sigmoid Gradient:**

- `sigmoid(x) * (1 - sigmoid(x))`

**Tanh Gradient:**

- `1 - tanh(x)^2`

## Best Practices

1. **Use ReLU as default** for hidden layers
2. **Use GELU** for transformer models
3. **Use Softmax** for classification outputs
4. **Use Sigmoid** for binary classification
5. **Avoid Sigmoid/Tanh in deep networks** (vanishing gradients)

## See Also

- [Layers](layers.md) - Neural network layers
- [Losses](losses.md) - Loss functions
- [AnyTensor Reference](../tensor.md) - Core tensor class
