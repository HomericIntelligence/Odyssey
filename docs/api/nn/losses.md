# Loss Functions

Loss functions for training neural networks.

## Overview

Loss functions measure the discrepancy between predicted and target values.
All loss functions support both functional and class forms.

## Classification Losses

### CrossEntropyLoss

Combines LogSoftmax and NLLLoss for multi-class classification.

```mojo
from odyssey.core import cross_entropy_loss
from odyssey.core.layers import CrossEntropyLoss
```

**Functional:**

```mojo
# logits: (batch, num_classes)
# targets: (batch,) - class indices
var loss = cross_entropy_loss(logits, targets)
```

**Class Form:**

```mojo
var criterion = CrossEntropyLoss()
var loss = criterion.forward(logits, targets)
```

**Parameters:**

- `weight`: Optional class weights for imbalanced datasets
- `reduction`: "mean" (default), "sum", or "none"
- `label_smoothing`: Smooth targets (default: 0.0)

**Example:**

```mojo
from odyssey.core import randn, zeros
from odyssey.core.layers import CrossEntropyLoss

var logits = randn[DType.float32](32, 10)  # 32 samples, 10 classes
var targets = zeros[DType.int32](32)  # Class indices

var criterion = CrossEntropyLoss()
var loss = criterion.forward(logits, targets)
print("Loss:", loss.item[DType.float32]())
```

### NLLLoss

Negative Log Likelihood Loss (expects log-probabilities).

```mojo
from odyssey.core import nll_loss
from odyssey.core.layers import NLLLoss

# log_probs: (batch, num_classes) - output of LogSoftmax
# targets: (batch,) - class indices
var loss = nll_loss(log_probs, targets)
```

**Use with LogSoftmax:**

```mojo
var log_probs = log_softmax(logits, dim=-1)
var loss = nll_loss(log_probs, targets)
```

### BinaryCrossEntropyLoss

Binary cross-entropy for binary classification.

```mojo
from odyssey.core import binary_cross_entropy
from odyssey.core.layers import BCELoss

# predictions: (batch,) - probabilities in [0, 1]
# targets: (batch,) - binary labels (0 or 1)
var loss = binary_cross_entropy(predictions, targets)
```

**With Logits (more stable):**

```mojo
from odyssey.core import binary_cross_entropy_with_logits
from odyssey.core.layers import BCEWithLogitsLoss

# logits: (batch,) - raw scores
# targets: (batch,) - binary labels
var loss = binary_cross_entropy_with_logits(logits, targets)
```

## Regression Losses

### MSELoss

Mean Squared Error: `mean((pred - target)^2)`

```mojo
from odyssey.core import mse_loss
from odyssey.core.layers import MSELoss
```

**Functional:**

```mojo
var loss = mse_loss(predictions, targets)
```

**Class Form:**

```mojo
var criterion = MSELoss(reduction="mean")
var loss = criterion.forward(predictions, targets)
```

**Parameters:**

- `reduction`: "mean" (default), "sum", or "none"

**Example:**

```mojo
from odyssey.core import randn
from odyssey.core.layers import MSELoss

var predictions = randn[DType.float32](32, 10)
var targets = randn[DType.float32](32, 10)

var criterion = MSELoss()
var loss = criterion.forward(predictions, targets)
```

### L1Loss

Mean Absolute Error: `mean(|pred - target|)`

```mojo
from odyssey.core import l1_loss
from odyssey.core.layers import L1Loss

var loss = l1_loss(predictions, targets)
```

**Properties:**

- More robust to outliers than MSE
- Less smooth gradient at zero

### SmoothL1Loss

Huber Loss: combines L1 and L2.

```mojo
from odyssey.core import smooth_l1_loss
from odyssey.core.layers import SmoothL1Loss

var criterion = SmoothL1Loss(beta=1.0)
var loss = criterion.forward(predictions, targets)
```

**Formula:**

- `0.5 * (x)^2 / beta` if `|x| < beta`
- `|x| - 0.5 * beta` otherwise

**Properties:**

- Smooth gradient everywhere
- Less sensitive to outliers than MSE

## Contrastive Losses

### TripletMarginLoss

Triplet loss for metric learning.

```mojo
from odyssey.core.layers import TripletMarginLoss

var criterion = TripletMarginLoss(margin=1.0)
# anchor, positive, negative: (batch, embedding_dim)
var loss = criterion.forward(anchor, positive, negative)
```

**Formula:** `max(0, d(a, p) - d(a, n) + margin)`

### CosineEmbeddingLoss

Cosine similarity loss.

```mojo
from odyssey.core.layers import CosineEmbeddingLoss

var criterion = CosineEmbeddingLoss(margin=0.0)
# x1, x2: (batch, embedding_dim)
# labels: (batch,) - 1 for similar, -1 for dissimilar
var loss = criterion.forward(x1, x2, labels)
```

## Reduction Modes

All losses support three reduction modes:

| Mode | Description | Output Shape |
| --- | --- | --- |
| `"mean"` | Average over all elements | `()` scalar |
| `"sum"` | Sum over all elements | `()` scalar |
| `"none"` | No reduction | Same as input |

**Example:**

```mojo
# Per-sample losses (useful for weighted averaging)
var criterion = MSELoss(reduction="none")
var per_sample_loss = criterion.forward(pred, target)  # Shape: (batch,)

# Custom weighting
var weights = randn[DType.float32](batch_size)
var weighted_loss = (per_sample_loss * weights).mean()
```

## Class Weights

Handle imbalanced datasets with class weights:

```mojo
# Higher weight for rare classes
var weights = full(List[Int](10), 1.0, DType.float32)
weights[List[Int](7)] = 5.0  # 5x weight for class 7

var criterion = CrossEntropyLoss(weight=weights)
```

## Label Smoothing

Regularize by softening hard targets:

```mojo
# Instead of [0, 0, 1, 0] -> [0.025, 0.025, 0.925, 0.025]
var criterion = CrossEntropyLoss(label_smoothing=0.1)
```

**Benefits:**

- Prevents overconfident predictions
- Improves generalization

## Custom Loss Functions

Create custom losses by combining existing operations:

```mojo
fn focal_loss(
    logits: AnyTensor,
    targets: AnyTensor,
    gamma: Float = 2.0,
    alpha: Float = 0.25,
) raises -> AnyTensor:
    """Focal loss for imbalanced classification."""
    var probs = softmax(logits, dim=-1)
    var ce = cross_entropy_loss(logits, targets, reduction="none")

    # Focus on hard examples
    var pt = gather(probs, dim=1, index=targets.unsqueeze(-1)).squeeze(-1)
    var focal_weight = alpha * (1 - pt) ** gamma

    return (focal_weight * ce).mean()
```

## Comparison Table

| Loss | Use Case | Input Type | Output |
| --- | --- | --- | --- |
| CrossEntropy | Multi-class | Logits + Class IDs | Scalar |
| NLLLoss | Multi-class | Log-probs + Class IDs | Scalar |
| BCELoss | Binary | Probabilities | Scalar |
| BCEWithLogits | Binary | Logits | Scalar |
| MSELoss | Regression | Continuous | Scalar |
| L1Loss | Regression | Continuous | Scalar |
| SmoothL1Loss | Regression | Continuous | Scalar |

## See Also

- [Layers](layers.md) - Neural network layers
- [Activations](activations.md) - Activation functions
- [Optimizers](../training/optimizers.md) - Parameter optimization
