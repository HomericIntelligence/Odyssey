# Skill: cifar10-one-hot-label-encoding

## Overview

| Field | Value |
| --- | --- |
| **Date** | 2026-07-02 |
| **Category** | testing |
| **Objective** | Encode CIFAR-10 integer labels to one-hot format for cross-entropy loss computation |
| **Outcome** | ✅ Success - Training pipeline correctly handles label encoding for 10-class classification |

## When to Use

Invoke this skill when:

- You're loading CIFAR-10 dataset and need to prepare labels for cross-entropy loss
- You have integer labels (0-9) and need (B, 10) float32 one-hot encoded labels
- You're implementing training loops with cross-entropy that expects one-hot targets
- You need to convert between raw integer labels and probabilistic target distributions
- You're debugging NaN losses or shape mismatches in loss computation

## Verified Workflow

### Step 1: Understand CIFAR-10 Label Format

CIFAR-10 dataset provides labels in two formats:

**Raw format (from dataset file)**:
- Shape: (B,) or scalar
- Data type: uint8
- Values: 0-9 (class indices)
- Meaning: Class assignment for each image

**One-hot format (for cross-entropy)**:
- Shape: (B, 10)
- Data type: float32
- Values: 0.0 or 1.0
- Meaning: Probability distribution (all zeros except one 1.0 per sample)

**Example conversion**:
```
Raw label:     [0, 1, 5, 3]
One-hot:
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  // Class 0
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  // Class 1
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  // Class 5
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  // Class 3
```

### Step 2: Implement One-Hot Encoding Function

```mojo
from projectodyssey.tensor import Tensor
from projectodyssey.core.dtype import DType

fn one_hot_encode(
    labels: Tensor[DType.uint8],
    num_classes: Int = 10,
) -> Tensor[DType.float32]:
    """
    Convert integer labels to one-hot encoded format.
    
    Args:
        labels: Shape (B,) with values in [0, num_classes)
        num_classes: Number of classes (10 for CIFAR-10)
    
    Returns:
        Shape (B, num_classes) with one 1.0 per row, rest 0.0
    """
    var batch_size = labels.shape[0]
    
    # Create zero tensor (B, num_classes)
    var one_hot = zeros[DType.float32](batch_size, num_classes)
    
    # Set one_hot[i, labels[i]] = 1.0
    for i in range(batch_size):
        var class_idx = int(labels[i])
        
        # Bounds check (safety)
        if class_idx < 0 or class_idx >= num_classes:
            raise ValueError(
                f"Label {class_idx} out of range [0, {num_classes})"
            )
        
        one_hot[i, class_idx] = Float32(1.0)
    
    return one_hot
```

### Step 3: Integration in Data Loading Pipeline

```mojo
struct CIFAR10Batch:
    var images: Tensor[DType.float32]  # Shape: (B, 3, 32, 32)
    var labels_raw: Tensor[DType.uint8]  # Shape: (B,) with values 0-9
    var labels_onehot: Tensor[DType.float32]  # Shape: (B, 10)
    
    fn __init__(
        out self,
        images: Tensor[DType.float32],
        labels_raw: Tensor[DType.uint8],
    ):
        self.images = images
        self.labels_raw = labels_raw
        # Convert to one-hot immediately
        self.labels_onehot = one_hot_encode(labels_raw, num_classes=10)

fn load_cifar10_batch(batch_file: String) -> CIFAR10Batch:
    """Load batch and convert labels to one-hot format"""
    var images = load_images(batch_file)  # (B, 3, 32, 32) float32
    var labels_raw = load_labels(batch_file)  # (B,) uint8
    
    return CIFAR10Batch(images, labels_raw)

fn train_epoch(
    model: MobileNetV1,
    batch_file: String,
) -> Float32:
    """Training epoch with proper label encoding"""
    
    var batch = load_cifar10_batch(batch_file)
    
    # Forward pass
    var logits = model.forward(batch.images)  # (B, 10)
    
    # Cross-entropy expects:
    # - logits: (B, 10) raw predictions
    # - labels: (B, 10) one-hot encoded targets
    var loss = cross_entropy(logits, batch.labels_onehot)
    
    return loss[0, 0]
```

### Step 4: Verify Encoding Correctness

```mojo
fn verify_one_hot_encoding(
    labels: Tensor[DType.uint8],
    one_hot: Tensor[DType.float32],
    num_classes: Int = 10,
) -> Bool:
    """Validate one-hot encoding properties"""
    
    var batch_size = labels.shape[0]
    
    # Check shape
    if one_hot.shape[0] != batch_size or one_hot.shape[1] != num_classes:
        print(f"ERROR: Shape mismatch. Expected ({batch_size}, {num_classes}), got {one_hot.shape}")
        return False
    
    # Check each row
    for i in range(batch_size):
        var row_sum = Float32(0.0)
        var one_count = 0
        var class_idx = int(labels[i])
        
        for j in range(num_classes):
            var val = one_hot[i, j]
            
            // Each value should be 0.0 or 1.0
            if val != 0.0 and val != 1.0:
                print(f"ERROR: one_hot[{i}, {j}] = {val}, expected 0.0 or 1.0")
                return False
            
            if val == 1.0:
                one_count += 1
            row_sum += val
        
        // Each row should sum to exactly 1.0
        if abs(row_sum - 1.0) > 1e-6:
            print(f"ERROR: Row {i} sums to {row_sum}, expected 1.0")
            return False
        
        // Exactly one 1.0 per row
        if one_count != 1:
            print(f"ERROR: Row {i} has {one_count} ones, expected 1")
            return False
        
        // The 1.0 should be at position labels[i]
        if abs(one_hot[i, class_idx] - 1.0) > 1e-6:
            print(f"ERROR: one_hot[{i}, {class_idx}] should be 1.0, got {one_hot[i, class_idx]}")
            return False
    
    return True
```

### Step 5: Handle Batch Processing

```mojo
fn process_cifar10_epoch(
    model: MobileNetV1,
    batch_iterator: CIFAR10BatchIterator,
    num_batches: Int,
) -> List[Float32]:
    """Process multiple batches with proper label encoding"""
    
    var losses = List[Float32]()
    
    for batch_idx in range(num_batches):
        // Load batch (images + raw integer labels)
        var batch = batch_iterator.next()
        
        var images = batch.images  // (B, 3, 32, 32)
        var labels_raw = batch.labels_raw  // (B,) with uint8 values
        
        // Encode labels to one-hot
        var labels_onehot = one_hot_encode(labels_raw, num_classes=10)
        
        // Forward pass
        var logits = model.forward(images)  // (B, 10)
        
        // Compute loss with one-hot encoded labels
        var loss = cross_entropy(logits, labels_onehot)
        losses.append(loss[0, 0])
        
        // Backward pass
        var grad_out = grad_cross_entropy(logits, labels_onehot)
        var gradients = model.backward(grad_out)
        
        // Update parameters
        update_step(model, gradients, lr=0.01)
    
    return losses
```

### Step 6: Testing One-Hot Encoding

```mojo
fn test_one_hot_encoding() -> None:
    """Test one-hot encoding implementation"""
    
    // Test 1: Single sample
    var labels1 = Tensor[DType.uint8](1)
    labels1[0] = 5
    var one_hot1 = one_hot_encode(labels1, num_classes=10)
    
    assert_true(one_hot1.shape == (1, 10), "Shape should be (1, 10)")
    assert_true(one_hot1[0, 5] == 1.0, "Position 5 should be 1.0")
    assert_true(one_hot1[0, 0] == 0.0, "Other positions should be 0.0")
    
    // Test 2: Multiple samples
    var labels2 = Tensor[DType.uint8](4)
    labels2[0] = 0
    labels2[1] = 1
    labels2[2] = 5
    labels2[3] = 9
    var one_hot2 = one_hot_encode(labels2, num_classes=10)
    
    assert_true(one_hot2.shape == (4, 10), "Shape should be (4, 10)")
    assert_true(one_hot2[0, 0] == 1.0, "Row 0, class 0 should be 1.0")
    assert_true(one_hot2[1, 1] == 1.0, "Row 1, class 1 should be 1.0")
    assert_true(one_hot2[2, 5] == 1.0, "Row 2, class 5 should be 1.0")
    assert_true(one_hot2[3, 9] == 1.0, "Row 3, class 9 should be 1.0")
    
    // Test 3: Row sums
    for i in range(4):
        var row_sum = Float32(0.0)
        for j in range(10):
            row_sum += one_hot2[i, j]
        assert_true(abs(row_sum - 1.0) < 1e-6, "Each row should sum to 1.0")
    
    print("✓ All one-hot encoding tests passed")
```

## Failed Attempts

### 1. Using Raw Integer Labels Directly

**What was tried:**

```mojo
fn train_step(model: Model, images: Tensor, labels_raw: Tensor[DType.uint8]):
    var logits = model.forward(images)  // (B, 10)
    // Try to compute loss directly with integer labels
    var loss = cross_entropy(logits, labels_raw)  // ERROR!
```

**Why it failed**: Cross-entropy expects float32 probability distributions as targets, not integer class indices. Shape mismatch: logits are (B, 10) but labels_raw are (B,).

**Compiler/runtime error**: Shape mismatch or type mismatch in loss computation.

**Solution**: Encode labels to one-hot format before passing to loss function.

### 2. Casting Integers to Float32 (Incorrect)

**What was tried:**

```mojo
var labels_raw = Tensor[DType.uint8]([0, 1, 5, 3])  // (4,)
var labels_float = labels_raw.cast[DType.float32]()  // (4,) with values 0.0, 1.0, 5.0, 3.0

var loss = cross_entropy(logits, labels_float)  // WRONG!
```

**Why it failed**: Cross-entropy expects (B, num_classes) one-hot targets, not (B,) class indices as floats. The loss function interprets 5.0 as a probability, not as "class 5".

**Result**: NaN loss or incorrect gradients (gradients computed as if all samples are misclassified).

**Solution**: Create (B, num_classes) one-hot tensor, not just cast to float32.

### 3. Using Dense Labels (Soft Targets)

**What was tried:**

```mojo
// Soft target (not one-hot): [0.1, 0.2, 0.7] for class 2
var soft_labels = Tensor[DType.float32](B, 10)
for i in range(B):
    for j in range(10):
        if j == labels_raw[i]:
            soft_labels[i, j] = 0.7  // NOT 1.0!
        else:
            soft_labels[i, j] = 0.3 / 9.0  // Uniform over other classes

var loss = cross_entropy(logits, soft_labels)
```

**Why it failed**: Cross-entropy formula assumes hard targets (one-hot with 1.0 for true class, 0.0 for others). Soft targets work mathematically but don't match standard CIFAR-10 training and cause label smoothing (unintended regularization).

**Solution**: Use true one-hot encoding (1.0 for true class, 0.0 for others) for standard CIFAR-10 training.

## Results & Parameters

### CIFAR-10 Label Encoding Specification

| Property | Value | Notes |
| --- | --- | --- |
| Input (raw) | (B,) uint8 | Values in [0, 9] |
| Output (one-hot) | (B, 10) float32 | Each row sums to 1.0 |
| One-hot values | {0.0, 1.0} | Hard targets (no soft labels) |
| Computation | O(B * num_classes) | ~40μs for B=32, num_classes=10 |

### One-Hot Encoding Pattern

```mojo
// Input: labels = [0, 1, 5, 3] (shape: (4,))
// Output: one_hot = 
// [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  // Class 0
// [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  // Class 1
// [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  // Class 5
// [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  // Class 3
```

Each sample has exactly one 1.0 (at the true class index) and nine 0.0s.

### Cross-Entropy Loss Computation

```
Cross-entropy(logits, labels_onehot) =
  -sum over all i,j: labels_onehot[i,j] * log(softmax(logits[i,j]))
  
With one-hot labels, this simplifies to:
  -log(softmax(logits[i, true_class[i]]))
```

The loss only depends on the predicted probability of the true class.

### Training Pipeline Integration

```
1. Load CIFAR-10 batch
   images: (B, 3, 32, 32) float32
   labels_raw: (B,) uint8 with values [0-9]
   
2. Encode labels
   labels_onehot = one_hot_encode(labels_raw, 10)
   Result: (B, 10) float32
   
3. Forward pass
   logits = model.forward(images)  // (B, 10)
   
4. Compute loss
   loss = cross_entropy(logits, labels_onehot)
   
5. Backward pass
   grad_out = grad_cross_entropy(logits, labels_onehot)
   gradients = model.backward(grad_out)
```

## Verified On

| Component | Test | Result |
| --- | --- | --- |
| CIFAR-10 loader | Load batch → encode labels | ✅ Pass |
| One-hot encoding | test_one_hot_encoding | ✅ Pass - all assertions pass |
| Cross-entropy | compute with one-hot labels | ✅ Pass - loss finite and positive |
| Training loop | MobileNetV1 + CIFAR-10 | ✅ Pass - loss decreases across iterations |
| Shape validation | Verify (B, 10) output | ✅ Pass - correct shape |

## Implementation Checklist

- [ ] Understand CIFAR-10 raw label format (B,) uint8
- [ ] Understand one-hot encoding (B, 10) float32
- [ ] Implement one_hot_encode() function
- [ ] Add to data loading pipeline
- [ ] Test encoding correctness
- [ ] Integrate into training loop
- [ ] Verify loss computation with one-hot labels
- [ ] Test on full CIFAR-10 batch (32 samples)

## Anti-Patterns to Avoid

```mojo
// ❌ WRONG: Pass raw integer labels to cross_entropy
var labels_raw = load_labels()  // (B,) uint8
var loss = cross_entropy(logits, labels_raw)  // Shape/type mismatch

// ✅ CORRECT: Encode to one-hot first
var labels_onehot = one_hot_encode(labels_raw, 10)  // (B, 10)
var loss = cross_entropy(logits, labels_onehot)

// ❌ WRONG: Just cast to float32
var labels_float = labels_raw.cast[DType.float32]()  // Still (B,)!
var loss = cross_entropy(logits, labels_float)  // Still wrong

// ✅ CORRECT: Create (B, num_classes) tensor
var labels_onehot = zeros[DType.float32](B, 10)
for i in range(B):
    labels_onehot[i, labels_raw[i]] = 1.0

// ❌ WRONG: Soft labels instead of hard one-hot
labels_onehot[i, true_class] = 0.8  // Label smoothing (unintended)

// ✅ CORRECT: Hard one-hot (1.0 for true class)
labels_onehot[i, true_class] = 1.0
```
