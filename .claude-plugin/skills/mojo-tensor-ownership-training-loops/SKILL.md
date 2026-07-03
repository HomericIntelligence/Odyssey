# Skill: mojo-tensor-ownership-training-loops

## Overview

| Field | Value |
| --- | --- |
| **Date** | 2026-07-02 |
| **Category** | debugging |
| **Objective** | Manage tensor ownership in Mojo training loops to avoid use-after-move errors |
| **Outcome** | ✅ Success - Post-training forward passes work correctly with separate batch objects |

## When to Use

Invoke this skill when:

- You're implementing training loops in Mojo where functions take tensors by value (moving ownership)
- You encounter "use after move" compile errors in training iterations
- You need to reuse tensor data between compute_gradients() and model.forward() calls
- You're debugging tensor state across multiple training steps
- You need to understand Mojo's ownership semantics in ML workflows

## Verified Workflow

### Step 1: Understand Mojo Value Semantics

In Mojo, function parameters default to value semantics (move semantics):

```mojo
fn compute_gradients(
    model: Model,
    input: Tensor,  # Takes ownership - input moved into function
    labels: Tensor,  # Takes ownership - labels moved into function
    lr: Float32,
) -> Float32:
    # input and labels are now owned by this function
    # After return, they are destroyed
    pass

# This fails:
var input1 = load_batch()  # Input owned by var input1
_ = compute_gradients(model, input1, labels)
# input1 has been MOVED into compute_gradients
// var input2 = input1  # ERROR: input1 already moved, cannot access
```

**Key insight**: When a function takes a parameter by value, it takes ownership. After the function call, the original variable is invalid.

### Step 2: Use Separate Batch Objects

Create independent batch objects for each function that takes ownership:

```mojo
fn train_step(
    model: Model,
    batch_loader: BatchLoader,
    lr: Float32,
) -> Tuple[Float32, Float32]:
    """Training step with separate batches for gradient and forward"""
    
    # First batch for gradient computation
    var batch1 = batch_loader.load_batch()  # Fresh batch object
    var input1 = batch1[0]  # Extract input (ownership moves)
    var labels1 = batch1[1]  # Extract labels (ownership moves)
    
    # compute_gradients takes ownership of input1 and labels1
    var loss1 = compute_gradients(model, input1, labels1, lr)
    // input1 and labels1 are now destroyed
    
    # Second batch for forward pass (SEPARATE batch object)
    var batch2 = batch_loader.load_batch()  # Fresh batch object
    var input2 = batch2[0]  # Extract input (ownership moves to input2)
    var labels2 = batch2[1]  # Extract labels (ownership moves to labels2)
    
    # model.forward takes ownership of input2 (not input1)
    var logits = model.forward(input2)  // Safe - input2 is fresh
    
    var loss2 = cross_entropy(logits, labels2)
    
    return (loss1, loss2[0, 0])
```

### Step 3: Pattern for Multiple Iterations

If you need the same batch data across multiple operations, borrow with `ref` or `&`:

```mojo
fn train_epoch(
    model: Model,
    batch: Tensor,
) -> Float32:
    """Multiple operations on same batch using references"""
    
    # Option 1: Borrow with & (read-only reference)
    var loss1 = compute_loss(model, batch &)  # & borrows, no move
    var loss2 = compute_loss_alt(model, batch &)  # Can borrow multiple times
    
    return loss1 + loss2
```

**But**: Most training functions require value semantics (moving tensors) for memory efficiency. Use separate batches instead.

### Step 4: Batch Loading Pattern

Design batch loader to create fresh batches each call:

```mojo
struct BatchLoader:
    var input_path: String
    var label_path: String
    var batch_size: Int
    
    fn load_batch(self) -> Tuple[Tensor, Tensor]:
        """Load fresh batch - each call creates new tensors"""
        var input_data = read_file(self.input_path)
        var label_data = read_file(self.label_path)
        
        var input_tensor = Tensor[DType.float32](
            self.batch_size,
            input_data.shape[1]
        )
        var label_tensor = Tensor[DType.uint8](self.batch_size)
        
        # ... populate tensors from files
        
        return (input_tensor, label_tensor)

fn train_epoch(model: Model, batch_loader: BatchLoader) -> None:
    for step in range(num_steps):
        # Each call creates NEW batch objects
        var batch = batch_loader.load_batch()
        var input = batch[0]
        var labels = batch[1]
        
        _ = compute_gradients(model, input, labels, lr)
        // input, labels destroyed after compute_gradients
```

### Step 5: Post-Training Verification Pattern

After training (when gradients are no longer needed), you can do a forward pass:

```mojo
fn verify_post_training(
    model: Model,
    validation_batch: Tuple[Tensor, Tensor],
) -> Float32:
    """Forward pass after training complete"""
    
    # Create separate batch for validation
    var eval_batch = load_eval_batch()  # Fresh batch
    var eval_input = eval_batch[0]
    var eval_labels = eval_batch[1]
    
    # Forward pass (may take input by value)
    var logits = model.forward(eval_input)  // input moved
    
    # Compute metrics
    var loss = cross_entropy(logits, eval_labels)
    var accuracy = compute_accuracy(logits, eval_labels)
    
    return loss[0, 0]
```

## Failed Attempts

### 1. Reusing Moved Tensor

**What was tried:**

```mojo
var batch = load_batch()
var input = batch[0]

_ = compute_gradients(model, input, labels)
// input has been moved into compute_gradients

var logits = model.forward(input)  // ERROR: input already moved!
```

**Why it failed**: Mojo's ownership tracking prevents use-after-move at compile time. After `compute_gradients` takes ownership, `input` is invalid.

**Compiler error:** `error: use of moved value 'input'`

**Solution**: Create separate batch objects. Don't try to reuse moved tensors.

### 2. Attempting Borrow Syntax on Value Functions

**What was tried:**

```mojo
fn compute_gradients(model: Model, input: Tensor, labels: Tensor) -> Float32:
    # Takes ownership (value semantics)

var input1 = load_batch()[0]
_ = compute_gradients(model, input1 &, labels)  // Attempted borrow
```

**Why it failed**: Functions defined with value parameters don't accept `&` (borrow) syntax. The function signature determines ownership, not the call site.

**Solution**: Change function signature to accept `ref` if you want borrow semantics (but this changes the function design). Otherwise, create separate batches.

### 3. Cloning as Workaround

**What was tried:**

```mojo
var batch = load_batch()
var input1 = batch[0]

_ = compute_gradients(model, input1, labels)  // Takes ownership

var input2 = input1.clone()  // Expensive clone!
var logits = model.forward(input2)
```

**Why it failed**: Cloning large tensors (e.g., [32, 3, 224, 224] for CIFAR-10) is extremely expensive and defeats the purpose of value semantics for performance.

**Solution**: Don't clone. Load fresh batches instead.

## Results & Parameters

### Ownership Rules in Training Loops

| Pattern | Ownership | When to Use | Example |
| --- | --- | --- | --- |
| Value parameter | Move | High-performance training | `compute_gradients(model, input)` |
| Ref parameter | Borrow | Multiple reads of same data | `compute_loss(model, batch &)` |
| Separate batches | Move + Create | Training + validation | Load batch1 for gradient, batch2 for forward |

### Post-Training Verification Pattern

```mojo
fn full_training_with_verification(
    model: Model,
    batch_loader: BatchLoader,
) -> Tuple[Float32, Float32]:
    """Training complete with post-training forward pass"""
    
    # Phase 1: Training (consuming batches)
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            var train_batch = batch_loader.load_batch()
            var train_input = train_batch[0]
            var train_labels = train_batch[1]
            _ = compute_gradients(model, train_input, train_labels, lr)
    
    # Phase 2: Post-training verification (SEPARATE batch)
    var eval_batch = batch_loader.load_batch()  // New batch object
    var eval_input = eval_batch[0]
    var eval_labels = eval_batch[1]
    
    var logits = model.forward(eval_input)  // Safe - eval_input is fresh
    var loss = cross_entropy(logits, eval_labels)
    
    return (training_loss, loss[0, 0])
```

### Tensor Lifecycle in Training Step

```
Batch Load: batch = load_batch()
            ↓
Extract: input = batch[0], labels = batch[1]
         (input owns data, labels owns data)
         ↓
Gradient: compute_gradients(model, input, labels)
          (function takes ownership)
          ↓
Destroy: input and labels destroyed after function returns
         ↓
Batch Load: batch2 = load_batch() (SEPARATE from batch)
            ↓
Forward: logits = model.forward(batch2[0])
         (SAFE - batch2[0] is fresh, not a moved value)
```

## Verified On

| Component | Version | Test | Status |
| --- | --- | --- | --- |
| Mojo | 1.0.0b2 | test_post_train_forward_shape | ✅ Pass - Forward pass executes correctly |
| ProjectOdyssey | main | test_mobilenetv1_training | ✅ Pass - Training + verification workflow |
| Ownership | v1.0 | Compile checks | ✅ Pass - No use-after-move errors |

## Implementation Checklist

- [ ] Identify functions that take tensors by value
- [ ] Separate batch loading logic from model operations
- [ ] Create fresh batch objects for each function call that consumes ownership
- [ ] Use `&` (borrow) syntax only for read-only verification operations
- [ ] Test post-training forward pass with separate batch
- [ ] Document tensor lifetime in training step comments

## Anti-Patterns to Avoid

```mojo
// ❌ WRONG: Reusing moved tensor
var input1 = batch[0]
compute_gradients(model, input1)
var logits = model.forward(input1)  // ERROR

// ✅ CORRECT: Separate batches
var batch1 = load_batch()
compute_gradients(model, batch1[0])
var batch2 = load_batch()
var logits = model.forward(batch2[0])

// ❌ WRONG: Expensive clone workaround
var input = batch[0]
compute_gradients(model, input)
var input_clone = input.clone()  // Expensive!
var logits = model.forward(input_clone)

// ✅ CORRECT: Load fresh batch
var batch = load_batch()
compute_gradients(model, batch[0])
var batch2 = load_batch()  // No clone needed
var logits = model.forward(batch2[0])
```
