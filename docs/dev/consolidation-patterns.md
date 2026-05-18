# Code Consolidation Patterns in ML Odyssey

This document describes the four primary consolidation patterns used across ML Odyssey to reduce code
duplication, improve maintainability, and enforce consistency.

## Overview

Consolidation patterns are recurring architectural solutions that centralize similar code into reusable
components. ML Odyssey implements four main patterns:

1. **Gradient Result Pattern** - Type-safe containers for multiple gradient returns
2. **DType Dispatch Pattern** - Compile-time specialized operations with runtime dispatch
3. **Constants Pattern** - Centralized mathematical and numerical constants
4. **Utility Module Pattern** - Shared functionality for cross-cutting concerns

## 1. Gradient Result Pattern

### Purpose

Eliminates duplicate tuple-return structures for backward pass functions that compute gradients with
respect to multiple inputs. Instead of returning multiple gradients as separate values or tuples
(which have incomplete support in Mojo), use specialized container types.

### Types

**GradientPair** - For binary operations returning 2 gradients
**GradientTriple** - For ternary operations returning 3 gradients
**GradientQuad** - For quaternary operations returning 4 gradients

### Location

`src/projectodyssey/core/gradient_types.mojo`

### Structure Example

```mojo
struct GradientPair(Copyable, Movable):
    """Container for gradients from binary operations.

    Used for backward functions that compute gradients with respect to
    two inputs (e.g., add_backward, multiply_backward).

    Attributes:
        grad_a: Gradient with respect to first input.
        grad_b: Gradient with respect to second input.
    """

    var grad_a: AnyTensor
    var grad_b: AnyTensor

    fn __init__(out self, var grad_a: AnyTensor, var grad_b: AnyTensor):
        self.grad_a = grad_a^
        self.grad_b = grad_b^
```

### Usage Example: Binary Operation

```mojo
# Without pattern (loses type information):
fn add_backward(grad_output: AnyTensor, shape_a: List[Int], shape_b: List[Int]) -> Tuple[AnyTensor, AnyTensor]:
    # ... compute gradients ...
    return (grad_a, grad_b)  # Loses semantic meaning

# With GradientPair pattern (clear intent):
fn add_backward(grad_output: AnyTensor, shape_a: List[Int], shape_b: List[Int]) -> GradientPair:
    # ... compute gradients ...
    var result = GradientPair(grad_a, grad_b)
    return result

# Usage:
var grads = add_backward(grad_output, a.shape(), b.shape())
var grad_a = grads.grad_a
var grad_b = grads.grad_b
```

### Usage Example: Linear Layer Backward

```mojo
# Use GradientTriple for layer backward passes
fn linear_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    weights: AnyTensor
) -> GradientTriple:
    """Compute gradients for linear layer.

    Args:
        grad_output: Gradient with respect to layer output.
        x: Input activation tensor.
        weights: Weight matrix.

    Returns:
        GradientTriple containing:
        - grad_input: Gradient with respect to input
        - grad_weights: Gradient with respect to weights
        - grad_bias: Gradient with respect to bias
    """
    # Compute gradients
    var grad_input = compute_grad_input(grad_output, weights)
    var grad_weights = compute_grad_weights(grad_output, x)
    var grad_bias = compute_grad_bias(grad_output)

    return GradientTriple(grad_input, grad_weights, grad_bias)

# Usage:
var grads = linear_backward(grad_out, input, weights)
var grad_x = grads.grad_input
var grad_w = grads.grad_weights
var grad_b = grads.grad_bias
```

### When to Use

- **Use GradientPair** for operations with exactly 2 inputs (add, sub, mul, div, pow, etc.)
- **Use GradientTriple** for layer backward passes (linear, conv2d, etc.) with input, weights, bias
- **Use GradientQuad** for operations with 4 inputs (reserved for future complex backward passes)
- **Don't use** Tuple returns for gradients - use one of these types instead

### Benefits

- Type-safe: Clear semantic meaning of each gradient
- Refactoring-safe: Field access is more robust than tuple unpacking
- Documentation: Type name itself documents the operation
- Extensible: Easy to add new gradient container types

## 2. DType Dispatch Pattern

### Purpose

Eliminates 40+ lines of repetitive dtype branching code by using compile-time specialized operations
with runtime dispatch. Instead of writing separate code paths for each dtype, write a single operation
and use dispatcher functions to handle runtime dtype selection.

### Reduction in Code Size

- **Before**: 500+ lines of dtype-specific code
- **After**: 100 lines with dispatch helpers
- **Reduction**: 80% fewer lines of code

### Location

`src/projectodyssey/core/dtype_dispatch.mojo`

### Core Concepts

**elementwise_unary[dtype, op]** - Compile-time specialized unary operation
**elementwise_binary[dtype, op]** - Compile-time specialized binary operation
**elementwise_scalar[dtype, op]** - Compile-time specialized scalar operation

**dispatch_unary[op]** - Runtime dispatch to unary operation
**dispatch_binary[op]** - Runtime dispatch to binary operation
**dispatch_scalar[op]** - Runtime dispatch to scalar operation

**dispatch_float_unary[op]** - Runtime dispatch for float-only unary operation
**dispatch_float_binary[op]** - Runtime dispatch for float-only binary operation
**dispatch_float_scalar[op]** - Runtime dispatch for float-only scalar operation

### Supported Dtypes

- Float types: float16, float32, float64
- Integer types: int8, int16, int32, int64
- Unsigned types: uint8, uint16, uint32, uint64

### Usage Example: Unary Operation

```mojo
# Step 1: Define the operation as a generic function
fn relu_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """ReLU activation: max(0, x)"""
    return max(Scalar[T](0), x)

# Step 2: Dispatch to compile-time specialized version (works for any dtype)
fn relu(tensor: AnyTensor) raises -> AnyTensor:
    return dispatch_unary[relu_op](tensor)

# Usage:
var x_f32 = full(List[Int](3, 4), 0.5, DType.float32)
var x_i32 = full(List[Int](3, 4), 5, DType.int32)
var result_f32 = relu(x_f32)  # Works!
var result_i32 = relu(x_i32)  # Works!
```

### Usage Example: Binary Operation

```mojo
# Step 1: Define operation for two tensors
fn add_op[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    return a + b

# Step 2: Use dispatch_binary for runtime dtype checking
fn add(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    return dispatch_binary[add_op](a, b)

# Usage:
var a = full(List[Int](2, 3), 1.0, DType.float32)
var b = full(List[Int](2, 3), 2.0, DType.float32)
var result = add(a, b)
```

### Usage Example: Scalar Operation

```mojo
# Step 1: Define operation between tensor and scalar
fn mul_op[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
    return a * b

# Step 2: Use dispatch_scalar for tensor-scalar operations
fn multiply(tensor: AnyTensor, scalar: Float64) raises -> AnyTensor:
    return dispatch_scalar[mul_op](tensor, scalar)

# Usage:
var x = full(List[Int](5,), 2.0, DType.float32)
var result = multiply(x, 3.5)  # Scalar automatically converted to float32
```

### Usage Example: Float-Only Operation

```mojo
# For operations that require floating-point types (exp, log, etc.)
fn sigmoid_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Sigmoid activation: 1 / (1 + exp(-x))

    Note: Only works with float types
    """
    var one = Scalar[T](1.0)
    var exp_neg_x = exp(-x)
    return one / (one + exp_neg_x)

# Step 2: Use dispatch_float_unary for float-only operations
fn sigmoid(tensor: AnyTensor) raises -> AnyTensor:
    return dispatch_float_unary[sigmoid_op](tensor)

# Usage:
var x = full(List[Int](3, 4), 0.5, DType.float32)
var result = sigmoid(x)  # Works for float32

# This would raise an error at runtime:
var x_int = full(List[Int](3, 4), 5, DType.int32)
try:
    var result_int = sigmoid(x_int)  # Error: operation only supports float16/32/64
except:
    print("Cannot apply sigmoid to integer tensor")
```

### Error Handling

Dispatch functions provide descriptive error messages that include:

- Which function failed (dispatch_unary, dispatch_binary, etc.)
- Which dtype was unsupported
- Expected dtype family (all, float-only, etc.)

### When to Use

- **dispatch_unary** - For operations that work on any supported dtype
- **dispatch_binary** - For element-wise binary operations on matching dtypes
- **dispatch_scalar** - For operations between tensor and scalar
- **dispatch_float_unary** - For float-only operations (exp, log, sigmoid, etc.)
- **dispatch_float_binary** - For float-only binary operations
- **dispatch_float_scalar** - For float-only scalar operations

### Benefits

- Code reduction: 80% fewer lines of dtype-specific code
- Single source of truth: One implementation for all dtypes
- Compile-time optimization: Zero runtime overhead vs hand-written branches
- Easy maintenance: Add new operations without dtype repetition
- Type safety: Compile-time specialization catches type errors early

## 3. Constants Pattern

### Purpose

Centralizes mathematical and numerical constants used across the codebase in dedicated modules.
This prevents magic number proliferation and ensures consistency across all uses.

### Module Organization

#### Math Constants

**Location**: `src/projectodyssey/core/math_constants.mojo`

Contains mathematical constants used in activation functions, initializers, and elementwise operations.

```mojo
# Pi and related constants
comptime PI: Float64 = 3.14159265358979323846

# Square roots
comptime SQRT_2: Float64 = 1.4142135623730951
comptime SQRT_2_OVER_PI: Float64 = 0.7978845608028654  # sqrt(2/pi) for GELU
comptime INV_SQRT_2PI: Float64 = 0.3989422804014327  # 1/sqrt(2*pi) for normal dist

# GELU activation constants
comptime GELU_COEFF: Float64 = 0.044715

# Logarithms
comptime LN2: Float64 = 0.6931471805599453
comptime LN10: Float64 = 2.302585092994046
```

#### Numerical Constants

**Location**: `src/projectodyssey/core/numerical_constants.mojo`

Contains epsilon and threshold values for numerical stability.

```mojo
# Division safety - prevents division by zero
comptime EPSILON_DIV: Float64 = 1e-10

# Loss function stability - for log operations in BCE, cross-entropy, etc.
comptime EPSILON_LOSS: Float64 = 1e-7

# Normalization stability - for BatchNorm, LayerNorm, GroupNorm, InstanceNorm
comptime EPSILON_NORM: Float64 = 1e-5

# Gradient safety thresholds
comptime GRADIENT_MAX_NORM: Float64 = 1000.0  # Threshold for gradient explosion
comptime GRADIENT_MIN_NORM: Float64 = 1e-7   # Threshold for gradient vanishing
```

### Usage Example: GELU Activation

```mojo
from projectodyssey.core.math_constants import GELU_COEFF, SQRT_2_OVER_PI

fn gelu_approximate[T: DType](x: Scalar[T]) -> Scalar[T]:
    """GELU approximation using precomputed constants.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))).
   """
    var one = Scalar[T](1.0)
    var half = Scalar[T](0.5)
    var coeff = Scalar[T](GELU_COEFF)
    var sqrt_term = Scalar[T](SQRT_2_OVER_PI)

    var x_cubed = x * x * x
    var inner = sqrt_term * (x + coeff * x_cubed)

    return half * x * (one + tanh(inner))
```

### Usage Example: Numerical Stability

```mojo
from projectodyssey.core.numerical_constants import EPSILON_LOSS, EPSILON_NORM

fn cross_entropy_loss(logits: AnyTensor, labels: AnyTensor) raises -> Float64:
    """Cross-entropy loss with numerical stability.

    Uses EPSILON_LOSS to prevent log(0).
    """
    # Softmax with stability...
    var log_probs = softmax(logits)

    # Clip to prevent log(0)
    var clipped = clip(log_probs, EPSILON_LOSS, 1.0 - EPSILON_LOSS)

    # Compute loss...
    return compute_loss(clipped, labels)

fn batch_norm_forward(
    x: AnyTensor,
    weight: AnyTensor,
    bias: AnyTensor,
    running_mean: AnyTensor,
    running_var: AnyTensor
) raises -> AnyTensor:
    """Batch normalization with numerical stability.

    Uses EPSILON_NORM to prevent division by zero.
    """
    # Normalize with stability
    var x_norm = (x - running_mean) / sqrt(running_var + EPSILON_NORM)

    # Scale and shift
    return weight * x_norm + bias
```

### When to Use

- **Use constants pattern** - For any value used in more than one function
- **Use math_constants.mojo** - For mathematical constants (π, √2, etc.)
- **Use numerical_constants.mojo** - For epsilon and threshold values
- **Don't use magic numbers** - Hard-coded constants should become aliased constants
- **Don't create new files** - Add new constants to existing files in the module

### Benefits

- Consistency: Same value used everywhere
- Maintainability: Single place to update constants
- Documentation: Constant names self-document their purpose
- Correctness: Prevents typos in frequently-used values

## 4. Utility Module Pattern

### Purpose

Groups related cross-cutting utilities into dedicated modules. This pattern centralizes functionality
that is used across multiple layers or modules without tight coupling.

### Location

`src/projectodyssey/autograd/grad_utils.mojo` (example)

### Pattern Structure

Utility modules contain sets of related functions that:

1. Share a common purpose (e.g., gradient clipping)
2. Are used across multiple components
3. Don't belong to a single layer's module
4. Have clear, specialized responsibilities

### Example: Gradient Clipping Utilities

```mojo
fn clip_grad_value_(mut grad: AnyTensor, max_value: Float64) raises:
    """Clip each gradient element to [-max_value, max_value].

    Args:
        grad: The gradient tensor to clip (modified in-place).
        max_value: Maximum absolute value allowed.

    Examples:
        var grad = ones(List[Int](3, 4), DType.float32)
        clip_grad_value_(grad, max_value=1.0).
   """
    if max_value < 0.0:
        raise Error("max_value must be non-negative")

    for i in range(grad.numel()):
        var val = grad._get_float64(i)
        if val > max_value:
            grad._set_float64(i, max_value)
        elif val < -max_value:
            grad._set_float64(i, -max_value)


fn clip_grad_norm_(mut grad: AnyTensor, max_norm: Float64) raises -> Float64:
    """Clip gradient if its L2 norm exceeds max_norm.

    Args:
        grad: The gradient tensor to clip (modified in-place).
        max_norm: Maximum allowed L2 norm.

    Returns:
        The original L2 norm of the gradient (before clipping).

    Examples:
        var grad = full(List[Int](100,), 1.0, DType.float32)
        var norm = clip_grad_norm_(grad, max_norm=1.0)
        # norm is approximately sqrt(100) = 10
        # grad is scaled by 0.1
    """
    # Compute L2 norm
    var sum_sq = Float64(0)
    for i in range(grad.numel()):
        var val = grad._get_float64(i)
        sum_sq += val * val

    var norm = sqrt(sum_sq)

    # Scale if needed
    if norm > max_norm:
        var scale = max_norm / norm
        for i in range(grad.numel()):
            var val = grad._get_float64(i)
            grad._set_float64(i, val * scale)

    return norm
```

### Usage Example

```mojo
from projectodyssey.autograd import clip_grad_value_, clip_grad_norm_, clip_grad_global_norm_

fn training_step(
    model: MyModel,
    x: AnyTensor,
    y: AnyTensor,
    optimizer: SGD
) raises:
    """Single training step with gradient clipping."""
    # Forward pass
    var logits = model(x)
    var loss = cross_entropy(logits, y)

    # Backward pass
    var grads = loss.backward()

    # Clip gradients to prevent explosion
    for ref grad in grads:
        clip_grad_value_(grad, max_value=1.0)
        clip_grad_norm_(grad, max_norm=5.0)

    # Update parameters
    optimizer.step(grads)
```

### Common Utility Modules

| Module | Purpose | Examples |
| --- | --- | --- |
| `grad_utils.mojo` | Gradient clipping and statistics | clip_grad_value_, clip_grad_norm_ |
| `dtype_utils.mojo` | Dtype-related utilities | dtype conversion helpers |
| `math_utils.mojo` | Mathematical utilities | Various math operations |
| `tensor_utils.mojo` | Tensor operations | Shape validation, broadcasting |

### When to Use

- **Create utility module** - For functions used across 3+ modules
- **Use in-module helper** - For functions used in only 1-2 modules
- **Group by concern** - Utilities that share purpose go in same module
- **Clear naming** - End with `_` for in-place operations (e.g., `clip_grad_value_`)
- **Don't mix concerns** - Don't combine math utilities with gradient utilities

### Benefits

- Centralized functionality: Single source of truth for shared operations
- No circular dependencies: Utilities don't depend on layer modules
- Reusability: Easy to reuse across different components
- Testability: Utilities can be tested independently
- Documentation: Clear purpose from module organization

## Integration Example: Complete Backward Pass

This example shows how all four consolidation patterns work together in a complete backward pass:

```mojo
from projectodyssey.core.gradient_types import GradientTriple
from projectodyssey.core.dtype_dispatch import dispatch_binary, dispatch_unary
from projectodyssey.core.numerical_constants import EPSILON_LOSS
from projectodyssey.autograd.grad_utils import clip_grad_norm_

fn linear_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    weights: AnyTensor
) -> GradientTriple:
    """Linear layer backward pass demonstrating all consolidation patterns.

    Uses:
    1. GradientTriple - Returns structured gradient container
    2. dtype_dispatch - Handles arbitrary dtypes
    3. Constants - Uses EPSILON_LOSS for stability
    4. grad_utils - Clips gradients by norm
    """
    # Compute gradients using dispatch pattern (avoids dtype branching)
    fn matmul_op[T: DType](a: Scalar[T], b: Scalar[T]) -> Scalar[T]:
        return a * b  # Simplified for example

    # grad_input = grad_output @ weights.T
    var grad_input = dispatch_binary[matmul_op](grad_output, weights)

    # grad_weights = x.T @ grad_output
    var grad_weights = dispatch_binary[matmul_op](x, grad_output)

    # grad_bias = sum(grad_output)
    var grad_bias = reduce_sum(grad_output)

    # Apply gradient clipping with stability constant
    clip_grad_norm_(grad_input, max_norm=5.0)
    clip_grad_norm_(grad_weights, max_norm=5.0)
    clip_grad_norm_(grad_bias, max_norm=5.0)

    # Return using GradientTriple pattern
    return GradientTriple(grad_input, grad_weights, grad_bias)
```

## Summary Table

| Pattern | Purpose | Location | Benefit |
| --- | --- | --- | --- |
| Gradient Result | Type-safe gradient containers | `src/projectodyssey/core/gradient_types.mojo` | Clear semantics, refactoring-safe |
| DType Dispatch | Compile-time specialization | `src/projectodyssey/core/dtype_dispatch.mojo` | 80% code reduction |
| Constants | Centralized values | `src/projectodyssey/core/math_constants.mojo`, `src/projectodyssey/core/numerical_constants.mojo` | Consistency, maintainability |
| Utility Module | Shared functionality | `src/projectodyssey/autograd/grad_utils.mojo`, etc. | No circular deps, reusability |

## Best Practices

1. **Always use GradientPair/Triple/Quad** - Never return naked tuples for gradients
2. **Always use dispatch helpers** - Never write dtype branching code manually
3. **Always centralize constants** - No magic numbers in implementation code
4. **Always create utility modules** - For functions used across 3+ modules
5. **Document consolidation patterns** - In module docstrings, not inline comments
6. **Keep patterns pure** - Don't mix unrelated utilities in one module
7. **Reuse existing patterns** - Check for existing consolidated code before creating new

## Related Documentation

- [Backward Pass Catalog](backward-pass-catalog.md)
- [Mojo Test Failure Patterns](mojo-test-failure-patterns.md)
- [Skills Architecture](skills-architecture.md)
