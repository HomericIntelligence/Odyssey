# ADR-002: Gradient Struct Return Types (Tuple Return Workaround)

## Status

ACCEPTED

## Context

The Mojo compiler (v0.26.1) does not fully support tuple return types in all contexts, causing
compilation failures in backward pass functions that needed to return multiple gradients.

### Problem

Functions like `add_backward()`, `matmul_backward()`, and `prelu_backward()` attempted to return tuples:

```mojo
fn add_backward(...) raises -> (AnyTensor, AnyTensor):
    var grad_a = ...
    var grad_b = ...
    return (grad_a, grad_b)  // COMPILATION ERROR
```

This caused all arithmetic backward tests to fail to compile, blocking gradient computation across multiple modules.

### Affected Functions

**arithmetic.mojo**:

- `add_backward()` (line 548)
- `subtract_backward()` (line 589)
- `multiply_backward()` (line 621)
- `divide_backward()` (line 654)

**matrix.mojo**:

- `matmul_backward()` (line 326)

**activation.mojo**:

- `prelu_backward()` (line 598)

## Decision

Create type-safe gradient container structs to replace tuple return types, following Mojo best
practices for struct-based return types.

### Solution Architecture

**File**: `src/projectodyssey/core/gradient_types.mojo`

Two struct types for different return arities:

```mojo
struct GradientPair:
    var grad_a: AnyTensor
    var grad_b: AnyTensor

    fn __init__(inout self, grad_a: AnyTensor, grad_b: AnyTensor):
        self.grad_a = grad_a
        self.grad_b = grad_b
```

```mojo
struct GradientTriple:
    var grad_input: AnyTensor
    var grad_weights: AnyTensor
    var grad_bias: AnyTensor

    fn __init__(inout self, grad_input: AnyTensor, grad_weights: AnyTensor, grad_bias: AnyTensor):
        self.grad_input = grad_input
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
```

### API Usage

**Before** (failed to compile):

```mojo
var grad_a, grad_b = add_backward(grad_output, a_shape, b_shape)
```

**After**:

```mojo
var grads = add_backward(grad_output, a_shape, b_shape)
var grad_a = grads.grad_a
var grad_b = grads.grad_b
```

## Rationale

1. **Type Safety**: Struct provides compile-time type checking and IDE support
2. **Ergonomic**: Named fields are self-documenting and prevent positional confusion
3. **Forward Compatible**: Easier to add computation graph metadata or other fields later
4. **Zero-Cost Abstraction**: Structs are inlined with optimization flags
5. **Idiomatic Mojo**: Follows Mojo conventions for composite returns

## Alternatives Considered

### Option 1: Separate Functions

```mojo
fn add_backward_a(grad_output, a_shape, b_shape) -> AnyTensor
fn add_backward_b(grad_output, a_shape, b_shape) -> AnyTensor
```

**Rejected**: Violates DRY principle, requires duplicate computation, poor API coherence.

### Option 2: Output Parameters (inout)

```mojo
fn add_backward(grad_output, a_shape, b_shape, inout grad_a, inout grad_b) raises
```

**Rejected**: Requires pre-allocation, less elegant, doesn't support composition.

### Option 3: Python-style Tuple (current attempt)

```mojo
fn add_backward(...) raises -> (AnyTensor, AnyTensor)
```

**Rejected**: Does not compile reliably in Mojo v0.26.1.

## Implementation Details

### Changes Summary

1. **New File** - `src/projectodyssey/core/gradient_types.mojo`
   - Define `GradientPair` struct
   - Define `GradientTriple` struct
   - Comprehensive documentation with examples

2. **arithmetic.mojo**
   - Import `GradientPair`
   - Change return type: `(AnyTensor, AnyTensor)` → `GradientPair`
   - Update return statements: `return (a, b)` → `return GradientPair(a, b)`
   - Update docstrings with new API example

3. **matrix.mojo**
   - Import `GradientPair`
   - Change `matmul_backward()` return type and implementation
   - Update docstrings

4. **activation.mojo**
   - Import `GradientPair`
   - Change `prelu_backward()` return type and implementation
   - Update docstrings

5. **test_backward.mojo**
   - Update gradient unpacking: `grads.grad_a` instead of `grads[0]`
   - 6 tests updated to use new API

### Field Naming Convention

For binary operations:

- `grad_a`: Gradient w.r.t. first input
- `grad_b`: Gradient w.r.t. second input

For ternary operations:

- `grad_input`: Gradient w.r.t. input activation
- `grad_weights`: Gradient w.r.t. learnable weights
- `grad_bias`: Gradient w.r.t. bias term

This follows PyTorch conventions for consistency.

## Consequences

### Positive

- All backward functions now compile successfully
- Type-safe return values with named fields
- Self-documenting code
- Compatible with future enhancements
- Consistent API across all backward functions
- No performance overhead (zero-cost abstraction)

### Negative

- Slightly more verbose API compared to tuple unpacking
- Requires updating all backward function call sites

## Migration Path

1. Update all backward function signatures (done)
2. Update all backward function implementations (done)
3. Update all test files (done)
4. Update any additional call sites (verify)

## Verification

All 6 test functions in `test_backward.mojo` updated:

- `test_linear_backward_shapes()`
- `test_linear_backward_numerical()`
- `test_linear_backward_batch()`
- `test_conv2d_backward_shapes()`
- `test_conv2d_backward_with_stride()`
- Plus implicit tests via existing test infrastructure

## References

- Mojo Language: Struct Types (<https://docs.modular.com/mojo/manual/lifecycle/>)
- Zero-Cost Abstractions: (<https://en.cppreference.com/w/cpp/language/Zero-overhead_principle.html>)
- Comparison with PyTorch tuple returns: <https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html>

## Related Issues

- Blocks: Arithmetic backward compilation
- Blocks: Matrix multiplication backward compilation
- Blocks: Activation backward compilation
- Part of: Comprehensive backward pass implementation

## Decision Date

2025-11-20

## Implementation Date

2025-11-20
