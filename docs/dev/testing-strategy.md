# Testing Strategy: Two-Tier Model Testing

This document describes ML Odyssey's comprehensive testing strategy for neural network models,
designed to balance fast PR validation with thorough weekly integration testing.

## Overview

ML Odyssey uses a two-tier testing strategy:

- **Tier 1**: Fast layerwise unit tests run on every PR (~12 minutes)
- **Tier 2**: Comprehensive E2E integration tests run weekly (~140 minutes parallelized)

This approach provides:

- Fast feedback on PRs (< 12 minutes)
- Comprehensive validation weekly
- Full gradient checking for all layers
- Real dataset integration testing
- 100% layer coverage across 7 models

## Architecture

### Two-Tier Design

```text
Pull Request → Layerwise Tests (12 min) → Merge
                     ↓
                Main Branch
                     ↓
Weekly Schedule → E2E Tests (140 min) → Report
```text
### Test Coverage Matrix

| Test Type | When | Purpose | Dataset | Runtime |
|-----------|------|---------|---------|---------|
| **Layerwise** | Every PR | Validate layer | Values | ~12 min |
| **E2E** | Weekly | Full integration | Data | ~140 min |

## Tier 1: Layerwise Unit Tests

### Purpose

Test each layer independently to catch bugs early without requiring full datasets.

### Test Structure

Each model has a `test_<model>_layers.mojo` file testing:

1. **Forward pass** - Output shape, dtype, no NaN/Inf
2. **Backward pass** - Gradient shape, numerical validation
3. **All layer types** - Conv, Linear, BatchNorm, ReLU, MaxPool, etc.

### Example: LeNet-5 Layerwise Tests

```mojo
fn test_conv1_forward() raises:
    """Test Conv1 layer (1→6, 5×5 kernel) forward pass."""
    # Create weights with proper shape
    var weights = create_conv1_weights()
    var bias = create_conv1_bias()

    # Test with special values
    for dtype in get_test_dtypes():
        LayerTester.test_conv_layer(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            input_h=28,
            input_w=28,
            weights=weights,
            bias=bias,
            dtype=dtype,
        )

fn test_conv1_backward() raises:
    """Test Conv1 layer backward pass with gradient checking."""
    var weights = create_conv1_weights()
    var bias = create_conv1_bias()

    LayerTester.test_conv_layer_backward(
        in_channels=1,
        out_channels=6,
        kernel_size=5,
        input_h=8,  # Small size for speed
        input_w=8,
        weights=weights,
        bias=bias,
        dtype=DType.float32,
    )
```text
### Key Features

- **Small tensor sizes**: 8×8 for backward (not 32×32) to prevent timeout
- **Special values**: 0.0, 0.5, 1.0, 1.5, -1.0, -0.5 for deterministic behavior
- **Multi-dtype**: Tests float32, float16, bfloat16 where applicable
- **Gradient checking**: Validates analytical vs numerical gradients

## Tier 2: End-to-End Integration Tests

### Purpose

Validate complete model behavior with real datasets and training scenarios.

### Test Structure

Each model has a `test_<model>_e2e.mojo` file testing:

1. **Forward with dataset** - Process real data batches
2. **Training convergence** - 5 epochs, ≥20% loss decrease
3. **Gradient flow** - All layers receive gradients
4. **Weight persistence** - Save/load functionality

### Example: LeNet-5 E2E Tests

```mojo
fn test_training_convergence() raises:
    """Verify loss decreases over epochs."""
    # Load EMNIST dataset (if available)
    var dataset = load_emnist_dataset()  # Skip if not available

    # Train for 5 epochs
    var initial_loss = train_epoch(model, dataset, epoch=0)
    # ... train epochs 1-4
    var final_loss = train_epoch(model, dataset, epoch=4)

    # Check convergence
    var loss_reduction = (initial_loss - final_loss) / initial_loss
    assert_true(loss_reduction >= 0.20, "Loss should decrease by ≥20%")
```text
### Datasets

- **LeNet-5**: EMNIST (47-class handwritten characters)
- **Other models**: CIFAR-10 (10-class object recognition)

Downloaded by CI in `prepare-datasets` job (weekly only).

## Special Values

### Why Special Values?

Values 0.0, 0.5, 1.0, 1.5, -1.0, -0.5 are **exactly representable** in IEEE 754:

- No rounding errors across dtypes
- Predictable behavior in all operations
- Identical results in FP4, FP8, FP16, FP32, BFloat16, Int8

### Value Categories

| Value | Binary | Purpose |
|-------|--------|---------|
| 0.0 | 0x00000000 | Additive identity |
| 0.5 | 0x3F000000 | Simple fraction |
| 1.0 | 0x3F800000 | Multiplicative id |
| 1.5 | 0x3FC00000 | 1 + 2^-1 |
| -1.0 | 0xBF800000 | ReLU gradient |
| -0.5 | 0xBF000000 | Negative frac |

### Usage Patterns

**Forward Pass**: Use positive special values

```mojo
var input = create_special_value_tensor([1, 3, 8, 8], dtype, 1.0)
var output = conv2d(input, weights, bias)
```text
**ReLU Gradient Testing**: Use negative values

```mojo
var input = create_alternating_pattern_tensor([2, 3, 4, 4], dtype)
# Pattern: -1.0, -0.5, 0.0, 0.5, 1.0, 1.5 (repeating)
var output = relu(input)
# Verify gradient=0 for negative inputs
```text
**Gradient Checking**: Use seeded random

```mojo
var input = create_seeded_random_tensor([1, 3, 8, 8], dtype, seed=42)
# Reproducible random values for numerical gradient validation
```text
## Gradient Checking

### How It Works

Compares analytical gradients (backpropagation) to numerical gradients (finite differences):

```text
Numerical Gradient ≈ [f(x + ε) - f(x - ε)] / (2ε)
```text
### Parameters

- **Epsilon**: 1e-5 for float32, 1e-4 for float16
- **Tolerance**: 1e-2 for float32, 1e-1 for lower precision
- **Method**: Central differences (more accurate than forward differences)

### Example

```mojo
fn test_linear_backward() raises:
    var weights = create_fc_weights()
    var bias = create_fc_bias()
    var input = create_seeded_random_tensor([1, 32], DType.float32, seed=42)

    # Define forward pass
    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return linear(t, weights, bias)^

    # Define backward pass
    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return linear_backward(grad, inp, weights)^

    # Check gradients match
    var passed = check_gradients(forward, backward, input, epsilon=1e-5, tolerance=1e-2)
    assert_true(passed, "Linear backward pass gradient mismatch")
```text

## Known Test Gotchas

This section documents edge cases that have caused false failures in the past.
Always check here before assuming a gradient mismatch is a real bug.

### BatchNorm2d: `ones_like(grad_output)` with `beta=0` Produces Zero Gradients

**Symptom**: Analytical gradient ≈ 0, numerical gradient ≈ 0.009 — apparent ~1000x mismatch.

**Root cause**: This is **not a bug**. The zero analytical gradient is mathematically correct,
and the numerical gradient is also essentially zero — the apparent mismatch is finite-difference
noise against an exactly-zero baseline.

#### Mathematical Proof

Batch normalization normalizes each channel over the batch and spatial dimensions:

```text
x_norm[b,c,h,w] = (x[b,c,h,w] - mean_c) / std_c
output[b,c,h,w] = gamma_c * x_norm[b,c,h,w] + beta_c
```

The key identity is the **zero-mean property**: normalized values per channel always sum to zero:

```text
sum over (b,h,w) of x_norm[b,c,h,w] = 0  (by definition of mean-subtraction)
```

Therefore:

```text
sum over (b,h,w) of output[b,c,h,w]
    = gamma_c * sum(x_norm) + N * beta_c
    = gamma_c * 0 + N * beta_c
    = N * beta_c
```

With `beta = 0`: **`sum(output) = 0` for any input x**.

#### Why the Gradient is Zero

Using the Kratzert three-term batch norm backward formula:

```text
grad_input[i] = (grad_output[i] - k/N - x_norm[i] * dotp/N) * gamma/std
```

where `k = sum(grad_output)` and `dotp = sum(grad_output * x_norm)`.

When `grad_output = ones_like(output)`:

- `k = N` (sum of N ones)
- `dotp = sum(1 * x_norm) = sum(x_norm) = 0`  (zero-mean property)
- Result: `grad_input[i] = (1 - N/N - x_norm[i] * 0/N) * gamma/std = 0`

This is **analytically exactly zero**, not approximately zero.

#### Why Numerical Gradient Also Appears Near Zero

The test loss `L = sum(output)` is itself degenerate when `beta=0`:
`L = N * beta = 0` identically, so numerical finite differences also return ~0
(up to floating-point noise). The ~0.009 discrepancy is pure numerical noise,
not a real gradient value.

#### Safe Alternatives

**Option 1**: Set `beta != 0` so `sum(output) != 0`

```mojo
var beta = create_constant_tensor([C], DType.float32, 0.1)  # Non-zero beta
```

**Option 2**: Use non-uniform `grad_output` instead of `ones_like`

```mojo
# Alternating pattern avoids sum(grad_output * x_norm) = 0 cancellation
var grad_output = zeros_like(output)
for i in range(output.numel()):
    var val = Float32(i % 4) * Float32(0.25) - Float32(0.3)
    grad_output._data.bitcast[Float32]()[i] = val
```

**Option 3**: Use a weighted scalar loss matched to the `grad_output`

```mojo
# Loss = sum(output * grad_output) — non-degenerate for any non-uniform grad_output
fn forward_for_grad(inp: ExTensor) raises -> ExTensor:
    var out = batch_norm2d(inp, gamma, beta, ...)[0]
    var weighted = multiply(out, grad_output)
    while weighted.dim() > 0:
        weighted = reduce_sum(weighted, axis=0, keepdims=False)
    return weighted
```

**Rule**: For normalization layers, never use `grad_output=ones_like(output)` in backward
gradient checks. See `docs/dev/testing-patterns.md` for the canonical correct pattern.

**Discovery context**: Found during PR #2724 / issue #3282. The `sum(x_norm) = 0` cancellation
also affects `layer_norm_backward` and any normalization layer using mean-centering.

---

## Layer Deduplication

### Rationale

Many models have repeated layer architectures. Testing every instance is wasteful.

### Strategy

Test **unique layer configurations** only, defined by:

- Layer type (Conv, Linear, BatchNorm, etc.)
- Input/output channels
- Kernel size, stride, padding
- Activation type

### Examples

**VGG-16**: 13 conv layers → 5 unique tests

- All use 3×3 kernels
- Differ only by channel count: 64, 128, 256, 512
- Test one conv per unique channel configuration

### Documentation

Each test clearly documents which layers it covers:

```mojo
fn test_conv_256_channels() raises:
    """Test Conv 256 channels (forward + backward).

    Covers:
    - Block 3, Layer 1: Conv(128→256)
    - Block 3, Layer 2: Conv(256→256)
    - Block 3, Layer 3: Conv(256→256)
    """
```text
## CI/CD Integration

### PR Workflow (`comprehensive-tests.yml`)

**Trigger**: Every pull request

**Test Groups** (21 parallel):

1. Core modules (types, conv, linear, etc.)
2. Shared utilities (autograd, data, testing)
3. **Model layerwise tests** (7 models)

**Runtime**: ~12 minutes (down from ~15 minutes, 20% reduction)

**No dataset downloads** - Uses special values only

### Weekly E2E Workflow (`model-e2e-tests-weekly.yml`)

**Trigger**: Sundays at 3 AM UTC

**Jobs**:

1. `prepare-datasets`: Download EMNIST and CIFAR-10
2. `test-model-e2e`: 7 parallel E2E tests
3. `e2e-report`: Aggregate results, upload with 365-day retention

**Runtime**: ~140 minutes (parallelized across 7 models)

**Artifacts**: Weekly reports with historical tracking

## Writing New Tests

### Adding a New Model

1. **Create layerwise test file**:

    ```bash
    touch tests/models/test_<model>_layers.mojo
    ```

2. **Structure**:

    ```mojo
    # Forward tests for each unique layer
    fn test_conv1_forward() raises: ...
    fn test_fc1_forward() raises: ...

    # Backward tests with gradient checking
    fn test_conv1_backward() raises: ...
    fn test_fc1_backward() raises: ...

    # Integration test
    fn test_all_layers_sequence() raises: ...

    fn main() raises:
        # Run all tests
    ```

3. **Create E2E test file**:

    ```bash
    touch tests/models/test_<model>_e2e.mojo
    ```

4. **Add to CI**:

   - Update `.github/workflows/comprehensive-tests.yml`
   - Update `.github/workflows/model-e2e-tests-weekly.yml`

### Best Practices

1. **Use LayerTester utilities**:

    ```mojo
    LayerTester.test_conv_layer(...)
    LayerTester.test_linear_layer_backward(...)
    ```

2. **Small tensor sizes for backward tests**:

    - Conv: 8×8 input (not 32×32)
    - Linear: 32 features (not 1024)

3. **Deduplicate heavily**:

    - Identify unique layer configurations
    - Document which layers each test covers

4. **Test all layer types**:

    - Parametric: Conv, Linear, BatchNorm (forward + backward)
    - Non-parametric: ReLU, MaxPool, Dropout (forward + property validation)

5. **Use seeded random for gradient checking**:

    ```mojo
    var input = create_seeded_random_tensor([1, 3, 8, 8], dtype, seed=42)
    ```

## References

- **Implementation**: `shared/testing/layer_testers.mojo`
- **Special Values**: `shared/testing/special_values.mojo`
- **Gradient Checker**: `shared/testing/gradient_checker.mojo`
- **DType Utils**: `shared/testing/dtype_utils.mojo`
- **PR Workflow**: `.github/workflows/comprehensive-tests.yml`
- **Weekly E2E**: `.github/workflows/model-e2e-tests-weekly.yml`
