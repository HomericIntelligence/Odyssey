# ADR-004: Two-Tier Testing Strategy

**Status**: Accepted

**Date**: 2025-12-28

**Decision Owner**: Chief Architect

## Executive Summary

This ADR documents ML Odyssey's two-tier testing strategy: fast layerwise unit tests that run on
every PR, and comprehensive end-to-end integration tests that run weekly. This approach balances
fast CI feedback with thorough validation.

## Context

### Problem Statement

ML model testing presents unique challenges:

1. **Test Runtime**: Full training runs take hours, incompatible with PR CI
2. **Determinism**: Floating-point operations vary across hardware
3. **Gradient Correctness**: Backward passes must be mathematically verified
4. **Integration Complexity**: End-to-end tests require datasets and full pipelines

### Requirements

1. **Fast PR Feedback**: CI must complete in under 15 minutes
2. **Comprehensive Coverage**: All layers tested for forward and backward correctness
3. **Gradient Verification**: Analytical gradients validated against numerical gradients
4. **Full Pipeline Validation**: Complete training cycles verified regularly
5. **Reproducibility**: Tests must be deterministic across runs

## Decision

### Two-Tier Architecture

**Tier 1: Layerwise Unit Tests (Every PR)**

- Run on every pull request via CI
- Target runtime: < 12 minutes total
- Test each layer independently (forward and backward)
- Use special FP-representable values for determinism
- Validate gradients numerically

**Tier 2: End-to-End Integration Tests (Weekly)**

- Run weekly via scheduled CI workflow
- Full model training with real datasets (EMNIST, CIFAR-10)
- Validate training convergence (loss decrease)
- Generate weekly report with 365-day retention

### Tier 1: Layerwise Unit Tests

**Test Categories**:

1. **Forward Pass Tests**: Verify output shapes and values
2. **Backward Pass Tests**: Verify gradient shapes and values
3. **Gradient Checking**: Compare analytical vs numerical gradients
4. **Edge Cases**: Empty tensors, single elements, large batches

**Special FP-Representable Values**:

To ensure determinism across dtypes, tests use values exactly representable in all formats:

- `0.0`, `0.5`, `1.0`, `1.5`: Positive values for forward pass
- `-1.0`, `-0.5`: Negative values for ReLU gradient testing
- Seeded random tensors (seed=42) for gradient checking

```mojo
# These values work identically in FP4, FP8, BF8, FP16, FP32, BFloat16, Int8
var test_values = [0.0, 0.5, 1.0, 1.5, -1.0, -0.5]
```

**Gradient Checking Pattern**:

```mojo
fn test_layer_backward():
    # Forward pass
    var output = layer.forward(input)

    # Backward pass (analytical gradient)
    var grad_output = ExTensor.ones_like(output)
    var grad_input = layer.backward(grad_output)

    # Numerical gradient (finite differences)
    var eps = 1e-5
    var numerical_grad = compute_numerical_gradient(layer, input, eps)

    # Compare
    assert_close(grad_input, numerical_grad, rtol=1e-2, atol=1e-4)
```

**Small Tensor Sizes**:

To prevent timeouts, tests use small but meaningful tensor dimensions:

- Convolutions: 8x8 input, 3x3 kernel
- Linear: 16 input features, 8 output features
- Batch size: 4 for gradient checking

### Tier 2: End-to-End Integration Tests

**Test Criteria**:

- Training for 5 epochs
- Loss decrease of at least 20%
- Real datasets (EMNIST, CIFAR-10)

**Weekly Schedule**:

- Runs Sundays at 3 AM UTC
- Downloads datasets if needed
- Generates HTML report
- Report retained for 365 days

**Models Covered**:

| Model       | Layers    | Layerwise Tests | E2E Tests | Runtime |
| ----------- | --------- | --------------- | --------- | ------- |
| LeNet-5     | 12 ops    | 25 tests        | 7 tests   | ~55s    |
| AlexNet     | 15 ops    | 42 tests        | 9 tests   | ~60s    |
| VGG-16      | 25 ops    | 16 tests        | 10 tests  | ~90s    |
| ResNet-18   | Residual  | 12 tests        | 9 tests   | ~90s    |
| MobileNetV1 | Depthwise | 26 tests        | 15 tests  | ~90s    |
| GoogLeNet   | Inception | 18 tests        | 15 tests  | ~90s    |

### Test Organization

```text
tests/
├── models/
│   ├── test_lenet5_conv_layers.mojo      # Tier 1: Layer tests
│   ├── test_lenet5_fc_layers.mojo        # Tier 1: Layer tests
│   ├── test_lenet5_pooling_layers.mojo   # Tier 1: Layer tests
│   ├── test_lenet5_e2e.mojo              # Tier 2: End-to-end
│   └── ...
├── shared/
│   └── testing/
│       ├── special_values.mojo           # FP-representable values
│       ├── layer_testers.mojo            # Reusable test patterns
│       └── gradient_checker.mojo         # Numerical gradient utils
└── helpers/
    └── fixtures.mojo                     # Test fixtures
```

## Rationale

### Why Two Tiers?

**Fast PR CI is Essential**:

- Developers need quick feedback (< 15 min)
- Long CI discourages frequent commits
- E2E tests are too slow for every PR

**Weekly E2E is Sufficient**:

- Catches integration issues before release
- Real datasets ensure practical correctness
- Scheduled runs don't block development

### Why Special FP Values?

Standard random values cause non-determinism:

- Float representation varies by precision
- Rounding differences across dtypes
- Hardware-specific FP implementations

Special values (0.0, 0.5, 1.0, 1.5, -1.0, -0.5) are exactly representable in all formats from
FP4 to FP64, ensuring identical behavior across all dtypes.

### Why Numerical Gradient Checking?

Analytical gradients can have bugs that are hard to spot:

- Sign errors
- Missing scale factors
- Incorrect broadcasting

Numerical gradients (finite differences) are simple and reliable:

```mojo
# Numerical gradient via finite differences
fn numerical_gradient(f: fn(x) -> y, x: ExTensor, eps: Float64) -> ExTensor:
    grad = ExTensor.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone(); x_plus[i] += eps
        x_minus = x.clone(); x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad
```

## Consequences

### Positive

- **Fast CI**: PR feedback in under 12 minutes
- **Comprehensive Coverage**: All layers tested for forward and backward
- **Deterministic**: Special values ensure reproducibility
- **Gradient Correctness**: Numerical checking catches bugs
- **Regular Validation**: Weekly E2E ensures integration health

### Negative

- **Two Test Suites**: Maintenance overhead for separate test types
- **Deferred E2E Feedback**: Integration issues may not surface until weekly run
- **Dataset Dependencies**: E2E tests require dataset downloads

### Neutral

- **Test Deduplication**: VGG-16 and ResNet reduce duplicate tests (13 conv layers
  tested by 5 unique test cases for VGG)
- **Parallel Execution**: 21 parallel test groups across CI matrix

## Alternatives Considered

### Alternative 1: E2E Tests on Every PR

**Description**: Run full training on every PR.

**Pros**:

- Immediate integration feedback
- No weekly schedule needed

**Cons**:

- 20+ minute CI for each model
- Total CI time: 2+ hours per PR
- Blocks developer velocity

**Why Rejected**: Too slow for practical development workflow.

### Alternative 2: Only Unit Tests

**Description**: Skip E2E tests entirely.

**Pros**:

- Fast CI
- Simple test infrastructure

**Cons**:

- Integration bugs slip through
- No training convergence validation
- Real dataset issues undiscovered

**Why Rejected**: Insufficient coverage for ML correctness.

### Alternative 3: Nightly E2E Tests

**Description**: Run E2E tests nightly instead of weekly.

**Pros**:

- Faster feedback than weekly
- Still doesn't block PRs

**Cons**:

- 7x more compute cost
- Most nights have no relevant changes
- Report noise

**Why Rejected**: Weekly is sufficient given change frequency.

## Implementation Details

### CI Workflows

**Tier 1 (PR CI)**: `.github/workflows/comprehensive-tests.yml`

```yaml
strategy:
  matrix:
    test-group:
      - path: "tests/models"
        pattern: "test_lenet5_*_layers.mojo"
      - path: "tests/models"
        pattern: "test_alexnet_layers.mojo"
      # ... 21 parallel groups
```

**Tier 2 (Weekly)**: `.github/workflows/model-e2e-tests-weekly.yml`

```yaml
on:
  schedule:
    - cron: '0 3 * * 0'  # Sundays 3 AM UTC

jobs:
  e2e-tests:
    steps:
      - name: Download datasets
      - name: Run E2E tests
      - name: Generate report
      - name: Upload artifact (365 days)
```

### Running Tests Locally

```bash
# Run Tier 1 layerwise tests
just test-group "tests/models" "test_lenet5_*_layers.mojo"

# Run Tier 2 E2E tests (requires datasets)
just test-group "tests/models" "test_lenet5_e2e.mojo"

# Run all layerwise tests
just test-mojo tests/models/test_*_layers.mojo

# Run all tests
just test-mojo
```

## References

### Related Files

- `tests/models/test_*_layers.mojo`: Tier 1 layerwise tests
- `tests/models/test_*_e2e.mojo`: Tier 2 E2E tests
- `.github/workflows/comprehensive-tests.yml`: PR CI workflow
- `.github/workflows/model-e2e-tests-weekly.yml`: Weekly E2E workflow

### Related ADRs

- [ADR-008](ADR-008-coverage-tool-blocker.md): Coverage tool limitations

### External Documentation

- [Gradient Checking](https://cs231n.github.io/neural-networks-3/#gradcheck): Stanford CS231n
- [Numerical Differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation)

## Revision History

| Version | Date       | Author          | Changes     |
| ------- | ---------- | --------------- | ----------- |
| 1.0     | 2025-12-28 | Chief Architect | Initial ADR |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-004-testing-strategy.md`
- **Status**: Accepted
- **Review Frequency**: Quarterly
- **Next Review**: 2026-03-28
- **Supersedes**: None
- **Superseded By**: None
