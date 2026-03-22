# ADR-011: Batch Normalization Gradient Loss Function Selection

**Status**: Accepted

**Date**: 2026-03-14

**Issue Reference**: [Issue #3666](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3666)

**Decision Owner**: ML Odyssey Team

## Executive Summary

For gradient checking of normalization layers (BatchNorm, LayerNorm), use `sum(output^2)` as the scalar
loss function instead of `sum(output)`. This produces non-zero upstream gradients (`dL/dY = 2 * output`)
that meaningfully test backward pass implementations, avoiding pathological zero-gradient cases that
can mask bugs.

## Context

### Problem Statement

Batch normalization and layer normalization have a mathematical property that causes the commonly-used
`sum(output)` loss to produce exactly zero gradients when testing gradient checking:

1. **Zero-mean property**: Normalized values per channel always sum to zero by definition of mean-subtraction
2. **Degenerate loss**: When `beta=0`, `sum(output) = gamma * sum(x_norm) + beta = 0` identically
3. **Test failure**: Gradient checking with this loss produces ~0 analytical gradients and ~0 numerical
   gradients (noise against zero baseline), appearing to pass trivially without validating actual
   backward pass correctness

### Constraints

- Must work with all normalization layer variants (BatchNorm2d, LayerNorm, GroupNorm, etc.)
- Must provide non-zero gradients for meaningful validation
- Must be deterministic and reproducible
- Must not introduce excessive computational overhead in tests
- Numerical gradient computation must be stable (epsilon=1e-5 for finite differences)

### Requirements

1. **Non-zero gradients**: Loss function must produce non-zero upstream gradients
2. **Mathematical correctness**: Should map to standard ML practices where applicable
3. **Symmetry**: Should apply to all normalization layers uniformly
4. **Documentation**: Must include derivation and rationale for future maintenance

## Decision

### Solution Overview

**Use `sum(output^2)` as the canonical scalar loss for normalization layer gradient checks.**

This choice:

1. Produces non-zero upstream gradients: `dL/dY = 2 * output`
2. Is mathematically well-defined for any `output` value (including zero-normalized outputs)
3. Follows standard ML practice (sum of squared errors / MSE loss)
4. Works uniformly across all normalization layer types
5. Avoids the zero-mean cancellation problem entirely

### Technical Details

#### Mathematical Derivation

For a normalization layer with output `Y`:

```text
Forward: L = sum(Y^2)

Backward (upstream gradient):
dL/dY[i] = 2 * Y[i]
```

This is non-zero for any non-zero output value, and provides meaningful gradients even when
the normalized values sum to zero.

#### Why `sum(output^2)` Avoids Zero Gradients

The Kratzert three-term batch norm backward formula:

```text
grad_input[i] = (grad_output[i] - k/N - x_norm[i] * dotp/N) * gamma/std
```

With `grad_output = 2 * output`:

- `k = sum(grad_output) = 2 * sum(output) = 2 * (gamma * sum(x_norm) + N * beta)`
- For `beta=0`: `k = 2 * gamma * 0 = 0` (still zero!)
- BUT: `dotp = sum(grad_output * x_norm) = 2 * sum(output * x_norm)`
  - This is **NOT zero** because `output * x_norm` doesn't have the same cancellation as `x_norm` alone

The key insight: While `sum(x_norm) = 0` and `sum(output) = N*beta`, the product
`sum(output * x_norm)` is generally non-zero, providing non-trivial gradients.

#### Implementation Pattern

In gradient checking test code:

```mojo
fn test_batch_norm_gradient() raises:
    """Test BatchNorm2d backward with sum(output^2) loss."""
    var layer = BatchNorm2dLayer(num_channels=4)
    var input = create_test_input(...)

    # Forward
    var output = layer.forward(input, training=True)

    # Loss: sum(output^2)
    fn forward_for_grad(inp: AnyTensor) raises -> AnyTensor:
        var out = layer.forward(inp, training=True)
        var squared = multiply(out, out)
        var result = squared
        while result.dim() > 0:
            result = reduce_sum(result, axis=0, keepdims=False)
        return result

    # Upstream gradient: dL/dY = 2 * output
    var two = full_like(output, 2.0)
    var grad_output = multiply(two, output)

    # Compute gradients and validate
    var analytical = compute_analytical_gradient(input, grad_output)
    var numerical = compute_numerical_gradient(forward_for_grad, input)

    assert_gradients_close(analytical, numerical, rtol=1e-2, atol=1e-5)
```

#### Parametric Validation

Test with varying parameters to ensure robustness:

1. **Different gamma/beta values**:
   - `gamma = [1.5, 2.0]` (non-trivial scaling)
   - `beta = [0.0, 0.5]` (both zero and non-zero shift)

2. **Various input distributions**:
   - Deterministic patterns (avoids randomness issues)
   - Values spread across dynamic range (0.1, 0.5, 1.0, 1.5, etc.)

3. **Multiple normalizations**:
   - Different batch sizes, channel counts, spatial dimensions
   - Test both training (batch stats) and inference (running stats) modes

## Rationale

### Key Factors

1. **Mathematical correctness**: `sum(output^2)` is a legitimate loss function with well-defined
   gradients, unlike the pathological `sum(output)` with `beta=0`

2. **Standard ML practice**: Sum-of-squared errors is widely used in ML (MSE loss), making this
   choice familiar to contributors

3. **Universal applicability**: Works for all normalization layers (BatchNorm, LayerNorm, GroupNorm)
   without special cases or parameter constraints

4. **Avoids subtle bugs**: Tests using this loss will catch real gradient computation errors that
   trivial `sum(output)` tests cannot detect

5. **Maintainability**: Clear mathematical reasoning prevents future contributors from re-introducing
   the trivial test pattern

### Trade-offs Accepted

1. **Slightly higher numerical cost**: Computing and squaring doubles computational overhead in
   test gradients, but cost is negligible in PR validation (~milliseconds)

2. **Different from some PyTorch patterns**: PyTorch tests sometimes use other loss functions,
   but the `sum(output^2)` pattern is equally valid and more robust for this use case

## Consequences

### Positive

- **Robust testing**: Gradient checks actually validate backward pass correctness
- **Bug detection**: Will catch real gradient computation errors that `sum(output)` tests miss
- **Future-proof**: Prevents future developers from re-introducing trivial tests
- **Clear documentation**: Mathematical derivation ensures understanding and maintenance
- **Consistent patterns**: Single canonical pattern for all normalization layers

### Negative

- **Migration effort**: Existing tests using `sum(output)` must be updated (manageable scope)
- **Slightly higher test cost**: Additional multiplication/reduction operations (negligible impact)

### Neutral

- **Learning curve**: Developers must understand why `sum(output^2)` is preferred (addressed by
  this ADR and inline documentation)

## Alternatives Considered

### Alternative 1: Non-uniform `grad_output` Pattern

**Description**: Use alternating or random pattern for `grad_output` instead of `ones_like`:

```mojo
var grad_output = zeros_like(output)
for i in range(output.numel()):
    var val = Float32(i % 4) * Float32(0.25) - Float32(0.3)
    grad_output._data.bitcast[Float32]()[i] = val
```

**Pros**:

- Avoids the `sum(grad_output)` cancellation by using non-uniform patterns
- Can work if patterns are carefully chosen

**Cons**:

- Complex and unintuitive (why this specific pattern?)
- Requires careful analysis to ensure no accidental cancellations
- Less standard in ML practice
- Harder to maintain and understand

**Why Rejected**: Less intuitive than `sum(output^2)`; requires pattern-specific analysis rather
than general mathematical principle.

### Alternative 2: Non-zero `beta` Parameter

**Description**: Set `beta != 0` so `sum(output) != 0`:

```mojo
var beta = create_constant_tensor([C], DType.float32, 0.1)
```

**Pros**:

- Simple to implement
- Uses familiar loss function

**Cons**:

- Tests become parameter-dependent (must remember to set beta)
- Masks the real issue (zero-mean cancellation in normalized values)
- Doesn't generalize to all normalization variants
- Feels like working around the problem rather than solving it

**Why Rejected**: Parameter-dependent workaround rather than principled solution; doesn't address
the fundamental zero-mean property.

### Alternative 3: Weighted Dot Product Loss

**Description**: Use `sum(output * grad_output)` where `grad_output` is chosen strategically.

**Pros**:

- More general formulation
- Can be tailored per test

**Cons**:

- Adds unnecessary indirection
- Less standard than MSE loss
- Requires pre-computing `grad_output` patterns

**Why Rejected**: `sum(output^2)` is simpler and more standard.

## Implementation Plan

### Phase 1: Documentation (Complete)

- [x] Create ADR-011 explaining the decision and mathematics
- [x] Add inline code comments to testing patterns
- [x] Update `docs/dev/testing-strategy.md` with canonical `sum(output^2)` pattern

### Phase 2: Migration (Future)

- [ ] Update `tests/shared/core/test_normalization.mojo` to use `sum(output^2)` pattern
- [ ] Update `tests/models/test_*_layers.mojo` (normalization tests)
- [ ] Verify gradient checks pass with new pattern
- [ ] Add validation tests to ensure pattern correctness

### Success Criteria

- [x] ADR documented and approved
- [ ] Testing pattern implemented in core normalization tests
- [ ] All normalization layer gradient checks use new pattern
- [ ] CI validates pattern correctness (gradient checks pass)
- [ ] No regressions in other tests

## References

### Related ADRs

- [ADR-004: Testing Strategy](ADR-004-testing-strategy.md) - Overall testing architecture
- [ADR-002: Gradient Struct Return Types](ADR-002-gradient-struct-return-types.md) - Gradient API

### Related Issues

- [Issue #3171](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3171) - Add meaningful
  batch_norm2d_backward gradient test
- [Issue #2724](https://github.com/HomericIntelligence/ProjectOdyssey/issues/2724) - Original batch
  norm gradient investigation
- [Issue #3282](https://github.com/HomericIntelligence/ProjectOdyssey/issues/3282) - LayerNorm gradient
  issues

### External Documentation

- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167) - Original batch norm definition
- [Kratzert Batch Norm Derivation](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
  - Three-term backward formula used in this ADR

## Revision History

| Version | Date       | Author            | Changes         |
| ------- | ---------- | ----------------- | --------------- |
| 1.0     | 2026-03-14 | ML Odyssey Team   | Initial ADR     |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-011-batch-norm-gradient-loss.md`
- **Status**: Accepted
- **Review Frequency**: As-needed
- **Next Review**: When normalization tests are migrated to new pattern
- **Supersedes**: None
- **Superseded By**: None
