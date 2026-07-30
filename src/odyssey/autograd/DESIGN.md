# Autograd Design and Implementation Status

## Overview

This document describes the autograd implementation for ML Odyssey, including
what's currently implemented, what's in progress, and what's deferred.

## Current Status (Committed)

### ✅ Implemented

1. **Variable wrapper** (`variable.mojo`)
   - Wraps AnyTensor with `requires_grad` flag
   - Stores gradients in `grad` field
   - Provides `zero_grad()`, `backward()`, `detach()` methods

2. **GradientTape structure** (`tape.mojo`)
   - TapeNode for representing operations
   - GradientTape for recording operations
   - `enable()`, `disable()`, `clear()`, `record()` methods

3. **SGD Optimizer** (`optimizers.mojo`)
   - Basic gradient descent with momentum support
   - `step()` and `zero_grad()` methods

4. **Integration with existing backward passes**
   - 27 backward functions in `src/odyssey/core/`
   - Loss functions with gradients
   - Comprehensive documentation

### 🚧 In Progress

1. **Automatic operation recording**
   - Requires Variable arithmetic operations (`__add__`, `__mul__`, etc.)
   - Requires global tape management
   - **Challenge**: Mojo's constraints on global mutable state

2. **Full backward() implementation**
   - Requires topological sort of computation graph
   - Requires backward function dispatch
   - Requires gradient accumulation
   - **Challenge**: Type system limitations, no Dict in collections

## Design Challenges

### Challenge 1: Global Mutable State

**Problem**: PyTorch-style autograd relies heavily on global mutable state:

- Global tape that's implicitly updated
- Global Variable registry
- Thread-local storage for gradients

**Mojo Constraint**: Mojo's ownership system and lack of mature global state
management makes this difficult.

**Solutions Considered**:

1. **Explicit tape passing** - Pass tape as argument to all operations
   - Pro: Works with Mojo's ownership
   - Con: Verbose API, not Pythonic

2. **Global Optional[GradientTape]** - Single global tape instance
   - Pro: Simpler API
   - Con: May have issues with Mojo's ownership rules

3. **Functional approach** - No mutable state, use closures
   - Pro: Clean, functional
   - Con: Different API from PyTorch, harder to use

### Challenge 2: Type System Limitations

**Problem**: PyTorch uses dynamic typing extensively:

- Operations return Union types
- Dict[int, Tensor] for gradient storage
- Dynamic dispatch based on operation type

**Mojo Constraint**: Static typing, limited generic support, no Dict in stdlib.

**Solutions Implemented**:

- `VariableRegistry`: Parallel DynamicVectors instead of Dict
- `GradientRegistry`: Same approach for gradients
- String-based operation dispatch (if/elif chains)

### Challenge 3: Operation Overloading

**Problem**: Need to override all AnyTensor operations for Variables.

**Mojo Support**: Has operator overloading (`__add__`, `__mul__`, etc.)

**Status**: Implemented in `variable_v2.mojo` (experimental)

## Recommended Approach

Given the challenges, I recommend a **pragmatic, phased approach**:

### Phase 1: Foundation ✅ (DONE)

- Variable wrapper
- Tape structure
- SGD optimizer
- Documentation
- Manual gradient example

**Value**: Provides clean API and documentation. Users can write gradients manually
with better structure than raw AnyTensor operations.

### Phase 2: Helper Functions (CURRENT)

- `compute_mse_gradient()` - Automatic MSE + mean backward
- `compute_bce_gradient()` - Automatic BCE backward
- `compute_ce_gradient()` - Automatic cross-entropy backward
- Pattern: One function per common loss + reduction combination

**Value**: Reduces boilerplate for common patterns without complex autograd.

**Status**: Started in `functional.mojo`

### Phase 3: Simple Computation Graphs (FUTURE)

- Explicit tape passed to operations
- Manual `tape.record()` calls
- Full `tape.backward()` implementation

**Value**: Semi-automatic gradients for custom operations.

**API**:

```mojo
var tape = GradientTape()
tape.enable()

# Manual recording
var z = add(x, y)
tape.record_add(x_id, y_id, z_id)

var loss = mean(z)
tape.record_mean(z_id, loss_id)

# Automatic backward
tape.backward(loss)
```

#### Experimental Sophia curvature-estimator surface (PR #5719)

PR #5719 adds `DiagonalHessianEstimator` as a narrow experimental seam for
the remaining Sophia work in #5683. The contract accepts
`List[Variable]`, a scalar objective `Variable`, a dedicated `GradientTape`
that recorded the objective, its batch size, and one requested result dtype
per parameter. This preserves parameter/objective connectivity, the scale
needed for the GNB `batch_size * gradient^2` estimate, and the destination
precision; detached `AnyTensor` parameters plus a detached loss value would
contain none of those derivative relationships.

The returned list is length- and shape-aligned with the parameters. Each
output dtype comes from the corresponding entry in `result_dtypes`. Callers
normally request float32 or the exact dtype of the Sophia Hessian-moment
buffer, including float64 state created with `force_f64`; neither parameter
storage nor scalar-objective precision implicitly chooses the result dtype.

The estimator tape is deliberately separate from the ordinary training tape.
At entry, the parameters and sampled objective must all be registered on the
dedicated tape, require gradients, and its registry must contain no
pre-existing gradients. A concrete estimator leaves only gradients from its
own backward pass in that registry. The caller consumes the estimate and then
clears or discards the dedicated tape, so training gradients are never
accumulated over or destroyed. The current `Variable` stores a numeric ID but
no owning-tape identity, so runtime validation can reject missing IDs but
cannot distinguish colliding IDs from two tapes; same-tape construction
remains an explicit caller precondition.

The estimator families have different derivative requirements:

- **Sophia-G (Gauss-Newton-Bartlett)** builds a sampled-label
  negative-log-likelihood on the tape, takes an ordinary backward pass, and
  uses the batch-scaled square of each parameter gradient. It does not require
  an HVP or JVP.
- **Sophia-H (Hutchinson)** multiplies by a Hessian and therefore does require
  an HVP (which may be implemented with higher-order reverse mode or JVP
  machinery).

Phase 1 includes only the trait and an inner-module
`PlaceholderDiagonalHessianEstimator`. The placeholder fails only when a
caller explicitly constructs and invokes it; no optimizer or dispatcher is
wired to select it. Consequently, PR #5719 does not make Sophia runnable end
to end and must not close #5683. Implementation follow-up #5717 remains under
that parent and is cross-referenced from `TODO.md`.

### Phase 4: Full Autograd (ASPIRATIONAL)

- Automatic operation recording via operator overloading
- Implicit global tape
- PyTorch-like API

**Blockers**:

- Mojo language maturity (global state, Dict, better generics)
- Significant engineering effort
- May need Mojo stdlib improvements

## Current Recommendation

**For immediate use**, provide:

1. ✅ Variable wrapper (done)
2. ✅ SGD optimizer (done)
3. ✅ Manual gradient example (done)
4. 🚧 Gradient helper functions for common patterns (in progress)
5. 📝 Clear documentation of limitations and path forward

**Value proposition**:

- Works today with current Mojo
- Reduces boilerplate compared to pure AnyTensor
- Clear API for training loops
- Foundation for future full autograd

## Files Status

| File | Status | Purpose |
| --- | --- | --- |
| `variable.mojo` | ✅ Committed | Variable wrapper (current version) |
| `variable_v2.mojo` | 🧪 Experimental | With operation overloading |
| `tape.mojo` | ✅ Committed | Tape structure (current version) |
| `registry.mojo` | 🧪 Experimental | ID->data mapping without Dict |
| `optimizers.mojo` | ✅ Committed | SGD optimizer |
| `functional.mojo` | 🚧 In progress | Gradient helper functions |
| `README.md` | ✅ Committed | User documentation |
| `DESIGN.md` | 📝 This file | Design rationale |

## Next Steps

1. **Complete functional.mojo** with common gradient helpers
2. **Update README.md** to reflect Phase 2 approach
3. **Add examples** using helper functions
4. **Test** the helper functions work correctly
5. **Commit** Phase 2 implementation
6. **Defer** Phase 3/4 until Mojo ecosystem matures

## Conclusion

Full PyTorch-style autograd is aspirational given current Mojo constraints.
The phased approach provides immediate value while maintaining a clear path
forward as the language and ecosystem mature.

**Current focus**: Practical gradient helpers that work today, not complex
autograd that's hard to maintain.
