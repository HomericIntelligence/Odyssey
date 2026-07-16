# Optimizers

This directory contains optimizer implementations for training neural networks in Odyssey.

## Available Optimizers

| Optimizer | State memory | Compute/step | Typical use case | Notes |
| --- | --- | --- | --- | --- |
| **SGD** | ~1x params | O(n) | Small batch, fast convergence | With momentum (default 0.9) |
| **Adam** | ~2x params (m + v) | O(n) | General-purpose, adaptive LR | Coupled weight decay |
| **AdamW** | ~2x params (m + v) | O(n) | General-purpose default | Decoupled weight decay (default 0.01) |
| **ADOPT** | ~2x params (m + v) | O(n) | General-purpose adaptive | Clamp-min denom (not sqrt(v)+eps); normalizes by previous v; arXiv:2411.02853 |
| **RMSprop** | ~1x params (v only) | O(n) | RNNs, non-convex problems | Adaptive learning rates |
| **LARS** | ~1x params | O(n) | Large-batch distributed training | Layer-wise adaptive rate scaling |
| **Muon** | ~1x params (matrix-only) | O(n) + Newton-Schulz | **Matrix-shaped weights only** | Newton-Schulz orthogonalization; Jordan et al. 2024 |
| **NorMuon** | ~1x params (matrix-only) | O(n) + Newton-Schulz + norms | Muon + per-row/col norm scaling | Improved LR stability vs Muon |
| **Muon Hyperball** | ~1x params (matrix-only) | O(n) + Newton-Schulz | LR transfer across model width/depth | Muon + Frobenius-norm clamps on the per-step update and the weight matrix (one-sided ball projections; radius <= 0 disables a clamp) |
| **Lion** | ~1x params (1 buffer) | O(n) | Memory-constrained, transfer learning | Signed momentum; **LR 3-10x SMALLER than AdamW** |
| **Adan** | ~4x params (exp_avg + exp_avg_diff + exp_avg_sq + prev_grad) | O(n) | General-purpose, fast convergence | Nesterov-style look-ahead + gradient-difference momentum; arXiv:2208.06677 |
| **Shampoo** | ~3x params (L [m×m] + R [n×n] + momentum [m×n]) | O(n³) + Newton-Schulz | Second-order baseline, matrix weights | Two-sided matrix preconditioner; Newton-Schulz inverse fourth root; **rank-2 params only** |
| **Sophia** (update step only) | ~2x params (momentum + hessian_moment) | O(n) | LLM pretraining / second-order-lite | Sophia-style clipped preconditioned update step, `clip(m / max(γ·h, ε), ±ρ)`; Hessian-diagonal estimates are CALLER-SUPPLIED (Sophia-H/G estimators need HVPs — not yet in Odyssey autograd, see `src/odyssey/autograd/TODO.md`); arXiv:2305.14342 |

## Quick Selection Guide

### By Parameter Type

```mojo
// Linear layer weights (784 x 128):
if is_muon_eligible(W_linear) {
    (W_linear, m_W) = muon_step(W_linear, grad_W, m_W, lr=0.01)
} else {
    (W_linear, m_W, v_W) = adamw_step(W_linear, grad_W, m_W, v_W, t, lr=0.01)
}

// Biases (128,): always AdamW
(b, m_b, v_b) = adamw_step(b, grad_b, m_b, v_b, t, lr=0.01)

// Embeddings (vocab_size x embed_dim): always AdamW
(embed, m_emb, v_emb) = adamw_step(embed, grad_emb, m_emb, v_emb, t, lr=0.01)

// Batch norm scales (128,): always AdamW
(scale, m_s, v_s) = adamw_step(scale, grad_s, m_s, v_s, t, lr=0.01)
```

### By Model Type

| Model | Weights | Biases | Embeddings | Notes |
| --- | --- | --- | --- | --- |
| **Vision CNN** (ResNet, ViT) | Muon | AdamW | — | Muon shows ~2% improvement on ImageNet |
| **Language LLM** (Transformer) | Muon | AdamW | AdamW | Muon matches AdamW speed, better perplexity |
| **Small benchmark** (MNIST, CIFAR-10) | AdamW or SGD | AdamW | — | Muon overkill for small datasets |
| **Fine-tuning** | Depends on LR | AdamW | AdamW | Use Muon if fine-tuning matrix-heavy layers |

## Muon Optimizer

### What is Muon?

Muon (Jordan et al. 2024) is a Newton-Schulz-based optimizer for matrix-shaped parameters.
It orthogonalizes the momentum buffer before each update, which:

- Improves conditioning of the weight matrix during training
- Reduces the effective rank of weight changes (more stable learning)
- Shows consistent 2-5% improvement over AdamW on large models
- Matches AdamW speed (same O(1) per-parameter complexity)

**Key papers / references:**

- Jordan et al. 2024, ["Muon: An optimizer for hidden layers in neural networks"]
  ([https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/))
- Reference implementation: [KellerJordan/Muon](https://github.com/KellerJordan/Muon)

### When to Use Muon

✅ **Use Muon for:**

- Linear layer weights: shape `[input_dim, output_dim]`
- Convolutional kernels: reshape from `[out_ch, in_ch, kH, kW]` to `[out_ch, in_ch*kH*kW]`
- Vision models (ResNets, ViT, etc.)
- Language models with large embedding matrices
- Any model where matrix-shaped parameters dominate

❌ **Do NOT use Muon for:**

- Embeddings: use AdamW instead
- Biases: use AdamW instead
- Batch norm / layer norm scales and shifts: use AdamW instead
- 1D/3D/4D parameters (reshaping allowed, but only rank-2 is eligible)
- Scalar parameters

### Example: Hybrid Muon+AdamW Optimizer

```mojo
from odyssey.training.optimizers import (
    muon_step, adamw_step,
    is_muon_eligible,
)

def optimizer_step(
    params: List[AnyTensor],
    gradients: List[AnyTensor],
    state_m: List[AnyTensor],  // Muon/SGD momentum or AdamW m
    state_v: List[AnyTensor],  // AdamW v (unused for Muon)
    t: Int,
    lr: Float64,
) raises -> Tuple[List[AnyTensor], List[AnyTensor], List[AnyTensor]]:
    """Hybrid optimizer: Muon for weights, AdamW for everything else."""
    var new_params = List[AnyTensor]()
    var new_m = List[AnyTensor]()
    var new_v = List[AnyTensor]()

    for i in range(len(params)):
        var p = params[i]
        var grad = gradients[i]
        var m = state_m[i]
        var v = state_v[i]

        if is_muon_eligible(p):
            // Muon for matrix-shaped parameters
            var (p_new, m_new) = muon_step(
                p, grad, m,
                learning_rate=lr,
                momentum_beta=0.95,
                weight_decay=0.01,
            )
            new_params.append(p_new)
            new_m.append(m_new)
            new_v.append(v)  // v unchanged (unused for this param)
        else:
            // AdamW for everything else
            var (p_new, m_new, v_new) = adamw_step(
                p, grad, m, v, t,
                learning_rate=lr,
                weight_decay=0.01,
            )
            new_params.append(p_new)
            new_m.append(m_new)
            new_v.append(v_new)

    return (new_params, new_m, new_v)
```

### Swapping Optimizers: Signature Note

**Important:** When replacing `adamw_step` with `muon_step`, update the destructuring:

```mojo
// Old (AdamW) — returns 3-tuple
(params, m, v) = adamw_step(params, grads, m, v, t, lr=...)

// New (Muon) — returns 2-tuple
(params, m) = muon_step(params, grads, m, lr=...)
```

AdamW maintains two state buffers (m and v). Muon maintains only one (momentum). The return
arity changes because **state cardinality differs**, not because of implementation style. This
is expected behavior; see the hybrid example above for how to handle mixed optimizer state.

### Hyperparameters

| Parameter | Default | Range | Notes |
| --- | --- | --- | --- |
| `learning_rate` | — | 1e-4 to 1e-2 | Tune per task; try 0.01 for medium models |
| `momentum_beta` | 0.95 | 0.9–0.99 | Paper default is 0.95 |
| `weight_decay` | 0.01 | 0.0–0.1 | Decoupled decay; 0.01 matches AdamW default |
| `ns_steps` | 5 | 3–8 | 5 is standard; more steps = better orthogonality, slower |
| `nesterov` | True | — | Recommended; set to False for heavy-ball momentum |

**Note on weight decay:** Muon's weight decay includes the learning rate factor (`lr * wd * p`),
differing from AdamW's form (`wd * p`). This is the Jordan et al. 2024 recipe. If you migrate
from AdamW and see different final accuracy, re-baseline hyperparameters; the decay formula
difference may require tuning.

## Pure Functional Design

All optimizers follow a **pure functional design**:

```mojo
// Old: in-place mutation
apply_weight_decay(mut params, wd=0.01)
update_momentum(mut momentum, grad=grad)

// New: pure functional
(params, m, v) = adamw_step(params, grads, m, v, t, lr=...)
```

Benefits:

- **No hidden state**: all state is explicit in function arguments and returns
- **Composable**: stack optimizers easily
- **Testable**: no side effects
- **Gradients-friendly**: no in-place ops to worry about in autograd

## Testing

Run optimizer tests:

```bash
# Test all optimizers
pixi run mojo test tests/odyssey/training/

# Test Lion (in the shared optimizer suite)
pixi run mojo test tests/odyssey/training/test_optimizers.mojo

# Test Muon specifically
pixi run mojo test tests/odyssey/training/test_muon.mojo

# Test a single Muon test
pixi run mojo test tests/odyssey/training/test_muon.mojo::test_muon_step_quadratic_descent
```

Tests verify:

- Shape and dtype correctness
- Descent on convex objectives (quadratic loss)
- Orthogonality convergence (Muon-specific)
- Error handling and validation
- Pure functional semantics (no input mutation)

## Implementation Notes

### Architecture

- **Pure functional**: Returns new state, does not mutate inputs
- **SIMD-optimized**: Uses `multiply_simd`, `add_simd`, etc. for element-wise ops
- **Dtype dispatch**: Works with float16, float32, float64
- **No allocations in tight loop**: Pre-allocate tensors before training loop

### Checkpoint Serialization

All optimizers use `AnyTensor` for state buffers, which integrate with `CheckpointManager`:

```mojo
// Save state
checkpoint.save("m", m)
checkpoint.save("v", v)

// Load state
m = checkpoint.load("m")
v = checkpoint.load("v")
```

No special serialization needed for Muon; the single momentum buffer `m` is handled the same as SGD's velocity.

## Lion Optimizer

Lion (Chen et al. 2023) is a memory-efficient optimizer that uses **signed momentum**. It maintains
a single momentum buffer (half the memory of AdamW) and applies the *sign* of an interpolated
momentum/gradient term as the update direction.

```mojo
from odyssey.training.optimizers import lion_step

var (new_params, new_momentum) = lion_step(
    params, gradients, momentum,
    learning_rate=0.0001,  // 3-10x lower than AdamW
    beta1=0.9,
    beta2=0.99,
    weight_decay=0.0,
)
```

**WARNING:** Lion is sensitive to learning rate. The signed update makes every step the same
magnitude (`lr`), so an AdamW learning rate of `1e-3` typically maps to a Lion learning rate of
`1e-4`. Start 3-10x lower and tune up.

State: 1 buffer (momentum), shape matches the parameter. Works on parameters of any shape.

## Shampoo Optimizer

Shampoo (Gupta, Koren, Singer 2018; Anil et al. 2020) is a **second-order optimizer** that
applies a two-sided matrix preconditioner computed via Newton-Schulz iteration. It is the
**full matrix-form** algorithm: it maintains separate left and right Gram matrix accumulators
and computes their inverse fourth roots to precondition each gradient update.

**Eligibility:** Only rank-2 parameter tensors with both dimensions ≥ 2 are eligible. Use
`is_shampoo_eligible(params)` to check. Biases, embeddings, and other non-matrix parameters
must use a different optimizer (e.g., AdamW).

**Fallback divergence (mixed-optimizer runs).** When a model contains Shampoo-ineligible
parameters (rank-4 conv kernels, rank-1 biases), those parameters must be updated by a
different rule for the same step. The `examples/grok/lenet_emnist` training loop applies
momentum-free `sgd_step_simple` to the ineligible parameters under `--optimizer shampoo`,
whereas the default `--optimizer sgd` path updates *every* parameter through
`model.update_parameters`, which uses SGD with momentum (default 0.9). The two paths therefore
update conv kernels and biases by different rules. This is intentional — Shampoo's matrix
state buffers cannot back rank-1/rank-4 tensors — but it means a Shampoo run is not a pure
drop-in replacement for the SGD run. When smoke-comparing optimizers, attribute differences in
conv/bias trajectories to this fallback rather than to the Shampoo preconditioner itself.

### State Initialization

```mojo
from odyssey.training.optimizers import (
    initialize_shampoo_state,
    shampoo_step,
    is_shampoo_eligible,
)

// params shape: [m, n]
// Returns three buffers: L [m, m], R [n, n], momentum [m, n]
var (L, R, momentum) = initialize_shampoo_state(params)
```

### Training Loop

```mojo
// Each step returns a 4-tuple: (params_new, L_new, R_new, momentum_new)
var (params, L, R, momentum) = shampoo_step(
    params, gradients, L, R, momentum,
    learning_rate=0.001,
    beta_precond=0.95,    // EMA decay for Gram matrix accumulators
    beta_momentum=0.95,   // EMA decay for momentum buffer
    weight_decay=0.0,
    ns_steps=8,           // Newton-Schulz iterations per inverse-root stage
    eps=1e-10,
    max_precond_norm=1e6, // Frobenius norm clamp for numerical stability
)
```

Or use the convenience wrapper with default hyperparameters:

```mojo
var (params, L, R, momentum) = shampoo_step_simple(
    params, gradients, L, R, momentum,
    learning_rate=0.001,
)
```

### Algorithm

Shampoo maintains two Gram matrix accumulators that track gradient covariance:

- `L [m, m]`: left accumulator, `L_t = β·L_{t-1} + G·Gᵀ`
- `R [n, n]`: right accumulator, `R_t = β·R_{t-1} + Gᵀ·G`

At each step, the inverse fourth roots are computed via Newton-Schulz iteration
(`L^{-1/4}` and `R^{-1/4}`), and the preconditioned gradient is:

```text
precond_grad = L^{-1/4} @ G @ R^{-1/4}
```

Momentum is accumulated on the preconditioned gradient, then the parameter update is:

```text
momentum_t = β_m · momentum_{t-1} + precond_grad
params_t   = params_{t-1} − lr · momentum_t
```

### Calling Convention

Shampoo's calling convention differs from most optimizers:

- `initialize_shampoo_state(params)` returns **3 buffers**: `(L, R, momentum)` — the
  caller continues to hold `params` separately.
- `shampoo_step(params, gradients, L, R, momentum, learning_rate, ...)` accepts
  **5 state arguments** and returns a **4-tuple**: `(params_new, L_new, R_new, momentum_new)`.
- `L_new` and `R_new` are the **unclamped** EMA accumulators; internal clamping only
  affects the inverse-root computation for numerical stability.

### Hyperparameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `learning_rate` | — | Tune per task; start around 0.001 |
| `beta_precond` | 0.95 | EMA decay for L and R accumulators (Anil et al. variant) |
| `beta_momentum` | 0.95 | EMA decay for the momentum buffer |
| `weight_decay` | 0.0 | Decoupled L2 regularization |
| `ns_steps` | 8 | Newton-Schulz iterations per inverse-root stage |
| `eps` | 1e-10 | Numerical stability floor for trace normalization |
| `max_precond_norm` | 1e6 | Frobenius norm clamp for L and R before inverse-root |

**Note on `beta_precond`:** The original Gupta et al. 2018 paper uses a pure running sum
(β = 1). The EMA form (β < 1) used here is the Anil et al. 2020 scalable variant, which
keeps accumulators bounded during long training runs.

State: 3 buffers per eligible parameter — `L [m, m]`, `R [n, n]`, `momentum [m, n]`.

## References

- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. arXiv:1711.05101 [AdamW]
- You, Y., Li, J., Reddi, S., et al. (2019). *LARS: Layer-wise Adaptive Rate Scaling*. arXiv:1708.03888
- Jordan, K., et al. (2024). *Muon: An optimizer for hidden layers in neural networks*.
  [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)
- Chen, X., et al. (2023). *Symbolic Discovery of Optimization Algorithms*. arXiv:2302.06675 [Lion]
- Anil, R., Gupta, V., Koren, T., & Singer, Y. (2020). *Scalable Second Order Optimization for
  Deep Learning*. arXiv:2002.09018 [Shampoo]
