# Optimizers

This directory contains optimizer implementations for training neural networks in ProjectOdyssey.

## Available Optimizers

| Optimizer | State memory | Compute/step | Typical use case | Notes |
| --- | --- | --- | --- | --- |
| **SGD** | ~1x params | O(n) | Small batch, fast convergence | With momentum (default 0.9) |
| **Adam** | ~2x params (m + v) | O(n) | General-purpose, adaptive LR | Coupled weight decay |
| **AdamW** | ~2x params (m + v) | O(n) | General-purpose default | Decoupled weight decay (default 0.01) |
| **RMSprop** | ~1x params (v only) | O(n) | RNNs, non-convex problems | Adaptive learning rates |
| **LARS** | ~1x params | O(n) | Large-batch distributed training | Layer-wise adaptive rate scaling |
| **Muon** | ~1x params (matrix-only) | O(n) + Newton-Schulz | **Matrix-shaped weights only** | Newton-Schulz orthogonalization; Jordan et al. 2024 |
| **NorMuon** | ~1x params (matrix-only) | O(n) + Newton-Schulz + norms | Muon + per-row/col norm scaling | Improved LR stability vs Muon |
| **Lion** | ~1x params (1 buffer) | O(n) | Memory-constrained, transfer learning | Signed momentum; **LR 3-10x SMALLER than AdamW** |
| **Shampoo** | ~1x params (1 H buffer, diagonal v1) | O(n) | Second-order baseline | Diagonal variant; sqrt(sqrt(H)) preconditioner; eps=1e-12 |

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
from projectodyssey.training.optimizers import (
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
pixi run mojo test tests/projectodyssey/training/

# Test Lion (in the shared optimizer suite)
pixi run mojo test tests/projectodyssey/training/test_optimizers.mojo

# Test Muon specifically
pixi run mojo test tests/projectodyssey/training/test_muon.mojo

# Test a single Muon test
pixi run mojo test tests/projectodyssey/training/test_muon.mojo::test_muon_step_quadratic_descent
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
from projectodyssey.training.optimizers import lion_step

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

Shampoo (Anil et al. 2020) is a **second-order optimizer** that uses gradient statistics to build
a preconditioner. This v1 implements the **diagonal element-wise variant**, which collapses
the full algorithm's separate left (H\_L) and right (H\_R) preconditioners into a single buffer H.

The diagonal variant is still a meaningful second-order baseline: the `sqrt(sqrt(H))`
preconditioner (exponent 1/4) differs from AdamW's `sqrt(v)` (exponent 1/2), giving distinct
optimization dynamics.

```mojo
from projectodyssey.training.optimizers import shampoo_step

var (new_params, new_H) = shampoo_step(
    params, gradients, H,
    t=global_step,
    learning_rate=0.001,
    beta2=0.999,
    epsilon=1e-12,
)
```

### Why H\_L and H\_R collapse in the diagonal variant

The full Shampoo maintains H\_L ≈ G G^T (shape [m×m]) and H\_R ≈ G^T G (shape [n×n]).
At **element granularity** (diagonal approximation), both reduce to EMA(g²) — they are
bit-identical buffers. Keeping two copies would use ~3x parameter memory with no algorithmic
gain. The diagonal variant uses a single H ≈ EMA(g²), halving state relative to full Shampoo.

Full matrix-form Shampoo (with inverse-pth-root via eigendecomposition) is tracked as a
follow-up issue.

State: 1 buffer H per parameter, shape matches the parameter.

## References

- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. arXiv:1711.05101 [AdamW]
- You, Y., Li, J., Reddi, S., et al. (2019). *LARS: Layer-wise Adaptive Rate Scaling*. arXiv:1708.03888
- Jordan, K., et al. (2024). *Muon: An optimizer for hidden layers in neural networks*.
  [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)
- Chen, X., et al. (2023). *Symbolic Discovery of Optimization Algorithms*. arXiv:2302.06675 [Lion]
- Anil, R., Gupta, V., Koren, T., & Singer, Y. (2020). *Scalable Second Order Optimization for
  Deep Learning*. arXiv:2002.09018 [Shampoo]
