# Adding a New Optimizer (Canonical Extension Point)

This guide is the **canonical extension point** for adding optimizers to
Odyssey. There is exactly one rule: **always implement the math as a pure
functional step first, then optionally wrap it with the autograd OO class.**

The codebase has 24+ optimizers already following this contract — see the
table in `src/odyssey/training/optimizers/README.md#lifecycle-init--step--updatestate`.

## Why functional first?

The legacy optimizers in `src/odyssey/autograd/optimizers.mojo` hand-rolled
their update math inline and duplicated it across files. After the
single-source-of-truth refactor, every OO optimizer lives in
`src/odyssey/autograd/optimizers_oo/<name>.mojo` and thin-delegates to its
functional core. The pattern today is:

| Layer | Where | Owns |
| --- | --- | --- |
| **Math (pure functional)** | `src/odyssey/training/optimizers/<name>.mojo` | The algorithm. Returns new state. |
| **State init (uniform)** | `init_<name>_state(...)` in the same file | The initialization convention. |
| **OO wrapper (autograd path)** | `src/odyssey/autograd/optimizers_oo/<name>.mojo` — always | Hyperparameters + per-param state buffers. Delivers the convenient `opt.step(params, tape)` shape. |

`src/odyssey/autograd/optimizers.mojo` is a **deprecation shim** — it
re-exports the five core OO optimizers (SGD, Adam, AdamW, AdaGrad,
RMSprop) from their canonical `optimizers_oo/` location so the historical
import path `from odyssey.autograd.optimizers import SGD, ...` keeps
working. New code should import from
`odyssey.autograd.optimizers_oo` directly.

The functional core is what gets reused: training loops, hybrid optimizers
(Muon + AdamW for matrix + non-matrix params), gradient checkers, and unit
tests all consume the pure step function. The OO wrapper is *only* needed
when the call site works with `Variable` + `GradientTape`.

## Recipe — three deliverables

### 1. Functional core (REQUIRED)

Create `src/odyssey/training/optimizers/<name>.mojo`:

```mojo
from odyssey.tensor.any_tensor import AnyTensor

def <name>_step(
    params: AnyTensor,
    gradients: AnyTensor,
    state0: AnyTensor,             # One entry per state buffer
    # ... additional state buffers ...
    t: Int,                         # If the rule needs a step counter
    learning_rate: Float64,
    # ... rule-specific hyperparameters ...
) raises -> Tuple[AnyTensor, AnyTensor, ...]:
    """Pure functional step — no mutation, returns new state."""
    ...
    return (new_params, new_state0, ...)
```

Conventions:

- Function name: `<name>_step`. Provide a `<name>_step_simple(...)`
  convenience that delegates with default hyperparameters.
- Pure functional: do not mutate inputs. Return the new state in a tuple.
- SIMD-optimize the math (use `multiply_simd`, `add_simd`, `divide_simd`,
  `subtract_simd`, `sqrt`, etc.) — follow the existing files.
- Validate shape/dtype contracts at function entry.
- Add `init_<name>_state(params_list, *, force_f64=False) -> List[List[AnyTensor]]`
  that returns a list-of-lists aligned with `params_list`. Each inner
  list has one buffer per state slot. See `init_adam_state` /
  `init_rmsprop_state` for the idiomatic implementation.

### 2. Re-export from the public package (REQUIRED)

Add a one-line re-export to `src/odyssey/training/optimizers/__init__.mojo`
under the appropriate section. Add an entry to the `Available Optimizers`
table in `src/odyssey/training/optimizers/README.md`.

### 3. OO wrapper (only if Variable/GradientTape API needed)

For a new optimizer, the OO wrapper **always lives at**
`src/odyssey/autograd/optimizers_oo/<name>.mojo`. The five legacy core
optimizers (SGD, Adam, AdamW, AdaGrad, RMSprop) were moved here from
`src/odyssey/autograd/optimizers.mojo`; that file is now a
backward-compatibility shim that re-exports them from `optimizers_oo/`.

Minimal template (this is exactly what `optimizers_oo/lion.mojo`,
`optimizers_oo/lars.mojo`, etc. look like):

```mojo
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros_like
from odyssey.autograd.variable import Variable
from odyssey.autograd.tape import GradientTape
from odyssey.autograd.optimizer_base import (
    Optimizer, zero_grad_impl, validate_learning_rate,
)
from odyssey.training.optimizers.<name> import <name>_step


@fieldwise_init
struct <Name>(Copyable, Movable, Optimizer):
    """<NAME> OO wrapper.

    State is held per parameter ID in a Dict so the same struct can drive
    a Variable list of arbitrary size. Math is delegated to the canonical
    `<name>_step` in `odyssey.training.optimizers.<name>`.
    """

    var learning_rate: Float64
    # ... rule-specific hyperparameters ...
    var state_buffers: Dict[Int, AnyTensor]   # one Dict per state slot

    def __init__(out self, learning_rate: Float64, ...):
        self.learning_rate = learning_rate
        ...
        self.state_buffers = Dict[Int, AnyTensor]()

    def step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        for i in range(len(parameters)):
            if not parameters[i].requires_grad:
                continue
            var param_id = parameters[i].id
            if not tape.registry.has_gradient(param_id):
                continue
            var grad = tape.registry.get_grad(param_id)
            var param_data = parameters[i].data
            if param_id not in self.state_buffers:
                self.state_buffers[param_id] = zeros_like(param_data)
            var state0 = self.state_buffers[param_id]
            var result = <name>_step(
                param_data, grad, state0,
                self.learning_rate, ...
            )
            parameters[i].data = result[0]
            self.state_buffers[param_id] = result[1]

    def zero_grad(self, mut tape: GradientTape):
        zero_grad_impl(tape)

    def get_lr(self) -> Float64:
        return self.learning_rate

    def set_lr(mut self, lr: Float64) raises:
        validate_learning_rate(lr)
        self.learning_rate = lr
```

State container rules:

- Always use `Dict[Int, AnyTensor]` keyed by `param.id` — never an index
  counter. The legacy `vel_idx++` pattern in the original
  `src/odyssey/autograd/optimizers.mojo` mixed state across parameter
  reordering and was replaced.
- One Dict per state slot (`m_buffers`, `v_buffers`, `momenta`, etc.).
- Lazy-initialize on first sight (`if param_id not in self.<buffers>:
  ...zeros_like(param_data)`).
- Increment any shared scalar (Adam's `t`, Prodigy's `r`/`d`, etc.) once
  per `step()` call, then pass to the canonical functional step.

Finally, re-export the OO struct from:

1. `src/odyssey/autograd/optimizers_oo/__init__.mojo`
2. `src/odyssey/autograd/__init__.mojo` (only if you want it auto-exported)
3. `src/odyssey/__init__.mojo` (only if you want the public-API surface
   to include it)

## Drift prevention

Once the OO wrapper delegates to the functional core, any improvement to
the algorithm in `<name>.mojo` automatically lands in every consumer:

- Training loops using `<name>_step` directly.
- The autograd OO wrapper used by `loss.backward(tape); opt.step(params, tape)`.
- Unit tests, gradient checkers, and hybrid optimizers.

Any DUPLICATED math (e.g. the original `SGD`/`Adam`/`AdaGrad`/`RMSprop`
classes in `src/odyssey/autograd/optimizers.mojo`) is a drift hazard and
should be replaced with a delegating wrapper per this guide. New
optimizers MUST NOT INLINE math in their OO wrapper — they must call
`<name>_step`.

## Reference list (canonical examples)

| Optimizer | Functional core | OO wrapper (canonical home: `optimizers_oo/`) |
| --- | --- | --- |
| SGD | `src/odyssey/training/optimizers/sgd.mojo` (`sgd_step`, `init_sgd_state`) | `src/odyssey/autograd/optimizers_oo/sgd.mojo` |
| Adam | `src/odyssey/training/optimizers/adam.mojo` (`adam_step`, `init_adam_state`) | `src/odyssey/autograd/optimizers_oo/adam.mojo` |
| AdamW | `src/odyssey/training/optimizers/adamw.mojo` (`adamw_step`, `init_adamw_state`) | `src/odyssey/autograd/optimizers_oo/adamw.mojo` |
| AdaGrad | `src/odyssey/training/optimizers/adagrad.mojo` (`adagrad_step`, `init_adagrad_state`) | `src/odyssey/autograd/optimizers_oo/adagrad.mojo` |
| RMSprop | `src/odyssey/training/optimizers/rmsprop.mojo` (`rmsprop_step`, `init_rmsprop_state`) | `src/odyssey/autograd/optimizers_oo/rmsprop.mojo` |
| Lion | `src/odyssey/training/optimizers/lion.mojo` | `src/odyssey/autograd/optimizers_oo/lion.mojo` |
| LARS | `src/odyssey/training/optimizers/lars.mojo` | `src/odyssey/autograd/optimizers_oo/lars.mojo` |
| FTRL | `src/odyssey/training/optimizers/ftrl.mojo` | `src/odyssey/autograd/optimizers_oo/ftrl.mojo` |
| Prodigy | `src/odyssey/training/optimizers/prodigy.mojo` | `src/odyssey/autograd/optimizers_oo/prodigy.mojo` |
| Muon family (5) | `src/odyssey/training/optimizers/muon*.mojo` | `src/odyssey/autograd/optimizers_oo/muon_family.mojo` |
| Shampoo family (4) | `src/odyssey/training/optimizers/{shampoo,soap,kl_shampoo,splus}.mojo` | `src/odyssey/autograd/optimizers_oo/shampoo_family.mojo` |
| Schedule-Free family (3) | `src/odyssey/training/optimizers/{schedule_free,schedule_free_plus,sf_normuon}.mojo` | `src/odyssey/autograd/optimizers_oo/schedule_free.mojo` |
| Sophia, Adan, AdOpt | `src/odyssey/training/optimizers/{sophia,adan,adopt}.mojo` | `src/odyssey/autograd/optimizers_oo/adam_family.mojo` |

> Historical note: The five core OO structs (SGD, Adam, AdamW, AdaGrad,
> RMSprop) previously lived inline in
> `src/odyssey/autograd/optimizers.mojo`. After the single-source-of-truth
> refactor they were moved to `optimizers_oo/<name>.mojo`; the legacy
> file is now a 5-line re-export shim. `from odyssey.autograd.optimizers
> import SGD, ...` still works for back-compat, but new code should
> import directly from `odyssey.autograd.optimizers_oo`.
