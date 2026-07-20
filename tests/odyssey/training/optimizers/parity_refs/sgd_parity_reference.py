"""SGD (momentum + weight-decay) parity reference.

Computes ONE SGD update step for a fixed (params, grad, velocity) under the
exact update Odyssey's `sgd_step` implements (sgd.mojo):

    effective_grad = grad + weight_decay * params          (wd applied first)
    velocity       = momentum * velocity + effective_grad
    params         = params - lr * velocity

and prints the resulting params + velocity as JSON, so the Mojo `sgd_step`
can be compared on identical inputs.

This is exactly `torch.optim.SGD` with the default flags
`dampening=0, nesterov=False, maximize=False` (PyTorch applies weight_decay to
the gradient BEFORE the momentum buffer update — see
torch/optim/sgd.py `_single_tensor_sgd`, and the SGD docs
https://pytorch.org/docs/stable/generated/torch.optim.SGD.html). We drive the
real `torch.optim.SGD` when torch is importable so the reference is
authoritative, and fall back to the closed-form numpy calculator (identical
algebra) when it is not, printing which path produced the numbers.

One wrinkle worth stating: torch seeds its momentum buffer with the raw
gradient on the FIRST step (`buf = grad.clone()`) rather than
`momentum*0 + grad`. Those coincide here because we pass velocity=0, so a
single hand-seeded step matches either reading; the numpy path uses the
Odyssey formula directly.
"""

import json

import numpy as np

# Fixed 6-vector inputs. Non-trivial momentum, weight_decay, and a starting
# velocity so the parity assert exercises every term of the update.
LR, MOMENTUM, WEIGHT_DECAY = 0.1, 0.9, 0.01

params = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60], dtype=np.float64)
grad = np.array([0.02, -0.03, 0.015, 0.025, -0.01, 0.04], dtype=np.float64)
velocity = np.array([0.005, -0.004, 0.003, -0.002, 0.001, -0.006], dtype=np.float64)


def _numpy_reference():
    eff_grad = grad + WEIGHT_DECAY * params
    new_velocity = MOMENTUM * velocity + eff_grad
    new_params = params - LR * new_velocity
    return new_params, new_velocity


def _torch_reference():
    import torch

    p = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    p.grad = torch.tensor(grad, dtype=torch.float64)
    opt = torch.optim.SGD(
        [p], lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, dampening=0.0
    )
    # Seed the momentum buffer to our fixed `velocity` so the single step is
    # deterministic and matches the Odyssey hand-seeded call.
    opt.state[p]["momentum_buffer"] = torch.tensor(velocity, dtype=torch.float64)
    opt.step()
    new_params = p.detach().numpy().copy()
    new_velocity = opt.state[p]["momentum_buffer"].detach().numpy().copy()
    return new_params, new_velocity


try:
    new_params, new_velocity = _torch_reference()
    source = "torch.optim.SGD"
except Exception:
    new_params, new_velocity = _numpy_reference()
    source = "numpy-closed-form"

print(
    json.dumps(
        {
            "source": source,
            "lr": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "params_in": params.tolist(),
            "grad": grad.tolist(),
            "velocity_in": velocity.tolist(),
            "new_params": new_params.tolist(),
            "new_velocity": new_velocity.tolist(),
        },
        indent=2,
    )
)
