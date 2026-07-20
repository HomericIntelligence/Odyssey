"""RMSprop parity reference (Tieleman & Hinton 2012 lecture 6.5).

Computes ONE RMSprop step for a fixed (params, grad, square_avg) under the
exact update Odyssey's `rmsprop_step` implements (rmsprop.mojo), for the
centered=False, momentum=0 path:

    effective_grad  = grad + weight_decay * params
    square_avg      = alpha * square_avg + (1 - alpha) * effective_grad**2
    params          = params - lr * effective_grad / (sqrt(square_avg) + epsilon)

and prints the resulting params + square_avg as JSON.

This matches `torch.optim.RMSprop(centered=False, momentum=0)`, which also
places eps outside the sqrt (`avg = square_avg.sqrt().add_(eps)`, see
torch/optim/rmsprop.py and
https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) and adds
coupled weight_decay to the gradient. We drive the real torch.optim.RMSprop
when importable (authoritative), else the identical numpy closed form.
"""

import json

import numpy as np

LR, ALPHA, EPS, WD = 0.01, 0.99, 1e-8, 0.0

params = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60], dtype=np.float64)
grad = np.array([0.02, -0.03, 0.015, 0.025, -0.01, 0.04], dtype=np.float64)
square_avg = np.array([1e-4, 2e-4, 1.5e-4, 3e-4, 0.5e-4, 4e-4], dtype=np.float64)


def _numpy_reference():
    eff = grad + WD * params
    new_sq = ALPHA * square_avg + (1.0 - ALPHA) * eff * eff
    new_params = params - LR * eff / (np.sqrt(new_sq) + EPS)
    return new_params, new_sq


def _torch_reference():
    import torch

    p = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    p.grad = torch.tensor(grad, dtype=torch.float64)
    opt = torch.optim.RMSprop([p], lr=LR, alpha=ALPHA, eps=EPS, weight_decay=WD, momentum=0.0, centered=False)
    opt.state[p]["step"] = torch.tensor(0.0)
    opt.state[p]["square_avg"] = torch.tensor(square_avg, dtype=torch.float64)
    opt.step()
    return (
        p.detach().numpy().copy(),
        opt.state[p]["square_avg"].detach().numpy().copy(),
    )


try:
    new_params, new_sq = _torch_reference()
    source = "torch.optim.RMSprop"
except Exception:
    new_params, new_sq = _numpy_reference()
    source = "numpy-closed-form"

print(
    json.dumps(
        {
            "source": source,
            "lr": LR,
            "alpha": ALPHA,
            "epsilon": EPS,
            "weight_decay": WD,
            "params_in": params.tolist(),
            "grad": grad.tolist(),
            "square_avg_in": square_avg.tolist(),
            "new_params": new_params.tolist(),
            "new_square_avg": new_sq.tolist(),
        },
        indent=2,
    )
)
