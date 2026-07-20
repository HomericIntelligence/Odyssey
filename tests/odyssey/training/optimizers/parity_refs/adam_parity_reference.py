"""Adam parity reference (arXiv:1412.6980).

Computes ONE Adam update step for a fixed (params, grad, m, v, t) under the
exact update Odyssey's `adam_step` implements (adam.mojo):

    effective_grad = grad + weight_decay * params        (coupled L2, wd added to grad)
    m      = beta1 * m + (1 - beta1) * effective_grad
    v      = beta2 * v + (1 - beta2) * effective_grad**2
    m_hat  = m / (1 - beta1**t)
    v_hat  = v / (1 - beta2**t)
    params = params - lr * m_hat / (sqrt(v_hat) + epsilon)   (eps OUTSIDE the sqrt)

and prints the resulting params + m + v as JSON.

This is exactly `torch.optim.Adam` with `amsgrad=False, maximize=False` — torch
also adds coupled weight_decay to the gradient and places eps outside the sqrt
(`denom = v_hat.sqrt().add_(eps)`, see torch/optim/adam.py `_single_tensor_adam`
and https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). We drive
the real `torch.optim.Adam` when torch is importable (authoritative), falling
back to the identical numpy closed form otherwise, and print which path was
used.

To make the single hand-seeded step deterministic we set the optimizer's
`step` state to t-1 before calling `.step()` so torch's internal
`state['step'] += 1` lands on t, and seed `exp_avg`/`exp_avg_sq` to our fixed
m/v.
"""

import json

import numpy as np

LR, B1, B2, EPS, WD = 0.001, 0.9, 0.999, 1e-8, 0.01
T = 3  # non-trivial timestep so bias correction is exercised

params = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60], dtype=np.float64)
grad = np.array([0.02, -0.03, 0.015, 0.025, -0.01, 0.04], dtype=np.float64)
m = np.array([0.005, -0.004, 0.003, -0.002, 0.001, -0.006], dtype=np.float64)
v = np.array([1e-4, 2e-4, 1.5e-4, 3e-4, 0.5e-4, 4e-4], dtype=np.float64)


def _numpy_reference():
    eff = grad + WD * params
    new_m = B1 * m + (1.0 - B1) * eff
    new_v = B2 * v + (1.0 - B2) * eff * eff
    m_hat = new_m / (1.0 - B1**T)
    v_hat = new_v / (1.0 - B2**T)
    new_params = params - LR * m_hat / (np.sqrt(v_hat) + EPS)
    return new_params, new_m, new_v


def _torch_reference():
    import torch

    p = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    p.grad = torch.tensor(grad, dtype=torch.float64)
    opt = torch.optim.Adam(
        [p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD, amsgrad=False
    )
    st = opt.state[p]
    st["step"] = torch.tensor(float(T - 1))
    st["exp_avg"] = torch.tensor(m, dtype=torch.float64)
    st["exp_avg_sq"] = torch.tensor(v, dtype=torch.float64)
    opt.step()
    return (
        p.detach().numpy().copy(),
        st["exp_avg"].detach().numpy().copy(),
        st["exp_avg_sq"].detach().numpy().copy(),
    )


try:
    new_params, new_m, new_v = _torch_reference()
    source = "torch.optim.Adam"
except Exception:
    new_params, new_m, new_v = _numpy_reference()
    source = "numpy-closed-form"

print(
    json.dumps(
        {
            "source": source,
            "lr": LR,
            "beta1": B1,
            "beta2": B2,
            "epsilon": EPS,
            "weight_decay": WD,
            "t": T,
            "params_in": params.tolist(),
            "grad": grad.tolist(),
            "m_in": m.tolist(),
            "v_in": v.tolist(),
            "new_params": new_params.tolist(),
            "new_m": new_m.tolist(),
            "new_v": new_v.tolist(),
        },
        indent=2,
    )
)
