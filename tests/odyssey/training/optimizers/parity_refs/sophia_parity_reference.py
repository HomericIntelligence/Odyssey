"""Sophia (SophiaH) parity reference.

Computes ONE SophiaH step for a fixed (params, grad, momentum, hessian_moment,
hessian) and prints the resulting params + momentum + hessian_moment, so the
Mojo `sophia_step` (+ `sophia_update_hessian_moment`) can be compared on
identical inputs.

Why not drive `pytorch_optimizer.SophiaH` directly: that class only creates its
`momentum` / `hessian_moment` state inside the Hessian-refresh branch, so a
single hand-seeded step raises `KeyError('momentum')` in the installed version.
Instead we transcribe SophiaH's exact per-parameter update math verbatim from
its source (`optimizer/sophia.py`), which is:

    momentum       = beta1 * momentum + (1 - beta1) * grad
    hessian_moment = beta2 * hessian_moment + (1 - beta2) * hessian   (on refresh)
    update         = clamp(momentum / clip(hessian_moment, min=eps), -rho, rho)
    params         = params - lr * update

so the numbers below are exactly what SophiaH computes. NumPy is used purely as
a calculator for that published algorithm.
"""

import json

import numpy as np

N = 5
LR, B1, B2, RHO, EPS = 0.06, 0.96, 0.99, 0.04, 1e-12

params = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
grad = np.array([0.05, 0.15, -0.25, 0.35, -0.45])
m = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
hm = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
hess = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

# SophiaH per-parameter update (source-verbatim math).
m2 = B1 * m + (1.0 - B1) * grad
hm2 = B2 * hm + (1.0 - B2) * hess  # Hessian refresh on this step
update = np.clip(m2 / np.clip(hm2, EPS, None), -RHO, RHO)
params2 = params - LR * update

print(
    json.dumps(
        {
            "inputs": {
                "params": params.tolist(),
                "grad": grad.tolist(),
                "m": m.tolist(),
                "hm": hm.tolist(),
                "hessian": hess.tolist(),
                "lr": LR,
                "beta1": B1,
                "beta2": B2,
                "rho": RHO,
                "eps": EPS,
            },
            "reference_out": {
                "params": params2.tolist(),
                "m": m2.tolist(),
                "hm": hm2.tolist(),
            },
        },
        indent=2,
    )
)
