"""Sophia clipped-preconditioned-update parity reference.

Computes ONE Sophia update step for a fixed (params, grad, momentum,
hessian_moment, hessian) and prints the resulting params + momentum +
hessian_moment, so the Mojo `sophia_step` (+ `sophia_update_hessian_moment`)
can be compared on identical inputs.

The modeled update is the form implemented in sophia.mojo:

    momentum       = beta1 * momentum + (1 - beta1) * grad
    hessian_moment = beta2 * hessian_moment + (1 - beta2) * hessian  (on refresh)
    update         = clamp(momentum / clip(gamma * hessian_moment, min=eps),
                           -rho, rho)
    params         = params - lr * update

which is the paper's clip(m / max(gamma * h, eps), .) update (Liu et al. 2023,
arXiv:2305.14342, Algorithm 3; official code github.com/Liuhong99/Sophia uses
gamma = rho * bs with the clamp at 1) expressed in the SophiaH-style
parametrization with an explicit +-rho clip (kozistr/pytorch_optimizer
SophiaH). gamma is set to a non-trivial 0.8 here so the parity test actually
asserts the gamma scaling.

Why not drive `pytorch_optimizer.SophiaH` directly: that class only creates
its `momentum` / `hessian_moment` state inside the Hessian-refresh branch, so
a single hand-seeded step raises `KeyError('momentum')` in the installed
version (and it has no gamma). NumPy is used purely as a calculator for the
published algorithm.

Vector regime: the inputs are chosen so that coordinates 0-4 land in the
UNCLIPPED regime (|m / (gamma * h)| < rho, so the preconditioned division is
what the parity assert exercises) while coordinate 5 saturates the +-rho clip.
The script prints per-coordinate raw updates and the clipped/unclipped split
so the regime is auditable.
"""

import json

import numpy as np

N = 6
LR, B1, B2, GAMMA, RHO, EPS = 0.06, 0.96, 0.99, 0.8, 0.04, 1e-12

params = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
grad = np.array([0.02, -0.03, 0.015, 0.025, -0.01, 0.5])
m = np.array([0.005, -0.004, 0.003, -0.002, 0.001, 0.05])
hm = np.array([1.0, 1.2, 0.9, 1.1, 0.8, 0.05])
hess = np.array([1.1, 1.0, 0.95, 1.05, 0.85, 0.04])

# Sophia per-parameter update (see module docstring for provenance).
m2 = B1 * m + (1.0 - B1) * grad
hm2 = B2 * hm + (1.0 - B2) * hess  # Hessian refresh on this step
raw = m2 / np.clip(GAMMA * hm2, EPS, None)
update = np.clip(raw, -RHO, RHO)
params2 = params - LR * update

clipped = np.abs(raw) >= RHO

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
                "gamma": GAMMA,
                "rho": RHO,
                "eps": EPS,
            },
            "regime": {
                "raw_update": raw.tolist(),
                "clipped_mask": clipped.tolist(),
                "n_clipped": int(clipped.sum()),
                "n_unclipped": int((~clipped).sum()),
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
