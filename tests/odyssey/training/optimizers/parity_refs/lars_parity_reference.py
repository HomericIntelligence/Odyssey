"""LARS parity reference (You, Gitman & Ginsburg 2017, arXiv:1708.03888).

Computes ONE LARS step for a fixed (params, grad, velocity) under the exact
update Odyssey's `lars_step` implements (lars.mojo):

    param_norm  = ||params||_2            (global L2 over the whole tensor)
    grad_norm   = ||grad||_2
    eff_grad    = grad + weight_decay * params
    trust_ratio = trust_coefficient * param_norm
                  / (grad_norm + weight_decay * param_norm + epsilon)
    velocity    = momentum * velocity + trust_ratio * eff_grad
    params      = params - lr * velocity

and prints the resulting params + velocity as JSON.

This is the layer-wise-adaptive LARS rule (the trust ratio scales the
per-layer update by the ratio of weight norm to gradient norm). The reference
is a numpy closed form of the published algorithm; `pytorch_optimizer`'s LARS
uses the same trust-ratio formulation. The single global norm here corresponds
to treating the whole tensor as one LARS "layer" (matching lars.mojo, which
takes the L2 norm of the entire params/grad tensor).
"""

import json

import numpy as np

LR, MOMENTUM, WD, TRUST, EPS = 0.1, 0.9, 0.0001, 0.001, 1e-8

params = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60], dtype=np.float64)
grad = np.array([0.02, -0.03, 0.015, 0.025, -0.01, 0.04], dtype=np.float64)
velocity = np.array([0.005, -0.004, 0.003, -0.002, 0.001, -0.006], dtype=np.float64)

param_norm = np.sqrt(np.sum(params * params))
grad_norm = np.sqrt(np.sum(grad * grad))
eff_grad = grad + WD * params
trust_ratio = TRUST * param_norm / (grad_norm + WD * param_norm + EPS)
new_velocity = MOMENTUM * velocity + trust_ratio * eff_grad
new_params = params - LR * new_velocity

print(
    json.dumps(
        {
            "source": "numpy-closed-form",
            "lr": LR,
            "momentum": MOMENTUM,
            "weight_decay": WD,
            "trust_coefficient": TRUST,
            "epsilon": EPS,
            "param_norm": param_norm,
            "grad_norm": grad_norm,
            "trust_ratio": trust_ratio,
            "params_in": params.tolist(),
            "grad": grad.tolist(),
            "velocity_in": velocity.tolist(),
            "new_params": new_params.tolist(),
            "new_velocity": new_velocity.tolist(),
        },
        indent=2,
    )
)
