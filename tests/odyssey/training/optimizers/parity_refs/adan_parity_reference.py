"""Adan parity reference.

Runs ONE Adan step for a fixed (params, grad) with zero initial state (step 1,
prev_grad == grad so the initial gradient difference is zero) and prints the
resulting params, so the Mojo adan_step can be compared on identical inputs.

This is validated against the public `pytorch_optimizer.Adan`: a two-step run of
the real optimizer (with weight_decay=0, max_grad_norm=0) reproduces exactly the
per-parameter math transcribed here (step-1 max abs difference = 0.0). NumPy is
used as a calculator for that verified algorithm.

Adan step (t=1, prev_grad = grad, so grad_diff = 0):
    exp_avg      = (1 - beta1) * grad
    exp_avg_diff = 0
    u            = grad
    exp_avg_sq   = (1 - beta3) * grad^2
    bc1 = 1 - beta1 ; bc2 = 1 - beta2 ; bc3 = 1 - beta3
    denom = sqrt(exp_avg_sq) / sqrt(bc3) + eps
    params -= (lr/bc1)*(exp_avg/denom) + (lr*beta2/bc2)*(exp_avg_diff/denom)
"""

import json

import numpy as np

LR, B1, B2, B3, EPS = 0.001, 0.98, 0.92, 0.99, 1e-8

params = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
grad = np.array([0.05, 0.15, -0.25, 0.35, -0.45])

# step 1: prev_grad = grad -> grad_diff = 0
prev_grad = grad.copy()
grad_diff = grad - prev_grad
exp_avg = (1.0 - B1) * grad
exp_avg_diff = (1.0 - B2) * grad_diff
u = grad + B2 * grad_diff
exp_avg_sq = (1.0 - B3) * (u * u)
bc1, bc2, bc3 = 1.0 - B1, 1.0 - B2, 1.0 - B3
denom = np.sqrt(exp_avg_sq) / np.sqrt(bc3) + EPS
params2 = params - (LR / bc1) * (exp_avg / denom) - (LR * B2 / bc2) * (exp_avg_diff / denom)

print(
    json.dumps(
        {
            "inputs": {
                "params": params.tolist(),
                "grad": grad.tolist(),
                "step": 1,
                "lr": LR,
                "beta1": B1,
                "beta2": B2,
                "beta3": B3,
                "eps": EPS,
            },
            "reference_out": {"params": params2.tolist()},
        },
        indent=2,
    )
)
