"""Adan parity reference.

Runs TWO Adan steps for fixed (params, grad) sequences with zero initial state
and prints the resulting params after each step, so the Mojo adan_step can be
compared on identical inputs.

Step 1 uses prev_grad == grad (initial gradient difference is zero), exercising
only the gradient-EMA / second-moment terms. Step 2 uses a DIFFERENT gradient so
that grad_diff != 0, exercising the difference-EMA (`exp_avg_diff`) and the
look-ahead term `u = grad + beta2 * grad_diff` that step 1 leaves at their zero
initial values.

This is validated against the public `pytorch_optimizer.Adan`: a two-step run of
the real optimizer (with weight_decay=0, max_grad_norm=0) reproduces exactly the
per-parameter math transcribed here. NumPy is used as a calculator for that
verified algorithm.

Adan step (per step t, 1-indexed):
    grad_diff    = grad - prev_grad
    exp_avg      = beta1 * exp_avg      + (1 - beta1) * grad
    exp_avg_diff = beta2 * exp_avg_diff + (1 - beta2) * grad_diff
    u            = grad + beta2 * grad_diff
    exp_avg_sq   = beta3 * exp_avg_sq   + (1 - beta3) * u^2
    bc1 = 1 - beta1^t ; bc2 = 1 - beta2^t ; bc3 = 1 - beta3^t
    denom  = sqrt(exp_avg_sq) / sqrt(bc3) + eps
    params = params - (lr/bc1)*(exp_avg/denom) - (lr*beta2/bc2)*(exp_avg_diff/denom)
    prev_grad = grad   (stored for the next step)
"""

import json

import numpy as np

LR, B1, B2, B3, EPS = 0.001, 0.98, 0.92, 0.99, 1e-8

params = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
grad1 = np.array([0.05, 0.15, -0.25, 0.35, -0.45])
# Step-2 gradient is DIFFERENT so grad_diff != 0 (exercises the difference-EMA).
grad2 = np.array([0.08, 0.05, -0.20, 0.40, -0.30])


def adan_step(params, grad, prev_grad, exp_avg, exp_avg_diff, exp_avg_sq, t):
    grad_diff = grad - prev_grad
    exp_avg = B1 * exp_avg + (1.0 - B1) * grad
    exp_avg_diff = B2 * exp_avg_diff + (1.0 - B2) * grad_diff
    u = grad + B2 * grad_diff
    exp_avg_sq = B3 * exp_avg_sq + (1.0 - B3) * (u * u)
    bc1 = 1.0 - B1**t
    bc2 = 1.0 - B2**t
    bc3 = 1.0 - B3**t
    denom = np.sqrt(exp_avg_sq) / np.sqrt(bc3) + EPS
    params = params - (LR / bc1) * (exp_avg / denom) - (LR * B2 / bc2) * (exp_avg_diff / denom)
    return params, grad.copy(), exp_avg, exp_avg_diff, exp_avg_sq


# Zero initial state.
exp_avg = np.zeros_like(params)
exp_avg_diff = np.zeros_like(params)
exp_avg_sq = np.zeros_like(params)

# Step 1: prev_grad = grad1 -> grad_diff = 0.
params1, prev_grad, exp_avg, exp_avg_diff, exp_avg_sq = adan_step(
    params, grad1, grad1.copy(), exp_avg, exp_avg_diff, exp_avg_sq, 1
)

# Step 2: prev_grad = grad1 (from step 1), grad = grad2 -> grad_diff != 0.
params2, prev_grad, exp_avg, exp_avg_diff, exp_avg_sq = adan_step(
    params1, grad2, prev_grad, exp_avg, exp_avg_diff, exp_avg_sq, 2
)

print(
    json.dumps(
        {
            "inputs": {
                "params": params.tolist(),
                "grad1": grad1.tolist(),
                "grad2": grad2.tolist(),
                "lr": LR,
                "beta1": B1,
                "beta2": B2,
                "beta3": B3,
                "eps": EPS,
            },
            "step1_out": {"params": params1.tolist()},
            "step2_out": {"params": params2.tolist()},
        },
        indent=2,
    )
)
