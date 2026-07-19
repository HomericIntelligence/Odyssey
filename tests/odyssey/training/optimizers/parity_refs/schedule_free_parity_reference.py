"""Schedule-Free parity reference (Defazio et al. 2024, "The Road Less Scheduled").

Runs THREE Schedule-Free SGD steps for a fixed (params, grad) sequence with
zero initial state and prints the resulting z, x, and next-query-point y after
each step, so the Mojo `schedule_free_step` can be compared on identical inputs.

The official `schedulefree` PyPI package is NOT available in the pinned
interpreter, so this reference is a hand-rolled NumPy transcription of the exact
update rule pinned in the tracking issue (mvillmow/Random#77). NumPy is used
only as a calculator for that verified algorithm — the issue's rule WINS over
any paper ambiguity.

Schedule-Free update rule (per step t, 1-indexed; three sequences y/z/x):
    y_t     = (1 - beta) * z_t + beta * x_t          (gradient query point)
    g_t     = grad( f, y_t )                          (caller supplies this)
    z_{t+1} = z_t - gamma * g_t                       (fast SGD-style sequence)
    c_{t+1} = (r + 1) / (t + r + 1)                   (averaging weight)
    x_{t+1} = (1 - c_{t+1}) * x_t + c_{t+1} * z_{t+1} (running average of z)

`z` and `x` are the persisted state; `y` is recomputed each step and is the
point the model uses for the forward/backward pass (train buffer). `x` is the
eval/checkpoint buffer. On step 1 the state is initialized z_1 = x_1 = params.

This transcription mirrors the reference `schedulefree.SGDScheduleFree`
(z <- z - gamma*g; x <- (1-c)*x + c*z; y = (1-beta)*z + beta*x), with the
weight schedule c_{t+1} = (r+1)/(t+r+1) — the r=0 default reduces to the
uniform average c_{t+1} = 1/(t+1).
"""

import json

import numpy as np

# Hyperparameters. beta ~ 0.9 (interpolation), r = 0 (uniform-average weight
# power), gamma = base learning rate.
GAMMA, BETA, R = 0.1, 0.9, 0.0

params = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
grads = [
    np.array([0.05, 0.15, -0.25, 0.35, -0.45]),
    np.array([0.08, 0.05, -0.20, 0.40, -0.30]),
    np.array([-0.02, 0.10, -0.15, 0.20, -0.25]),
]


def schedule_free_step(z, x, grad, t, gamma=GAMMA, beta=BETA, r=R):
    """One Schedule-Free step. Returns (z_next, x_next, y_next).

    `grad` is the gradient evaluated at y_t = (1-beta)*z_t + beta*x_t; the
    caller is responsible for having formed y_t and evaluated the gradient
    there. This function advances the z/x state and returns the NEXT query
    point y_{t+1} for convenience.
    """
    z_next = z - gamma * grad
    c = (r + 1.0) / (t + r + 1.0)
    x_next = (1.0 - c) * x + c * z_next
    y_next = (1.0 - beta) * z_next + beta * x_next
    return z_next, x_next, y_next


# Step 1 state initialization: z_1 = x_1 = params, so y_1 = params.
z = params.copy()
x = params.copy()

outs = []
for i, g in enumerate(grads):
    t = i + 1
    z, x, y = schedule_free_step(z, x, g, t)
    outs.append({"z": z.tolist(), "x": x.tolist(), "y_next": y.tolist()})

print(
    json.dumps(
        {
            "inputs": {
                "params": params.tolist(),
                "grads": [g.tolist() for g in grads],
                "gamma": GAMMA,
                "beta": BETA,
                "r": R,
            },
            "step1_out": outs[0],
            "step2_out": outs[1],
            "step3_out": outs[2],
        },
        indent=2,
    )
)
