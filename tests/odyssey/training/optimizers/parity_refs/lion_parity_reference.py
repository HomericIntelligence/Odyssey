"""Lion parity reference (Chen et al. 2023, arXiv:2302.06675).

Computes ONE Lion step for a fixed (params, grad, momentum) under the exact
update Odyssey's `lion_step` implements (lion.mojo):

    update  = sign(beta1 * momentum + (1 - beta1) * grad)
    params  = params - lr * update
    params  = params - lr * weight_decay * params        (decoupled)
    momentum = beta2 * momentum + (1 - beta2) * grad      (EMA updated AFTER)

and prints the resulting params + momentum as JSON.

This matches the reference Lion (lucidrains/lion-pytorch and
kozistr/pytorch_optimizer Lion): the update direction is the SIGN of the
beta1-interpolated momentum, the momentum EMA uses beta2 and is updated after
the step, and decoupled weight decay is lr-scaled. Note the sign of exactly
zero: numpy `np.sign(0)=0` and Odyssey's `sign` likewise yields 0, so we pick
inputs where no interpolated coordinate is exactly zero to avoid a
sign(0)-convention ambiguity. NumPy is used as the calculator for the
published algorithm (a single hand-seeded step of pytorch_optimizer.Lion
needs its state pre-created; the closed form is unambiguous).
"""

import json

import numpy as np

LR, B1, B2, WD = 0.001, 0.9, 0.99, 0.01

params = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60], dtype=np.float64)
grad = np.array([0.02, -0.03, 0.015, 0.025, -0.01, 0.04], dtype=np.float64)
momentum = np.array([0.05, -0.04, 0.03, -0.02, 0.01, -0.06], dtype=np.float64)

interp = B1 * momentum + (1.0 - B1) * grad
update = np.sign(interp)
# assert no exact-zero coordinate so sign() convention is unambiguous
assert np.all(interp != 0.0), "pick inputs with no zero interpolation"

new_params = params - LR * update
new_params = new_params - LR * WD * params  # decoupled, lr-scaled, on original p
new_momentum = B2 * momentum + (1.0 - B2) * grad

print(
    json.dumps(
        {
            "source": "numpy-closed-form",
            "lr": LR,
            "beta1": B1,
            "beta2": B2,
            "weight_decay": WD,
            "params_in": params.tolist(),
            "grad": grad.tolist(),
            "momentum_in": momentum.tolist(),
            "sign_update": update.tolist(),
            "new_params": new_params.tolist(),
            "new_momentum": new_momentum.tolist(),
        },
        indent=2,
    )
)
