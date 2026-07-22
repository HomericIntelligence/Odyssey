"""AdamW parity reference (Loshchilov & Hutter 2019, arXiv:1711.05101).

Computes ONE AdamW update step for a fixed (params, grad, m, v, t) under the
exact update Odyssey's `adamw_step` implements (adamw.mojo):

    m      = beta1 * m + (1 - beta1) * grad            (grad NOT decayed)
    v      = beta2 * v + (1 - beta2) * grad**2
    m_hat  = m / (1 - beta1**t)
    v_hat  = v / (1 - beta2**t)
    p'     = params - lr * m_hat / (sqrt(v_hat) + epsilon)
    params = p' - weight_decay * p'                    (decoupled, on p')

and prints the resulting params + m + v as JSON.

IMPORTANT — this deviates from `torch.optim.AdamW` in the weight-decay term:
torch applies `p *= (1 - lr * weight_decay)` to the PRE-update parameters
(decoupled decay scaled by lr, see torch/optim/adamw.py), whereas Odyssey's
adamw.mojo applies `p' -= weight_decay * p'` — i.e. `p' * (1 - weight_decay)`
— to the POST-gradient-update parameters, with NO lr factor. So this reference
models Odyssey's implemented formula (numpy closed form) and does NOT drive
torch.optim.AdamW, because the two disagree by design at the pinned SHA. The
docstring records the divergence so a future maintainer who wants exact
torch-AdamW semantics knows the gap is in the decay term, not the Adam core
(which IS torch-equivalent — see adam_parity_reference.py).
"""

import json

import numpy as np

LR, B1, B2, EPS, WD = 0.001, 0.9, 0.999, 1e-8, 0.01
T = 3

params = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60], dtype=np.float64)
grad = np.array([0.02, -0.03, 0.015, 0.025, -0.01, 0.04], dtype=np.float64)
m = np.array([0.005, -0.004, 0.003, -0.002, 0.001, -0.006], dtype=np.float64)
v = np.array([1e-4, 2e-4, 1.5e-4, 3e-4, 0.5e-4, 4e-4], dtype=np.float64)

# Odyssey adamw formula (grad NOT decayed into the moments; decoupled decay
# applied to the post-gradient-update params without an lr factor).
new_m = B1 * m + (1.0 - B1) * grad
new_v = B2 * v + (1.0 - B2) * grad * grad
m_hat = new_m / (1.0 - B1**T)
v_hat = new_v / (1.0 - B2**T)
p_after_grad = params - LR * m_hat / (np.sqrt(v_hat) + EPS)
new_params = p_after_grad - WD * p_after_grad

print(
    json.dumps(
        {
            "source": "numpy-odyssey-formula",
            "note": "decoupled decay applied to post-update params, no lr factor "
            "(differs from torch.optim.AdamW; see docstring)",
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
