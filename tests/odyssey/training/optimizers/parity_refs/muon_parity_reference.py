"""Muon parity reference (Jordan et al. 2024; https://github.com/KellerJordan/Muon).

Computes ONE Muon step for a fixed rank-2 (params, grad, momentum) under the
exact update Odyssey's `muon_step` implements (muon.mojo), including a faithful
numpy replica of `newton_schulz_orthogonalize`:

    new_momentum = momentum_beta * momentum + grad
    dir          = grad + momentum_beta * new_momentum   (nesterov=True; else new_momentum)
    u_orth       = newton_schulz(dir, steps=5)
    scale        = 0.2 * max(R, C) / sqrt(R * C)
    params       = params - lr * scale * u_orth
    params       = params - lr * weight_decay * params    (on original params)

Newton-Schulz (Jordan quintic, coeffs a=3.4445, b=-4.7750, c=2.0315):
    1. Y = X / (||X||_F + 1e-7)          (Frobenius pre-normalization)
    2. if rows > cols: iterate on Y^T (shorter dim), transpose back at the end
    3. repeat 5x:  A = Y @ Y^T;  B = b*A + c*(A @ A);  Y = a*Y + B @ Y

This is the reference Muon (KellerJordan/Muon torch optim). numpy is used as the
calculator because the Jordan iteration deliberately does NOT converge to exact
orthogonality (singular values end in ~[0.68, 1.13]) — so the "parity" is
against Odyssey's implemented iteration, not an idealized orthogonal matrix.
The 2x3 shape (rows < cols) exercises the non-transposed NS path.
"""

import json

import numpy as np

LR, BETA, WD, NS_STEPS = 0.01, 0.95, 0.01, 5
A_COEF, B_COEF, C_COEF = 3.4445, -4.7750, 2.0315

# 2x3 matrix (rows < cols → no transpose inside NS).
params = np.array([[0.10, -0.20, 0.30], [-0.40, 0.50, -0.60]], dtype=np.float64)
grad = np.array([[0.02, -0.03, 0.015], [0.025, -0.01, 0.04]], dtype=np.float64)
momentum = np.array([[0.05, -0.04, 0.03], [-0.02, 0.01, -0.06]], dtype=np.float64)


def newton_schulz(X, steps=5):
    rows, cols = X.shape
    transposed = rows > cols
    Y = X.T.copy() if transposed else X.copy()
    norm = np.sqrt(np.sum(Y * Y))  # Frobenius norm
    Y = Y / (norm + 1e-7)
    for _ in range(steps):
        A = Y @ Y.T
        B = B_COEF * A + C_COEF * (A @ A)
        Y = A_COEF * Y + B @ Y
    return Y.T if transposed else Y


new_momentum = BETA * momentum + grad
direction = grad + BETA * new_momentum  # nesterov=True
u_orth = newton_schulz(direction, NS_STEPS)

R, C = float(params.shape[0]), float(params.shape[1])
scale = 0.2 * max(R, C) / np.sqrt(R * C)
params_after_grad = params - LR * scale * u_orth
new_params = params_after_grad - LR * WD * params  # wd on original params

print(
    json.dumps(
        {
            "source": "numpy-newton-schulz-replica",
            "lr": LR,
            "momentum_beta": BETA,
            "weight_decay": WD,
            "ns_steps": NS_STEPS,
            "nesterov": True,
            "params_in": params.tolist(),
            "grad": grad.tolist(),
            "momentum_in": momentum.tolist(),
            "scale": scale,
            "u_orth": u_orth.tolist(),
            "new_params": new_params.tolist(),
            "new_momentum": new_momentum.tolist(),
        },
        indent=2,
    )
)
