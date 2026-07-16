"""Muon Hyperball parity reference (numpy transcription).

Muon lacks a hand-seedable public reference whose exact Newton-Schulz convention
matches Odyssey's `muon_step` (the momentum/coefficient conventions across public
Muon implementations differ). So this reference transcribes Odyssey's own Muon core
VERBATIM — the same coefficients (a=3.4445, b=-4.7750, c=2.0315), the same
transpose-to-shorter-dim, the same spectral pre-normalization X/(||X||_F + 1e-7),
the same shape-invariant scale 0.2*max(R,C)/sqrt(R*C), and the same lr*wd*p weight
decay — then applies the hyperball projections on top:

    dW = W_muon - W;  dW *= min(1, update_norm_max / ||dW||_F)
    W_new = W + dW;   W_new *= min(1, weight_norm_max / ||W_new||_F)

A non-positive radius disables the corresponding constraint. This validates the
hyperball wrapper arithmetic against an independent implementation of the exact same
math the Mojo path runs. Emits single-step results for a fixed ramp W/grad in two
regimes:

- Case 1 (saturated): weight_norm_max=1.0 < ||W_new||, so the weight projection
  rescales the result onto the ball surface (||new_params||_F == weight_norm_max).
- Case 2 (non-saturated): weight_norm_max=10.0 > ||W_new||, so the weight projection
  is the identity and the parity assertion pins the unclamped muon+update-clamp
  arithmetic exactly (no final radial rescale can mask an upstream error).

Run:
    python tests/odyssey/training/optimizers/parity_refs/muon_hyperball_parity_reference.py
"""

import json

import numpy as np


def newton_schulz(X, steps=5):
    """Odyssey's newton_schulz_orthogonalize, transcribed."""
    rows, cols = X.shape
    transposed = rows > cols
    Y = X.T.copy() if transposed else X.copy()
    norm = np.sqrt(np.sum(Y * Y))
    Y = Y / (norm + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = Y @ Y.T
        A2 = A @ A
        B = b * A + c * A2
        Y = a * Y + B @ Y
    return Y.T if transposed else Y


def muon_step(params, grad, momentum, lr, beta=0.95, wd=0.01, ns_steps=5, nesterov=True):
    """Odyssey's muon_step, transcribed."""
    new_m = beta * momentum + grad
    update = grad + beta * new_m if nesterov else new_m
    u_orth = newton_schulz(update, ns_steps)
    R, C = params.shape
    scale = 0.2 * max(R, C) / np.sqrt(R * C)
    new_p = params - lr * scale * u_orth
    if wd > 0.0:
        new_p = new_p - (lr * wd) * params
    return new_p, new_m


def project_to_ball(x, radius):
    if radius <= 0.0:
        return x
    norm = np.sqrt(np.sum(x * x))
    if norm <= radius:
        return x
    return x * (radius / norm)


def muon_hyperball_step(
    params,
    grad,
    momentum,
    lr,
    weight_norm_max=1.0,
    update_norm_max=0.1,
    beta=0.95,
    wd=0.01,
    ns_steps=5,
    nesterov=True,
):
    w_muon, new_m = muon_step(params, grad, momentum, lr, beta, wd, ns_steps, nesterov)
    dW = project_to_ball(w_muon - params, update_norm_max)
    w_new = params + dW
    w_new = project_to_ball(w_new, weight_norm_max)
    return w_new, new_m


R, C = 3, 4
W = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5
G = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.05 - 0.3
M = np.zeros((R, C), dtype=np.float64)
LR = 0.1
UPDATE_NORM_MAX = 0.1

# Case 1: weight ball SATURATED (radius below the unclamped result norm).
# Case 2: weight ball NOT saturated (radius far above the result norm) — the
# projection is the identity, so the reference pins the raw arithmetic exactly.
CASES = {"saturated": 1.0, "non_saturated": 10.0}

out = {}
for name, weight_norm_max in CASES.items():
    new_p, new_m = muon_hyperball_step(W, G, M, LR, weight_norm_max, UPDATE_NORM_MAX)
    out[name] = {
        "config": {
            "R": R,
            "C": C,
            "lr": LR,
            "weight_norm_max": weight_norm_max,
            "update_norm_max": UPDATE_NORM_MAX,
        },
        "W": W.flatten().tolist(),
        "G": G.flatten().tolist(),
        "new_params": new_p.flatten().tolist(),
        "new_momentum": new_m.flatten().tolist(),
        "new_params_fro_norm": float(np.sqrt(np.sum(new_p * new_p))),
    }

print(json.dumps(out, indent=2))
