"""MGUP-Muon parity reference (numpy transcription).

Transcribes Odyssey's Muon core VERBATIM, then applies the MGUP selective-update
mechanism exactly as the Mojo path does: compute the Muon update dW, select the
`selected_fraction` of coordinates with the largest |dW| via the k-th-largest
threshold rule (k = max(1, floor(fraction * N)), threshold = k-th largest |dW|,
selected = |dW| >= threshold), and multiply those coordinates' step by
`select_scale`. Emits a single-step result for a fixed ramp W/grad.

Note: the threshold is a magnitude value, so ties at the boundary select ALL entries
equal to it (both Mojo and numpy apply the same >= comparison, so they agree).

Run:
    python tests/odyssey/training/optimizers/parity_refs/mgup_muon_parity_reference.py
"""

import json
import math

import numpy as np


def newton_schulz(X, steps=5):
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
    new_m = beta * momentum + grad
    update = grad + beta * new_m if nesterov else new_m
    u_orth = newton_schulz(update, ns_steps)
    R, C = params.shape
    scale = 0.2 * max(R, C) / np.sqrt(R * C)
    new_p = params - lr * scale * u_orth
    if wd > 0.0:
        new_p = new_p - (lr * wd) * params
    return new_p, new_m


def select_threshold(abs_update, fraction):
    flat = abs_update.flatten()
    n = flat.size
    if n == 0:
        return 0.0
    if fraction <= 0.0:
        return float(flat.max()) + 1.0
    if fraction >= 1.0:
        return 0.0
    k = int(math.floor(n * fraction))
    if k < 1:
        k = 1
    # k-th largest magnitude.
    order = np.sort(flat)[::-1]
    return float(order[k - 1])


def mgup_muon_step(
    params,
    grad,
    momentum,
    lr,
    selected_fraction=0.25,
    select_scale=2.0,
    beta=0.95,
    wd=0.01,
    ns_steps=5,
    nesterov=True,
):
    w_muon, new_m = muon_step(params, grad, momentum, lr, beta, wd, ns_steps, nesterov)
    update = w_muon - params
    if selected_fraction <= 0.0 or select_scale == 1.0:
        return w_muon, new_m
    abs_u = np.abs(update)
    threshold = select_threshold(abs_u, selected_fraction)
    new_update = np.where(abs_u >= threshold, update * select_scale, update)
    return params + new_update, new_m


R, C = 3, 4
W = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5
G = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.05 - 0.3
M = np.zeros((R, C), dtype=np.float64)
LR = 0.1
SELECTED_FRACTION, SELECT_SCALE = 0.25, 2.0

new_p, new_m = mgup_muon_step(W, G, M, LR, SELECTED_FRACTION, SELECT_SCALE)

# Number of coordinates actually amplified (for the test's sanity check).
w_muon, _ = muon_step(W, G, M, LR)
abs_u = np.abs(w_muon - W)
thr = select_threshold(abs_u, SELECTED_FRACTION)
n_selected = int((abs_u >= thr).sum())

print(
    json.dumps(
        {
            "config": {
                "R": R,
                "C": C,
                "lr": LR,
                "selected_fraction": SELECTED_FRACTION,
                "select_scale": SELECT_SCALE,
                "n_selected": n_selected,
            },
            "W": W.flatten().tolist(),
            "G": G.flatten().tolist(),
            "new_params": new_p.flatten().tolist(),
            "new_momentum": new_m.flatten().tolist(),
        },
        indent=2,
    )
)
