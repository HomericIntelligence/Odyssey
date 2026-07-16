"""LionMuon parity reference (numpy transcription).

Transcribes Odyssey's Lion and Muon cores VERBATIM and runs the LionMuon
alternation for three steps with period=4:

    step 0: step_index % 4 == 0 -> Muon
    step 1: -> Lion
    step 2: -> Lion

The SAME momentum buffer is threaded through all three steps, so this validates
both branches AND the shared-buffer continuity across the alternation. Emits the
parameters and the shared momentum buffer after each of the three steps.

Run:
    python tests/odyssey/training/optimizers/parity_refs/lionmuon_parity_reference.py
"""

import json

import numpy as np


def newton_schulz(X, steps=5):
    """Odyssey newton_schulz_orthogonalize, transcribed."""
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


def muon_step(params, grad, momentum, lr, beta=0.95, wd=0.0, ns_steps=5, nesterov=True):
    """Odyssey muon_step, transcribed."""
    new_m = beta * momentum + grad
    update = grad + beta * new_m if nesterov else new_m
    u_orth = newton_schulz(update, ns_steps)
    R, C = params.shape
    scale = 0.2 * max(R, C) / np.sqrt(R * C)
    new_p = params - lr * scale * u_orth
    if wd > 0.0:
        new_p = new_p - (lr * wd) * params
    return new_p, new_m


def lion_step(params, grad, momentum, lr, beta1=0.9, beta2=0.99, wd=0.0):
    """Odyssey lion_step, transcribed."""
    new_m = beta2 * momentum + (1.0 - beta2) * grad
    update = np.sign(beta1 * momentum + (1.0 - beta1) * grad)
    new_p = params - lr * update
    if wd != 0.0:
        new_p = new_p - (wd * lr) * params
    return new_p, new_m


def lionmuon_step(params, grad, momentum, lr, step_index, period=4):
    if step_index % period == 0:
        return muon_step(params, grad, momentum, lr, beta=0.95, wd=0.0)
    return lion_step(params, grad, momentum, lr, beta1=0.9, beta2=0.99, wd=0.0)


R, C = 3, 4
W = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5
G = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.05 - 0.3
M = np.zeros((R, C), dtype=np.float64)
LR = 0.1

steps_out = []
for s in range(3):
    W, M = lionmuon_step(W, G, M, LR, s, period=4)
    steps_out.append(
        {
            "step": s,
            "branch": "muon" if s % 4 == 0 else "lion",
            "params": W.flatten().tolist(),
            "momentum": M.flatten().tolist(),
        }
    )

print(
    json.dumps(
        {
            "config": {"R": R, "C": C, "lr": LR, "period": 4},
            "steps": steps_out,
        },
        indent=2,
    )
)
