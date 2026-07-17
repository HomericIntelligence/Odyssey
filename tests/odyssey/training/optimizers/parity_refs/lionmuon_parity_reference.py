"""LionMuon parity reference (numpy transcription).

Transcribes Odyssey's Lion and Muon cores VERBATIM and runs the LionMuon
alternation with SEPARATE per-rule momentum buffers (matching
src/odyssey/training/optimizers/lionmuon.mojo): Muon steps read/write only the
Muon heavy-ball buffer, Lion steps read/write only the Lion EMA buffer, and the
inactive buffer passes through unchanged.

Emits two sections:

1. "parity3" — three steps with period=4 (step 0: Muon, steps 1-2: Lion).
   Parameters and both momentum buffers after each step. These are the ref0 /
   ref1 / ref2 values asserted in test_lionmuon.mojo::test_parity_three_step.

2. "steady9" — nine steps with period=3 and a CONSTANT gradient (Muon at steps
   0, 3, 6; Lion elsewhere), covering >2 full periods so the Muon buffer has
   accumulated well past ||g|| (heavy-ball steady state ~ ||g||/(1-beta)).
   Parameters, both buffers, and each Lion step's update sign pattern after
   each step. Asserted in test_lionmuon.mojo::test_steady_state_two_periods —
   the buffer trajectories are what distinguish the two-buffer design from the
   old (unsound) shared-buffer design, which let Muon's large accumulator leak
   into Lion's sign updates.

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


def lionmuon_step(params, grad, lion_m, muon_m, lr, step_index, period=4):
    """Two-buffer LionMuon: each branch touches only its own buffer."""
    if step_index % period == 0:
        new_p, new_muon_m = muon_step(params, grad, muon_m, lr, beta=0.95, wd=0.0)
        return new_p, lion_m, new_muon_m, "muon"
    new_p, new_lion_m = lion_step(params, grad, lion_m, lr, beta1=0.9, beta2=0.99, wd=0.0)
    return new_p, new_lion_m, muon_m, "lion"


def run(n_steps, period):
    R, C = 3, 4
    W = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5
    G = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.05 - 0.3
    LM = np.zeros((R, C), dtype=np.float64)
    MM = np.zeros((R, C), dtype=np.float64)
    LR = 0.1

    steps_out = []
    for s in range(n_steps):
        W_prev = W
        W, LM, MM, branch = lionmuon_step(W, G, LM, MM, LR, s, period=period)
        entry = {
            "step": s,
            "branch": branch,
            "params": W.flatten().tolist(),
            "lion_momentum": LM.flatten().tolist(),
            "muon_momentum": MM.flatten().tolist(),
        }
        if branch == "lion":
            # Lion moves every element by exactly +/- lr; record the signs.
            entry["lion_update_signs"] = np.sign(W_prev - W).flatten().astype(int).tolist()
        steps_out.append(entry)
    return {"config": {"R": R, "C": C, "lr": LR, "period": period}, "steps": steps_out}


print(
    json.dumps(
        {
            "parity3": run(3, period=4),
            "steady9": run(9, period=3),
        },
        indent=2,
    )
)
