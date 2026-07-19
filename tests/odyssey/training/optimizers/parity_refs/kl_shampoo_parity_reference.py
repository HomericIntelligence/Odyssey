"""KL-Shampoo parity reference (numpy transcription).

Transcribes the idealized KL-Shampoo step (Lin et al. 2025, arXiv:2509.03378,
Eq. 5) for a 2-D weight exactly as the Mojo path computes it:

    S_A ← (1−β) S_A + (β / d_B) · G  S_B⁻¹ Gᵀ        (uses OLD S_B, d_B = C)
    S_B ← (1−β) S_B + (β / d_A) · Gᵀ S_A⁻¹ G         (uses OLD S_A, d_A = R)
    W   ← W − γ · S_A^{-1/2} G S_B^{-1/2}            (updated factors)
    W   ← W − (γ·wd) · W_pre                         (decoupled weight decay)

Both factors are initialized to the identity. Factor inverses and inverse-square-
roots are formed from a symmetric eigendecomposition with the SAME ridge floor the
Mojo path uses (Q diag(1/(λ+ridge)) Qᵀ and Q diag((λ+ridge)^{-1/2}) Qᵀ), so the two
implementations agree to eigensolver precision. This validates the Mojo arithmetic
(coupled cross-factor whitening, inverse-square-root preconditioning, weight decay,
and cross-step state threading), NOT any approximation.

Runs THREE steps (state accumulates across steps; the factors depart from the
identity so the inverse-square-root path is genuinely exercised). Emits params
after each step as JSON to stdout.

Run:
    python tests/odyssey/training/optimizers/parity_refs/kl_shampoo_parity_reference.py
"""

import json

import numpy as np


def sym_inv(s, ridge):
    """Q diag(1/(λ+ridge)) Qᵀ for a symmetric PSD matrix (matches Mojo _sym_inv)."""
    w, q = np.linalg.eigh(s)
    return q @ np.diag(1.0 / (w + ridge)) @ q.T


def sym_inv_sqrt(s, ridge):
    """Q diag((λ+ridge)^{-1/2}) Qᵀ (matches Mojo _sym_inv_sqrt)."""
    w, q = np.linalg.eigh(s)
    return q @ np.diag(1.0 / np.sqrt(w + ridge)) @ q.T


def kl_shampoo_step(W, g, s_a, s_b, lr, beta=0.95, weight_decay=0.0, ridge=1e-8):
    R, C = W.shape

    # Inverses of the OLD factors (coupled update reads pre-update state).
    s_a_inv = sym_inv(s_a, ridge)
    s_b_inv = sym_inv(s_b, ridge)

    # Factor updates (Eq. 5): whiten G by the CROSS factor's inverse.
    a_term = g @ s_b_inv @ g.T  # R×R
    new_s_a = (1.0 - beta) * s_a + (beta / C) * a_term
    b_term = g.T @ s_a_inv @ g  # C×C
    new_s_b = (1.0 - beta) * s_b + (beta / R) * b_term

    # Preconditioned step with the UPDATED factors.
    s_a_inv_sqrt = sym_inv_sqrt(new_s_a, ridge)
    s_b_inv_sqrt = sym_inv_sqrt(new_s_b, ridge)
    precond = s_a_inv_sqrt @ g @ s_b_inv_sqrt

    new_W = W - lr * precond
    if weight_decay != 0.0:
        new_W = new_W - (lr * weight_decay) * W

    return (new_W, new_s_a, new_s_b)


R, C = 3, 4
W = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5
LR = 0.1

s_a = np.eye(R, dtype=np.float64)
s_b = np.eye(C, dtype=np.float64)

steps_out = []
for s in range(1, 4):  # 1-indexed steps
    # Fixed but step-varying gradient so each step differs.
    g = (np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.05 - 0.3) + s * 0.01
    (W, s_a, s_b) = kl_shampoo_step(W, g, s_a, s_b, LR)
    steps_out.append({"step": s, "params": W.flatten().tolist()})

print(
    json.dumps(
        {
            "config": {"R": R, "C": C, "lr": LR, "beta": 0.95, "ridge": 1e-8},
            "W0": (np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5).flatten().tolist(),
            "steps": steps_out,
        },
        indent=2,
    )
)
