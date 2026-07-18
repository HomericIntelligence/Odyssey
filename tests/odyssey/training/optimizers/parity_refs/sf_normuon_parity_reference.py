"""SF-NorMuon parity reference (numpy transcription).

Transcribes Odyssey's SF-NorMuon core VERBATIM (matching
src/odyssey/training/optimizers/sf_normuon.mojo) and runs the schedule-free
y/z/x three-sequence recurrence combined with NorMuon-style row-wise-normalized
orthogonalized (polar / Newton-Schulz) updates. Emits the per-step z, x, y and
the polar factor P over SEVERAL steps on a small 4x3 weight so that all three
sequences, the orthogonalization, and the row-wise normalization evolve.

The Newton-Schulz iteration below (coefficients + 5-step count + Frobenius
pre-normalization + transpose-on-shorter-dim) is transcribed VERBATIM from
Odyssey's `newton_schulz_orthogonalize` (muon.mojo), which the Mojo SF-NorMuon
step reuses directly, so parity is exact by construction. The row-wise L2
normalization (axis=0, eps=1e-8) mirrors NorMuon's `_normalize_tensor_by_axis`.

Update rule (per step t, 1-indexed; issue mvillmow/Random#79):
    y_t     = beta*x_t + (1-beta)*z_t                (schedule-free query point)
    g_t     = grad(f, y_t)                            (caller supplies this)
    m_t     = mu*m_{t-1} + (1-mu)*g_t                 (momentum EMA)
    P_t     = rownorm( newton_schulz(m_t) )           (polar + row-wise norm)
    z_{t+1} = (1 - lambda_wd)*z_t - eta*P_t           (WEIGHT DECAY ON z_t)
    c_{t+1} = (r + 1)/(t + r + 1)                     (averaging weight)
    x_{t+1} = (1 - c_{t+1})*x_t + c_{t+1}*z_{t+1}      (running average)

On step 1, z_1 = x_1 = params (so y_1 = params) and m_0 = 0.

Run:
    python tests/odyssey/training/optimizers/parity_refs/sf_normuon_parity_reference.py
"""

import json

import numpy as np


def newton_schulz(X, steps=5):
    """Odyssey newton_schulz_orthogonalize (muon.mojo), transcribed verbatim."""
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


def row_normalize(delta, eps=1e-8):
    """NorMuon _normalize_tensor_by_axis(axis=0): divide each row by its L2 norm.

    Matches normuon.mojo: norm_i = sqrt(sum_j delta[i,j]^2 + eps).
    """
    out = np.zeros_like(delta)
    m, n = delta.shape
    for i in range(m):
        sum_sq = 0.0
        for j in range(n):
            sum_sq += delta[i, j] * delta[i, j]
        norm_val = np.sqrt(sum_sq + eps)
        out[i, :] = delta[i, :] / norm_val
    return out


def polar(m, ns_steps=5, eps=1e-8):
    """P = rownorm(newton_schulz(m)) — the NorMuon-style polar factor."""
    return row_normalize(newton_schulz(m, ns_steps), eps=eps)


def sf_normuon_step(z, x, m, grad_fn, step, eta, beta, mu, lam_wd, r=0.0):
    """One SF-NorMuon step. grad_fn(y) supplies g_t evaluated at the query point."""
    y = beta * x + (1.0 - beta) * z
    g = grad_fn(y)
    new_m = mu * m + (1.0 - mu) * g
    P = polar(new_m)
    new_z = (1.0 - lam_wd) * z - eta * P
    c = (r + 1.0) / (float(step) + r + 1.0)
    new_x = (1.0 - c) * x + c * new_z
    y_next = beta * new_x + (1.0 - beta) * new_z
    return new_z, new_x, new_m, y, P, y_next, c


def run(n_steps):
    R, C = 4, 3
    # params (== z_1 == x_1): a small ramp so rows are non-degenerate.
    P0 = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5
    # A fixed "gradient field": g(y) = 0.5*y + G_const, so g_t evolves with y_t
    # (exercises the y-sequence coupling) but stays deterministic.
    G_const = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.03 - 0.2

    def grad_fn(y):
        return 0.5 * y + G_const

    eta, beta, mu, lam_wd, r = 0.05, 0.9, 0.95, 0.01, 0.0

    z = P0.copy()
    x = P0.copy()
    m = np.zeros((R, C), dtype=np.float64)

    steps_out = []
    for s in range(1, n_steps + 1):
        z, x, m, y, P, y_next, c = sf_normuon_step(z, x, m, grad_fn, s, eta, beta, mu, lam_wd, r)
        steps_out.append(
            {
                "step": s,
                "c": c,
                "y": y.flatten().tolist(),
                "momentum": m.flatten().tolist(),
                "polar": P.flatten().tolist(),
                "z": z.flatten().tolist(),
                "x": x.flatten().tolist(),
                "y_next": y_next.flatten().tolist(),
            }
        )
    return {
        "config": {
            "R": R,
            "C": C,
            "eta": eta,
            "beta": beta,
            "mu": mu,
            "lambda_wd": lam_wd,
            "weight_power": r,
        },
        "steps": steps_out,
    }


print(json.dumps({"parity4": run(4)}, indent=2))
