"""FTRL-Proximal parity reference: an independent numpy transcription of McMahan
et al. 2013 Algorithm 1 (Per-Coordinate FTRL-Proximal with L1 and L2), run for
TWO steps on fixed inputs, printing the resulting weights so the Mojo `ftrl_step`
can be compared against identical inputs.

FTRL is not in torch.optim, and its update is an exact closed-form algorithm, so
the numpy transcription of Algorithm 1 IS the authoritative reference (no
framework dependency). Step 1 seeds the (z, n) state from zero; step 2 is the
general step. Both steps are compared in the Mojo test.

Run:  python tests/odyssey/training/optimizers/parity_refs/ftrl_parity_reference.py
"""

import json

import numpy as np

# Fixed deterministic inputs (replayed exactly in the Mojo test).
N = 6
params0 = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60], dtype=np.float64)
grad_a = np.array([0.05, 0.15, -0.25, 0.35, -0.45, 0.55], dtype=np.float64)  # step 1
grad_b = np.array([-0.02, 0.08, 0.12, -0.18, 0.22, -0.28], dtype=np.float64)  # step 2

ALPHA, BETA, L1, L2, LR = 0.1, 1.0, 0.02, 0.01, 1.0


def ftrl_step(w, g, z, n, alpha=ALPHA, beta=BETA, l1=L1, l2=L2, lr=LR):
    """One per-coordinate FTRL-Proximal step (McMahan 2013, Algorithm 1)."""
    n_new = n + g * g
    sigma = (np.sqrt(n_new) - np.sqrt(n)) / alpha
    z_new = z + g - sigma * w
    # Closed-form weight solve. max(|z| - l1, 0) is exactly 0 on the
    # |z| <= l1 branch, so the piecewise "w = 0 if |z| <= l1" rule needs no
    # control flow (produces exact zeros — the FTRL sparsity property).
    shrunk = np.maximum(np.abs(z_new) - l1, 0.0)
    numer = -np.sign(z_new) * shrunk
    denom = (beta + np.sqrt(n_new)) / alpha + l2
    w_new = numer / denom
    if lr != 1.0:
        w_new = lr * w_new
    return w_new, z_new, n_new


def main():
    z = np.zeros(N)
    n = np.zeros(N)
    w1, z, n = ftrl_step(params0, grad_a, z, n)
    w2, z2, n2 = ftrl_step(w1, grad_b, z, n)
    out = {
        "N": N,
        "params0": params0.tolist(),
        "grad_a": grad_a.tolist(),
        "grad_b": grad_b.tolist(),
        "alpha": ALPHA,
        "beta": BETA,
        "lambda1": L1,
        "lambda2": L2,
        "lr": LR,
        "w_step1": [round(x, 10) for x in w1.tolist()],
        "w_step2": [round(x, 10) for x in w2.tolist()],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
