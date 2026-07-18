"""SPlus parity reference (numpy transcription).

Transcribes the SPlus step for a 2-D weight exactly as the Mojo path computes it.
SPlus (Frans, Levine, Abbeel 2025 — "A Stable Whitening Optimizer for Efficient
Neural Network Training", arXiv:2506.07254) is a whitening optimizer in the
Shampoo/SOAP family. It fixes the long-cache divergence of naive Shampoo by
replacing the divide-by-sqrt-eigenvalue whitening with an ELEMENT-WISE SIGN of the
update projected into the historical eigenbasis (which bounds the max spectral
magnitude), adds shape-aware scaling for width-invariant learning-rate transfer,
and evaluates on an EMA (iterate-averaged) copy of the parameters while the live
parameters move at a high learning rate.

The update rule transcribed here follows the tracking issue (mvillmow/Random#76),
whose wording wins over any paper ambiguity:

    # Kronecker factors + their eigenbases (historical eigenbasis, as in SOAP):
    L        = EMA_b2(g gᵀ)            # R×R
    M        = EMA_b2(gᵀ g)            # C×C
    Q_L, Q_R = eig(L), eig(M)          # refreshed every `precondition_frequency`

    # Momentum on the RAW gradient (first moment), projected into the eigenbasis:
    m        = EMA_b1(g)               # R×C
    m'       = Q_Lᵀ m Q_R             # project first moment into eigenbasis

    # ELEMENT-WISE SIGN of the projected update (instead of / sqrt(eigenvalue)):
    u_proj   = sign(m')               # bounds max spectral magnitude
                                      # (deadzone: |m'| < sign_eps -> 0, so
                                      # numerically-zero eigen-directions map to 0
                                      # rather than an arbitrary +-1)
    u        = Q_L u_proj Q_Rᵀ        # rotate the signed update back

    # Shape-aware scaling by the layer's dimensional ratio (2/(R+C)):
    scale    = 2 / (R + C)

    # Live params at high LR; EMA sequence is what you evaluate on:
    θ        = θ - lr * scale * u - lr*wd*θ     # decoupled weight decay on live θ
    θ_ema    = ema_rate * θ_ema + (1-ema_rate) * θ

The eigenbases are refreshed on step 1 and every `precondition_frequency` steps
afterwards, matching SOAP. All matrix state is float64 (Odyssey's matmul raises on
mixed dtypes; the Mojo path up-casts to f64 and keeps every matrix in f64).

Runs THREE steps with precondition_frequency=100 (eigenbasis built once on step 1,
reused on steps 2 and 3) so the sign/EMA-of-iterates behaviour is exercised across
several steps of state evolution. Emits both the live params θ and the EMA params
θ_ema after each step.

**Fixture design (why a SQUARE, full-rank, direction-varying gradient).** The
Kronecker factors are `L = g gᵀ (R×R)` and `M = gᵀ g (C×C)`; if `R != C` the smaller
of the two is rank-deficient and its eigenbasis has an arbitrary null-space
orientation that differs between numpy's `eigh` and the Mojo Jacobi solver. Because
SPlus's `sign(m')` is a NONLINEAR op, that null-space ambiguity flips signs and
breaks cross-implementation parity. A square `R == C == 4` gradient of full rank 4
makes BOTH factors full-rank with distinct eigenvalues, so the eigenbasis is unique
up to per-column sign (which `sign(m')`'s rotate-back cancels). The gradient DIRECTION
also changes each step so the frozen step-1 eigenbasis no longer diagonalizes the
accumulated momentum — that keeps every projected entry well away from zero (min
|m'| ~ 4e-3 on steps 2–3), where `sign` is stable and implementation-independent.

Run:
    python tests/odyssey/training/optimizers/parity_refs/splus_parity_reference.py
"""

import json

import numpy as np


def eigh_flipped(gg):
    """Full symmetric eigendecomposition, columns flipped to descending eigenvalue."""
    _, q = np.linalg.eigh(gg)  # ascending
    return np.flip(q, axis=1)  # descending columns (torch.flip(dims=[1]))


def sign_deadzone(x, eps):
    """Element-wise sign with a deadzone: |x| < eps -> 0, else +-1.

    The deadzone suppresses the sign of numerically-zero projected components (the
    eigen-directions in which the momentum has no energy — e.g. every off-diagonal
    on step 1, where the freshly-built eigenbasis exactly diagonalizes the gradient).
    Without it, `sign` of a ~1e-18 value returns an arbitrary +-1 whose sign differs
    between eigensolvers, breaking cross-implementation parity. The genuine
    components here are ~2e-3 and larger, ~15 orders above eps, so the classification
    is unambiguous.
    """
    out = np.sign(x)
    out[np.abs(x) < eps] = 0.0
    return out


def splus_step(
    W,
    W_ema,
    g,
    exp_avg,
    gg_left,
    gg_right,
    q_left,
    q_right,
    step,
    lr,
    beta1=0.9,
    beta2=0.99,
    ema_rate=0.999,
    weight_decay=0.0,
    precondition_frequency=100,
    sign_eps=1e-12,
):
    # 1. Kronecker factors: L = EMA(g gᵀ), M = EMA(gᵀ g).
    new_gg_left = beta2 * gg_left + (1.0 - beta2) * (g @ g.T)
    new_gg_right = beta2 * gg_right + (1.0 - beta2) * (g.T @ g)

    # 2. Eigenbasis (build on step 1, refresh every precondition_frequency).
    new_q_left, new_q_right = q_left, q_right
    if step == 1 or step % precondition_frequency == 0:
        new_q_left = eigh_flipped(new_gg_left)
        new_q_right = eigh_flipped(new_gg_right)

    # 3. Momentum on the raw gradient (first moment).
    new_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g

    # 4. Project the first moment into the eigenbasis, then SIGN it.
    m_proj = new_q_left.T @ new_exp_avg @ new_q_right
    u_proj = sign_deadzone(m_proj, sign_eps)  # sign — bounds spectral magnitude

    # 5. Rotate the signed update back.
    u = new_q_left @ u_proj @ new_q_right.T

    # 6. Shape-aware scaling by the layer's dimensional ratio.
    R, C = W.shape
    scale = 2.0 / (R + C)

    # 7. Live params at high LR (decoupled weight decay), then EMA-average.
    new_W = W - lr * scale * u
    if weight_decay != 0.0:
        new_W = new_W - (lr * weight_decay) * W
    new_W_ema = ema_rate * W_ema + (1.0 - ema_rate) * new_W

    return (new_W, new_W_ema, new_exp_avg, new_gg_left, new_gg_right, new_q_left, new_q_right)


R, C = 4, 4
W = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5
LR = 0.1

# Full-rank, direction-VARYING gradients (a distinct rank-4 pattern per step). The
# changing direction keeps the frozen step-1 eigenbasis misaligned with the
# accumulated momentum, so every projected entry stays well away from zero (min
# |m'| ~ 4e-3 on steps 2-3) — the regime where sign() is stable across the numpy
# and Mojo eigensolvers. A degenerate/aligned gradient would put sign() on a zero.
GRADS = {
    1: np.array(
        [
            [0.9, -0.4, 0.2, -0.7],
            [-0.3, 0.8, -0.6, 0.1],
            [0.5, -0.2, 0.7, -0.9],
            [-0.8, 0.6, -0.1, 0.4],
        ]
    ),
    2: np.array(
        [
            [0.2, 0.7, -0.5, 0.3],
            [0.6, -0.1, 0.4, -0.8],
            [-0.7, 0.9, -0.3, 0.2],
            [0.4, -0.6, 0.8, -0.2],
        ]
    ),
    3: np.array(
        [
            [-0.5, 0.3, 0.9, -0.2],
            [0.8, -0.7, 0.1, 0.5],
            [-0.4, 0.6, -0.9, 0.7],
            [0.2, -0.3, 0.4, -0.6],
        ]
    ),
}

W_ema = W.copy()  # EMA sequence starts at the initial params
exp_avg = np.zeros((R, C))
gg_left = np.zeros((R, R))
gg_right = np.zeros((C, C))
q_left = np.zeros((R, R))
q_right = np.zeros((C, C))

steps_out = []
for s in range(1, 4):  # 1-indexed steps
    g = GRADS[s]
    (W, W_ema, exp_avg, gg_left, gg_right, q_left, q_right) = splus_step(
        W,
        W_ema,
        g,
        exp_avg,
        gg_left,
        gg_right,
        q_left,
        q_right,
        s,
        LR,
        precondition_frequency=100,
    )
    steps_out.append(
        {
            "step": s,
            "params": W.flatten().tolist(),
            "params_ema": W_ema.flatten().tolist(),
        }
    )

print(
    json.dumps(
        {
            "config": {
                "R": R,
                "C": C,
                "lr": LR,
                "beta1": 0.9,
                "beta2": 0.99,
                "ema_rate": 0.999,
                "precondition_frequency": 100,
            },
            "W0": (np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5).flatten().tolist(),
            "steps": steps_out,
        },
        indent=2,
    )
)
