"""SOAP parity reference (numpy transcription).

Transcribes the SOAP step for a 2-D weight exactly as the Mojo path computes it:
Kronecker factors L=EMA(g gᵀ), M=EMA(gᵀ g); eigenbases Q_L, Q_R from a FULL symmetric
eigendecomposition (numpy.linalg.eigh, columns flipped to descending — matching the
Mojo `symmetric_eigh` + `_flip_columns`); Adam with the first moment on the raw grad
and the second moment on the projected grad squared; the rotated-Adam update projected
back; bias-corrected step size; decoupled weight decay.

Runs TWO steps with precondition_frequency=100 (so the eigenbasis is built once on
step 1 and reused on step 2) to exercise the projection path and cross-step state
threading. Emits params after each step.

Note on fidelity: pytorch_optimizer.SOAP refreshes the basis via power-iteration+QR;
both this reference AND the Mojo implementation use the exact full eigendecomposition,
so they agree with each other (this validates the Mojo arithmetic, not the QR approx).

Run:
    python tests/odyssey/training/optimizers/parity_refs/soap_parity_reference.py
"""

import json

import numpy as np


def eigh_flipped(gg):
    """Full symmetric eigendecomposition, columns flipped to descending eigenvalue."""
    _, q = np.linalg.eigh(gg)  # ascending
    return np.flip(q, axis=1)  # descending columns (torch.flip(dims=[1]))


def soap_step(
    W,
    g,
    exp_avg,
    exp_avg_sq,
    gg_left,
    gg_right,
    q_left,
    q_right,
    step,
    lr,
    beta1=0.95,
    beta2=0.95,
    shampoo_beta=0.95,
    weight_decay=0.01,
    precondition_frequency=100,
    eps=1e-8,
    correct_bias=True,
):
    # 1. Kronecker factors.
    new_gg_left = shampoo_beta * gg_left + (1.0 - shampoo_beta) * (g @ g.T)
    new_gg_right = shampoo_beta * gg_right + (1.0 - shampoo_beta) * (g.T @ g)

    # 2. Eigenbasis (build on step 1, refresh every precondition_frequency).
    new_q_left, new_q_right = q_left, q_right
    if step == 1 or step % precondition_frequency == 0:
        new_q_left = eigh_flipped(new_gg_left)
        new_q_right = eigh_flipped(new_gg_right)

    # 3. Project gradient.
    g_proj = new_q_left.T @ g @ new_q_right

    # 4. Adam moments (m on raw grad, v on projected grad squared).
    new_exp_avg = beta1 * exp_avg + (1.0 - beta1) * g
    new_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (g_proj * g_proj)

    # 5. Rotated Adam update, projected back.
    denom = np.sqrt(new_exp_avg_sq) + eps
    m_proj = new_q_left.T @ new_exp_avg @ new_q_right
    norm_grad = new_q_left @ (m_proj / denom) @ new_q_right.T

    # 6. Step size + decoupled weight decay.
    step_size = lr
    if correct_bias:
        bc1 = 1.0 - beta1**step
        bc2_sq = np.sqrt(1.0 - beta2**step)
        step_size = step_size * bc2_sq / bc1
    new_W = W - step_size * norm_grad
    if weight_decay != 0.0:
        new_W = new_W - (lr * weight_decay) * W

    return (new_W, new_exp_avg, new_exp_avg_sq, new_gg_left, new_gg_right, new_q_left, new_q_right)


R, C = 3, 4
W = np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5
LR = 0.1

exp_avg = np.zeros((R, C))
exp_avg_sq = np.zeros((R, C))
gg_left = np.zeros((R, R))
gg_right = np.zeros((C, C))
q_left = np.zeros((R, R))
q_right = np.zeros((C, C))

steps_out = []
for s in range(1, 3):  # 1-indexed steps
    # Fixed but step-varying gradient so step 2 differs from step 1.
    g = (np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.05 - 0.3) + s * 0.01
    (W, exp_avg, exp_avg_sq, gg_left, gg_right, q_left, q_right) = soap_step(
        W,
        g,
        exp_avg,
        exp_avg_sq,
        gg_left,
        gg_right,
        q_left,
        q_right,
        s,
        LR,
        precondition_frequency=100,
    )
    steps_out.append({"step": s, "params": W.flatten().tolist()})

print(
    json.dumps(
        {
            "config": {"R": R, "C": C, "lr": LR, "precondition_frequency": 100},
            "W0": (np.arange(R * C, dtype=np.float64).reshape(R, C) * 0.1 - 0.5).flatten().tolist(),
            "steps": steps_out,
        },
        indent=2,
    )
)
