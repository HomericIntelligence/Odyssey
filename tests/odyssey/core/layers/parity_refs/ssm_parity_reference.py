"""Diagonal state-space (S4-style) block parity reference.

Builds ramp-seeded parameters for a real-valued diagonal LTI SSM block and runs
the ground-truth recurrence in explicit torch ops (torch has no built-in S4), so
the Mojo `DiagonalSSM` layer can assert against it to 1e-5.

Model (per-feature independent SISO channels; batch-first [batch, seq, dim]):

    A       = -exp(A_log)                 # negative real diagonal, stable   [D, N]
    dt      =  exp(log_dt)                 # per-channel timestep             [D]
    dA      =  exp(dt * A)                 # ZOH discretized state matrix     [D, N]
    dB      =  (dA - 1) / A * B            # ZOH discretized input matrix     [D, N]
    x_t[d]  =  dA[d] * x_{t-1}[d] + dB[d] * u_t[d]           # state, [D, N]
    y_t[d]  =  sum_n C[d, n] * x_t[d, n] + D_skip[d] * u_t[d]

This is the S4 diagonal parameterization (Gu, Goel, Re 2022, arXiv:2111.00396,
Sec. 2-3: LTI x' = A x + B u, y = C x + D u with a diagonalizable structured A),
zero-order-hold discretized (their Eq. for the discrete-time SSM). We use a real
negative diagonal A (the stable diagonal-real variant, DSS/S4D style) rather than
the complex HiPPO-diagonal-plus-low-rank, which keeps the layer real-valued f32.

Numerics: the discretization (exp of A*dt, and the (dA-1)/A ZOH factor) is
accumulated here in float64 to define ground truth; the Mojo layer runs f32 and
the test tolerance is 1e-5, which the f32 recurrence meets at this tiny scale.

Config: dim=3, state=2, batch=2, seq=4, dtype=float64. Emits BOTH a single-step
output (seq index 0, from zero initial state) and the full multi-step sequence
(state carried across all 4 steps).

Run:
    python tests/odyssey/core/layers/parity_refs/ssm_parity_reference.py

Writes the JSON fixture next to this script (ssm_parity_reference.json) and also
prints it to stdout. Re-run and diff against the committed fixture before push.
"""

import json
import os

import torch

DIM, STATE, B, SEQ = 3, 2, 2, 4


def ramp(rows, cols, scale, off):
    return torch.arange(rows * cols, dtype=torch.float64).reshape(rows, cols) * scale + off


# --- ramp-seeded parameters (float64 ground truth) -----------------------------
# A_log [D, N]; A = -exp(A_log) is a stable negative-real diagonal.
A_log = ramp(DIM, STATE, 0.05, -0.30)
Bmat = ramp(DIM, STATE, 0.02, 0.10)  # input matrix   [D, N]
Cmat = ramp(DIM, STATE, 0.03, -0.15)  # output matrix  [D, N]
Dskip = torch.arange(DIM, dtype=torch.float64) * 0.04 - 0.05  # skip [D]
log_dt = torch.arange(DIM, dtype=torch.float64) * 0.10 - 0.20  # [D]

# Input sequence U [B, SEQ, D], ramp-seeded.
U = ramp(B, SEQ * DIM, 0.07, -0.25).reshape(B, SEQ, DIM)


def run_ssm(u):
    """Explicit diagonal-SSM recurrence in torch ops. u: [B, SEQ, D] -> y: same."""
    A = -torch.exp(A_log)  # [D, N]
    dt = torch.exp(log_dt)  # [D]
    dA = torch.exp(dt.unsqueeze(1) * A)  # [D, N]
    dB = (dA - 1.0) / A * Bmat  # [D, N]  (ZOH; A != 0 by construction)

    bsz, seqlen, dim = u.shape
    x = torch.zeros(bsz, dim, STATE, dtype=torch.float64)  # state [B, D, N]
    ys = []
    for t in range(seqlen):
        u_t = u[:, t, :]  # [B, D]
        # x_t = dA * x_{t-1} + dB * u_t   (broadcast dA,dB over batch; u over N)
        x = dA.unsqueeze(0) * x + dB.unsqueeze(0) * u_t.unsqueeze(2)  # [B, D, N]
        # y_t = sum_n C * x   +  Dskip * u_t
        y_t = (Cmat.unsqueeze(0) * x).sum(dim=2) + Dskip.unsqueeze(0) * u_t
        ys.append(y_t)
    return torch.stack(ys, dim=1)  # [B, SEQ, D]


Y = run_ssm(U)  # multi-step, state carried across all SEQ steps
# single step: first timestep only, from zero initial state
Y0 = Y[:, 0, :]  # [B, D]

fixture = {
    "config": {"dim": DIM, "state": STATE, "batch": B, "seq": SEQ},
    "discretization": "ZOH",
    "A_parameterization": "A = -exp(A_log) (stable negative-real diagonal)",
    "reference": "Gu, Goel, Re 2022, S4, arXiv:2111.00396",
    "A_log": A_log.flatten().tolist(),
    "B": Bmat.flatten().tolist(),
    "C": Cmat.flatten().tolist(),
    "D": Dskip.tolist(),
    "log_dt": log_dt.tolist(),
    "U": U.flatten().tolist(),
    "y_single_step": Y0.flatten().tolist(),  # [B, D]
    "y_sequence": Y.flatten().tolist(),  # [B, SEQ, D]
}

_out = json.dumps(fixture, indent=2)
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssm_parity_reference.json")
with open(_path, "w") as f:
    f.write(_out + "\n")
print(_out)
