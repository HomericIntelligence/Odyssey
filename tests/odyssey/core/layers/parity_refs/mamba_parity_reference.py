"""Mamba selective-SSM (S6) block parity reference.

Builds ramp-seeded parameters for a single selective state-space (S6) Mamba block
and runs the ground-truth selective scan in explicit torch ops (no `mamba-ssm`
package dependency), so the Mojo `MambaBlock` layer can assert against it to 1e-5.

Reference transcribed from Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces" (arXiv:2312.00752), Sec. 3.2, Algorithm 2 (SSM +
Selection). We implement the *selective* SSM recurrence explicitly; we do NOT use
the fused hardware-aware CUDA scan (that only changes the compute schedule, not
the mathematical result), so this recurrence IS the ground truth for both.

Model (batch-first [batch, seq, dim]; input dim == output dim == D):

    # --- gate branch ---
    z      = u @ Wz + bz                                  # [B, L, D]  gate pre-act
    # --- causal depthwise conv over the x-branch (x-branch = u itself) ---
    xc[b,t,d] = SiLU( sum_{j=0..K-1} conv_w[d,j] * u_pad[b, t+j, d] + conv_b[d] )
                with u_pad left-padded by (K-1) zeros (CAUSAL)          # [B, L, D]
    # --- input-dependent (selective) SSM parameters from the conv'd x ---
    Bmat   = xc @ WB                                      # [B, L, N]  (shared over D)
    Cmat   = xc @ WC                                      # [B, L, N]  (shared over D)
    delta  = softplus( xc @ Wdt + dt_bias )              # [B, L, D]  per-channel Δ
    A      = -exp(A_log)                                  # [D, N] stable neg diagonal
    # --- discretize (Mamba simplified ZOH) + selective scan ---
    dA[b,t,d,n] = exp( delta[b,t,d] * A[d,n] )
    dB[b,t,d,n] = delta[b,t,d] * Bmat[b,t,n]
    h[b,t,d,n]  = dA * h[b,t-1,d,n] + dB * xc[b,t,d]
    y[b,t,d]    = sum_n Cmat[b,t,n] * h[b,t,d,n] + D_skip[d] * xc[b,t,d]
    # --- gated output projection ---
    y_gated = y * SiLU(z)                                 # [B, L, D]
    out     = y_gated @ Wo + bo                           # [B, L, D]

This is the selective S6 form: B, C, and Δ are functions of the input (Gu & Dao
2023, Sec. 3.1 "Selection"), unlike the LTI S4 sibling (`ssm.mojo`) whose B/C/Δ
are fixed parameters. A stays a fixed diagonal (Gu & Dao keep A input-independent;
selection enters via Δ, B, C — Sec. 3.2, Algorithm 2). We use the simplified
`dB = Δ·B` discretization from the official implementation (the reference impl's
`dB = Δ ⊗ B`, an Euler / first-order-ZOH approximation), NOT the `(dA-1)/A·B`
S4 factor, which is the intentional S6-vs-S4 discretization difference.

Numerics: ground truth accumulated in float64; the Mojo layer runs f32 and the
test tolerance is 1e-5, met at this tiny scale. `softplus` and `exp` here operate
on O(1) ramp-seeded inputs (see the magnitude note in the Mojo docstring): exp of
`delta*A` with delta,|A| in ~(0, 2) stays in (0, 1]; softplus argument stays O(1),
well clear of the f32 exp-overflow ceiling (~89).

Config: dim(D)=4, state(N)=3, conv_kernel(K)=3, batch=2, seq=4, dtype=float64.
Emits BOTH a single-step output (seq index 0) and the full multi-step sequence
(state carried across all 4 steps). NOTE: because B/C/Δ and the causal conv are
input-dependent, the "single step" here is timestep t=0 of the full forward
(conv left-pad makes t=0 well-defined), not an independent RNN-cell call.

Run:
    python tests/odyssey/core/layers/parity_refs/mamba_parity_reference.py

Writes the JSON fixture next to this script (mamba_parity_reference.json) and also
prints it to stdout. Re-run and diff against the committed fixture before push.
"""

import json
import os

import torch

DIM, STATE, KERNEL, B, SEQ = 4, 3, 3, 2, 4


def ramp(rows, cols, scale, off):
    return torch.arange(rows * cols, dtype=torch.float64).reshape(rows, cols) * scale + off


def silu(x):
    return x * torch.sigmoid(x)


def softplus(x):
    # log(1 + exp(x)), numerically stable: max(0,x) + log1p(exp(-|x|))
    return torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-torch.abs(x)))


# --- ramp-seeded parameters (float64 ground truth) -----------------------------
# Gate-branch projection Wz [D, D], bz [D].
Wz = ramp(DIM, DIM, 0.011, -0.07)
bz = torch.arange(DIM, dtype=torch.float64) * 0.03 - 0.04
# Causal depthwise conv: per-channel kernel [D, K] and bias [D].
conv_w = ramp(DIM, KERNEL, 0.02, 0.05)
conv_b = torch.arange(DIM, dtype=torch.float64) * 0.01 - 0.02
# Selective input->B and input->C projections [D, N] (map conv'd x -> B/C).
WB = ramp(DIM, STATE, 0.013, -0.06)
WC = ramp(DIM, STATE, 0.017, -0.05)
# Selective Δ projection [D, D] and dt_bias [D].
Wdt = ramp(DIM, DIM, 0.009, -0.03)
dt_bias = torch.arange(DIM, dtype=torch.float64) * 0.02 - 0.10
# A_log [D, N]; A = -exp(A_log) stable negative-real diagonal.
A_log = ramp(DIM, STATE, 0.05, -0.30)
# Direct skip D_skip [D].
Dskip = torch.arange(DIM, dtype=torch.float64) * 0.04 - 0.05
# Output projection Wo [D, D], bo [D].
Wo = ramp(DIM, DIM, 0.015, -0.08)
bo = torch.arange(DIM, dtype=torch.float64) * 0.025 - 0.03

# Input sequence U [B, SEQ, D], ramp-seeded.
U = ramp(B, SEQ * DIM, 0.07, -0.25).reshape(B, SEQ, DIM)


def run_mamba(u):
    """Explicit selective-SSM (S6) forward in torch ops. u:[B,L,D]->y:[B,L,D]."""
    bsz, seqlen, dim = u.shape

    # gate branch z = u @ Wz + bz
    z = torch.einsum("bld,de->ble", u, Wz) + bz  # [B, L, D]

    # causal depthwise conv over the sequence, per channel, then SiLU.
    # u_pad left-padded by (K-1) zeros so output[t] depends on inputs <= t.
    u_pad = torch.cat([torch.zeros(bsz, KERNEL - 1, dim, dtype=torch.float64), u], dim=1)  # [B, L+K-1, D]
    xc = torch.zeros(bsz, seqlen, dim, dtype=torch.float64)
    for t in range(seqlen):
        acc = torch.zeros(bsz, dim, dtype=torch.float64)
        for j in range(KERNEL):
            # conv_w[d, j] multiplies u_pad[:, t+j, :] (causal window ending at t)
            acc = acc + conv_w[:, j].unsqueeze(0) * u_pad[:, t + j, :]
        acc = acc + conv_b.unsqueeze(0)
        xc[:, t, :] = silu(acc)

    # selective params from conv'd x
    Bmat = torch.einsum("bld,dn->bln", xc, WB)  # [B, L, N]
    Cmat = torch.einsum("bld,dn->bln", xc, WC)  # [B, L, N]
    delta = softplus(torch.einsum("bld,de->ble", xc, Wdt) + dt_bias)  # [B, L, D]

    A = -torch.exp(A_log)  # [D, N]

    # selective scan
    h = torch.zeros(bsz, dim, STATE, dtype=torch.float64)  # state [B, D, N]
    ys = []
    for t in range(seqlen):
        delta_t = delta[:, t, :]  # [B, D]
        # dA[b,d,n] = exp(delta[b,d] * A[d,n])
        dA = torch.exp(delta_t.unsqueeze(2) * A.unsqueeze(0))  # [B, D, N]
        # dB[b,d,n] = delta[b,d] * B[b,n]     (Mamba simplified ZOH)
        dB = delta_t.unsqueeze(2) * Bmat[:, t, :].unsqueeze(1)  # [B, D, N]
        xt = xc[:, t, :]  # [B, D]
        h = dA * h + dB * xt.unsqueeze(2)  # [B, D, N]
        # y_t = sum_n C * h + Dskip * x
        y_t = (Cmat[:, t, :].unsqueeze(1) * h).sum(dim=2) + Dskip.unsqueeze(0) * xt
        ys.append(y_t)
    y = torch.stack(ys, dim=1)  # [B, L, D]

    # gated output projection
    y_gated = y * silu(z)  # [B, L, D]
    out = torch.einsum("bld,de->ble", y_gated, Wo) + bo  # [B, L, D]
    return out


Y = run_mamba(U)  # multi-step, state carried across all SEQ steps
Y0 = Y[:, 0, :]  # single step: first timestep only

fixture = {
    "config": {
        "dim": DIM,
        "state": STATE,
        "conv_kernel": KERNEL,
        "batch": B,
        "seq": SEQ,
    },
    "variant": "selective S6 (input-dependent B/C/Delta, causal depthwise conv, SiLU gate, diagonal A)",
    "discretization": "Mamba simplified ZOH: dA=exp(delta*A), dB=delta*B",
    "A_parameterization": "A = -exp(A_log) (stable negative-real diagonal)",
    "reference": "Gu & Dao 2023, Mamba, arXiv:2312.00752, Sec. 3.2 Algorithm 2",
    "Wz": Wz.flatten().tolist(),
    "bz": bz.tolist(),
    "conv_w": conv_w.flatten().tolist(),
    "conv_b": conv_b.tolist(),
    "WB": WB.flatten().tolist(),
    "WC": WC.flatten().tolist(),
    "Wdt": Wdt.flatten().tolist(),
    "dt_bias": dt_bias.tolist(),
    "A_log": A_log.flatten().tolist(),
    "D": Dskip.tolist(),
    "Wo": Wo.flatten().tolist(),
    "bo": bo.tolist(),
    "U": U.flatten().tolist(),
    "y_single_step": Y0.flatten().tolist(),  # [B, D]
    "y_sequence": Y.flatten().tolist(),  # [B, SEQ, D]
}

_out = json.dumps(fixture, indent=2)
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamba_parity_reference.json")
with open(_path, "w") as f:
    f.write(_out + "\n")
print(_out)
