"""LTC cell parity reference (Liquid Time-constant Networks, arXiv:2006.04439).

Hand-rolled numpy transcription of Odyssey's `LTCCell` fused ODE solver (no
`ncps` package assumed). The solver and gating network are transcribed EXACTLY
so parity with the Mojo cell is exact-by-construction:

    f(x, I)     = tanh(I @ Wi + bi + x @ Wh)          # single bias mu = bi
    sub-step dt = elapsed / L
    x <- (x + dt * f * A) / (1 + dt * (1/tau + f))    # L fused sub-steps

Odyssey Linear uses W of shape (in, out) and applies y = x @ W + b directly, so
NO transpose is needed here (unlike the packed torch GRU/LSTM refs). The
recurrent projection bias is zero (LTC has a single bias mu, carried by Wi).

Emits BOTH a single-step output and a multi-step sequence (3 steps with hidden
carried across steps) so the Mojo test asserts single-step AND sequence-carry.

Config: input=3, hidden=4, batch=2, solver_steps L=6, elapsed=1.0, dtype=float64.

Run:
    python tests/odyssey/core/layers/parity_refs/ltc_parity_reference.py
"""

import json

import numpy as np

IN, HID, B = 3, 4, 2
L = 6
ELAPSED = 1.0


def ramp(rows, cols, scale, off):
    return np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * scale + off


def ramp_vec(n, scale, off):
    return np.arange(n, dtype=np.float64) * scale + off


# Odyssey-layout (in, out) weights + biases. Recurrent bias is zero (single mu).
Wi = ramp(IN, HID, 0.01, -0.05)  # input->hidden gamma
bi = ramp_vec(HID, 0.01, -0.02)  # mu (single gating bias)
Wh = ramp(HID, HID, 0.007, -0.03)  # hidden->hidden gamma_r
tau = ramp_vec(HID, 0.05, 0.8)  # time constants (all > 0)
A = ramp_vec(HID, 0.03, -0.05)  # equilibrium bias A


def f_gate(x, inp):  # inp = input current I(t) in the paper
    return np.tanh(inp @ Wi + bi + x @ Wh)


def ltc_step(x, inp):
    dt = ELAPSED / L
    inv_tau = 1.0 / tau
    for _ in range(L):
        f = f_gate(x, inp)
        num = x + dt * (f * A)
        denom = 1.0 + dt * (inv_tau + f)
        x = num / denom
    return x


# --- single step from a given hidden state ---
X1 = ramp(B, IN, 0.1, -0.2)
H0 = ramp(B, HID, 0.05, -0.1)
out_single = ltc_step(H0, X1)

# --- multi-step sequence (3 steps), hidden carried across steps ---
seq_inputs = [
    ramp(B, IN, 0.1, -0.2),
    ramp(B, IN, 0.05, 0.1),
    ramp(B, IN, -0.03, 0.2),
]
h = np.zeros((B, HID), dtype=np.float64)
seq_outs = []
for xt in seq_inputs:
    h = ltc_step(h, xt)
    seq_outs.append(h.copy())

print(
    json.dumps(
        {
            "config": {
                "input_size": IN,
                "hidden_size": HID,
                "batch": B,
                "solver_steps": L,
                "elapsed": ELAPSED,
            },
            "Wi": Wi.flatten().tolist(),
            "bi": bi.tolist(),
            "Wh": Wh.flatten().tolist(),
            "tau": tau.tolist(),
            "A": A.tolist(),
            "X1": X1.flatten().tolist(),
            "H0": H0.flatten().tolist(),
            "out_single": out_single.flatten().tolist(),
            "seq_in_0": seq_inputs[0].flatten().tolist(),
            "seq_in_1": seq_inputs[1].flatten().tolist(),
            "seq_in_2": seq_inputs[2].flatten().tolist(),
            "seq_out_0": seq_outs[0].flatten().tolist(),
            "seq_out_1": seq_outs[1].flatten().tolist(),
            "seq_out_2": seq_outs[2].flatten().tolist(),
        },
        indent=2,
    )
)
