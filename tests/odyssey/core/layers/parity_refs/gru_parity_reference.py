"""GRU cell parity reference (torch.nn.GRUCell).

Sets six ramp weight matrices (Odyssey layout: each Linear W is (in, out)) into a
torch.nn.GRUCell (which packs r, z, n gates into W_ih (3H, in) and W_hh (3H, H);
torch uses (out, in) layout, so each Odyssey (in, out) block is transposed before
being copied in). Prints the single-step output for the Mojo GRUCell test to
assert against. Config: input=3, hidden=4, batch=2, dtype=float64.

Run:
    python tests/odyssey/core/layers/parity_refs/gru_parity_reference.py
"""

import json

import torch
import torch.nn as nn

IN, HID, B = 3, 4, 2


def ramp(rows, cols, scale, off):
    return torch.arange(rows * cols, dtype=torch.float64).reshape(rows, cols) * scale + off


# Odyssey-layout (in, out) weight blocks + biases for the six projections.
Wir = ramp(IN, HID, 0.01, -0.05)
bir = torch.arange(HID, dtype=torch.float64) * 0.01 - 0.02
Wiz = ramp(IN, HID, 0.011, -0.06)
biz = torch.arange(HID, dtype=torch.float64) * 0.012 - 0.03
Win = ramp(IN, HID, 0.009, -0.04)
bin_ = torch.arange(HID, dtype=torch.float64) * 0.008 - 0.01
Whr = ramp(HID, HID, 0.007, -0.03)
bhr = torch.arange(HID, dtype=torch.float64) * 0.006 - 0.02
Whz = ramp(HID, HID, 0.008, -0.04)
bhz = torch.arange(HID, dtype=torch.float64) * 0.005 - 0.01
Whn = ramp(HID, HID, 0.006, -0.02)
bhn = torch.arange(HID, dtype=torch.float64) * 0.004 - 0.015

X = ramp(B, IN, 0.1, -0.2)
H = ramp(B, HID, 0.05, -0.1)

cell = nn.GRUCell(IN, HID).double()
with torch.no_grad():
    # torch packs gates in [r, z, n] order; weight rows are (out, in), so each
    # Odyssey (in, out) block is transposed before concatenation.
    cell.weight_ih.copy_(torch.cat([Wir.T, Wiz.T, Win.T], dim=0))
    cell.weight_hh.copy_(torch.cat([Whr.T, Whz.T, Whn.T], dim=0))
    cell.bias_ih.copy_(torch.cat([bir, biz, bin_]))
    cell.bias_hh.copy_(torch.cat([bhr, bhz, bhn]))
    out = cell(X, H)

print(
    json.dumps(
        {
            "config": {"input_size": IN, "hidden_size": HID, "batch": B},
            "Wir": Wir.flatten().tolist(),
            "bir": bir.tolist(),
            "Wiz": Wiz.flatten().tolist(),
            "biz": biz.tolist(),
            "Win": Win.flatten().tolist(),
            "bin": bin_.tolist(),
            "Whr": Whr.flatten().tolist(),
            "bhr": bhr.tolist(),
            "Whz": Whz.flatten().tolist(),
            "bhz": bhz.tolist(),
            "Whn": Whn.flatten().tolist(),
            "bhn": bhn.tolist(),
            "X": X.flatten().tolist(),
            "H": H.flatten().tolist(),
            "out": out.flatten().tolist(),
        },
        indent=2,
    )
)
