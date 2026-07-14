"""LSTM cell parity reference (torch.nn.LSTMCell).

Sets eight ramp weight matrices (Odyssey layout: each Linear W is (in, out)) into a
torch.nn.LSTMCell (which packs i, f, g, o gates into W_ih (4H, in) and W_hh (4H, H);
torch uses (out, in) layout, so each Odyssey (in, out) block is transposed before
being copied in). Prints the single-step (h', c') outputs for the Mojo LSTMCell test
to assert against. Config: input=3, hidden=4, batch=2, dtype=float64.

Run:
    python tests/odyssey/core/layers/parity_refs/lstm_parity_reference.py
"""

import json

import torch
import torch.nn as nn

IN, HID, B = 3, 4, 2


def ramp(rows, cols, scale, off):
    return torch.arange(rows * cols, dtype=torch.float64).reshape(rows, cols) * scale + off


# Odyssey-layout (in, out) weight blocks + biases for the eight projections.
Wii = ramp(IN, HID, 0.01, -0.05)
bii = torch.arange(HID, dtype=torch.float64) * 0.01 - 0.02
Wif = ramp(IN, HID, 0.011, -0.06)
bif = torch.arange(HID, dtype=torch.float64) * 0.012 - 0.03
Wig = ramp(IN, HID, 0.009, -0.04)
big = torch.arange(HID, dtype=torch.float64) * 0.008 - 0.01
Wio = ramp(IN, HID, 0.012, -0.055)
bio = torch.arange(HID, dtype=torch.float64) * 0.009 - 0.025
Whi = ramp(HID, HID, 0.007, -0.03)
bhi = torch.arange(HID, dtype=torch.float64) * 0.006 - 0.02
Whf = ramp(HID, HID, 0.008, -0.04)
bhf = torch.arange(HID, dtype=torch.float64) * 0.005 - 0.01
Whg = ramp(HID, HID, 0.006, -0.02)
bhg = torch.arange(HID, dtype=torch.float64) * 0.004 - 0.015
Who = ramp(HID, HID, 0.0075, -0.035)
bho = torch.arange(HID, dtype=torch.float64) * 0.0045 - 0.012

X = ramp(B, IN, 0.1, -0.2)
H = ramp(B, HID, 0.05, -0.1)
C = ramp(B, HID, 0.03, -0.06)

cell = nn.LSTMCell(IN, HID).double()
with torch.no_grad():
    # torch packs gates in [i, f, g, o] order; weight rows are (out, in), so each
    # Odyssey (in, out) block is transposed before concatenation.
    cell.weight_ih.copy_(torch.cat([Wii.T, Wif.T, Wig.T, Wio.T], dim=0))
    cell.weight_hh.copy_(torch.cat([Whi.T, Whf.T, Whg.T, Who.T], dim=0))
    cell.bias_ih.copy_(torch.cat([bii, bif, big, bio]))
    cell.bias_hh.copy_(torch.cat([bhi, bhf, bhg, bho]))
    h_out, c_out = cell(X, (H, C))

print(
    json.dumps(
        {
            "config": {"input_size": IN, "hidden_size": HID, "batch": B},
            "Wii": Wii.flatten().tolist(),
            "bii": bii.tolist(),
            "Wif": Wif.flatten().tolist(),
            "bif": bif.tolist(),
            "Wig": Wig.flatten().tolist(),
            "big": big.tolist(),
            "Wio": Wio.flatten().tolist(),
            "bio": bio.tolist(),
            "Whi": Whi.flatten().tolist(),
            "bhi": bhi.tolist(),
            "Whf": Whf.flatten().tolist(),
            "bhf": bhf.tolist(),
            "Whg": Whg.flatten().tolist(),
            "bhg": bhg.tolist(),
            "Who": Who.flatten().tolist(),
            "bho": bho.tolist(),
            "X": X.flatten().tolist(),
            "H": H.flatten().tolist(),
            "C": C.flatten().tolist(),
            "h_out": h_out.flatten().tolist(),
            "c_out": c_out.flatten().tolist(),
        },
        indent=2,
    )
)
