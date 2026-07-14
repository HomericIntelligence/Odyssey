"""Vanilla RNN cell parity reference.

Computes h_t = tanh(x @ W_ih + b_ih + h @ W_hh + b_hh) in PyTorch with FIXED
ramp weights + input + hidden, and prints the flattened output so the Mojo
RNNCell test can set the same weights (Odyssey Linear layout (in, out); torch
nn.RNNCell uses (out, in), so torch weights are the transpose) and assert.

Config: input_size=3, hidden_size=4, batch=2.
"""

import json

import torch
import torch.nn as nn

torch.manual_seed(0)
IN, HID, B = 3, 4, 2

# Odyssey layout: W_ih (in, hid), W_hh (hid, hid); ramp values
W_ih = torch.arange(IN * HID, dtype=torch.float64).reshape(IN, HID) * 0.02 - 0.1
b_ih = torch.arange(HID, dtype=torch.float64) * 0.03 - 0.05
W_hh = torch.arange(HID * HID, dtype=torch.float64).reshape(HID, HID) * 0.01 - 0.06
b_hh = torch.arange(HID, dtype=torch.float64) * 0.02 - 0.03
X = torch.arange(B * IN, dtype=torch.float64).reshape(B, IN) * 0.1 - 0.2
H = torch.arange(B * HID, dtype=torch.float64).reshape(B, HID) * 0.05 - 0.1

cell = nn.RNNCell(IN, HID, nonlinearity="tanh").double()
with torch.no_grad():
    cell.weight_ih.copy_(W_ih.T)  # torch (out, in) = transpose of Odyssey (in, out)
    cell.bias_ih.copy_(b_ih)
    cell.weight_hh.copy_(W_hh.T)
    cell.bias_hh.copy_(b_hh)

with torch.no_grad():
    out = cell(X, H)

print(
    json.dumps(
        {
            "config": {"input_size": IN, "hidden_size": HID, "batch": B},
            "W_ih": W_ih.flatten().tolist(),
            "b_ih": b_ih.tolist(),
            "W_hh": W_hh.flatten().tolist(),
            "b_hh": b_hh.tolist(),
            "X": X.flatten().tolist(),
            "H": H.flatten().tolist(),
            "out": out.flatten().tolist(),
        },
        indent=2,
    )
)
