"""FeedForward (Transformer FFN) parity reference.

Computes the forward pass of a Transformer position-wise FFN
(Linear -> GELU(exact) -> Linear) in PyTorch with FIXED weights and a FIXED
input, and prints the flattened output so the Mojo FeedForward test can set the
same weights and assert equality.

Odyssey `Linear` uses y = x @ W + b with W of shape (in, out).
torch.nn.Linear uses y = x @ W_t^T + b with W_t of shape (out, in), so we set
torch's weight to the TRANSPOSE of the Odyssey weight to make the two identical.

Config: d_model=4, d_ff=8, batch=2, exact GELU (matches torch.nn.GELU()).
Weights are simple deterministic ramps so they transcribe cleanly into Mojo.
"""

import json

import torch
import torch.nn as nn

torch.manual_seed(0)
D_MODEL, D_FF, B = 4, 8, 2

# Deterministic ramp weights/inputs (Odyssey layout: W1 (d_model,d_ff), W2 (d_ff,d_model))
W1 = torch.arange(D_MODEL * D_FF, dtype=torch.float64).reshape(D_MODEL, D_FF) * 0.01 - 0.15
b1 = torch.arange(D_FF, dtype=torch.float64) * 0.02 - 0.08
W2 = torch.arange(D_FF * D_MODEL, dtype=torch.float64).reshape(D_FF, D_MODEL) * 0.005 - 0.08
b2 = torch.arange(D_MODEL, dtype=torch.float64) * 0.03 - 0.05
X = torch.arange(B * D_MODEL, dtype=torch.float64).reshape(B, D_MODEL) * 0.1 - 0.3

fc1 = nn.Linear(D_MODEL, D_FF).double()
fc2 = nn.Linear(D_FF, D_MODEL).double()
with torch.no_grad():
    fc1.weight.copy_(W1.T)  # torch stores (out, in) => transpose of Odyssey (in, out)
    fc1.bias.copy_(b1)
    fc2.weight.copy_(W2.T)
    fc2.bias.copy_(b2)

with torch.no_grad():
    hidden = fc1(X)
    activated = torch.nn.functional.gelu(hidden, approximate="none")  # exact GELU
    out = fc2(activated)

print(
    json.dumps(
        {
            "config": {"d_model": D_MODEL, "d_ff": D_FF, "batch": B},
            "W1": W1.flatten().tolist(),
            "b1": b1.tolist(),
            "W2": W2.flatten().tolist(),
            "b2": b2.tolist(),
            "X": X.flatten().tolist(),
            "out": out.flatten().tolist(),
        },
        indent=2,
    )
)
