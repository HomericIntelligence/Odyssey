"""Permutation-equivariant (Deep Sets) block parity reference.

Computes the forward pass of a single permutation-equivariant linear layer
(Zaheer et al. 2017, "Deep Sets", arXiv:1703.06114, equivariant layer, Eq. in
Sec. "Permutation Equivariance") in PyTorch with FIXED ramp weights and a FIXED
input, and prints the flattened output so the Mojo DeepSetsEquivariant test can
set the same weights and assert equality.

Layer (issue mvillmow/Random#64, ARCH-14 variant — SUM pool, ReLU activation):

    y_i = relu( x_i @ Lambda  +  (sum_j x_j) @ Gamma  +  b )

with the set axis being axis 1 of a batch-first [batch, set_size, dim] input.
`Lambda` and `Gamma` are both (dim, out) in Odyssey (in, out) layout; the pooled
term (sum over the set axis) is broadcast back across every set element before
the elementwise transform, which is exactly what makes the map equivariant:
permuting the set elements of `x` permutes the rows of `y` identically, because
the pooled term is permutation-INVARIANT.

Config: batch=2, set_size=4, dim=6, out=5, dtype=float64. Ramp weights are
simple deterministic sequences so they transcribe cleanly into the Mojo test.

Run:
    python tests/odyssey/core/layers/parity_refs/deepsets_parity_reference.py
"""

import json

import torch

B, S, D, OUT = 2, 4, 6, 5

# Odyssey-layout (in, out) ramp weight matrices + bias, and a ramp input.
Lam = torch.arange(D * OUT, dtype=torch.float64).reshape(D, OUT) * 0.01 - 0.15
Gam = torch.arange(D * OUT, dtype=torch.float64).reshape(D, OUT) * 0.007 - 0.10
b = torch.arange(OUT, dtype=torch.float64) * 0.02 - 0.04
X = torch.arange(B * S * D, dtype=torch.float64).reshape(B, S, D) * 0.01 - 0.2

with torch.no_grad():
    # Per-element transform: [B, S, D] @ [D, OUT] -> [B, S, OUT]
    per_elem = X @ Lam
    # Permutation-INVARIANT pooled term: sum over the set axis (axis=1), then
    # the pooled [B, D] is projected and broadcast back over the set axis.
    pooled = X.sum(dim=1)  # [B, D]
    pooled_proj = pooled @ Gam  # [B, OUT]
    out = torch.relu(per_elem + pooled_proj.unsqueeze(1) + b)  # [B, S, OUT]

print(
    json.dumps(
        {
            "config": {"batch": B, "set_size": S, "dim": D, "out": OUT},
            "Lambda": Lam.flatten().tolist(),
            "Gamma": Gam.flatten().tolist(),
            "bias": b.tolist(),
            "X": X.flatten().tolist(),
            "out": out.flatten().tolist(),
        },
        indent=2,
    )
)
