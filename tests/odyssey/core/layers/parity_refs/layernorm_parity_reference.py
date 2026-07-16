"""LayerNorm parity reference (torch.nn.LayerNorm).

Normalizes a fixed ramp input over the last (feature) dimension with learnable
per-feature gamma/beta, matching torch.nn.LayerNorm(normalized_shape=features).
Two cases are emitted:

  - default affine (gamma=1, beta=0)
  - custom affine (gamma, beta set to fixed ramps)

Config: batch=2, features=4, epsilon=1e-5, dtype=float64.

Run:
    python tests/odyssey/core/layers/parity_refs/layernorm_parity_reference.py
"""

import json

import torch
import torch.nn as nn

B, F, EPS = 2, 4, 1e-5

X = torch.arange(B * F, dtype=torch.float64).reshape(B, F) * 0.3 - 1.0

# Default affine (gamma=1, beta=0).
ln_default = nn.LayerNorm(F, eps=EPS).double()
with torch.no_grad():
    out_default = ln_default(X)

# Custom affine.
gamma = torch.arange(F, dtype=torch.float64) * 0.5 + 0.75  # 0.75, 1.25, 1.75, 2.25
beta = torch.arange(F, dtype=torch.float64) * 0.1 - 0.15  # -0.15, -0.05, 0.05, 0.15
ln_custom = nn.LayerNorm(F, eps=EPS).double()
with torch.no_grad():
    ln_custom.weight.copy_(gamma)
    ln_custom.bias.copy_(beta)
    out_custom = ln_custom(X)

print(
    json.dumps(
        {
            "config": {"batch": B, "features": F, "epsilon": EPS},
            "X": X.flatten().tolist(),
            "gamma": gamma.tolist(),
            "beta": beta.tolist(),
            "out_default": out_default.flatten().tolist(),
            "out_custom": out_custom.flatten().tolist(),
        },
        indent=2,
    )
)
