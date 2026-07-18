"""MLP-Mixer block parity reference.

Computes the forward pass of ONE MLP-Mixer block (Tolstikhin et al. 2021,
arXiv:2105.01601, §2, Eq. 1-2) in PyTorch with FIXED ramp weights and a FIXED
input, and prints the flattened output as JSON so the Mojo MLPMixerBlock test can
set the same weights and assert equality.

Block transcribed (batch-first x of shape [B, S, D]):

    n1 = LayerNorm_D(x)                        # normalize over feature axis D
    t  = n1.transpose(1, 2)                    # [B, D, S]
    t  = fc_t2(gelu(fc_t1(t)))                 # token MLP: S -> Ht -> S
    u  = x + t.transpose(1, 2)                 # residual, back to [B, S, D]
    n2 = LayerNorm_D(u)                        # normalize over feature axis D
    y  = u + fc_c2(gelu(fc_c1(n2)))            # channel MLP: D -> Hc -> D, residual

Conventions matched to Odyssey:
  * Odyssey `Linear` uses y = x @ W + b with W of shape (in, out); torch.nn.Linear
    stores weight as (out, in), so we copy the TRANSPOSE of each Odyssey W.
  * GELU is exact (erf-based), i.e. torch.nn.functional.gelu(..., approximate="none"),
    matching Odyssey's default FeedForward activation.
  * LayerNorm epsilon = 1e-5, normalized over the last (feature) dim only — matches
    Odyssey layernorm.mojo default (biased/population variance, torch default).
  * gamma=1, beta=0 for both LayerNorms (Odyssey LayerNorm init), so they are
    identity-affine here and the parity isolates the mixing MLPs + normalization.

Config: batch=2, seq=4, dim=8, token_hidden=6, channel_hidden=16, float64.
Weights/inputs are deterministic ramps so they transcribe cleanly into Mojo.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
B, S, D = 2, 4, 8
HT, HC = 6, 16  # token_hidden, channel_hidden
EPS = 1e-5

# --- Deterministic ramp params (Odyssey layout: W (in, out)) ---
# Token MLP: fc_t1 (S, HT), fc_t2 (HT, S)
WT1 = torch.arange(S * HT, dtype=torch.float64).reshape(S, HT) * 0.01 - 0.1
bT1 = torch.arange(HT, dtype=torch.float64) * 0.02 - 0.05
WT2 = torch.arange(HT * S, dtype=torch.float64).reshape(HT, S) * 0.005 - 0.06
bT2 = torch.arange(S, dtype=torch.float64) * 0.03 - 0.04

# Channel MLP: fc_c1 (D, HC), fc_c2 (HC, D)
WC1 = torch.arange(D * HC, dtype=torch.float64).reshape(D, HC) * 0.002 - 0.12
bC1 = torch.arange(HC, dtype=torch.float64) * 0.01 - 0.07
WC2 = torch.arange(HC * D, dtype=torch.float64).reshape(HC, D) * 0.003 - 0.09
bC2 = torch.arange(D, dtype=torch.float64) * 0.02 - 0.03

# Input [B, S, D]
X = torch.arange(B * S * D, dtype=torch.float64).reshape(B, S, D) * 0.01 - 0.15

# --- Build torch modules (weights = transpose of Odyssey layout) ---
ln1 = nn.LayerNorm(D, eps=EPS).double()  # gamma=1, beta=0 by default
ln2 = nn.LayerNorm(D, eps=EPS).double()
fc_t1 = nn.Linear(S, HT).double()
fc_t2 = nn.Linear(HT, S).double()
fc_c1 = nn.Linear(D, HC).double()
fc_c2 = nn.Linear(HC, D).double()

with torch.no_grad():
    fc_t1.weight.copy_(WT1.T)
    fc_t1.bias.copy_(bT1)
    fc_t2.weight.copy_(WT2.T)
    fc_t2.bias.copy_(bT2)
    fc_c1.weight.copy_(WC1.T)
    fc_c1.bias.copy_(bC1)
    fc_c2.weight.copy_(WC2.T)
    fc_c2.bias.copy_(bC2)

with torch.no_grad():
    # Token-mixing sublayer
    n1 = ln1(X)  # [B, S, D], normalized over D
    t = n1.transpose(1, 2)  # [B, D, S]
    t = fc_t2(F.gelu(fc_t1(t), approximate="none"))  # [B, D, S]
    u = X + t.transpose(1, 2)  # [B, S, D]
    # Channel-mixing sublayer
    n2 = ln2(u)  # [B, S, D]
    y = u + fc_c2(F.gelu(fc_c1(n2), approximate="none"))  # [B, S, D]

print(
    json.dumps(
        {
            "config": {
                "batch": B,
                "seq": S,
                "dim": D,
                "token_hidden": HT,
                "channel_hidden": HC,
                "eps": EPS,
            },
            "WT1": WT1.flatten().tolist(),
            "bT1": bT1.tolist(),
            "WT2": WT2.flatten().tolist(),
            "bT2": bT2.tolist(),
            "WC1": WC1.flatten().tolist(),
            "bC1": bC1.tolist(),
            "WC2": WC2.flatten().tolist(),
            "bC2": bC2.tolist(),
            "X": X.flatten().tolist(),
            "out": y.flatten().tolist(),
        },
        indent=2,
    )
)
