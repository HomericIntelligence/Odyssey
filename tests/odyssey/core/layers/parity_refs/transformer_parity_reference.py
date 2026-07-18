"""Pre-LN Transformer encoder block parity reference.

Computes the forward pass of ONE pre-LN Transformer encoder block (Vaswani et al.
2017, arXiv:1706.03762, §3.1-3.3; pre-norm placement per Xiong et al. 2020,
arXiv:2002.04745) in PyTorch with FIXED ramp weights and a FIXED batch-first
input, and prints the flattened output as JSON so the Mojo
`TransformerEncoderBlock` test can set the same weights and assert equality to
1e-5.

Block transcribed (batch-first x of shape [B, S, d_model]):

    u = x + Attention( LayerNorm_1(x) )     # self-attention sublayer
    y = u + FFN(       LayerNorm_2(u) )      # position-wise FFN sublayer

where Attention is explicit scaled dot-product self-attention
(softmax(QKᵀ/√d_k)V with the SAME QKV/output projections and −∞ causal mask as
the attention parity reference) and FFN is fc2(gelu(fc1(·))) with exact (erf)
GELU — matching Odyssey's `FeedForward` default.

Conventions matched to Odyssey:
  * Odyssey `Linear` uses y = x @ W + b with W of shape (in, out); torch.nn.Linear
    stores weight as (out, in), so every torch weight is set to the TRANSPOSE of
    the Odyssey ramp weight.
  * GELU is exact (erf-based), F.gelu(..., approximate="none") — Odyssey
    FeedForward default (use_gelu=True).
  * LayerNorm eps = 1e-5, normalized over the last (feature) dim only, with
    gamma=1/beta=0 (Odyssey LayerNorm init) — biased/population variance, torch
    default. The two LayerNorms are therefore identity-affine here, so the parity
    isolates the mixing (attention + FFN) plus normalization.
  * Attention −∞ causal mask via torch.triu(diagonal=1).masked_fill(-inf),
    matching the attention layer's true-−∞ additive convention.

Two fixtures are emitted (mirroring the attention reference):
  - no_mask (num_heads=1, causal=False)
  - causal  (num_heads=2, causal=True)

Both use d_model=4, seq=3, batch=2, d_ff=8, float64. Weights are deterministic
ramps; the INPUT is a ramp plus a per-position quadratic curvature term (see
`build_input`) so the sequence positions are NOT affine-equivalent — otherwise
LayerNorm collapses every position to one vector and the causal mask has no
observable effect. Weights are seeded in
the SAME flat order the Mojo test uses:
  norm1 (gamma, beta),
  attn  (q.w, q.b, k.w, k.b, v.w, v.b, out.w, out.b),
  norm2 (gamma, beta),
  ffn   (fc1.w, fc1.b, fc2.w, fc2.b),
then the input X.

Regenerate + diff before push:
    /tmp/claude-1000/venvs/torch_venv/bin/python \
        tests/odyssey/core/layers/parity_refs/transformer_parity_reference.py \
        > tests/odyssey/core/layers/parity_refs/transformer_parity_reference.json
    git diff --exit-code \
        tests/odyssey/core/layers/parity_refs/transformer_parity_reference.json
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

D_MODEL, SEQ, B, D_FF = 4, 3, 2, 8
EPS = 1e-5


def ramp(n, scale, off):
    """value[i] = i * scale + off, as a float64 tensor of length n."""
    return torch.arange(n, dtype=torch.float64) * scale + off


def build_attn_projections():
    """Four torch Linear projections (q/k/v/out) with ramp weights.

    Weights are the transpose of the Odyssey (in, out) ramp layout. Returns the
    torch layers plus the flat Odyssey-order ramp lists for the JSON fixture.
    Ramp specs match the attention parity reference so the two are cross-checkable.
    """
    # Larger, more-varied ramps than the attention reference: with the pre-norm
    # LayerNorm normalizing every position to comparable scale, small QKV weights
    # yield near-uniform softmax weights and the causal mask barely moves the
    # output. These scales spread the attention scores across keys so the causal
    # fixture visibly differs from the no-mask fixture (verified in the test).
    specs = {
        "q": (0.11, -0.7, 0.05, -0.2),
        "k": (0.13, -0.6, 0.04, -0.15),
        "v": (0.09, -0.5, 0.06, -0.25),
        "out": (0.07, -0.4, 0.05, -0.1),
    }
    layers = {}
    flat = {}
    for name, (ws, wo, bs, bo) in specs.items():
        w = ramp(D_MODEL * D_MODEL, ws, wo).reshape(D_MODEL, D_MODEL)
        b = ramp(D_MODEL, bs, bo)
        lin = nn.Linear(D_MODEL, D_MODEL).double()
        with torch.no_grad():
            lin.weight.copy_(w.T)  # torch stores (out, in) => transpose
            lin.bias.copy_(b)
        layers[name] = lin
        flat[name + "_w"] = w.flatten().tolist()
        flat[name + "_b"] = b.tolist()
    return layers, flat


def attention_forward(layers, x, num_heads, causal):
    """Explicit scaled dot-product self-attention, batch-first [B, S, d_model]."""
    q = layers["q"](x)
    k = layers["k"](x)
    v = layers["v"](x)
    d_k = D_MODEL // num_heads

    def split(t):
        return t.reshape(B, SEQ, num_heads, d_k).permute(0, 2, 1, 3)

    qh, kh, vh = split(q), split(k), split(v)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(d_k)
    if causal:
        mask = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, vh)
    merged = context.permute(0, 2, 1, 3).reshape(B, SEQ, D_MODEL)
    return layers["out"](merged)


def build_ffn():
    """FFN as fc2(gelu(fc1(·))): fc1 (d_model, d_ff), fc2 (d_ff, d_model).

    Returns torch layers (weights = transpose of Odyssey layout) + flat ramps.
    """
    w1 = ramp(D_MODEL * D_FF, 0.007, -0.09).reshape(D_MODEL, D_FF)
    b1 = ramp(D_FF, 0.01, -0.04)
    w2 = ramp(D_FF * D_MODEL, 0.004, -0.07).reshape(D_FF, D_MODEL)
    b2 = ramp(D_MODEL, 0.02, -0.03)
    fc1 = nn.Linear(D_MODEL, D_FF).double()
    fc2 = nn.Linear(D_FF, D_MODEL).double()
    with torch.no_grad():
        fc1.weight.copy_(w1.T)
        fc1.bias.copy_(b1)
        fc2.weight.copy_(w2.T)
        fc2.bias.copy_(b2)
    flat = {
        "ffn_w1": w1.flatten().tolist(),
        "ffn_b1": b1.tolist(),
        "ffn_w2": w2.flatten().tolist(),
        "ffn_b2": b2.tolist(),
    }
    return fc1, fc2, flat


def build_input():
    """Input [B, S, d_model] whose positions are NOT affine-equivalent.

    A pure flat ramp reshaped to [B, S, d_model] makes every position a shifted
    ramp, and LayerNorm (per-position mean-subtract + std-divide) collapses them
    to one identical normalized vector — the causal mask then never changes the
    output (all V rows equal). To make masking observable, each position gets a
    DISTINCT non-affine pattern: a per-position quadratic curvature term is added
    so no two rows are related by an affine map and LayerNorm keeps them distinct.

    Returns the [B, S, d_model] tensor and its flat list for the JSON fixture.
    """
    x = torch.empty(B, SEQ, D_MODEL, dtype=torch.float64)
    for b in range(B):
        for s in range(SEQ):
            for c in range(D_MODEL):
                base = (b * SEQ + s) * D_MODEL + c
                # linear ramp + position-dependent quadratic curvature: the
                # (s+1) coefficient makes each position's shape differ, so it
                # survives LayerNorm's affine normalization.
                x[b, s, c] = base * 0.1 - 0.3 + (s + 1) * (c - 1.5) ** 2 * 0.05
    return x, x.flatten().tolist()


def make_fixture(num_heads, causal):
    layers, attn_flat = build_attn_projections()
    fc1, fc2, ffn_flat = build_ffn()
    # Two LayerNorms, identity-affine (gamma=1, beta=0), eps=1e-5.
    ln1 = nn.LayerNorm(D_MODEL, eps=EPS).double()
    ln2 = nn.LayerNorm(D_MODEL, eps=EPS).double()
    x, x_flat = build_input()
    with torch.no_grad():
        # u = x + Attention(LayerNorm_1(x))
        n1 = ln1(x)
        u = x + attention_forward(layers, n1, num_heads, causal)
        # y = u + FFN(LayerNorm_2(u)), FFN = fc2(gelu(fc1(·)))
        n2 = ln2(u)
        y = u + fc2(F.gelu(fc1(n2), approximate="none"))
    gamma = [1.0] * D_MODEL
    beta = [0.0] * D_MODEL
    return {
        "config": {
            "d_model": D_MODEL,
            "seq": SEQ,
            "batch": B,
            "d_ff": D_FF,
            "num_heads": num_heads,
            "causal": causal,
            "eps": EPS,
        },
        "norm1_gamma": gamma,
        "norm1_beta": beta,
        **attn_flat,
        "norm2_gamma": gamma,
        "norm2_beta": beta,
        **ffn_flat,
        "X": x_flat,
        "out": y.flatten().tolist(),
    }


if __name__ == "__main__":
    fixture = {
        "no_mask": make_fixture(num_heads=1, causal=False),
        "causal": make_fixture(num_heads=2, causal=True),
    }
    print(json.dumps(fixture, indent=2))
