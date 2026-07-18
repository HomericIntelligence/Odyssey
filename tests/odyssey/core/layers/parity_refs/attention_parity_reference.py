"""MultiHeadAttention (scaled dot-product self-attention) parity reference.

Computes the forward pass of a standalone Transformer self-attention block
(Q/K/V projections -> softmax(QKᵀ/√d_k)V -> output projection; Vaswani et al.
2017, arXiv:1706.03762 §3.2) in PyTorch with FIXED ramp weights and a FIXED
batch-first input, and prints the flattened output as JSON so the Mojo
attention test can set the same weights and assert equality to 1e-5.

Odyssey `Linear` uses y = x @ W + b with W of shape (in, out).
torch.nn.Linear uses y = x @ W_t^T + b with W_t of shape (out, in), so we set
torch's weight to the TRANSPOSE of the Odyssey weight to make the two identical.

Two fixtures are emitted:
  - single-head (num_heads=1, causal=False)
  - multi-head  (num_heads=2, causal=True)

Both use d_model=4, seq=3, batch=2 and deterministic ramp weights so they
transcribe cleanly into Mojo. Weights are seeded in the SAME flat order the Mojo
test uses: q_proj.weight, q_proj.bias, k_proj.weight, k_proj.bias, v_proj.weight,
v_proj.bias, out_proj.weight, out_proj.bias, then the input X.

Regenerate + diff before push:
    /tmp/claude-1000/venvs/torch_venv/bin/python \
        tests/odyssey/core/layers/parity_refs/attention_parity_reference.py \
        > tests/odyssey/core/layers/parity_refs/attention_parity_reference.json
    git diff --exit-code tests/odyssey/core/layers/parity_refs/attention_parity_reference.json
"""

import json
import math

import torch
import torch.nn as nn

torch.manual_seed(0)

D_MODEL, SEQ, B = 4, 3, 2


def ramp(n, scale, off):
    """value[i] = i * scale + off, as a float64 tensor of length n."""
    return torch.arange(n, dtype=torch.float64) * scale + off


def build_projections():
    """Deterministic ramp weights in Odyssey (in, out) layout + torch layers.

    Returns the four torch Linear layers (query/key/value/output) with weights
    set to the transpose of the Odyssey ramp weights, plus the flat Odyssey-order
    ramp lists for the JSON fixture.
    """
    # Odyssey layout: W (d_model, d_model), b (d_model,). Distinct ramps per proj
    # so an accidental Q/K/V swap is caught.
    specs = {
        "q": (0.01, -0.15, 0.02, -0.08),
        "k": (0.013, -0.12, 0.015, -0.05),
        "v": (0.008, -0.10, 0.011, -0.06),
        "out": (0.006, -0.08, 0.03, -0.05),
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
    q = layers["q"](x)  # [B, S, d_model]
    k = layers["k"](x)
    v = layers["v"](x)
    d_k = D_MODEL // num_heads

    def split(t):
        # [B, S, d_model] -> [B, H, S, d_k]
        return t.reshape(B, SEQ, num_heads, d_k).permute(0, 2, 1, 3)

    qh, kh, vh = split(q), split(k), split(v)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,S,S]
    if causal:
        mask = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)  # True strictly above diagonal
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, vh)  # [B, H, S, d_k]
    merged = context.permute(0, 2, 1, 3).reshape(B, SEQ, D_MODEL)
    return layers["out"](merged)


def make_fixture(num_heads, causal):
    layers, flat = build_projections()
    x = ramp(B * SEQ * D_MODEL, 0.1, -0.3).reshape(B, SEQ, D_MODEL)
    with torch.no_grad():
        out = attention_forward(layers, x, num_heads, causal)
    return {
        "config": {
            "d_model": D_MODEL,
            "seq": SEQ,
            "batch": B,
            "num_heads": num_heads,
            "causal": causal,
        },
        **flat,
        "X": x.flatten().tolist(),
        "out": out.flatten().tolist(),
    }


if __name__ == "__main__":
    fixture = {
        "single_head": make_fixture(num_heads=1, causal=False),
        "multi_head": make_fixture(num_heads=2, causal=True),
    }
    print(json.dumps(fixture, indent=2))
