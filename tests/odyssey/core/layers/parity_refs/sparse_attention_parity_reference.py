"""SparseAttention (strided factorized self-attention) parity reference.

Computes the forward pass of a 1-layer *sparse* Transformer self-attention block
(Q/K/V projections -> softmax((QKᵀ/√d_k) + sparse_mask) V -> output projection;
Child, Gray, Radford & Sutskever 2019, arXiv:1904.10509 §4.2 "strided") in
PyTorch with FIXED ramp weights and a FIXED batch-first input, and prints the
flattened output as JSON so the Mojo test can set the same weights and assert
equality to 1e-5.

Sparse (strided) connectivity — query i attends to key j iff:
    j <= i  AND  ( (i - j) < window  OR  (i - j) % stride == 0 )
i.e. the union of the local window A_i^(1) = {previous `window` positions} and
the strided set A_i^(2) = {every `stride`-th earlier position} (§4.2). Masked
(i, j) get −inf added before the softmax, so their post-softmax weight is 0.

Odyssey `Linear` uses y = x @ W + b with W of shape (in, out).
torch.nn.Linear uses y = x @ W_t^T + b with W_t of shape (out, in), so we set
torch's weight to the TRANSPOSE of the Odyssey weight to make the two identical.

Three fixtures are emitted:
  - single_head : num_heads=1, window=2, stride=2 (a genuinely sparse pattern)
  - multi_head  : num_heads=2, window=2, stride=3 (head split + sparse pattern)
  - dense_equiv : num_heads=1, window=seq, stride=1 (window >= seq collapses the
                  union mask to plain lower-triangular causal — must equal dense
                  causal attention on the same weights)

All use d_model=4, seq=4, batch=2 and deterministic ramp weights so they
transcribe cleanly into Mojo. seq=4 (not 3) so the window/stride pattern has room
to be non-trivial. Weights are seeded in the SAME flat order the Mojo test uses:
q_proj.weight, q_proj.bias, k_proj.weight, k_proj.bias, v_proj.weight,
v_proj.bias, out_proj.weight, out_proj.bias, then the input X.

Regenerate + diff before push:
    /tmp/claude-1000/venvs/torch_venv/bin/python \\
        tests/odyssey/core/layers/parity_refs/sparse_attention_parity_reference.py \\
        > tests/odyssey/core/layers/parity_refs/sparse_attention_parity_reference.json
    git diff --exit-code \\
        tests/odyssey/core/layers/parity_refs/sparse_attention_parity_reference.json
"""

import json
import math

import torch
import torch.nn as nn

torch.manual_seed(0)

D_MODEL, SEQ, B = 4, 4, 2


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


def sparse_mask(seq, window, stride):
    """Additive [seq, seq] mask: 0 on attended (i, j), −inf off the pattern.

    Attended iff  j <= i  AND ( (i - j) < window  OR  (i - j) % stride == 0 ).
    Mirrors SparseAttention.is_attended exactly.
    """
    m = torch.full((seq, seq), float("-inf"), dtype=torch.float64)
    for i in range(seq):
        for j in range(seq):
            if j <= i and ((i - j) < window or (i - j) % stride == 0):
                m[i, j] = 0.0
    return m


def attention_forward(layers, x, num_heads, window, stride):
    """Explicit sparse (strided) scaled dot-product self-attention."""
    q = layers["q"](x)  # [B, S, d_model]
    k = layers["k"](x)
    v = layers["v"](x)
    d_k = D_MODEL // num_heads

    def split(t):
        # [B, S, d_model] -> [B, H, S, d_k]
        return t.reshape(B, SEQ, num_heads, d_k).permute(0, 2, 1, 3)

    qh, kh, vh = split(q), split(k), split(v)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,S,S]
    mask = sparse_mask(SEQ, window, stride)  # [S, S] broadcast over B, H
    scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, vh)  # [B, H, S, d_k]
    merged = context.permute(0, 2, 1, 3).reshape(B, SEQ, D_MODEL)
    return layers["out"](merged), weights


def make_fixture(num_heads, window, stride):
    layers, flat = build_projections()
    x = ramp(B * SEQ * D_MODEL, 0.1, -0.3).reshape(B, SEQ, D_MODEL)
    with torch.no_grad():
        out, weights = attention_forward(layers, x, num_heads, window, stride)
    return {
        "config": {
            "d_model": D_MODEL,
            "seq": SEQ,
            "batch": B,
            "num_heads": num_heads,
            "window": window,
            "stride": stride,
        },
        **flat,
        "X": x.flatten().tolist(),
        "out": out.flatten().tolist(),
        "weights": weights.flatten().tolist(),  # [B, H, S, S] for the zero test
    }


if __name__ == "__main__":
    fixture = {
        "single_head": make_fixture(num_heads=1, window=2, stride=2),
        "multi_head": make_fixture(num_heads=2, window=2, stride=3),
        "dense_equiv": make_fixture(num_heads=1, window=SEQ, stride=1),
    }
    print(json.dumps(fixture, indent=2))
