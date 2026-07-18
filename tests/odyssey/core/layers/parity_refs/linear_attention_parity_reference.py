"""LinearAttention (kernel-feature self-attention) parity reference.

Computes the forward pass of a standalone linear-attention block (Q/K/V
projections -> kernelized attention with feature map phi(x) = elu(x) + 1 ->
output projection; Katharopoulos et al. 2020, "Transformers are RNNs: Fast
Autoregressive Transformers with Linear Attention", arXiv:2006.16236, ICML 2020,
sec. 3.2 "Linearized Attention") in PyTorch with FIXED ramp weights and a FIXED
batch-first input, and prints the flattened output as JSON so the Mojo
LinearAttention test can set the same weights and assert equality to 1e-5.

Linear attention replaces softmax(QK^T/sqrt(d_k))V with the kernelized form

    V'_i = ( phi(Q_i)^T sum_j phi(K_j) V_j^T )
           / ( phi(Q_i)^T sum_j phi(K_j) )                         (eq. 5)

with the feature map (eq. 7)

    phi(x) = elu(x) + 1     (> 0 for all x, so the normalizer is positive).

This reference implements the NON-CAUSAL (full-sequence, unmasked) form: each
query attends to every key. A small epsilon (EPS) is added to the denominator
before the division to guard against a vanishing normalizer; the Mojo layer
MIRRORS the same epsilon, so parity holds to 1e-5.

Odyssey `Linear` uses y = x @ W + b with W of shape (in, out).
torch.nn.Linear uses y = x @ W_t^T + b with W_t of shape (out, in), so we set
torch's weight to the TRANSPOSE of the Odyssey weight to make the two identical.

Two fixtures are emitted:
  - single-head (num_heads=1)
  - multi-head  (num_heads=2)

Both use d_model=8, seq=4, batch=2 and deterministic ramp weights so they
transcribe cleanly into Mojo. Weights are seeded in the SAME flat order the Mojo
test uses: q_proj.weight, q_proj.bias, k_proj.weight, k_proj.bias, v_proj.weight,
v_proj.bias, out_proj.weight, out_proj.bias, then the input X.

Regenerate + diff before push:
    /tmp/claude-1000/venvs/torch_venv/bin/python \
        tests/odyssey/core/layers/parity_refs/linear_attention_parity_reference.py \
        > tests/odyssey/core/layers/parity_refs/linear_attention_parity_reference.json
    git diff --exit-code \
        tests/odyssey/core/layers/parity_refs/linear_attention_parity_reference.json
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

D_MODEL, SEQ, B = 8, 4, 2
# Denominator guard. Mirrored EXACTLY in the Mojo layer's default eps so the two
# stay bit-comparable to 1e-5. phi = elu + 1 > 0 keeps the normalizer positive;
# eps only guards the degenerate all-near-zero-key case.
EPS = 1e-6


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
        "q": (0.003, -0.20, 0.02, -0.08),
        "k": (0.004, -0.18, 0.015, -0.05),
        "v": (0.005, -0.16, 0.011, -0.06),
        "out": (0.002, -0.12, 0.03, -0.05),
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


def linear_attention_forward(layers, x, num_heads):
    """Explicit non-causal linear (kernel-feature) self-attention.

    Batch-first [B, S, d_model]. Uses the O(N) associativity: compute the
    key-value summary KV = phi(K)^T V once per head, then numerator = phi(Q) KV.
    """
    q = layers["q"](x)  # [B, S, d_model]
    k = layers["k"](x)
    v = layers["v"](x)
    d_k = D_MODEL // num_heads

    def split(t):
        # [B, S, d_model] -> [B, H, S, d_k]
        return t.reshape(B, SEQ, num_heads, d_k).permute(0, 2, 1, 3)

    qh, kh, vh = split(q), split(k), split(v)

    # phi(x) = elu(x) + 1  (elementwise, alpha=1.0)
    phi_q = F.elu(qh, alpha=1.0) + 1.0  # [B, H, S, d_k]
    phi_k = F.elu(kh, alpha=1.0) + 1.0  # [B, H, S, d_k]

    # KV summary: phi(K)^T V -> [B, H, d_k, d_k], computed ONCE (the O(N) trick).
    kv = torch.matmul(phi_k.transpose(-2, -1), vh)  # [B, H, d_k, d_k]
    numerator = torch.matmul(phi_q, kv)  # [B, H, S, d_k]

    # Denominator: phi(Q) . sum_j phi(K_j)  -> [B, H, S, 1]
    k_sum = phi_k.sum(dim=-2, keepdim=True)  # [B, H, 1, d_k]
    denominator = torch.matmul(phi_q, k_sum.transpose(-2, -1))  # [B, H, S, 1]
    context = numerator / (denominator + EPS)  # broadcast [B, H, S, d_k]

    merged = context.permute(0, 2, 1, 3).reshape(B, SEQ, D_MODEL)
    return layers["out"](merged)


def make_fixture(num_heads):
    layers, flat = build_projections()
    x = ramp(B * SEQ * D_MODEL, 0.05, -0.4).reshape(B, SEQ, D_MODEL)
    with torch.no_grad():
        out = linear_attention_forward(layers, x, num_heads)
    return {
        "config": {
            "d_model": D_MODEL,
            "seq": SEQ,
            "batch": B,
            "num_heads": num_heads,
            "eps": EPS,
        },
        **flat,
        "X": x.flatten().tolist(),
        "out": out.flatten().tolist(),
    }


if __name__ == "__main__":
    fixture = {
        "single_head": make_fixture(num_heads=1),
        "multi_head": make_fixture(num_heads=2),
    }
    print(json.dumps(fixture, indent=2))
