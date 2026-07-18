"""Standalone (multi-head) linear (kernel-feature) self-attention block.

A single self-attention sub-layer that replaces the softmax kernel of ordinary
scaled dot-product attention with a **linear (kernelized) attention** as
introduced by Katharopoulos et al. 2020 ("Transformers are RNNs: Fast
Autoregressive Transformers with Linear Attention", ICML 2020, arXiv:2006.16236,
§3.2 "Linearized Attention"). Four `Linear` projections (query, key, value, and
a final output projection) surround the kernelized attention

    V'_i = ( φ(Q_i)ᵀ Σ_j φ(K_j) V_jᵀ ) / ( φ(Q_i)ᵀ Σ_j φ(K_j) )      (§3.2 eq. 5)

with the feature map (§3.2 eq. 7)

    φ(x) = elu(x) + 1                       (> 0 for all x).

Because φ is applied elementwise and the numerator factors as
φ(Q)·(φ(K)ᵀV), the key–value summary S = Σ_j φ(K_j) V_jᵀ (shape [d_k, d_v],
equal to [d_k, d_k] here since d_v = d_k) and the key summary Z = Σ_j φ(K_j)
(shape [d_k]) are each computed ONCE per head,
turning the cost from O(S²·d_k) (the naive (φ(Q)φ(K)ᵀ)V order) into O(S·d_k²).
This is the associativity trick that gives linear attention its O(N) complexity
in the sequence length. The two evaluation orders are mathematically identical;
`test_linear_attention.mojo` asserts the O(N) path equals the naive O(N²) order
to tight tolerance.

Feature-map positivity + denominator guard: φ(x) = elu(x) + 1 > 0 for every x
(elu(x) > −1), so Z has strictly positive entries and the normalizer
φ(Q_i)ᵀZ is positive whenever φ(Q_i) has any positive entry — which it always
does, since φ(Q_i) > 0. A small epsilon `eps` (default 1e-6) is nonetheless
added to the denominator before the division to guard the numerically
degenerate all-near-zero case; the PyTorch parity reference mirrors the SAME
epsilon so parity holds to 1e-5.

For `num_heads > 1` the model dimension `d_model` is split into `num_heads`
independent heads of width `d_k = d_model / num_heads`; each head attends
separately and the concatenated results are mixed by the output projection.
`num_heads = 1` recovers ordinary single-head linear attention.

Tensors are batch-first: inputs and outputs are `[batch, seq, d_model]`. This is
a *self*-attention block: Q, K and V are all projections of the same input. This
module implements the **non-causal (full-sequence) form** — every query attends
to every key (§3.2 eq. 5). The paper's causal/autoregressive variant (§3.3,
where S and Z accumulate as running RNN-like states) is a separate recurrence
and is deliberately out of scope for this bare, full-sequence primitive.

No residual connection and no LayerNorm are applied here — this is the bare
attention primitive so its behavior can be measured in isolation; a full
Transformer block composes this with `LayerNorm` and `FeedForward` externally.

Relationship to the softmax attention primitives: this is the linear-kernel
sibling of scaled dot-product attention. It reuses the same QKV-projection /
head-split / head-merge conventions but replaces the softmax score matrix with
the kernel-feature factorization above, so no S×S score matrix is ever
materialized. It does NOT call the functional softmax attention core
(`core/attention.mojo`), which is broken at this SHA (Odyssey#5648) and computes
softmax attention regardless. (A `Module`-conforming softmax `MultiHeadAttention`
layer is pending in PR #5640; when it lands it will sit alongside this one at
`core/layers/attention.mojo` — this module does not depend on it.)

Cached surface (for a downstream PC activation-cache wrapper): unlike the
softmax block, there is no `scores`/`weights` S×S matrix here. A PC wrapper that
needs the pre/post caches should stash, per forward: `q, k, v` (post-projection,
pre-split), `phi_q, phi_k` (post-feature-map), the key–value summary `S` and key
summary `Z` (the O(N) intermediates), `context` (post-normalization, pre-merge),
and the two projection inputs (`input`, `concat`). None are stored on `self`
here — `forward` is pure and recomputes everything each call.

Odyssey `Linear` uses y = x @ W + b with W of shape (in, out). torch.nn.Linear
uses W of shape (out, in), so a PyTorch parity reference sets torch's weights to
the transpose of the Odyssey weights (see
`tests/odyssey/core/layers/parity_refs/linear_attention_parity_reference.py`).

Reference:
    Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers
    are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
    arXiv:2006.16236, §3.2 "Linearized Attention" (eq. 5, eq. 7).
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.module import Module
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import elu
from odyssey.core.matrix import matmul, transpose
from odyssey.core.arithmetic_simd import add_simd, divide_simd
from odyssey.core.reduction import sum as reduce_sum


struct LinearAttention[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Standalone linear (kernel-feature) self-attention block (arXiv:2006.16236).

    Parameters:
        dtype: Data type for weights and biases (default: float32). The
            compile-time `Self.dtype` fixes the tensor element type of every
            projection weight/bias and of all intermediates; float32 is the
            trained/inference default and float64 is used by the parity test for
            bit-comparable ground truth.

    Attributes:
        q_proj: Query projection Linear (d_model -> d_model).
        k_proj: Key projection Linear (d_model -> d_model).
        v_proj: Value projection Linear (d_model -> d_model).
        out_proj: Output projection Linear (d_model -> d_model).
        d_model: Model (input/output) feature dimension.
        num_heads: Number of attention heads (d_model must be divisible by it).
        d_k: Per-head feature width = d_model / num_heads.
        eps: Denominator guard added before the normalizing division.
    """

    var q_proj: Linear[Self.dtype]
    var k_proj: Linear[Self.dtype]
    var v_proj: Linear[Self.dtype]
    var out_proj: Linear[Self.dtype]
    var d_model: Int
    var num_heads: Int
    var d_k: Int
    var eps: Float64

    def __init__(
        out self,
        d_model: Int,
        num_heads: Int = 1,
        eps: Float64 = 1e-6,
    ) raises:
        """Initialize the linear-attention block.

        Args:
            d_model: Model (input and output) dimension.
            num_heads: Number of attention heads (default 1 = single head).
                `d_model` must be divisible by `num_heads`.
            eps: Positive guard added to the normalizer before division
                (default 1e-6). φ = elu + 1 > 0 already keeps the normalizer
                positive; `eps` only guards the degenerate all-near-zero case.

        Raises:
            Error: If d_model <= 0, num_heads <= 0, d_model % num_heads != 0,
                or Linear construction fails.

        Example:
            ```mojo
            var attn = LinearAttention[DType.float32](16, num_heads=4)
            var y = attn.forward(x)   # x: [batch, seq, 16] -> [batch, seq, 16]
            ```
        """
        if d_model <= 0:
            raise Error("LinearAttention: d_model must be positive")
        if num_heads <= 0:
            raise Error("LinearAttention: num_heads must be positive")
        if d_model % num_heads != 0:
            raise Error(
                "LinearAttention: d_model must be divisible by num_heads"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.eps = eps
        self.q_proj = Linear[Self.dtype](d_model, d_model)
        self.k_proj = Linear[Self.dtype](d_model, d_model)
        self.v_proj = Linear[Self.dtype](d_model, d_model)
        self.out_proj = Linear[Self.dtype](d_model, d_model)

    @staticmethod
    def _project(
        mut proj: Linear[Self.dtype], input: AnyTensor, d_model: Int
    ) raises -> AnyTensor:
        """Apply a Linear projection to a batch-first [B, S, d_model] tensor.

        `Linear.forward` matmuls a 2D or 1D input by its 2D weight, so a 3D
        input is flattened to [B*S, d_model], projected, then restored to
        [B, S, d_model]. The projection is position-wise (each (b, s) row is
        transformed by the same weight).

        A static method taking `proj` explicitly (rather than a `mut self`
        method) so `forward` can pass one projection field as `mut` without also
        aliasing the whole `self`.

        Args:
            proj: The Linear layer to apply.
            input: Input tensor of shape [B, S, d_model].
            d_model: Model feature dimension (the input's last axis).

        Returns:
            Projected tensor of shape [B, S, d_model].

        Raises:
            Error: If tensor operations fail.
        """
        var b = input.shape()[0]
        var s = input.shape()[1]
        var flat = input.reshape([b * s, d_model])
        var projected = proj.forward(flat)
        return projected.reshape([b, s, d_model])

    def _split_heads(
        self, x: AnyTensor, batch: Int, seq: Int
    ) raises -> AnyTensor:
        """Reshape [B, S, d_model] -> [B, H, S, d_k] for per-head attention.

        Splits the feature axis into (num_heads, d_k) then moves the head axis
        in front of the sequence axis so each head is an independent [S, d_k]
        matrix in the batched matmul.

        Args:
            x: Tensor of shape [B, S, d_model].
            batch: Batch size B.
            seq: Sequence length S.

        Returns:
            Tensor of shape [B, num_heads, S, d_k].

        Raises:
            Error: If tensor operations fail.
        """
        var split = x.reshape([batch, seq, self.num_heads, self.d_k])
        # [B, S, H, d_k] -> [B, H, S, d_k]
        var perm = List[Int]()
        perm.append(0)
        perm.append(2)
        perm.append(1)
        perm.append(3)
        return transpose(split, perm^)

    @staticmethod
    def _feature_map(x: AnyTensor) raises -> AnyTensor:
        """Kernel feature map φ(x) = elu(x) + 1 (§3.2 eq. 7).

        Applied elementwise; φ(x) > 0 for all x (elu(x) > −1), which keeps the
        attention normalizer strictly positive.

        Args:
            x: Any tensor.

        Returns:
            `elu(x, alpha=1.0) + 1`, same shape as `x`.

        Raises:
            Error: If tensor operations fail.
        """
        var e = elu(x, alpha=1.0)
        return add_simd(e, full_like(e, 1.0))

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Non-causal linear-attention forward pass (§3.2 eq. 5, O(N) order).

        Computes, for each head h, with φ(x) = elu(x) + 1:

            S_h = φ(K_h)ᵀ V_h              # [d_k, d_v] key–value summary (d_v = d_k)
            Z_h = Σ_j φ(K_h)_j             # [d_k]      key summary
            out_h = (φ(Q_h) S_h) / (φ(Q_h) Z_hᵀ + eps)

        computing the S_h and Z_h summaries ONCE per head (the O(N)
        associativity trick), then concatenates the heads back to `d_model`
        width and applies the output projection.

        Args:
            input: Input tensor of shape [batch, seq, d_model].

        Returns:
            Output tensor of shape [batch, seq, d_model].

        Raises:
            Error: If the input is not 3D, its last dim != d_model, or a tensor
                operation fails.
        """
        if input.dim() != 3:
            raise Error(
                "LinearAttention.forward expects a 3D [batch, seq, d_model]"
                " tensor"
            )
        if input.shape()[2] != self.d_model:
            raise Error("LinearAttention.forward: last dim must equal d_model")
        var batch = input.shape()[0]
        var seq = input.shape()[1]

        # Position-wise Q/K/V projections, then split into heads. Each
        # projection is applied in its own statement so the `mut` borrow of one
        # projection field does not alias the immutable `self` read in
        # `_split_heads`.
        var q_flat = Self._project(self.q_proj, input, self.d_model)
        var k_flat = Self._project(self.k_proj, input, self.d_model)
        var v_flat = Self._project(self.v_proj, input, self.d_model)
        var q = self._split_heads(q_flat, batch, seq)
        var k = self._split_heads(k_flat, batch, seq)
        var v = self._split_heads(v_flat, batch, seq)

        # Feature map φ(x) = elu(x) + 1 on Q and K.  [B, H, S, d_k]
        var phi_q = Self._feature_map(q)
        var phi_k = Self._feature_map(k)

        # Key–value summary S = φ(K)ᵀ V, computed ONCE per head. Transpose the
        # last two axes of φ(K): [B, H, S, d_k] -> [B, H, d_k, S], then
        # [B, H, d_k, S] @ [B, H, S, d_k] -> [B, H, d_k, d_k].
        var kt_perm = List[Int]()
        kt_perm.append(0)
        kt_perm.append(1)
        kt_perm.append(3)
        kt_perm.append(2)
        var phi_k_t = transpose(phi_k, kt_perm^)
        var kv = matmul(phi_k_t, v)  # [B, H, d_k, d_k]

        # Numerator = φ(Q) S  ->  [B, H, S, d_k].
        var numerator = matmul(phi_q, kv)

        # Key summary Z = Σ_j φ(K)_j over the sequence axis -> [B, H, 1, d_k].
        var z = reduce_sum(phi_k, axis=2, keepdims=True)
        # Denominator = φ(Q) Zᵀ  ->  [B, H, S, 1]. Transpose Z's last two axes:
        # [B, H, 1, d_k] -> [B, H, d_k, 1].
        var z_perm = List[Int]()
        z_perm.append(0)
        z_perm.append(1)
        z_perm.append(3)
        z_perm.append(2)
        var z_t = transpose(z, z_perm^)
        var denominator = matmul(phi_q, z_t)  # [B, H, S, 1]
        # Guard: add eps before dividing. denominator is > 0 already (φ > 0);
        # eps only guards the degenerate all-near-zero case.
        denominator = add_simd(denominator, full_like(denominator, self.eps))

        # Broadcast-divide [B, H, S, d_k] / [B, H, S, 1] -> [B, H, S, d_k].
        var context = divide_simd(numerator, denominator)

        # Merge heads back: [B, H, S, d_k] -> [B, S, H, d_k] -> [B, S, d_model].
        var merge_perm = List[Int]()
        merge_perm.append(0)
        merge_perm.append(2)
        merge_perm.append(1)
        merge_perm.append(3)
        var merged = transpose(context, merge_perm^)
        var concat = merged.reshape([batch, seq, self.d_model])

        return Self._project(self.out_proj, concat, self.d_model)

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from the four projections.

        Returns:
            List of 8 tensors: weight+bias of q_proj, k_proj, v_proj, out_proj.

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        for p in self.q_proj.parameters():
            params.append(p)
        for p in self.k_proj.parameters():
            params.append(p)
        for p in self.v_proj.parameters():
            params.append(p)
        for p in self.out_proj.parameters():
            params.append(p)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; sub-layers are stateless in mode)."""
        self.q_proj.train()
        self.k_proj.train()
        self.v_proj.train()
        self.out_proj.train()

    def eval(mut self):
        """Switch to inference mode (no-op; sub-layers are stateless in mode).
        """
        self.q_proj.eval()
        self.k_proj.eval()
        self.v_proj.eval()
        self.out_proj.eval()
