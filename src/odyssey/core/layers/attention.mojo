"""Standalone (multi-head) scaled dot-product self-attention block.

A single self-attention sub-layer as introduced by the Transformer (Vaswani et
al., "Attention Is All You Need", 2017, §3.2): four `Linear` projections (query,
key, value, and a final output projection) around the scaled dot-product
attention operation

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V                    (§3.2.1)

For `num_heads > 1` the model dimension `d_model` is split into `num_heads`
independent heads of width `d_k = d_model / num_heads`; each head attends
separately and the concatenated results are mixed by the output projection
(§3.2.2, "Multi-Head Attention"). `num_heads = 1` recovers ordinary single-head
scaled dot-product self-attention.

Tensors are batch-first: inputs and outputs are `[batch, seq, d_model]`. This is
a *self*-attention block: Q, K and V are all projections of the same input. An
optional causal (autoregressive) mask forbids each position from attending to
later positions by adding −∞ to the upper-triangular scores before the softmax
(§3.2.3, "Masked" attention in the decoder self-attention sub-layer).

No residual connection and no LayerNorm are applied here — this is the bare
attention primitive so its behavior can be measured in isolation; a full
Transformer block composes this with `LayerNorm` and `FeedForward` externally.

Odyssey `Linear` uses y = x @ W + b with W of shape (in, out). torch.nn.Linear
uses W of shape (out, in), so a PyTorch parity reference sets torch's weights to
the transpose of the Odyssey weights (see
`tests/odyssey/core/layers/parity_refs/attention_parity_reference.py`).

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 2017.
    arXiv:1706.03762, §3.2 "Attention".
"""

from std.math import sqrt

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like, neg_inf_tensor
from odyssey.core.module import Module
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import softmax
from odyssey.core.matrix import matmul, transpose
from odyssey.core.arithmetic_simd import add_simd, multiply_simd


struct MultiHeadAttention[dtype: DType = DType.float32](
    Copyable, Module, Movable
):
    """Standalone scaled dot-product self-attention block (Vaswani 2017 §3.2).

    Parameters:
        dtype: Data type for weights and biases (default: float32).

    Attributes:
        q_proj: Query projection Linear (d_model -> d_model).
        k_proj: Key projection Linear (d_model -> d_model).
        v_proj: Value projection Linear (d_model -> d_model).
        out_proj: Output projection Linear (d_model -> d_model).
        d_model: Model (input/output) feature dimension.
        num_heads: Number of attention heads (d_model must be divisible by it).
        d_k: Per-head feature width = d_model / num_heads.
        causal: If True, apply an autoregressive (upper-triangular) mask.
    """

    var q_proj: Linear[Self.dtype]
    var k_proj: Linear[Self.dtype]
    var v_proj: Linear[Self.dtype]
    var out_proj: Linear[Self.dtype]
    var d_model: Int
    var num_heads: Int
    var d_k: Int
    var causal: Bool

    def __init__(
        out self,
        d_model: Int,
        num_heads: Int = 1,
        causal: Bool = False,
    ) raises:
        """Initialize the attention block.

        Args:
            d_model: Model (input and output) dimension.
            num_heads: Number of attention heads (default 1 = single head).
                `d_model` must be divisible by `num_heads`.
            causal: If True, forbid attending to future positions (default off).

        Raises:
            Error: If d_model <= 0, num_heads <= 0, d_model % num_heads != 0,
                or Linear construction fails.

        Example:
            ```mojo
            var attn = MultiHeadAttention[DType.float32](16, num_heads=4)
            var y = attn.forward(x)   # x: [batch, seq, 16] -> [batch, seq, 16]
            ```
        """
        if d_model <= 0:
            raise Error("MultiHeadAttention: d_model must be positive")
        if num_heads <= 0:
            raise Error("MultiHeadAttention: num_heads must be positive")
        if d_model % num_heads != 0:
            raise Error(
                "MultiHeadAttention: d_model must be divisible by num_heads"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal
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
        transformed by the same weight), which is exactly the Transformer's
        per-position Q/K/V/output projection.

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

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Scaled dot-product self-attention forward pass (§3.2).

        Computes, for each head h,

            head_h = softmax(Q_h K_hᵀ / √d_k) V_h

        concatenates the heads back to `d_model` width, and applies the output
        projection. When `causal` is set, positions may only attend to earlier
        or equal positions (an upper-triangular −∞ mask on the scores).

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
                "MultiHeadAttention.forward expects a 3D [batch, seq, d_model]"
                " tensor"
            )
        if input.shape()[2] != self.d_model:
            raise Error(
                "MultiHeadAttention.forward: last dim must equal d_model"
            )
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

        # scores = Q Kᵀ / √d_k  ->  [B, H, S, S]
        # transpose the last two axes of K: [B, H, S, d_k] -> [B, H, d_k, S]
        var k_perm = List[Int]()
        k_perm.append(0)
        k_perm.append(1)
        k_perm.append(3)
        k_perm.append(2)
        var k_t = transpose(k, k_perm^)
        var scores = matmul(q, k_t)
        var inv_sqrt = 1.0 / sqrt(Float64(self.d_k))
        scores = multiply_simd(scores, full_like(scores, inv_sqrt))

        if self.causal:
            scores = self._apply_causal_mask(scores, batch, seq)

        # softmax over the key axis (last), then weight the values.
        var weights = softmax(scores, axis=-1)
        var context = matmul(weights, v)  # [B, H, S, d_k]

        # Merge heads back: [B, H, S, d_k] -> [B, S, H, d_k] -> [B, S, d_model]
        var merge_perm = List[Int]()
        merge_perm.append(0)
        merge_perm.append(2)
        merge_perm.append(1)
        merge_perm.append(3)
        var merged = transpose(context, merge_perm^)
        var concat = merged.reshape([batch, seq, self.d_model])

        return Self._project(self.out_proj, concat, self.d_model)

    def _apply_causal_mask(
        self, scores: AnyTensor, batch: Int, seq: Int
    ) raises -> AnyTensor:
        """Add −∞ to the strictly-upper-triangular scores (future positions).

        For scores of shape [B, H, S, S], every entry (i, j) with j > i (a query
        at position i attending to a key at a *later* position j) is set to −∞ so
        it contributes zero probability mass after the softmax (§3.2.3).

        Args:
            scores: Attention scores of shape [B, H, S, S].
            batch: Batch size B.
            seq: Sequence length S.

        Returns:
            Masked scores of the same shape.

        Raises:
            Error: If tensor operations fail.
        """
        # Additive mask: 0 on/below the diagonal, −∞ strictly above it.
        var mask = neg_inf_tensor([batch, self.num_heads, seq, seq], Self.dtype)
        var head_stride = seq * seq
        for bh in range(batch * self.num_heads):
            var base = bh * head_stride
            for i in range(seq):
                for j in range(seq):
                    if j <= i:
                        mask.store[Self.dtype](base + i * seq + j, 0)
        return add_simd(scores, mask)

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
