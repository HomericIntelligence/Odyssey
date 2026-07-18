"""1-layer sparse (factorized) self-attention block (Child et al., 2019).

A single self-attention sub-layer whose dense O(n²) attention connectivity is
restricted to a fixed *sparse* pattern, following the **strided** factorization
of Child, Gray, Radford & Sutskever, "Generating Long Sequences with Sparse
Transformers" (2019, arXiv:1904.10509, §4.2). The query/key/value/output
projection machinery, head split/merge, scaling, softmax and the additive true−∞
pre-softmax mask are identical to `MultiHeadAttention` (Vaswani 2017 §3.2); the
ONLY difference is which key positions a query is allowed to attend to.

Sparse connectivity (strided pattern, arXiv:1904.10509 §4.2)
------------------------------------------------------------
For a causal (autoregressive) sequence, query position `i` attends to key
position `j` iff `j <= i` AND `j` lies in the union of two sets:

  * local window  A_i^(1) = { t, t+1, ..., i },  t = max(0, i − window + 1)
    — the previous `window` positions (§4.2, "the previous l locations").
  * strided       A_i^(2) = { j : (i − j) mod stride == 0 }
    — every `stride`-th earlier position (§4.2, "every l-th location").

A query attends to `j` iff  j <= i  AND  ( (i − j) < window  OR  (i − j) mod
stride == 0 ). Every other (masked) score is set to true −∞ before the softmax,
so its post-softmax attention weight is EXACTLY 0. Because `i − i == 0` is always
< window (for `window >= 1`) the self-position `j == i` is always kept, so no
query row is ever fully masked — the softmax can never see an all−∞ row (which
would yield NaN). This is enforced by the `window >= 1` constructor guard and
covered by `test_no_all_masked_row`.

Dense-equivalence limit
-----------------------
When `window >= seq`, the local window already covers every `j <= i`, so the
union mask degenerates to the plain lower-triangular causal mask and the block
computes exactly the same output as dense causal `MultiHeadAttention` on the same
weights. `test_dense_equivalence` asserts this numerically; the parity fixture
emits a matching `dense_equiv` case.

Relationship to `MultiHeadAttention`
------------------------------------
This block mirrors `MultiHeadAttention`'s QKV/head/softmax structure rather than
calling it, because that layer's `forward` takes no external mask — its mask is
the internal upper-triangular causal fill. Sparse attention needs a *different*
additive mask (the strided-union pattern above), so the mask-construction step is
replaced while the surrounding pipeline (projections, scale, softmax, merge) is
kept byte-for-byte identical (same true−∞ convention, same `Linear` (in, out)
layout, same 8-parameter order). The two layers are validated against a shared
PyTorch reference; the `window >= seq` case is checked to equal dense causal
attention.

Odyssey `Linear` uses y = x @ W + b with W of shape (in, out). torch.nn.Linear
uses W of shape (out, in), so a PyTorch parity reference sets torch's weights to
the transpose of the Odyssey weights (see
`tests/odyssey/core/layers/parity_refs/sparse_attention_parity_reference.py`).

Reference:
    Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating Long
    Sequences with Sparse Transformers. arXiv:1904.10509, §4.2 ("strided"
    factorized attention).
"""

from std.math import sqrt

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like, neg_inf_tensor
from odyssey.core.module import Module
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import softmax
from odyssey.core.matrix import matmul, transpose
from odyssey.core.arithmetic_simd import add_simd, multiply_simd


struct SparseAttention[dtype: DType = DType.float32](Copyable, Module, Movable):
    """1-layer sparse (strided) self-attention block (Child et al. 2019 §4.2).

    Parameters:
        dtype: Data type for weights and biases (default: float32). The public
            API is float32; parity is verified in float64. `Self.dtype` is the
            single dtype used for every parameter and intermediate tensor.

    Attributes:
        q_proj: Query projection Linear (d_model -> d_model).
        k_proj: Key projection Linear (d_model -> d_model).
        v_proj: Value projection Linear (d_model -> d_model).
        out_proj: Output projection Linear (d_model -> d_model).
        d_model: Model (input/output) feature dimension.
        num_heads: Number of attention heads (d_model must be divisible by it).
        d_k: Per-head feature width = d_model / num_heads.
        window: Local-window size (previous-`window` positions are attended).
        stride: Strided period (every `stride`-th earlier position is attended).
    """

    var q_proj: Linear[Self.dtype]
    var k_proj: Linear[Self.dtype]
    var v_proj: Linear[Self.dtype]
    var out_proj: Linear[Self.dtype]
    var d_model: Int
    var num_heads: Int
    var d_k: Int
    var window: Int
    var stride: Int

    def __init__(
        out self,
        d_model: Int,
        num_heads: Int = 1,
        window: Int = 4,
        stride: Int = 4,
    ) raises:
        """Initialize the sparse attention block.

        Args:
            d_model: Model (input and output) dimension.
            num_heads: Number of attention heads (default 1 = single head).
                `d_model` must be divisible by `num_heads`.
            window: Local-window size — a query attends to the previous `window`
                positions (Child et al. §4.2 local set A_i^(1)). Must be >= 1 so
                the self-position is always attended (no all-masked rows).
            stride: Strided period — a query additionally attends to every
                `stride`-th earlier position (§4.2 strided set A_i^(2)). Must be
                >= 1.

        Raises:
            Error: If d_model <= 0, num_heads <= 0, d_model % num_heads != 0,
                window < 1, stride < 1, or Linear construction fails.

        Example:
            ```mojo
            var attn = SparseAttention[DType.float32](16, num_heads=4)
            var y = attn.forward(x)   # x: [batch, seq, 16] -> [batch, seq, 16]
            ```
        """
        if d_model <= 0:
            raise Error("SparseAttention: d_model must be positive")
        if num_heads <= 0:
            raise Error("SparseAttention: num_heads must be positive")
        if d_model % num_heads != 0:
            raise Error(
                "SparseAttention: d_model must be divisible by num_heads"
            )
        if window < 1:
            raise Error(
                "SparseAttention: window must be >= 1 (keeps the self-position"
                " so no query row is fully masked)"
            )
        if stride < 1:
            raise Error("SparseAttention: stride must be >= 1")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window = window
        self.stride = stride
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
    def is_attended(i: Int, j: Int, window: Int, stride: Int) -> Bool:
        """Return True iff query `i` attends to key `j` (strided pattern).

        Implements the union of Child et al. (2019) §4.2's two sets, restricted
        to the causal region `j <= i`:

            j <= i  AND  ( (i − j) < window  OR  (i − j) mod stride == 0 )

        The first disjunct is the local window A_i^(1) (previous `window`
        positions); the second is the strided set A_i^(2) (every `stride`-th
        earlier position). `j == i` is always attended (i − j == 0 < window for
        window >= 1), guaranteeing no all-masked query row.

        Args:
            i: Query position.
            j: Key position.
            window: Local-window size.
            stride: Strided period.

        Returns:
            True if the (i, j) score is KEPT (finite), False if it is masked.
        """
        if j > i:
            return False
        var delta = i - j
        if delta < window:
            return True
        return delta % stride == 0

    def _apply_sparse_mask(
        self, scores: AnyTensor, batch: Int, seq: Int
    ) raises -> AnyTensor:
        """Add true −∞ to every score outside the strided sparse pattern.

        For scores of shape [B, H, S, S], entry (i, j) is set to −∞ (so its
        post-softmax weight is EXACTLY 0) unless `is_attended(i, j)` holds. Kept
        entries get an additive 0. The mask is identical across batch and head
        (a *fixed* structural pattern, Child et al. §4.2), so the per-(i, j)
        keep/mask decision is computed once and broadcast over all B·H slices.

        Mask convention: a true −∞ sentinel (numerically honest;
        `softmax`'s max-subtraction maps exp(−∞) = 0 without overflow), matching
        `MultiHeadAttention`'s causal mask (NOT the functional core's −1e9).

        Args:
            scores: Attention scores of shape [B, H, S, S].
            batch: Batch size B.
            seq: Sequence length S.

        Returns:
            Masked scores of the same shape.

        Raises:
            Error: If tensor operations fail.
        """
        # Additive mask: 0 on kept (i, j), −∞ on masked (i, j).
        var mask = neg_inf_tensor([batch, self.num_heads, seq, seq], Self.dtype)
        var head_stride = seq * seq
        for bh in range(batch * self.num_heads):
            var base = bh * head_stride
            for i in range(seq):
                for j in range(seq):
                    if Self.is_attended(i, j, self.window, self.stride):
                        mask.store[Self.dtype](base + i * seq + j, 0)
        return add_simd(scores, mask)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Sparse (strided) scaled dot-product self-attention forward pass.

        Computes, for each head h,

            head_h = softmax( (Q_h K_hᵀ / √d_k) + M ) V_h

        where `M` is the additive sparse mask (0 on attended (i, j), −∞ off the
        pattern; Child et al. §4.2 strided). The heads are concatenated back to
        `d_model` width and mixed by the output projection.

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
                "SparseAttention.forward expects a 3D [batch, seq, d_model]"
                " tensor"
            )
        if input.shape()[2] != self.d_model:
            raise Error("SparseAttention.forward: last dim must equal d_model")
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

        scores = self._apply_sparse_mask(scores, batch, seq)

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

    def attention_weights(mut self, input: AnyTensor) raises -> AnyTensor:
        """Return the post-softmax attention weights [B, H, S, S].

        Exposes the sparse attention matrix directly (before the value-weighting
        and output projection) so a test can assert masked positions are EXACTLY
        zero. Recomputes Q/K/scores/mask/softmax with the same pipeline as
        `forward` but stops after the softmax.

        Args:
            input: Input tensor of shape [batch, seq, d_model].

        Returns:
            Attention weights of shape [batch, num_heads, seq, seq]; each row
            sums to 1 and is 0 at every masked (i, j).

        Raises:
            Error: If the input is not 3D, its last dim != d_model, or a tensor
                operation fails.
        """
        if input.dim() != 3:
            raise Error(
                "SparseAttention.attention_weights expects a 3D"
                " [batch, seq, d_model] tensor"
            )
        if input.shape()[2] != self.d_model:
            raise Error(
                "SparseAttention.attention_weights: last dim must equal d_model"
            )
        var batch = input.shape()[0]
        var seq = input.shape()[1]

        var q_flat = Self._project(self.q_proj, input, self.d_model)
        var k_flat = Self._project(self.k_proj, input, self.d_model)
        var q = self._split_heads(q_flat, batch, seq)
        var k = self._split_heads(k_flat, batch, seq)

        var k_perm = List[Int]()
        k_perm.append(0)
        k_perm.append(1)
        k_perm.append(3)
        k_perm.append(2)
        var k_t = transpose(k, k_perm^)
        var scores = matmul(q, k_t)
        var inv_sqrt = 1.0 / sqrt(Float64(self.d_k))
        scores = multiply_simd(scores, full_like(scores, inv_sqrt))
        scores = self._apply_sparse_mask(scores, batch, seq)
        return softmax(scores, axis=-1)

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from the four projections.

        Returns:
            List of 8 tensors: weight+bias of q_proj, k_proj, v_proj, out_proj
            (the same 8-tensor order `MultiHeadAttention` uses).

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
        """Switch to inference mode (no-op; sub-layers stateless in mode)."""
        self.q_proj.eval()
        self.k_proj.eval()
        self.v_proj.eval()
        self.out_proj.eval()
