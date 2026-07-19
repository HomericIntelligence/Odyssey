"""1-layer pre-LN Transformer encoder block (self-attention + FFN).

A single Transformer encoder block (Vaswani et al., "Attention Is All You Need",
2017, arXiv:1706.03762, §3.1 "Encoder and Decoder Stacks") in the **pre-norm**
(pre-LN) arrangement popularized by later work (Xiong et al. 2020, "On Layer
Normalization in the Transformer Architecture", arXiv:2002.04745): LayerNorm is
applied to the *input* of each sub-layer rather than to its output, which
stabilizes training of deep stacks. The block operates on a batch-first sequence
`[batch, seq, d_model]` and is shape-preserving.

The two residual sub-layers (batch-first x of shape [B, S, d_model]):

    u = x + MultiHeadAttention( LayerNorm_1(x) )        # self-attention sublayer
    y = u + FeedForward(        LayerNorm_2(u) )        # position-wise FFN sublayer

This differs from the *original* (post-LN) Transformer, which normalizes the
sub-layer OUTPUT: `x + Sublayer(x)` then `LayerNorm(·)` (§3.1, Eq. following the
residual definition). Pre-LN is the modern default and the variant pinned by the
tracking issue (mvillmow/Random#51).

Composition / reuse posture (the crux of this block):
  * This is a THIN COMPOSITION of three existing `Module`s — `MultiHeadAttention`
    (self-attention sub-layer, `attention.mojo`), `LayerNorm` (`layernorm.mojo`),
    and `FeedForward` (`feedforward.mojo`). None of them is reimplemented here;
    the block only wires them into the two pre-norm residual sub-layers and
    forwards their parameters. This is the same reuse posture `FeedForward` itself
    takes toward `Linear` (`feedforward.mojo`) and `ReLULayer` takes toward the
    functional `relu` (`relu.mojo`) — compose existing `Module`s, add no new math.
  * `MultiHeadAttention.forward` consumes the 3D `[B, S, d_model]` sequence
    directly (it does its own head split/merge and −∞ causal masking).
  * `LayerNorm.forward` and `FeedForward.forward` are POSITION-WISE 2D ops
    (they matmul/normalize over the last feature axis and reject a 3D input), so
    each is applied by flattening `[B, S, d_model] -> [B*S, d_model]`, running the
    sub-layer, and reshaping back. The transform is identical for every (b, s)
    position, which is the intended per-position semantics of both LayerNorm and
    the FFN (the MLP-Mixer block, `core/layers/mlp_mixer.mojo`, applies its
    channel-mixing MLP the same way).

Design/convention notes:
  * Batch-first `[batch, seq, d_model]`; `forward()` consumes the FULL sequence at
    once (all `seq` positions present) — the established convention for
    sequence-consuming Odyssey layers; there is no per-step/streaming interface.
  * The FFN activation, the LayerNorm epsilon (1e-5), and the attention mask
    convention (true −∞ additive causal mask) are inherited UNCHANGED from the
    reused sub-modules; this block introduces none of its own.
  * No BatchNorm anywhere (BN couples batch elements and is inappropriate for a
    per-position sequence block); LayerNorm only.
  * `causal` forwards to the attention sub-layer's autoregressive mask so the same
    block serves both encoder (non-causal) and decoder-style (causal) use.

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 2017.
    arXiv:1706.03762, §3.1-3.3.
    Xiong, R., et al. (2020). On Layer Normalization in the Transformer
    Architecture. ICML 2020. arXiv:2002.04745 (pre-LN placement).
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.module import Module
from odyssey.core.layers.attention import MultiHeadAttention
from odyssey.core.layers.feedforward import FeedForward
from odyssey.core.layers.layernorm import LayerNorm


struct TransformerEncoderBlock[dtype: DType = DType.float32](
    Copyable, Module, Movable
):
    """A single pre-LN Transformer encoder block (Vaswani 2017; pre-norm).

    Two residual sub-layers over a batch-first `[batch, seq, d_model]` sequence:

        u = x + MultiHeadAttention(LayerNorm_1(x))
        y = u + FeedForward(LayerNorm_2(u))

    Input and output are `[batch, seq, d_model]` and shape-preserving.

    Parameters:
        dtype: Data type for all weights, biases, and LayerNorm params
            (default: float32).

    Attributes:
        norm1: Pre-norm LayerNorm for the self-attention sub-layer (over d_model).
        attn: Multi-head self-attention sub-layer.
        norm2: Pre-norm LayerNorm for the feed-forward sub-layer (over d_model).
        ffn: Position-wise feed-forward sub-layer (d_model -> d_ff -> d_model).
        d_model: Model (input/output) feature dimension.
        num_heads: Number of attention heads (d_model must be divisible by it).
        d_ff: Inner (hidden) width of the feed-forward sub-layer.
        causal: If True, the attention sub-layer applies an autoregressive mask.
    """

    var norm1: LayerNorm[Self.dtype]
    var attn: MultiHeadAttention[Self.dtype]
    var norm2: LayerNorm[Self.dtype]
    var ffn: FeedForward[Self.dtype]
    var d_model: Int
    var num_heads: Int
    var d_ff: Int
    var causal: Bool

    def __init__(
        out self,
        d_model: Int,
        num_heads: Int = 1,
        d_ff: Int = -1,
        use_gelu: Bool = True,
        causal: Bool = False,
    ) raises:
        """Initialize the Transformer encoder block.

        Args:
            d_model: Model (input and output) dimension. Must be positive and
                divisible by `num_heads`.
            num_heads: Number of attention heads (default 1 = single head).
            d_ff: Inner hidden width of the feed-forward sub-layer. If <= 0,
                defaults to 4 * d_model (the ratio used in the original
                Transformer, §3.3).
            use_gelu: Feed-forward activation — exact (erf) GELU when True
                (default), ReLU (the original Transformer choice) when False.
            causal: If True, the attention sub-layer forbids attending to future
                positions (default off = bidirectional encoder self-attention).

        Raises:
            Error: If d_model <= 0, num_heads <= 0, d_model % num_heads != 0, or
                any sub-layer construction fails.

        Example:
            ```mojo
            var block = TransformerEncoderBlock[DType.float32](16, num_heads=4)
            var y = block.forward(x)   # x: [batch, seq, 16] -> [batch, seq, 16]
            ```
        """
        if d_model <= 0:
            raise Error("TransformerEncoderBlock: d_model must be positive")
        # num_heads / divisibility are validated by MultiHeadAttention's
        # constructor; d_ff <= 0 is resolved to 4 * d_model by FeedForward. The
        # concrete widths are read back from the sub-modules below so the
        # attributes report the actual sizes rather than the -1/default sentinels.
        self.d_model = d_model
        self.num_heads = num_heads
        self.causal = causal
        self.norm1 = LayerNorm[Self.dtype](d_model)
        self.attn = MultiHeadAttention[Self.dtype](
            d_model, num_heads=num_heads, causal=causal
        )
        self.norm2 = LayerNorm[Self.dtype](d_model)
        self.ffn = FeedForward[Self.dtype](d_model, d_ff, use_gelu)
        self.d_ff = self.ffn.d_ff

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass through one pre-LN Transformer encoder block.

        Computes

            u = input + MultiHeadAttention(LayerNorm_1(input))
            y = u     + FeedForward(LayerNorm_2(u))

        LayerNorm and FeedForward are position-wise 2D ops, so each is applied by
        flattening the `[B, S, d_model]` sequence to `[B*S, d_model]`, running the
        sub-layer, and reshaping back. Attention consumes the 3D sequence directly.

        Args:
            input: Batch-first tensor of shape `[batch, seq, d_model]`, where the
                last axis equals `d_model`. The block consumes the whole sequence.

        Returns:
            Tensor of shape `[batch, seq, d_model]` (same shape as `input`).

        Raises:
            Error: If `input` is not rank-3, its last axis != d_model, or a tensor
                op fails.
        """
        var shape = input.shape()
        if len(shape) != 3:
            raise Error(
                "TransformerEncoderBlock: input must be rank-3 [batch, seq,"
                " d_model], got rank "
                + String(len(shape))
            )
        var batch = shape[0]
        var seq = shape[1]
        var d = shape[2]
        if d != self.d_model:
            raise Error(
                "TransformerEncoderBlock: input feature dim ("
                + String(d)
                + ") does not match d_model ("
                + String(self.d_model)
                + ")"
            )

        # --- Self-attention sub-layer: u = x + attn(LayerNorm_1(x)) ---
        # LayerNorm is a position-wise 2D op: flatten [B, S, D] -> [B*S, D],
        # normalize, reshape back to the 3D sequence for attention.
        var x_flat = input.reshape([batch * seq, d])
        var norm1_flat = self.norm1.forward(x_flat)
        var norm1 = norm1_flat.reshape([batch, seq, d])
        var attn_out = self.attn.forward(norm1)  # [B, S, D]
        var u = input + attn_out

        # --- Feed-forward sub-layer: y = u + ffn(LayerNorm_2(u)) ---
        # Both LayerNorm and FFN are position-wise: flatten, apply, reshape back.
        var u_flat = u.reshape([batch * seq, d])
        var norm2_flat = self.norm2.forward(u_flat)
        var ffn_flat = self.ffn.forward(norm2_flat)
        var ffn_out = ffn_flat.reshape([batch, seq, d])
        var y = u + ffn_out
        return y

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from all four sub-modules.

        Returns:
            List of parameters in order: norm1 (gamma, beta),
            attn (q/k/v/out weight+bias = 8 tensors), norm2 (gamma, beta),
            ffn (fc1.w/b, fc2.w/b) — 16 tensors total.

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        for p in self.norm1.parameters():
            params.append(p)
        for p in self.attn.parameters():
            params.append(p)
        for p in self.norm2.parameters():
            params.append(p)
        for p in self.ffn.parameters():
            params.append(p)
        return params^

    def train(mut self):
        """Switch all sub-modules to training mode (all are mode-independent).
        """
        self.norm1.train()
        self.attn.train()
        self.norm2.train()
        self.ffn.train()

    def eval(mut self):
        """Switch all sub-modules to inference mode (all are mode-independent).
        """
        self.norm1.eval()
        self.attn.eval()
        self.norm2.eval()
        self.ffn.eval()
