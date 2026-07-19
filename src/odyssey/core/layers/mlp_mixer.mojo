"""1-layer MLP-Mixer block (token-mixing + channel-mixing MLPs).

A single Mixer layer from Tolstikhin et al. 2021 ("MLP-Mixer: An all-MLP
Architecture for Vision", §2 "Mixer Architecture", Eq. 1-2). The block operates
on a sequence of tokens (patch embeddings) laid out batch-first as
`[batch, seq, dim]` and mixes information along BOTH axes with pure MLPs — no
convolution, no self-attention:

    U = X + TokenMLP( LayerNorm_1(X)^T )^T      # token-mixing (across `seq`)
    Y = U + ChannelMLP( LayerNorm_2(U) )        # channel-mixing (across `dim`)

Both sublayers are pre-normalized (LayerNorm applied to the input of the MLP,
not its output) and wrapped in a residual/skip connection, exactly as in the
reference. Each MLP is the standard two-layer expansion MLP
`fc2(act(fc1(x)))` with a GELU non-linearity (the paper uses GELU; §2).

Transcription of the two mixing directions (the crux of the block):

  * Channel-mixing MLP acts on the feature (`dim`) axis. Since `dim` is the last
    axis, this is the ordinary position-wise MLP: reshape `[batch, seq, dim]`
    -> `[batch*seq, dim]`, apply the MLP, reshape back. `fc1: dim -> channel_hidden`,
    `fc2: channel_hidden -> dim`.

  * Token-mixing MLP acts on the token (`seq`) axis. The reference applies it to
    the TRANSPOSE of the (LayerNorm'd) input so the MLP's linear layers contract
    over `seq`: transpose `[batch, seq, dim]` -> `[batch, dim, seq]`, reshape to
    `[batch*dim, seq]`, apply the MLP (`fc1: seq -> token_hidden`,
    `fc2: token_hidden -> seq`), reshape back to `[batch, dim, seq]` and transpose
    to `[batch, seq, dim]`. The LayerNorm itself still normalizes over the
    feature (`dim`) axis — the transpose is applied only to feed the token MLP,
    matching Eq. 1 where LayerNorm precedes the transpose.

Design/convention notes:
  * Batch-first `[batch, seq, dim]`. `forward()` consumes a FULL sequence at once
    (all `seq` tokens present) — established convention for sequence-consuming
    Odyssey layers; there is no per-step/streaming interface.
  * This block is a thin composition of two `LayerNorm` modules and two
    `FeedForward` MLPs, so it reuses their initialization (Xavier-style Linear,
    gamma=1/beta=0 LayerNorm) and epsilon convention (1e-5) unchanged. LayerNorm
    is reused verbatim from `layernorm.mojo` — the pre-norm sublayers are NOT a
    reimplementation.
  * No BatchNorm anywhere (BN couples batch elements and is inappropriate for
    this token/channel-local block); LayerNorm only.

Reference:
    Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X.,
    Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J.,
    Lucic, M., & Dosovitskiy, A. (2021). MLP-Mixer: An all-MLP Architecture for
    Vision. NeurIPS 2021. arXiv:2105.01601, §2 "Mixer Architecture", Eq. 1-2.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.module import Module
from odyssey.core.layers.feedforward import FeedForward
from odyssey.core.layers.layernorm import LayerNorm


struct MLPMixerBlock[dtype: DType = DType.float32](Copyable, Module, Movable):
    """A single MLP-Mixer block: token-mixing MLP then channel-mixing MLP.

    Both sublayers are pre-LayerNorm with a residual connection. Input and
    output are batch-first `[batch, seq, dim]` and shape-preserving.

    Parameters:
        dtype: Data type for all weights, biases, and LayerNorm params
            (default: float32).

    Attributes:
        norm_token: Pre-norm LayerNorm for the token-mixing sublayer (over `dim`).
        token_mlp: Token-mixing MLP (contracts over `seq`; seq -> token_hidden -> seq).
        norm_channel: Pre-norm LayerNorm for the channel-mixing sublayer (over `dim`).
        channel_mlp: Channel-mixing MLP (contracts over `dim`; dim -> channel_hidden -> dim).
        seq_len: Number of tokens (sequence length) the block is built for.
        dim: Feature (channel) dimension per token.
        token_hidden: Hidden width of the token-mixing MLP.
        channel_hidden: Hidden width of the channel-mixing MLP.
    """

    var norm_token: LayerNorm[Self.dtype]
    var token_mlp: FeedForward[Self.dtype]
    var norm_channel: LayerNorm[Self.dtype]
    var channel_mlp: FeedForward[Self.dtype]
    var seq_len: Int
    var dim: Int
    var token_hidden: Int
    var channel_hidden: Int

    def __init__(
        out self,
        seq_len: Int,
        dim: Int,
        token_hidden: Int = -1,
        channel_hidden: Int = -1,
    ) raises:
        """Initialize the Mixer block.

        Args:
            seq_len: Number of tokens (sequence length). Must be positive.
            dim: Feature dimension per token. Must be positive.
            token_hidden: Hidden width of the token-mixing MLP. If <= 0, defaults
                to `4 * seq_len` (the FeedForward 4x expansion default).
            channel_hidden: Hidden width of the channel-mixing MLP. If <= 0,
                defaults to `4 * dim`.

        Raises:
            Error: If seq_len <= 0 or dim <= 0, or if sub-layer construction fails.

        Example:
            ```mojo
            var mixer = MLPMixerBlock(seq_len=4, dim=8, token_hidden=6, channel_hidden=16)
            var y = mixer.forward(x)   # x: [2, 4, 8] -> y: [2, 4, 8]
            ```
        """
        if seq_len <= 0:
            raise Error("MLPMixerBlock: seq_len must be positive")
        if dim <= 0:
            raise Error("MLPMixerBlock: dim must be positive")
        self.seq_len = seq_len
        self.dim = dim
        # FeedForward already applies the 4x default when its d_ff arg is <= 0,
        # but resolve the concrete widths here so the attributes report the
        # actual sizes rather than the -1 sentinel.
        self.token_hidden = token_hidden if token_hidden > 0 else 4 * seq_len
        self.channel_hidden = channel_hidden if channel_hidden > 0 else 4 * dim
        # Pre-norm LayerNorms normalize over the feature (dim) axis in both
        # sublayers, per Eq. 1-2.
        self.norm_token = LayerNorm[Self.dtype](dim)
        self.norm_channel = LayerNorm[Self.dtype](dim)
        # Token MLP contracts over seq; channel MLP contracts over dim.
        self.token_mlp = FeedForward[Self.dtype](seq_len, self.token_hidden)
        self.channel_mlp = FeedForward[Self.dtype](dim, self.channel_hidden)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass through one Mixer block.

        Args:
            input: Batch-first tensor of shape `[batch, seq, dim]`, where
                `seq == seq_len` and `dim == dim`. The block consumes the whole
                sequence at once.

        Returns:
            Tensor of shape `[batch, seq, dim]` (same shape as `input`).

        Raises:
            Error: If `input` is not rank-3, or its `seq`/`dim` axes do not match
                the block's configured `seq_len`/`dim`, or a tensor op fails.
        """
        var shape = input.shape()
        if len(shape) != 3:
            raise Error(
                "MLPMixerBlock: input must be rank-3 [batch, seq, dim], got"
                " rank "
                + String(len(shape))
            )
        var batch = shape[0]
        var seq = shape[1]
        var d = shape[2]
        if seq != self.seq_len:
            raise Error(
                "MLPMixerBlock: input seq ("
                + String(seq)
                + ") does not match seq_len ("
                + String(self.seq_len)
                + ")"
            )
        if d != self.dim:
            raise Error(
                "MLPMixerBlock: input dim ("
                + String(d)
                + ") does not match dim ("
                + String(self.dim)
                + ")"
            )

        # --- Token-mixing sublayer: U = X + TokenMLP( LayerNorm_1(X)^T )^T ---
        # LayerNorm normalizes over the last (dim) axis. The MLP operating
        # position-wise over the feature axis means it acts on [*, dim]; reshape
        # [batch, seq, dim] -> [batch*seq, dim] for the norm, then back.
        var x_flat = input.reshape([batch * seq, d])
        var norm1_flat = self.norm_token.forward(x_flat)
        var norm1 = norm1_flat.reshape([batch, seq, d])
        # Transpose seq<->dim so the token MLP contracts over seq: [batch, dim, seq].
        var norm1_t = norm1.transpose(1, 2)
        # reshape needs a contiguous flat buffer; transpose returns a view.
        from odyssey.core.shape import as_contiguous

        var norm1_t_c = as_contiguous(norm1_t)
        var tok_in = norm1_t_c.reshape([batch * d, seq])
        var tok_out_flat = self.token_mlp.forward(tok_in)
        var tok_out = tok_out_flat.reshape([batch, d, seq])
        # Transpose back to [batch, seq, dim] and add the residual.
        var tok_out_t = as_contiguous(tok_out.transpose(1, 2))
        var u = input + tok_out_t

        # --- Channel-mixing sublayer: Y = U + ChannelMLP( LayerNorm_2(U) ) ---
        var u_flat = u.reshape([batch * seq, d])
        var norm2_flat = self.norm_channel.forward(u_flat)
        var chan_out_flat = self.channel_mlp.forward(norm2_flat)
        var chan_out = chan_out_flat.reshape([batch, seq, d])
        var y = u + chan_out
        return y

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from all four sub-modules.

        Returns:
            List of parameters in order: norm_token (gamma, beta),
            token_mlp (fc1.w/b, fc2.w/b), norm_channel (gamma, beta),
            channel_mlp (fc1.w/b, fc2.w/b) — 12 tensors total.

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        for p in self.norm_token.parameters():
            params.append(p)
        for p in self.token_mlp.parameters():
            params.append(p)
        for p in self.norm_channel.parameters():
            params.append(p)
        for p in self.channel_mlp.parameters():
            params.append(p)
        return params^

    def train(mut self):
        """Switch all sub-modules to training mode (all are mode-independent).
        """
        self.norm_token.train()
        self.token_mlp.train()
        self.norm_channel.train()
        self.channel_mlp.train()

    def eval(mut self):
        """Switch all sub-modules to inference mode (all are mode-independent).
        """
        self.norm_token.eval()
        self.token_mlp.eval()
        self.norm_channel.eval()
        self.channel_mlp.eval()
