"""Transformer feed-forward network (FFN / MLP) block.

The position-wise feed-forward sub-layer of a Transformer block (Vaswani et al.,
"Attention Is All You Need", 2017, §3.3): two linear projections with a
non-linearity in between, applied identically to every position.

    FFN(x) = activation(x W1 + b1) W2 + b2

with an inner (hidden) dimension `d_ff` that is conventionally larger than the
model dimension `d_model` (the paper uses d_ff = 4 * d_model). The default
activation is GELU (as used by BERT/GPT); ReLU (the original Transformer choice)
is available via `use_gelu=False`.

This is a thin composition of two `Linear` layers and an activation, so it
inherits their initialization and matmul/bias-broadcast behavior; it exists as a
first-class `Module` so a Transformer block can hold it directly and so its
parameters are collected uniformly by optimizers.

Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 2017.
    arXiv:1706.03762, §3.3 "Position-wise Feed-Forward Networks".
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.module import Module
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import gelu, relu


struct FeedForward[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Transformer position-wise feed-forward block: Linear -> act -> Linear.

    Parameters:
        dtype: Data type for weights and biases (default: float32).

    Attributes:
        fc1: First linear projection (d_model -> d_ff).
        fc2: Second linear projection (d_ff -> d_model).
        d_model: Input/output feature dimension.
        d_ff: Inner (hidden) feature dimension.
        use_gelu: If True use GELU activation, else ReLU.
    """

    var fc1: Linear[Self.dtype]
    var fc2: Linear[Self.dtype]
    var d_model: Int
    var d_ff: Int
    var use_gelu: Bool

    def __init__(
        out self,
        d_model: Int,
        d_ff: Int = -1,
        use_gelu: Bool = True,
    ) raises:
        """Initialize the feed-forward block.

        Args:
            d_model: Model (input and output) dimension.
            d_ff: Inner hidden dimension. If <= 0, defaults to 4 * d_model
                (the ratio used in the original Transformer).
            use_gelu: Use GELU (default, BERT/GPT style) when True; ReLU (the
                original Transformer activation) when False.

        Raises:
            Error: If d_model <= 0, or if layer construction fails.

        Example:
            ```mojo
            var ffn = FeedForward(512)          # d_ff defaults to 2048
            var out = ffn.forward(x)            # x: [batch, 512] -> [batch, 512]
            ```
        """
        if d_model <= 0:
            raise Error("FeedForward: d_model must be positive")
        var inner = d_ff if d_ff > 0 else 4 * d_model
        self.d_model = d_model
        self.d_ff = inner
        self.use_gelu = use_gelu
        self.fc1 = Linear[Self.dtype](d_model, inner)
        self.fc2 = Linear[Self.dtype](inner, d_model)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: FFN(x) = activation(x W1 + b1) W2 + b2.

        Args:
            input: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of shape (..., d_model) (same shape as input).

        Raises:
            Error: If tensor operations fail or the last dim != d_model.
        """
        var hidden = self.fc1.forward(input)
        var activated = gelu(hidden) if self.use_gelu else relu(hidden)
        return self.fc2.forward(activated)

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from both linear sub-layers.

        Returns:
            List of [fc1.weight, fc1.bias, fc2.weight, fc2.bias].

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        for p in self.fc1.parameters():
            params.append(p)
        for p in self.fc2.parameters():
            params.append(p)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; sub-layers are stateless in mode)."""
        self.fc1.train()
        self.fc2.train()

    def eval(mut self):
        """Switch to inference mode (no-op; sub-layers are stateless in mode).
        """
        self.fc1.eval()
        self.fc2.eval()
