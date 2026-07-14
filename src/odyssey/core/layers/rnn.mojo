"""Vanilla (Elman) RNN cell.

A single recurrent step of a vanilla Elman RNN (Elman, 1990):

    h_t = tanh(x_t @ W_ih + b_ih + h_{t-1} @ W_hh + b_hh)

The cell holds the input-to-hidden and hidden-to-hidden projections as two
`Linear` layers and applies a tanh nonlinearity to their sum. `forward` computes
one step given the current input `x_t` and previous hidden state `h_{t-1}`; a
sequence is processed by the caller looping `forward` over time steps (the same
convention as PyTorch's `nn.RNNCell`).

Reference:
    Elman, J. L. (1990). Finding Structure in Time. Cognitive Science, 14(2),
    179-211. (Vanilla / Elman RNN.)
    Interface mirrors `torch.nn.RNNCell` (nonlinearity='tanh').
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.module import Module
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import tanh


struct RNNCell[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Vanilla Elman RNN cell: h_t = tanh(x_t W_ih + b_ih + h_{t-1} W_hh + b_hh).

    Parameters:
        dtype: Data type for weights and biases (default: float32).

    Attributes:
        ih: Input-to-hidden projection (input_size -> hidden_size).
        hh: Hidden-to-hidden projection (hidden_size -> hidden_size).
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
    """

    var ih: Linear[Self.dtype]
    var hh: Linear[Self.dtype]
    var input_size: Int
    var hidden_size: Int

    def __init__(out self, input_size: Int, hidden_size: Int) raises:
        """Initialize the RNN cell.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.

        Raises:
            Error: If input_size or hidden_size <= 0, or layer construction fails.

        Example:
            ```mojo
            var cell = RNNCell(3, 4)
            var h1 = cell.forward(x, h0)   # x: [batch, 3], h0: [batch, 4]
            ```
        """
        if input_size <= 0:
            raise Error("RNNCell: input_size must be positive")
        if hidden_size <= 0:
            raise Error("RNNCell: hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = Linear[Self.dtype](input_size, hidden_size)
        self.hh = Linear[Self.dtype](hidden_size, hidden_size)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Module `forward` for a zero initial hidden state.

        Computes one step from a zero hidden state:
        `h = tanh(input @ W_ih + b_ih)`. Use `step(input, hidden)` to thread a
        real recurrent hidden state across time.

        Args:
            input: Input tensor of shape (batch, input_size).

        Returns:
            Hidden state of shape (batch, hidden_size).

        Raises:
            Error: If tensor operations fail.
        """
        return tanh(self.ih.forward(input))

    def step(mut self, input: AnyTensor, hidden: AnyTensor) raises -> AnyTensor:
        """One recurrent step: h_t = tanh(x_t W_ih + b_ih + h_{t-1} W_hh + b_hh).

        Args:
            input: Input tensor x_t of shape (batch, input_size).
            hidden: Previous hidden state h_{t-1} of shape (batch, hidden_size).

        Returns:
            New hidden state h_t of shape (batch, hidden_size).

        Raises:
            Error: If tensor operations fail or shapes are incompatible.
        """
        var pre = self.ih.forward(input) + self.hh.forward(hidden)
        return tanh(pre)

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from both projections.

        Returns:
            List of [ih.weight, ih.bias, hh.weight, hh.bias].

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        for p in self.ih.parameters():
            params.append(p)
        for p in self.hh.parameters():
            params.append(p)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; sub-layers are stateless in mode)."""
        self.ih.train()
        self.hh.train()

    def eval(mut self):
        """Switch to inference mode (no-op; sub-layers are stateless in mode).
        """
        self.ih.eval()
        self.hh.eval()
