"""Long Short-Term Memory (LSTM) cell.

A single recurrent step of an LSTM (Hochreiter & Schmidhuber, 1997), matching the
`torch.nn.LSTMCell` convention:

    i = sigmoid(x W_ii + b_ii + h W_hi + b_hi)     # input gate
    f = sigmoid(x W_if + b_if + h W_hf + b_hf)     # forget gate
    g = tanh(x W_ig + b_ig + h W_hg + b_hg)        # candidate cell
    o = sigmoid(x W_io + b_io + h W_ho + b_ho)     # output gate
    c' = f * c + i * g                             # new cell state
    h' = o * tanh(c')                              # new hidden state

Each of the four gates uses a separate input-to-hidden and hidden-to-hidden `Linear`
projection (eight projections total). `step(input, hidden, cell)` computes one step
and returns the new `(hidden, cell)` pair; a sequence is processed by the caller
looping over time steps (same convention as `torch.nn.LSTMCell`).

Reference:
    Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural
    Computation, 9(8), 1735-1780. Interface mirrors `torch.nn.LSTMCell`.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like, zeros
from odyssey.core.module import Module
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import sigmoid, tanh
from odyssey.core.arithmetic_simd import (
    multiply_simd,
    add_simd,
)


struct LSTMCell[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Long Short-Term Memory cell (torch.nn.LSTMCell convention).

    Parameters:
        dtype: Data type for weights and biases (default: float32).

    Attributes:
        ii, if_, ig, io: input-to-hidden projections for input/forget/candidate/output.
        hi, hf, hg, ho: hidden-to-hidden projections for input/forget/candidate/output.
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
    """

    var ii: Linear[Self.dtype]
    var if_: Linear[Self.dtype]
    var ig: Linear[Self.dtype]
    var io: Linear[Self.dtype]
    var hi: Linear[Self.dtype]
    var hf: Linear[Self.dtype]
    var hg: Linear[Self.dtype]
    var ho: Linear[Self.dtype]
    var input_size: Int
    var hidden_size: Int

    def __init__(out self, input_size: Int, hidden_size: Int) raises:
        """Initialize the LSTM cell.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.

        Raises:
            Error: If input_size or hidden_size <= 0, or construction fails.

        Example:
            ```mojo
            var cell = LSTMCell(3, 4)
            var hc = cell.step(x, h0, c0)   # x: [B, 3]; h0, c0: [B, 4]
            ```
        """
        if input_size <= 0:
            raise Error("LSTMCell: input_size must be positive")
        if hidden_size <= 0:
            raise Error("LSTMCell: hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ii = Linear[Self.dtype](input_size, hidden_size)
        self.if_ = Linear[Self.dtype](input_size, hidden_size)
        self.ig = Linear[Self.dtype](input_size, hidden_size)
        self.io = Linear[Self.dtype](input_size, hidden_size)
        self.hi = Linear[Self.dtype](hidden_size, hidden_size)
        self.hf = Linear[Self.dtype](hidden_size, hidden_size)
        self.hg = Linear[Self.dtype](hidden_size, hidden_size)
        self.ho = Linear[Self.dtype](hidden_size, hidden_size)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Module `forward` from a zero initial hidden and cell state.

        Runs a single `step` from all-zero hidden and cell states, so this is
        exactly the hidden output of `step(input, zeros, zeros)` — the
        hidden-to-hidden projections still contribute their biases (a zero hidden
        zeros only the weight terms, not `b_hi`/`b_hf`/`b_hg`/`b_ho`). Returns only
        the new hidden state (the new cell state is discarded). Use
        `step(input, hidden, cell)` to thread real recurrent state.

        Args:
            input: Input tensor of shape (batch, input_size).

        Returns:
            Hidden state of shape (batch, hidden_size).

        Raises:
            Error: If tensor operations fail.
        """
        var batch = input.shape()[0]
        var h0 = zeros([batch, self.hidden_size], Self.dtype)
        var c0 = zeros([batch, self.hidden_size], Self.dtype)
        var hc = self.step(input, h0, c0)
        return hc[0]

    def step(
        mut self, input: AnyTensor, hidden: AnyTensor, cell: AnyTensor
    ) raises -> Tuple[AnyTensor, AnyTensor]:
        """One LSTM step (torch.nn.LSTMCell convention).

        Args:
            input: Input tensor x_t of shape (batch, input_size).
            hidden: Previous hidden state h_{t-1} of shape (batch, hidden_size).
            cell: Previous cell state c_{t-1} of shape (batch, hidden_size).

        Returns:
            Tuple (h_t, c_t): new hidden state and new cell state, each of shape
            (batch, hidden_size).

        Raises:
            Error: If tensor operations fail or shapes are incompatible.
        """
        var i = sigmoid(
            add_simd(self.ii.forward(input), self.hi.forward(hidden))
        )
        var f = sigmoid(
            add_simd(self.if_.forward(input), self.hf.forward(hidden))
        )
        var g = tanh(add_simd(self.ig.forward(input), self.hg.forward(hidden)))
        var o = sigmoid(
            add_simd(self.io.forward(input), self.ho.forward(hidden))
        )
        # c' = f * c + i * g
        var c_new = add_simd(multiply_simd(f, cell), multiply_simd(i, g))
        # h' = o * tanh(c')
        var h_new = multiply_simd(o, tanh(c_new))
        return (h_new, c_new)

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from all eight projections.

        Returns:
            List of 16 tensors (weight+bias of ii, if_, ig, io, hi, hf, hg, ho).

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        for p in self.ii.parameters():
            params.append(p)
        for p in self.if_.parameters():
            params.append(p)
        for p in self.ig.parameters():
            params.append(p)
        for p in self.io.parameters():
            params.append(p)
        for p in self.hi.parameters():
            params.append(p)
        for p in self.hf.parameters():
            params.append(p)
        for p in self.hg.parameters():
            params.append(p)
        for p in self.ho.parameters():
            params.append(p)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; sub-layers are stateless in mode)."""
        self.ii.train()
        self.if_.train()
        self.ig.train()
        self.io.train()
        self.hi.train()
        self.hf.train()
        self.hg.train()
        self.ho.train()

    def eval(mut self):
        """Switch to inference mode (no-op; sub-layers are stateless in mode).
        """
        self.ii.eval()
        self.if_.eval()
        self.ig.eval()
        self.io.eval()
        self.hi.eval()
        self.hf.eval()
        self.hg.eval()
        self.ho.eval()
