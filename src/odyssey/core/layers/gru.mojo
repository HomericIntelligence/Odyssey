"""Gated Recurrent Unit (GRU) cell.

A single recurrent step of a GRU (Cho et al., 2014), matching the
`torch.nn.GRUCell` convention:

    r = sigmoid(x W_ir + b_ir + h W_hr + b_hr)          # reset gate
    z = sigmoid(x W_iz + b_iz + h W_hz + b_hz)          # update gate
    n = tanh(x W_in + b_in + r * (h W_hn + b_hn))       # candidate state
    h' = (1 - z) * n + z * h                            # new hidden state

Note the reset gate `r` multiplies ONLY the hidden contribution to the candidate
`n` (after its bias), which is the PyTorch / cuDNN convention. Each of the three
gates uses a separate input-to-hidden and hidden-to-hidden `Linear` projection.

`step(input, hidden)` computes one step; a sequence is processed by the caller
looping over time steps (same convention as `torch.nn.GRUCell`).

Reference:
    Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
    Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN
    Encoder-Decoder for Statistical Machine Translation. arXiv:1406.1078.
    Interface mirrors `torch.nn.GRUCell`.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like, zeros
from odyssey.core.module import Module
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import sigmoid, tanh
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
)


struct GRUCell[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Gated Recurrent Unit cell (torch.nn.GRUCell convention).

    Parameters:
        dtype: Data type for weights and biases (default: float32).

    Attributes:
        ir, iz, in_: input-to-hidden projections for reset/update/candidate.
        hr, hz, hn: hidden-to-hidden projections for reset/update/candidate.
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
    """

    var ir: Linear[Self.dtype]
    var iz: Linear[Self.dtype]
    var in_: Linear[Self.dtype]
    var hr: Linear[Self.dtype]
    var hz: Linear[Self.dtype]
    var hn: Linear[Self.dtype]
    var input_size: Int
    var hidden_size: Int

    def __init__(out self, input_size: Int, hidden_size: Int) raises:
        """Initialize the GRU cell.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.

        Raises:
            Error: If input_size or hidden_size <= 0, or construction fails.

        Example:
            ```mojo
            var cell = GRUCell(3, 4)
            var h1 = cell.step(x, h0)   # x: [batch, 3], h0: [batch, 4]
            ```
        """
        if input_size <= 0:
            raise Error("GRUCell: input_size must be positive")
        if hidden_size <= 0:
            raise Error("GRUCell: hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ir = Linear[Self.dtype](input_size, hidden_size)
        self.iz = Linear[Self.dtype](input_size, hidden_size)
        self.in_ = Linear[Self.dtype](input_size, hidden_size)
        self.hr = Linear[Self.dtype](hidden_size, hidden_size)
        self.hz = Linear[Self.dtype](hidden_size, hidden_size)
        self.hn = Linear[Self.dtype](hidden_size, hidden_size)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Module `forward` from a zero initial hidden state.

        Runs a single `step` from an all-zero hidden state, so this is exactly
        `step(input, zeros)` — the hidden-to-hidden projections still contribute
        their biases (h=0 zeros only the weight terms, not `b_hr`/`b_hz`/`b_hn`).
        Use `step(input, hidden)` to thread a real recurrent hidden state.

        Args:
            input: Input tensor of shape (batch, input_size).

        Returns:
            Hidden state of shape (batch, hidden_size).

        Raises:
            Error: If tensor operations fail.
        """
        var batch = input.shape()[0]
        var h0 = zeros([batch, self.hidden_size], Self.dtype)
        return self.step(input, h0)

    def step(mut self, input: AnyTensor, hidden: AnyTensor) raises -> AnyTensor:
        """One GRU step (torch.nn.GRUCell convention).

        Args:
            input: Input tensor x_t of shape (batch, input_size).
            hidden: Previous hidden state h_{t-1} of shape (batch, hidden_size).

        Returns:
            New hidden state h_t of shape (batch, hidden_size).

        Raises:
            Error: If tensor operations fail or shapes are incompatible.
        """
        var r = sigmoid(
            add_simd(self.ir.forward(input), self.hr.forward(hidden))
        )
        var z = sigmoid(
            add_simd(self.iz.forward(input), self.hz.forward(hidden))
        )
        # candidate: r gates ONLY the hidden contribution (after its bias)
        var n = tanh(
            add_simd(
                self.in_.forward(input),
                multiply_simd(r, self.hn.forward(hidden)),
            )
        )
        # h' = (1 - z) * n + z * h
        var one_minus_z = subtract_simd(full_like(z, 1.0), z)
        return add_simd(multiply_simd(one_minus_z, n), multiply_simd(z, hidden))

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from all six projections.

        Returns:
            List of 12 tensors (weight+bias of ir, iz, in_, hr, hz, hn).

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        for p in self.ir.parameters():
            params.append(p)
        for p in self.iz.parameters():
            params.append(p)
        for p in self.in_.parameters():
            params.append(p)
        for p in self.hr.parameters():
            params.append(p)
        for p in self.hz.parameters():
            params.append(p)
        for p in self.hn.parameters():
            params.append(p)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; sub-layers are stateless in mode)."""
        self.ir.train()
        self.iz.train()
        self.in_.train()
        self.hr.train()
        self.hz.train()
        self.hn.train()

    def eval(mut self):
        """Switch to inference mode (no-op; sub-layers are stateless in mode).
        """
        self.ir.eval()
        self.iz.eval()
        self.in_.eval()
        self.hr.eval()
        self.hz.eval()
        self.hn.eval()
