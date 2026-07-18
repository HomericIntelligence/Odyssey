"""Liquid Time-constant (LTC) recurrent cell.

A single continuous-time recurrent step of a Liquid Time-constant network
(Hasani et al., 2021), integrated with the paper's *fused* (semi-implicit)
ODE solver.

Formulation (Hasani et al. 2021, arXiv:2006.04439):

    LTC ODE (Eq. 1):
        dx/dt = -[1/tau + f(x, I, theta)] * x(t) + f(x, I, theta) * A

    where
        x     : hidden state              (batch, hidden)
        I     : input                     (batch, input)
        tau   : learnable time-constant   (hidden,)   (elementwise, > 0)
        A     : learnable bias vector     (hidden,)   (bounds the equilibrium)
        f     : bounded gating network    f = tanh(I @ gamma + x @ gamma_r + mu)
                (arXiv:2006.04439 §3; tanh is the paper's primary choice)

    Fused ODE solver (arXiv:2006.04439, Algorithm 1 / Eq. for the fused solver),
    the exact discrete per-sub-step used here — element-wise over (batch, hidden):

        x(t+dt) = ( x(t) + dt * f(x, I) * A )
                  / ( 1 + dt * (1/tau + f(x, I)) )

    This semi-implicit update is stable for any dt > 0 (denominator > 1 whenever
    1/tau + f > 0). We unfold it `solver_steps` (L) times per `step()` call over a
    total elapsed time `elapsed`, with sub-step dt = elapsed / L. The input I is
    held constant across the L sub-steps of one `step()` call (zero-order hold),
    matching the paper's per-time-step solver unfolding.

Forward contract (documented explicitly — the single-step vs full-sequence
ambiguity was flagged in a sibling recurrent-cell review):

    `step(input, hidden)` advances the state by ONE input time step (running L
    fused-solver sub-steps internally) and returns the new hidden state. A full
    sequence is processed by the CALLER looping over time steps and threading the
    returned hidden state — identical convention to `RNNCell` / `GRUCell` /
    `LSTMCell` in this package. There is no built-in full-sequence `forward`;
    `forward(input)` is the Module entry point and runs exactly ONE step from an
    all-zero hidden state (so `forward(x) == step(x, zeros)`).

Numerical assumptions:
    - `tau` must be positive; it is initialized to 1.0 (so 1/tau = 1.0) and is a
      trainable parameter. Callers that train it should keep it positive (e.g.
      via a softplus reparameterization in the caller) — the cell itself does NOT
      clamp tau, to preserve byte-identical forward behavior.
    - With dt = elapsed/L <= 1 the denominator 1 + dt*(1/tau + f) > 0 for every
      tau > 0 (since 1/tau + f > -1 >= -1/dt); the paper's stronger 1/tau + f > 0
      property additionally holds when tau < 1.
    - Denominator positivity is guaranteed only for dt <= 1. With elapsed/L > 1
      (i.e. elapsed > solver_steps) and a learned tau > 1/(1 - 1/dt), the
      denominator can reach <= 0 and the update blows up — so keep
      elapsed <= solver_steps (or clamp tau in the caller). The cell adds NO
      runtime guard: a clamp would break the byte-identical forward contract, so
      this is a documented caller responsibility.
    - Scalar loads use the tensor's own dtype (`Self.dtype`) — the compile-time
      dtype contract used across this package.

Reference:
    Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021).
    Liquid Time-constant Networks. AAAI-21. arXiv:2006.04439.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full, full_like, zeros
from odyssey.core.module import Module
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import tanh
from odyssey.core.arithmetic_simd import (
    add_simd,
    subtract_simd,
    multiply_simd,
    divide_simd,
)


struct LTCCell[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Liquid Time-constant cell with the paper's fused ODE solver.

    Parameters:
        dtype: Data type for weights and parameters (default: float32).

    Attributes:
        wi: input-to-hidden projection (carries the shared bias mu).
        wh: hidden-to-hidden (recurrent) projection (bias forced to zero, so the
            gating network has the single bias mu of the paper's f).
        tau: learnable time-constant vector, shape (hidden_size,), init 1.0.
        a: learnable equilibrium-bias vector A, shape (hidden_size,), init 0.0.
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        solver_steps: Number of fused-solver sub-steps L per `step()`.
        elapsed: Total elapsed time per `step()` (sub-step dt = elapsed/L).
    """

    var wi: Linear[Self.dtype]
    var wh: Linear[Self.dtype]
    var tau: AnyTensor
    var a: AnyTensor
    var input_size: Int
    var hidden_size: Int
    var solver_steps: Int
    var elapsed: Float64

    def __init__(
        out self,
        input_size: Int,
        hidden_size: Int,
        solver_steps: Int = 6,
        elapsed: Float64 = 1.0,
    ) raises:
        """Initialize the LTC cell.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.
            solver_steps: Fused-solver unfolding steps L per `step()` (default 6,
                a common LTC setting; arXiv:2006.04439 uses a variable L).
            elapsed: Total elapsed time advanced per `step()` (default 1.0). The
                per-sub-step dt is `elapsed / solver_steps`.

        Raises:
            Error: If input_size, hidden_size, or solver_steps <= 0, or elapsed
                <= 0, or construction fails.

        Example:
            ```mojo
            var cell = LTCCell(3, 4)
            var h1 = cell.step(x, h0)   # x: [batch, 3], h0: [batch, 4]
            ```
        """
        if input_size <= 0:
            raise Error("LTCCell: input_size must be positive")
        if hidden_size <= 0:
            raise Error("LTCCell: hidden_size must be positive")
        if solver_steps <= 0:
            raise Error("LTCCell: solver_steps must be positive")
        if elapsed <= 0.0:
            raise Error("LTCCell: elapsed must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.solver_steps = solver_steps
        self.elapsed = elapsed
        self.wi = Linear[Self.dtype](input_size, hidden_size)
        self.wh = Linear[Self.dtype](hidden_size, hidden_size)
        # The gating network f has a single bias mu, carried by wi. Linear
        # initializes its bias to zeros, so wh's bias is already zero here; the
        # cell never sets it, keeping f = tanh(I@gamma + x@gamma_r + mu) with the
        # single bias mu. (Any caller that later trains wh.bias breaks the
        # single-bias formulation; the parity reference keeps it at zero.)
        # tau initialized to 1.0 (=> 1/tau = 1.0); A initialized to 0.0.
        self.tau = full([hidden_size], 1.0, Self.dtype)
        self.a = zeros([hidden_size], Self.dtype)

    def _f(mut self, input: AnyTensor, hidden: AnyTensor) raises -> AnyTensor:
        """Gating network f = tanh(input @ gamma + hidden @ gamma_r + mu).

        Args:
            input: Input tensor of shape (batch, input_size).
            hidden: Hidden state of shape (batch, hidden_size).

        Returns:
            f evaluated element-wise, shape (batch, hidden_size).

        Raises:
            Error: If tensor operations fail.
        """
        return tanh(add_simd(self.wi.forward(input), self.wh.forward(hidden)))

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Module `forward`: ONE LTC step from an all-zero initial hidden state.

        Runs a single `step` from an all-zero hidden state, so this is exactly
        `step(input, zeros)`. Use `step(input, hidden)` to thread a real
        recurrent hidden state across a sequence.

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
        """One LTC time step via the fused ODE solver (L sub-steps).

        Advances the hidden state by one input time step (total time `elapsed`,
        split into `solver_steps` fused-solver sub-steps of dt = elapsed/L). The
        input is held constant across the sub-steps (zero-order hold). Each
        sub-step applies, element-wise over (batch, hidden):

            x <- (x + dt * f(x, I) * A) / (1 + dt * (1/tau + f(x, I)))

        Args:
            input: Input tensor x_t of shape (batch, input_size).
            hidden: Previous hidden state h_{t-1} of shape (batch, hidden_size).

        Returns:
            New hidden state h_t of shape (batch, hidden_size).

        Raises:
            Error: If tensor operations fail or shapes are incompatible.
        """
        var dt = self.elapsed / Float64(self.solver_steps)
        # inv_tau = 1 / tau  (shape (hidden,), broadcasts over batch)
        var inv_tau = divide_simd(full_like(self.tau, 1.0), self.tau)
        var x = hidden
        for _ in range(self.solver_steps):
            var f = self._f(input, x)  # (batch, hidden)
            # numerator = x + dt * f * A
            var num = add_simd(
                x, multiply_simd(full_like(f, dt), multiply_simd(f, self.a))
            )
            # denominator = 1 + dt * (1/tau + f)
            var denom = add_simd(
                full_like(f, 1.0),
                multiply_simd(full_like(f, dt), add_simd(inv_tau, f)),
            )
            x = divide_simd(num, denom)
        return x

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters.

        Returns:
            List of 6 tensors: wi weight+bias, wh weight+bias, tau, A.

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        for p in self.wi.parameters():
            params.append(p)
        for p in self.wh.parameters():
            params.append(p)
        params.append(self.tau)
        params.append(self.a)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; sub-layers are stateless in mode)."""
        self.wi.train()
        self.wh.train()

    def eval(mut self):
        """Switch to inference mode (no-op; sub-layers are stateless in mode).
        """
        self.wi.eval()
        self.wh.eval()
