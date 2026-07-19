"""Diagonal state-space (S4-style) block: a linear time-invariant SSM layer.

A single structured state-space (SSM) block in the S4 family (Gu, Goel, Re 2022,
"Efficiently Modeling Long Sequences with Structured State Spaces", arXiv:2111.00396).
It realizes the continuous-time LTI system from that paper (Sec. 2-3):

    x'(t) = A x(t) + B u(t),    y(t) = C x(t) + D u(t)

with a *diagonal* state matrix A (the diagonalizable structured variant; here the
real negative-diagonal S4D/DSS form, which keeps the layer real-valued), discretized
with zero-order hold (ZOH):

    A       = -exp(A_log)                 # stable negative-real diagonal   [dim, state]
    dt      =  exp(log_dt)                # per-channel timestep            [dim]
    dA      =  exp(dt * A)                # discretized state matrix        [dim, state]
    dB      =  (dA - 1) / A * B           # discretized input matrix        [dim, state]
    x_t[d]  =  dA[d] * x_{t-1}[d] + dB[d] * u_t[d]         # recurrence, per channel
    y_t[d]  =  sum_n C[d, n] * x_t[d, n] + D_skip[d] * u_t[d]

Channels are independent SISO systems (one diagonal SSM per input feature), so the
block maps `[batch, seq, dim] -> [batch, seq, dim]` (batch-first, like the rest of
Odyssey's sequence layers). This is the recurrent (scan) form of the S4 block; the
convolutional form of S4 computes the identical LTI output, so the recurrence here
is the ground-truth reference for both.

`forward(input)` runs the full sequence from a zero initial state (state carried
across time steps internally). `step(input, state)` advances one timestep and
returns `(y_t, x_t)` so a caller can thread state across calls (torch-RNN-cell
convention). A sequence therefore equals repeated `step` calls threading the state.

No normalization is applied inside the block (LTI by construction; add LayerNorm
outside if desired) and no BatchNorm is used.

Reference:
    Gu, A., Goel, K., & Re, C. (2022). Efficiently Modeling Long Sequences with
    Structured State Spaces. ICLR 2022. arXiv:2111.00396. (S4; diagonal-structured
    A, ZOH discretization, LTI x'=Ax+Bu, y=Cx+Du.)
"""

from std.math import exp as math_exp

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, randn
from odyssey.core.module import Module


@always_inline
def _exp_scalar[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Scalar exp with a dtype-concrete cast (proves the float constraint).

    Mirrors `odyssey.core.elementwise._exp_op`: cast to a concrete floating
    width before calling `math.exp`, so a generic `T` still type-checks.
    """
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_exp(Float32(x)))
    else:
        return Scalar[T](math_exp(Float64(x)))


struct DiagonalSSM[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Diagonal LTI state-space (S4-style) block, batch-first [batch, seq, dim].

    Each of the `dim` input features is an independent SISO diagonal SSM with
    `state` internal states, discretized with zero-order hold. Input and output
    feature dimensions are equal (`dim`).

    Parameters:
        dtype: Data type for parameters (default: float32).

    Attributes:
        a_log: [dim, state]  A = -exp(a_log) (stable negative-real diagonal).
        b: [dim, state]      input matrix B.
        c: [dim, state]      output matrix C.
        d: [dim]             direct skip term D.
        log_dt: [dim]        per-channel log timestep; dt = exp(log_dt).
        dim: Feature dimension (input == output).
        state: State dimension per channel.
    """

    var a_log: AnyTensor
    var b: AnyTensor
    var c: AnyTensor
    var d: AnyTensor
    var log_dt: AnyTensor
    var dim: Int
    var state: Int

    def __init__(out self, dim: Int, state: Int) raises:
        """Initialize the diagonal SSM block.

        Args:
            dim: Feature (channel) dimension; input and output are both `dim`.
            state: Number of internal states per channel.

        Raises:
            Error: If dim or state <= 0, or tensor construction fails.

        Example:
            ```mojo
            var ssm = DiagonalSSM(16, 4)
            var y = ssm.forward(u)   # u: [batch, seq, 16] -> y: [batch, seq, 16]
            ```
        """
        if dim <= 0:
            raise Error("DiagonalSSM: dim must be positive")
        if state <= 0:
            raise Error("DiagonalSSM: state must be positive")
        self.dim = dim
        self.state = state
        # A_log ~ small so A = -exp(A_log) starts near -1 (stable). B, C random;
        # D and log_dt zero-initialized. Callers seed exact values for tests.
        self.a_log = zeros([dim, state], Self.dtype)
        self.b = randn([dim, state], Self.dtype)
        self.c = randn([dim, state], Self.dtype)
        self.d = zeros([dim], Self.dtype)
        self.log_dt = zeros([dim], Self.dtype)

    def _discretize(self) raises -> Tuple[AnyTensor, AnyTensor]:
        """ZOH-discretize the diagonal SSM into (dA, dB), each [dim, state].

            A  = -exp(a_log);  dt = exp(log_dt)
            dA = exp(dt * A);  dB = (dA - 1) / A * B

        `a_log` and `log_dt` are assumed O(1): the stable regime dA in (0, 1]
        holds for such values, but the pre-discretization `exp(a_log)` /
        `exp(log_dt)` overflow to +inf in float32 once either exceeds ~89.

        Returns:
            Tuple (dA, dB) of shape [dim, state].

        Raises:
            Error: If tensor operations fail.
        """
        var da = zeros([self.dim, self.state], Self.dtype)
        var db = zeros([self.dim, self.state], Self.dtype)
        for ch in range(self.dim):
            var dt = _exp_scalar[Self.dtype](self.log_dt.load[Self.dtype](ch))
            for n in range(self.state):
                var idx = ch * self.state + n
                var a = -_exp_scalar[Self.dtype](
                    self.a_log.load[Self.dtype](idx)
                )
                var da_val = _exp_scalar[Self.dtype](dt * a)
                da.store[Self.dtype](idx, da_val)
                # ZOH input factor (dA - 1) / A; A < 0 strictly, no divide-by-zero.
                var db_val = (da_val - 1.0) / a * self.b.load[Self.dtype](idx)
                db.store[Self.dtype](idx, db_val)
        return (da^, db^)

    def step(
        mut self, input: AnyTensor, state: AnyTensor
    ) raises -> Tuple[AnyTensor, AnyTensor]:
        """One SSM timestep (RNN-cell convention), threading the state.

        Args:
            input: Input u_t of shape (batch, dim).
            state: Previous state x_{t-1} of shape (batch, dim, state), row-major.

        Returns:
            Tuple (y_t, x_t): output (batch, dim) and new state (batch, dim, state).

        Raises:
            Error: If tensor operations fail or shapes are incompatible.
        """
        var batch = input.shape()[0]
        var da_db = self._discretize()
        var da = da_db[0]
        var db = da_db[1]
        var y = zeros([batch, self.dim], Self.dtype)
        var x_new = zeros([batch, self.dim, self.state], Self.dtype)
        for bi in range(batch):
            for ch in range(self.dim):
                var u_val = input.load[Self.dtype](bi * self.dim + ch)
                var acc = self.d.load[Self.dtype](ch) * u_val
                for n in range(self.state):
                    var pidx = ch * self.state + n
                    var sidx = (bi * self.dim + ch) * self.state + n
                    var x_val = (
                        da.load[Self.dtype](pidx) * state.load[Self.dtype](sidx)
                        + db.load[Self.dtype](pidx) * u_val
                    )
                    x_new.store[Self.dtype](sidx, x_val)
                    acc += self.c.load[Self.dtype](pidx) * x_val
                y.store[Self.dtype](bi * self.dim + ch, acc)
        return (y^, x_new^)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Run the full sequence from a zero initial state.

        Convention note: unlike the RNN-cell siblings (`GRUCell.forward` /
        `LSTMCell.forward`), whose `forward` takes a rank-2 SINGLE timestep
        `(batch, dim)` and delegates to one `step`, this `forward` takes the
        rank-3 FULL sequence `(batch, seq, dim)` and scans internally over all
        timesteps — so a polymorphic caller must pass a full sequence here, not
        a single step. Use `step(u_t, x)` for the per-timestep interface.

        Args:
            input: Input tensor of shape (batch, seq, dim), batch-first.

        Returns:
            Output tensor of shape (batch, seq, dim); state is carried across
            all timesteps internally and discarded after the last step.

        Raises:
            Error: If tensor operations fail or the input is not rank-3.
        """
        if len(input.shape()) != 3:
            raise Error("DiagonalSSM.forward: input must be [batch, seq, dim]")
        var batch = input.shape()[0]
        var seq = input.shape()[1]
        var in_dim = input.shape()[2]
        if in_dim != self.dim:
            raise Error("DiagonalSSM.forward: input dim mismatch")

        var da_db = self._discretize()
        var da = da_db[0]
        var db = da_db[1]
        var out = zeros([batch, seq, self.dim], Self.dtype)
        # state x: [batch, dim, state], zero initial.
        var x = zeros([batch, self.dim, self.state], Self.dtype)
        for t in range(seq):
            for bi in range(batch):
                for ch in range(self.dim):
                    var u_val = input.load[Self.dtype](
                        (bi * seq + t) * self.dim + ch
                    )
                    var acc = self.d.load[Self.dtype](ch) * u_val
                    for n in range(self.state):
                        var pidx = ch * self.state + n
                        var sidx = (bi * self.dim + ch) * self.state + n
                        var x_val = (
                            da.load[Self.dtype](pidx) * x.load[Self.dtype](sidx)
                            + db.load[Self.dtype](pidx) * u_val
                        )
                        x.store[Self.dtype](sidx, x_val)
                        acc += self.c.load[Self.dtype](pidx) * x_val
                    out.store[Self.dtype]((bi * seq + t) * self.dim + ch, acc)
        return out^

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters.

        Returns:
            List [a_log, b, c, d, log_dt] (5 tensors).

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        params.append(self.a_log)
        params.append(self.b)
        params.append(self.c)
        params.append(self.d)
        params.append(self.log_dt)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; the block has no train/eval state).
        """
        pass

    def eval(mut self):
        """Switch to inference mode (no-op; the block has no train/eval state).
        """
        pass
