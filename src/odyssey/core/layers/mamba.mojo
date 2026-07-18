"""Mamba selective state-space (S6) block: an input-dependent SSM layer.

A single selective state-space (S6) block from Gu & Dao 2023, "Mamba:
Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752),
Sec. 3.2, Algorithm 2 (SSM + Selection). Unlike an LTI S4 block (such as the
planned S4-LTI sibling `DiagonalSSM` in PR #5636), whose B/C/Δ are fixed
parameters, the S6 block makes
B, C, and Δ *functions of the input* (Sec. 3.1, "Selection") while keeping the
diagonal state matrix A input-independent. This block also adds the two structural
pieces that surround the selective scan in Mamba: a short CAUSAL depthwise
convolution over the sequence (SiLU-activated) and a gated (SiLU) branch.

Batch-first [batch, seq, dim]; input feature dim == output feature dim == `dim`.
This is a research-primitive-scale block (no expansion factor / no `d_inner`
projection; the x-branch is the input directly and the gate branch is a single
learned linear), kept small-dim-friendly (dim ~8-32, state ~4-16) for the
optimizer×clip×LR ablation — it is NOT the full production Mamba module.

Forward computation (per batch element, all shapes batch-first):

    z        = u @ Wz + bz                              # gate pre-activation [B,L,D]
    # causal depthwise conv over the x-branch (x-branch = u), then SiLU:
    xc[.,t,d] = SiLU( sum_{j} conv_w[d,j] * u_pad[.,t+j,d] + conv_b[d] )   # [B,L,D]
                (u_pad = u left-padded by K-1 zeros -> output[t] sees inputs <= t)
    # input-dependent (selective) SSM parameters from the conv'd x:
    B        = xc @ WB                                  # [B,L,N]  (shared over D)
    C        = xc @ WC                                  # [B,L,N]  (shared over D)
    delta    = softplus( xc @ Wdt + dt_bias )           # [B,L,D]  per-channel Δ > 0
    A        = -exp(A_log)                              # [D,N] stable neg. diagonal
    # discretize (Mamba simplified ZOH) + selective scan:
    dA[.,t,d,n] = exp( delta[.,t,d] * A[d,n] )
    dB[.,t,d,n] = delta[.,t,d] * B[.,t,n]
    h[.,t,d,n]  = dA * h[.,t-1,d,n] + dB * xc[.,t,d]
    y[.,t,d]    = sum_n C[.,t,n] * h[.,t,d,n] + D_skip[d] * xc[.,t,d]
    # gated output projection:
    out      = (y * SiLU(z)) @ Wo + bo                  # [B,L,D]

The discretization uses the official implementation's simplified ZOH
`dB = Δ·B` (a first-order / Euler approximation), NOT the S4 `(dA-1)/A·B`
factor — this is the intentional S6-vs-S4 discretization difference.

`forward(input)` runs the FULL sequence from a zero initial SSM state, scanning
internally over all timesteps. Because B/C/Δ and the causal conv are all
input-dependent, there is no independent RNN-cell `step` interface (the selective
parameters are recomputed per timestep from the conv'd input); a caller must pass
a full sequence [batch, seq, dim] to `forward`. This follows the same
full-sequence `forward` convention planned for the S4-LTI sibling
(`DiagonalSSM.forward`, PR #5636): `forward` consumes a
FULL sequence, not a single timestep.

Dtypes: parameters and all internal accumulation are the layer's `dtype` (default
float32). No float64 is used inside the layer; the parity test's float64 ground
truth lives only in the generator. All scalar loads use `Self.dtype` — this is a
COMPILE-TIME contract: `AnyTensor.load[dtype]` is a raw bitcast that does not
check the runtime dtype, so a mismatched load would silently misread bytes. The
layer never loads a tensor at a dtype other than `Self.dtype`.

Numerical-stability assumptions (documented per the SSM review's exp-overflow
flag): `A_log`, `Wdt`/`dt_bias`, and `conv`/`Wz` params are assumed O(1) so that
(a) `exp(A_log)` stays finite (float32 `exp` overflows to +inf past ~89), and
(b) the selective `Δ = softplus(...)` and `dA = exp(Δ·A)` stay in the stable
regime: with Δ > 0 and A < 0, `Δ·A < 0` so `dA ∈ (0, 1]` (no growth). Softplus is
computed in the numerically stable form `max(0,x) + log(1 + exp(-|x|))`, finite
for all inputs. These assumptions hold for the ramp-seeded test scale and for
normally-initialized small-dim research configs; extreme parameter magnitudes
(|A_log| or |Wdt·x + dt_bias| ≳ 80) can overflow and are out of scope.

No normalization is applied inside the block, and no BatchNorm is used (BatchNorm
couples per-batch-element updates and is disallowed for the PC locality property).

Reference:
    Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces. arXiv:2312.00752. (S6 selective SSM: input-dependent B/C/Δ,
    causal depthwise conv, SiLU gate, diagonal A, simplified-ZOH discretization;
    Sec. 3.2, Algorithm 2.)
"""

from std.math import exp as math_exp, log1p as math_log1p

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, randn
from odyssey.core.module import Module


@always_inline
def _exp_scalar[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Scalar exp with a dtype-concrete cast (proves the float constraint).

    Casts to a concrete floating width before calling `math.exp`, so a generic
    `T` still type-checks (a bare `math.exp(Scalar[T])` does not).
    """
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_exp(Float32(x)))
    else:
        return Scalar[T](math_exp(Float64(x)))


@always_inline
def _sigmoid_scalar[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Numerically stable scalar sigmoid: 1 / (1 + exp(-x))."""
    return Scalar[T](1.0) / (Scalar[T](1.0) + _exp_scalar[T](-x))


@always_inline
def _silu_scalar[T: DType](x: Scalar[T]) -> Scalar[T]:
    """SiLU / swish: x * sigmoid(x)."""
    return x * _sigmoid_scalar[T](x)


@always_inline
def _softplus_scalar[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Numerically stable scalar softplus: max(0,x) + log(1 + exp(-|x|)).

    Finite for all inputs; the `exp(-|x|)` argument is always <= 0 so it never
    overflows (log1p keeps small-argument precision).
    """
    var pos = x if x > Scalar[T](0.0) else Scalar[T](0.0)
    var ax = x if x >= Scalar[T](0.0) else -x
    comptime if T == DType.float16 or T == DType.float32:
        var lt = Float32(math_log1p(Float32(math_exp(Float32(-ax)))))
        return pos + Scalar[T](lt)
    else:
        var lt = Float64(math_log1p(Float64(math_exp(Float64(-ax)))))
        return pos + Scalar[T](lt)


struct MambaBlock[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Mamba selective state-space (S6) block, batch-first [batch, seq, dim].

    Input feature dimension equals output feature dimension (`dim`). Internal SSM
    state has `state` dimensions per channel; the causal depthwise conv has a
    `conv_kernel`-wide per-channel kernel. B, C, and Δ are input-dependent
    (selective); A is a fixed stable diagonal.

    Parameters:
        dtype: Data type for parameters (default: float32).

    Attributes:
        wz: [dim, dim]           gate-branch projection weight (row-major, in->out)
        bz: [dim]                gate-branch bias
        conv_w: [dim, conv_kernel]  per-channel causal depthwise conv kernel
        conv_b: [dim]            per-channel conv bias
        w_b: [dim, state]        input->B selective projection
        w_c: [dim, state]        input->C selective projection
        w_dt: [dim, dim]         input->Δ selective projection
        dt_bias: [dim]           Δ bias
        a_log: [dim, state]      A = -exp(a_log) (stable negative-real diagonal)
        d: [dim]                 direct skip term D
        wo: [dim, dim]           output projection weight (row-major, in->out)
        bo: [dim]                output projection bias
        dim: Feature dimension (input == output).
        state: SSM state dimension per channel.
        conv_kernel: Causal depthwise conv width.
    """

    var wz: AnyTensor
    var bz: AnyTensor
    var conv_w: AnyTensor
    var conv_b: AnyTensor
    var w_b: AnyTensor
    var w_c: AnyTensor
    var w_dt: AnyTensor
    var dt_bias: AnyTensor
    var a_log: AnyTensor
    var d: AnyTensor
    var wo: AnyTensor
    var bo: AnyTensor
    var dim: Int
    var state: Int
    var conv_kernel: Int

    def __init__(out self, dim: Int, state: Int, conv_kernel: Int = 4) raises:
        """Initialize the Mamba selective-SSM block.

        Args:
            dim: Feature (channel) dimension; input and output are both `dim`.
            state: Number of internal SSM states per channel.
            conv_kernel: Causal depthwise conv width (default: 4).

        Raises:
            Error: If dim, state, or conv_kernel <= 0, or tensor construction
                fails.

        Example:
            ```mojo
            var mamba = MambaBlock(16, 4, 3)
            var y = mamba.forward(u)   # u:[batch,seq,16] -> y:[batch,seq,16]
            ```
        """
        if dim <= 0:
            raise Error("MambaBlock: dim must be positive")
        if state <= 0:
            raise Error("MambaBlock: state must be positive")
        if conv_kernel <= 0:
            raise Error("MambaBlock: conv_kernel must be positive")
        self.dim = dim
        self.state = state
        self.conv_kernel = conv_kernel
        # A_log ~ small so A = -exp(A_log) starts near -1 (stable). Projections
        # random; biases and skip zero-initialized. Callers seed exact values
        # for tests.
        self.wz = randn([dim, dim], Self.dtype)
        self.bz = zeros([dim], Self.dtype)
        self.conv_w = randn([dim, conv_kernel], Self.dtype)
        self.conv_b = zeros([dim], Self.dtype)
        self.w_b = randn([dim, state], Self.dtype)
        self.w_c = randn([dim, state], Self.dtype)
        self.w_dt = randn([dim, dim], Self.dtype)
        self.dt_bias = zeros([dim], Self.dtype)
        self.a_log = zeros([dim, state], Self.dtype)
        self.d = zeros([dim], Self.dtype)
        self.wo = randn([dim, dim], Self.dtype)
        self.bo = zeros([dim], Self.dtype)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Run the full selective-SSM (S6) block over a sequence.

        Convention note (same full-sequence convention planned for the S4-LTI
        sibling `DiagonalSSM.forward`, PR #5636): this `forward` takes the
        rank-3 FULL sequence `(batch, seq, dim)` and scans internally over all
        timesteps. There is no per-timestep `step` interface — the selective
        parameters B/C/Δ and the causal conv are input-dependent and recomputed
        across the whole sequence — so a polymorphic caller MUST pass a full
        sequence here.

        Args:
            input: Input tensor of shape (batch, seq, dim), batch-first.

        Returns:
            Output tensor of shape (batch, seq, dim); the SSM state is carried
            across all timesteps internally and discarded after the last step.

        Raises:
            Error: If tensor operations fail or the input is not rank-3
                with a matching feature dimension.
        """
        if len(input.shape()) != 3:
            raise Error("MambaBlock.forward: input must be [batch, seq, dim]")
        var batch = input.shape()[0]
        var seq = input.shape()[1]
        var in_dim = input.shape()[2]
        if in_dim != self.dim:
            raise Error("MambaBlock.forward: input dim mismatch")

        var D = self.dim
        var N = self.state
        var K = self.conv_kernel

        # --- A = -exp(a_log), [D, N] ---
        var a_mat = zeros([D, N], Self.dtype)
        for idx in range(D * N):
            a_mat.store[Self.dtype](
                idx, -_exp_scalar[Self.dtype](self.a_log.load[Self.dtype](idx))
            )

        var out = zeros([batch, seq, D], Self.dtype)
        # SSM state h: [batch, D, N], zero initial.
        var h = zeros([batch, D, N], Self.dtype)

        for bi in range(batch):
            for t in range(seq):
                # --- gate branch: z = u_t @ Wz + bz, [D] ---
                # --- causal depthwise conv over x-branch (x = u), then SiLU ---
                var xc = zeros([D], Self.dtype)  # conv'd x at this (bi, t), [D]
                var z_silu = zeros([D], Self.dtype)  # SiLU(z_t), [D]
                for d0 in range(D):
                    # gate pre-activation z[d0] = sum_e u_t[e]*Wz[e,d0] + bz[d0]
                    var z_acc = self.bz.load[Self.dtype](d0)
                    for e in range(D):
                        var u_e = input.load[Self.dtype]((bi * seq + t) * D + e)
                        z_acc += u_e * self.wz.load[Self.dtype](e * D + d0)
                    z_silu.store[Self.dtype](
                        d0, _silu_scalar[Self.dtype](z_acc)
                    )
                    # causal conv on channel d0: sum_j conv_w[d0,j]*u_pad[t+j,d0]
                    # u_pad left-padded K-1 zeros; window index tt = t-(K-1)+j.
                    var c_acc = self.conv_b.load[Self.dtype](d0)
                    for j in range(K):
                        var tt = t - (K - 1) + j
                        if tt >= 0:
                            var u_val = input.load[Self.dtype](
                                (bi * seq + tt) * D + d0
                            )
                            c_acc += (
                                self.conv_w.load[Self.dtype](d0 * K + j) * u_val
                            )
                    xc.store[Self.dtype](d0, _silu_scalar[Self.dtype](c_acc))

                # --- selective B, C from conv'd x: [N] each (shared over D) ---
                var b_vec = zeros([N], Self.dtype)
                var c_vec = zeros([N], Self.dtype)
                for n in range(N):
                    var b_acc = Scalar[Self.dtype](0.0)
                    var c_acc2 = Scalar[Self.dtype](0.0)
                    for d0 in range(D):
                        var xcd = xc.load[Self.dtype](d0)
                        b_acc += xcd * self.w_b.load[Self.dtype](d0 * N + n)
                        c_acc2 += xcd * self.w_c.load[Self.dtype](d0 * N + n)
                    b_vec.store[Self.dtype](n, b_acc)
                    c_vec.store[Self.dtype](n, c_acc2)

                # --- selective Δ per channel: softplus(xc @ Wdt + dt_bias) ---
                var delta = zeros([D], Self.dtype)
                for d0 in range(D):
                    var dt_acc = self.dt_bias.load[Self.dtype](d0)
                    for e in range(D):
                        dt_acc += xc.load[Self.dtype](e) * self.w_dt.load[
                            Self.dtype
                        ](e * D + d0)
                    delta.store[Self.dtype](
                        d0, _softplus_scalar[Self.dtype](dt_acc)
                    )

                # --- discretize + selective scan; y_t = C·h + D·xc ---
                for d0 in range(D):
                    var delta_d = delta.load[Self.dtype](d0)
                    var xc_d = xc.load[Self.dtype](d0)
                    var y_acc = self.d.load[Self.dtype](d0) * xc_d
                    for n in range(N):
                        var a_dn = a_mat.load[Self.dtype](d0 * N + n)
                        var da = _exp_scalar[Self.dtype](delta_d * a_dn)
                        var db = delta_d * b_vec.load[Self.dtype](n)
                        var sidx = (bi * D + d0) * N + n
                        var h_new = da * h.load[Self.dtype](sidx) + db * xc_d
                        h.store[Self.dtype](sidx, h_new)
                        y_acc += c_vec.load[Self.dtype](n) * h_new
                    # gated: y_t[d0] *= SiLU(z_t[d0])
                    y_acc = y_acc * z_silu.load[Self.dtype](d0)
                    # stash pre-out-projection gated y into out buffer slot
                    out.store[Self.dtype]((bi * seq + t) * D + d0, y_acc)

                # --- output projection: out_t = y_gated @ Wo + bo, [D] ---
                # read the just-written gated y_t row, project, overwrite in place.
                var y_gated = zeros([D], Self.dtype)
                for d0 in range(D):
                    y_gated.store[Self.dtype](
                        d0, out.load[Self.dtype]((bi * seq + t) * D + d0)
                    )
                for d0 in range(D):
                    var o_acc = self.bo.load[Self.dtype](d0)
                    for e in range(D):
                        o_acc += y_gated.load[Self.dtype](e) * self.wo.load[
                            Self.dtype
                        ](e * D + d0)
                    out.store[Self.dtype]((bi * seq + t) * D + d0, o_acc)

        return out^

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters.

        Returns:
            List [wz, bz, conv_w, conv_b, w_b, w_c, w_dt, dt_bias, a_log, d,
            wo, bo] (12 tensors).

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        params.append(self.wz)
        params.append(self.bz)
        params.append(self.conv_w)
        params.append(self.conv_b)
        params.append(self.w_b)
        params.append(self.w_c)
        params.append(self.w_dt)
        params.append(self.dt_bias)
        params.append(self.a_log)
        params.append(self.d)
        params.append(self.wo)
        params.append(self.bo)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; the block has no train/eval state).
        """
        pass

    def eval(mut self):
        """Switch to inference mode (no-op; the block has no train/eval state).
        """
        pass
