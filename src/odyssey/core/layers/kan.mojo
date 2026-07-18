"""1-layer Kolmogorov-Arnold Network (KAN) block.

A KAN layer replaces the MLP's fixed node non-linearities + linear edge weights
with *learnable univariate functions on each edge* (Liu et al. 2024, "KAN:
Kolmogorov-Arnold Networks", arXiv:2404.19756). Each scalar edge (i -> j), from
input feature `i` to output feature `j`, carries an activation function

    phi_{j,i}(x) = w_base_{j,i} * silu(x) + w_spline_{j,i} * spline_{j,i}(x)

("residual activation function", paper §2.2 Eq. 2.10), where the base branch
`silu(x) = x * sigmoid(x)` is a fixed shortcut and

    spline_{j,i}(x) = sum_{m=0}^{G+k-1} c_{j,i,m} * B_{m,k}(x)

is a B-spline of order `k` (degree `k`, so cubic for k=3) on a uniform grid of
`G` intervals. `B_{m,k}(x)` are the order-`k` B-spline basis functions defined by
the Cox-de Boor recursion on the knot vector; the `c_{j,i,m}` are the learnable
spline coefficients. The layer output is the sum over input edges:

    y_j = sum_{i=0}^{in-1} phi_{j,i}(x_i)

Positioning (2D vs 3D). KAN is position-wise: each scalar feature is transformed
independently and summed across the input axis, with no dependence on any
sequence position. We therefore adopt the same 2D `[batch, in_features]` ->
`[batch, out_features]` contract as `Linear`/`FeedForward` in this package, so a
KAN block composes as a drop-in dense-block replacement in the sibling layers'
stacks. A `[batch, seq, features]` caller flattens the leading axes before the
call, exactly as they would for `Linear`.

Grid-range behavior. The B-spline basis has *compact support*: an input `x`
outside the closed grid range `[grid_min, grid_max]` evaluates every basis
`B_{m,k}(x)` to 0, so the spline branch contributes nothing there and the edge
output degenerates to the base branch `w_base * silu(x)`. This is the natural
"extrapolate via the base branch" behavior (paper §2.2 motivates the base
shortcut precisely as a well-behaved fallback). We deliberately do NOT clamp the
input into range: clamping would fold out-of-range points onto the boundary knot
and distort the learned shape. The numerical note is that the transition at the
boundary is continuous (the boundary basis values reach 0 there), so there is no
discontinuity as `x` crosses `grid_max`/`grid_min` — only a loss of the spline
term. Training the grid-range/adaptivity (paper's grid extension) is out of scope
for this 1-layer research block; the grid is fixed at construction.

dtype contract. The struct is generic over `dtype` (default float32, the package
API dtype). Scalar loads/stores use the tensor's own `Self.dtype` — the
compile-time `Self.dtype` contract used across this package — so the float32 API
and a float64 parity build share one code path with no runtime dtype dispatch.
The B-spline recursion accumulates in `Self.dtype`; the float32 default is the
honest training dtype (float64 is used only by the parity test for exactness).

Reference:
    Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljacic, M.,
    Hou, T. Y., & Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks.
    arXiv:2404.19756, §2.2 (residual activation, Eq. 2.10; B-spline
    parametrization).
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.module import Module


struct KAN[dtype: DType = DType.float32](Copyable, Module, Movable):
    """1-layer KAN block: learnable per-edge B-spline activations + base branch.

    Parameters:
        dtype: Data type for all parameters (default: float32, the package API
            dtype). Scalar arithmetic uses `Self.dtype` throughout.

    Attributes:
        base_weight: `[in_features, out_features]` — the `w_base` coefficient of
            the fixed `silu` shortcut on each edge.
        spline_weight: `[in_features, out_features]` — the `w_spline` scalar that
            scales each edge's spline branch.
        spline_coeff: `[in_features, out_features, n_coeff]` (flattened) — the
            B-spline coefficients `c_{j,i,m}`; `n_coeff = grid_size + spline_order`.
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        grid_size: Number of uniform grid *intervals* `G`.
        spline_order: B-spline order/degree `k` (cubic = 3).
        grid_min, grid_max: Closed range spanned by the interior grid.
    """

    var base_weight: AnyTensor
    var spline_weight: AnyTensor
    var spline_coeff: AnyTensor
    var in_features: Int
    var out_features: Int
    var grid_size: Int
    var spline_order: Int
    var grid_min: Float64
    var grid_max: Float64

    def __init__(
        out self,
        in_features: Int,
        out_features: Int,
        grid_size: Int = 5,
        spline_order: Int = 3,
        grid_min: Float64 = -1.0,
        grid_max: Float64 = 1.0,
    ) raises:
        """Initialize a KAN layer with zeroed parameters.

        Parameters are zero-initialized (a deterministic, reproducible start; the
        research harness overwrites them with a seeded schedule, and the parity
        test writes fixed ramps). The number of B-spline coefficients per edge is
        `grid_size + spline_order` — the standard count of order-`k` basis
        functions over `G` intervals.

        Args:
            in_features: Number of input features (research-grade: ~4-8).
            out_features: Number of output features (~4-8).
            grid_size: Number of uniform grid intervals `G` (default 5, the
                paper's default).
            spline_order: B-spline order/degree `k` (default 3, cubic — the
                paper's default).
            grid_min: Lower bound of the interior grid range (default -1.0).
            grid_max: Upper bound of the interior grid range (default 1.0).

        Raises:
            Error: If any dimension is non-positive or grid_max <= grid_min.

        Example:
            ```mojo
            var kan = KAN[DType.float32](4, 4)   # grid_size=5, spline_order=3
            var y = kan.forward(x)               # x: [batch, 4] -> [batch, 4]
            ```
        """
        if in_features <= 0 or out_features <= 0:
            raise Error("KAN: in_features and out_features must be positive")
        if grid_size <= 0 or spline_order <= 0:
            raise Error("KAN: grid_size and spline_order must be positive")
        if grid_max <= grid_min:
            raise Error("KAN: grid_max must be greater than grid_min")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_min = grid_min
        self.grid_max = grid_max

        var n_coeff = grid_size + spline_order
        self.base_weight = zeros([in_features, out_features], Self.dtype)
        self.spline_weight = zeros([in_features, out_features], Self.dtype)
        self.spline_coeff = zeros(
            [in_features * out_features * n_coeff], Self.dtype
        )

    def _knot(self, idx: Int) -> Scalar[Self.dtype]:
        """Knot `t_idx` of the open-uniform knot vector.

        The knot vector uniformly partitions `[grid_min, grid_max]` into
        `grid_size` intervals and extends by `spline_order` extra knots on each
        side (uniform continuation of the same step `h`). With that extension the
        `grid_size + spline_order` order-`k` basis functions tile the interior
        range. Knot index runs `0 .. grid_size + 2*spline_order`.

        Args:
            idx: Knot index (may be negative-extended via the formula).

        Returns:
            The knot position `t_idx` in `Self.dtype`.
        """
        var h = (self.grid_max - self.grid_min) / Float64(self.grid_size)
        var t = self.grid_min + Float64(idx - self.spline_order) * h
        return t.cast[Self.dtype]()

    def _bspline_basis(self, x: Scalar[Self.dtype]) -> List[Scalar[Self.dtype]]:
        """Cox-de Boor: return `basis[m]` = B_{m,k}(x), m in [0, n_coeff).

        Order-`k` (degree-`k`) B-spline basis via the standard Cox-de Boor
        recursion:

            B_{m,0}(x) = 1 if t_m <= x < t_{m+1} else 0
            B_{m,p}(x) =  (x - t_m)/(t_{m+p} - t_m)       * B_{m,p-1}(x)
                        + (t_{m+p+1} - x)/(t_{m+p+1} - t_{m+1}) * B_{m+1,p-1}(x)

        with the convention that a zero denominator contributes a zero term. `x`
        outside `[grid_min, grid_max]` yields all-zero basis (compact support),
        which is the documented out-of-range behavior. This is transcribed
        IDENTICALLY in the torch/numpy parity reference so parity is
        exact-by-construction.

        Args:
            x: Scalar input value in `Self.dtype`.

        Returns:
            List of length `n_coeff` with the order-`k` basis values.
        """
        var k = self.spline_order
        var n_coeff = self.grid_size + self.spline_order
        var n_knots = self.grid_size + 2 * self.spline_order + 1

        # Degree-0 basis over all knot spans: b0[m] = 1 on [t_m, t_{m+1}).
        var b = List[Scalar[Self.dtype]]()
        for m in range(n_knots - 1):
            var lo = self._knot(m)
            var hi = self._knot(m + 1)
            if x >= lo and x < hi:
                b.append(Scalar[Self.dtype](1))
            else:
                b.append(Scalar[Self.dtype](0))

        # Raise degree 1..k via Cox-de Boor.
        for p in range(1, k + 1):
            var nb = List[Scalar[Self.dtype]]()
            for m in range(n_knots - 1 - p):
                var tm = self._knot(m)
                var tmp = self._knot(m + p)
                var tm1 = self._knot(m + 1)
                var tmp1 = self._knot(m + p + 1)
                var left = Scalar[Self.dtype](0)
                var den_l = tmp - tm
                if den_l != 0:
                    left = (x - tm) / den_l * b[m]
                var right = Scalar[Self.dtype](0)
                var den_r = tmp1 - tm1
                if den_r != 0:
                    right = (tmp1 - x) / den_r * b[m + 1]
                nb.append(left + right)
            b = nb^

        var result = List[Scalar[Self.dtype]]()
        for m in range(n_coeff):
            result.append(b[m])
        return result^

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: y_j = sum_i (w_base_{ij} silu(x_i) + w_spline_{ij} spline_{ij}(x_i)).

        Args:
            input: Input tensor of shape `[batch, in_features]`.

        Returns:
            Output tensor of shape `[batch, out_features]`.

        Raises:
            Error: If the input is not 2D or its last dim != in_features.
        """
        var shp = input.shape()
        if len(shp) != 2:
            raise Error("KAN.forward expects a 2D [batch, in_features] input")
        var batch = shp[0]
        if shp[1] != self.in_features:
            raise Error("KAN.forward: input last dim must equal in_features")

        var n_coeff = self.grid_size + self.spline_order
        var output = zeros([batch, self.out_features], Self.dtype)

        for bi in range(batch):
            for j in range(self.out_features):
                var acc = Scalar[Self.dtype](0)
                for i in range(self.in_features):
                    var x = input.load[Self.dtype](bi * self.in_features + i)

                    # Base branch: silu(x) = x * sigmoid(x). The sigmoid is
                    # computed in Float64 (like the package's `_sigmoid_op`) and
                    # cast back to Self.dtype — this sidesteps a generic
                    # floating-point constraint on the struct's dtype while
                    # keeping the honest float32 API result.
                    var x64 = Float64(x)
                    var sig64 = 1.0 / (1.0 + _exp_neg(x64))
                    var silu = x * sig64.cast[Self.dtype]()
                    var wb = self.base_weight.load[Self.dtype](
                        i * self.out_features + j
                    )

                    # Spline branch: sum_m c_{ijm} B_{m,k}(x).
                    var basis = self._bspline_basis(x)
                    var spline = Scalar[Self.dtype](0)
                    var coeff_base = (i * self.out_features + j) * n_coeff
                    for m in range(n_coeff):
                        var c = self.spline_coeff.load[Self.dtype](
                            coeff_base + m
                        )
                        spline = spline + c * basis[m]
                    var ws = self.spline_weight.load[Self.dtype](
                        i * self.out_features + j
                    )

                    acc = acc + wb * silu + ws * spline
                output.store[Self.dtype](bi * self.out_features + j, acc)
        return output

    def parameters(self) raises -> List[AnyTensor]:
        """Get trainable parameters: [base_weight, spline_weight, spline_coeff].

        Returns:
            List of the three parameter tensors.

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        params.append(self.base_weight)
        params.append(self.spline_weight)
        params.append(self.spline_coeff)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; KAN has no mode-dependent state)."""
        pass

    def eval(mut self):
        """Switch to inference mode (no-op; KAN has no mode-dependent state)."""
        pass


def _exp_neg(x: Float64) -> Float64:
    """Compute exp(-x) in Float64 (for silu's sigmoid, mirroring `_sigmoid_op`).

    Args:
        x: Input value (already upcast to Float64).

    Returns:
        exp(-x).
    """
    from std.math import exp

    return exp(-x)
