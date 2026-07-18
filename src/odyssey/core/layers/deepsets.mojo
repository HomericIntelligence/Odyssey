"""Permutation-equivariant (Deep Sets) linear block.

A single permutation-equivariant linear layer from Zaheer et al. 2017,
"Deep Sets" (NeurIPS 2017, arXiv:1703.06114). The layer maps a *set* of feature
vectors to a set of the same size, and is **equivariant** to permutations of the
set elements: permuting the input elements permutes the output elements
identically. This is the defining property (see `tests/.../test_deepsets.mojo`,
`test_permutation_equivariance`).

Equivariant form (this block, ARCH-14 variant — SUM pool, ReLU activation):

    y_i = relu( x_i @ Lambda  +  (sum_j x_j) @ Gamma  +  b )

Here `x_i` is the i-th element of an input set, `Lambda` and `Gamma` are learnable
(dim, out) projections, and `b` is a learnable (out,) bias. The pooled term
`sum_j x_j` is permutation-INVARIANT, and it is broadcast identically onto every
set element; that invariant + shared-per-element structure is exactly the
necessary-and-sufficient condition for permutation equivariance derived in the
paper (Sec. "Permutation Equivariance"; the parameter-tied linear map there is
`Lambda = lambda*I`, `Gamma = gamma*(1 1^T)` — this block uses the general
learnable dense form, which is equivariant for ANY `Lambda`, `Gamma`).

Batch-first convention `[batch, set_size, dim]`: the SET axis (axis 1) plays the
role the sequence axis plays in the recurrent layers, and pooling reduces over it.

The forward pass is documented over the full set at once (no per-element loop is
exposed): the per-element `Lambda` transform is a single batched matmul, and the
pooled `Gamma` term is one reduction + one matmul, broadcast back over the set.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, randn
from odyssey.core.module import Module


struct DeepSetsEquivariant[dtype: DType = DType.float32](
    Copyable, Module, Movable
):
    """Permutation-equivariant Deep Sets linear block.

    Implements `y_i = relu(x_i @ Lambda + (sum_j x_j) @ Gamma + b)` over a
    batch-first `[batch, set_size, dim]` input, equivariant to permutations of
    the set axis (axis 1).

    Compile-time contract: `Self.dtype` fixes the element type of every
    parameter and of the forward computation. The public API is float32 by
    default; float64 is available for tight-tolerance parity checks. Inputs
    passed to `forward` must share `Self.dtype` (matmul rejects mixed dtypes).

    Parameters:
        dtype: Element type for weights, bias, and compute (default: float32).

    Attributes:
        lam: Per-element projection `Lambda`, shape (dim, out).
        gam: Pooled (sum-over-set) projection `Gamma`, shape (dim, out).
        bias: Bias vector, shape (out,).
        dim: Input feature dimension of each set element.
        out_features: Output feature dimension of each set element.
    """

    var lam: AnyTensor
    var gam: AnyTensor
    var bias: AnyTensor
    var dim: Int
    var out_features: Int

    def __init__(out self, dim: Int, out_features: Int) raises:
        """Initialize with random `Lambda`/`Gamma` and zero bias.

        Args:
            dim: Input feature dimension of each set element.
            out_features: Output feature dimension of each set element.

        Raises:
            Error: If dim <= 0 or out_features <= 0, or if tensor creation fails.

        Example:
            ```mojo
            var layer = DeepSetsEquivariant(6, 5)  # dim 6 -> out 5
            ```
        """
        if dim <= 0:
            raise Error("DeepSetsEquivariant: dim must be positive")
        if out_features <= 0:
            raise Error("DeepSetsEquivariant: out_features must be positive")
        self.dim = dim
        self.out_features = out_features
        self.lam = randn([dim, out_features], Self.dtype)
        self.gam = randn([dim, out_features], Self.dtype)
        self.bias = zeros([out_features], Self.dtype)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass over the full set.

        Computes `y_i = relu(x_i @ Lambda + (sum_j x_j) @ Gamma + b)` for every
        set element `i`, where the pooled term `sum_j x_j` is reduced over the
        set axis and broadcast back over every element.

        Args:
            input: Input tensor of shape `[batch, set_size, dim]`. Its dtype must
                equal `Self.dtype`.

        Returns:
            Output tensor of shape `[batch, set_size, out_features]`.

        Raises:
            Error: If `input` is not rank-3, if its last dim != `dim`, or if any
                tensor operation fails.

        Example:
            ```mojo
            var layer = DeepSetsEquivariant[DType.float32](6, 5)
            var x = zeros([2, 4, 6], DType.float32)  # batch 2, set 4, dim 6
            var y = layer.forward(x)                 # shape [2, 4, 5]
            ```
        """
        from odyssey.core.matrix import matmul
        from odyssey.core.reduction import sum as reduce_sum
        from odyssey.core.activation import relu

        var shape = input.shape()
        if len(shape) != 3:
            raise Error(
                "DeepSetsEquivariant.forward expects rank-3 [batch, set_size,"
                " dim] input"
            )
        var batch = shape[0]
        var set_size = shape[1]
        var feat = shape[2]
        if feat != self.dim:
            raise Error(
                "DeepSetsEquivariant.forward: last input dim ("
                + String(feat)
                + ") != layer dim ("
                + String(self.dim)
                + ")"
            )

        # Per-element transform: reshape [B, S, D] -> [B*S, D], apply Lambda,
        # reshape back to [B, S, O]. matmul is 2D-only for the broadcast case, so
        # we fold the set axis into the batch axis for this shared transform.
        var flat = input.reshape([batch * set_size, self.dim])
        var per_elem = matmul(flat, self.lam)  # [B*S, O]
        var per_elem_3d = per_elem.reshape([batch, set_size, self.out_features])

        # Permutation-INVARIANT pooled term: sum over the set axis (axis 1),
        # project with Gamma, broadcast back over the set axis.
        var pooled = reduce_sum(input, axis=1, keepdims=False)  # [B, D]
        var pooled_proj = matmul(pooled, self.gam)  # [B, O]
        var pooled_3d = pooled_proj.reshape(
            [batch, 1, self.out_features]
        )  # broadcast over set axis

        # Broadcast-add pooled term + bias, then activation.
        var pre = (per_elem_3d + pooled_3d) + self.bias
        return relu(pre)

    def parameters(self) raises -> List[AnyTensor]:
        """Get trainable parameters.

        Returns:
            List `[lam, gam, bias]` in a deterministic order.

        Raises:
            Error: If tensor copying fails.
        """
        var params = List[AnyTensor]()
        params.append(self.lam)
        params.append(self.gam)
        params.append(self.bias)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; block has no mode-dependent state).
        """
        pass

    def eval(mut self):
        """Switch to inference mode (no-op; block has no mode-dependent state).
        """
        pass
