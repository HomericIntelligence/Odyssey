"""Layer Normalization layer.

A `Module`-conforming wrapper around the functional `layer_norm` op that owns
learnable per-feature scale (`gamma`) and shift (`beta`) parameters, matching the
`torch.nn.LayerNorm` convention for a 1-D `normalized_shape`:

    mean = mean(x[i])                              # over the feature dimension
    var  = var(x[i])                               # biased (population) variance
    x_norm[i] = (x[i] - mean) / sqrt(var + eps)
    y[i] = gamma * x_norm[i] + beta

Unlike batch norm, layer norm needs no running statistics — each sample is
normalized independently, so training and inference behave identically.

Reference:
    Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization.
    arXiv:1607.06450. Interface mirrors `torch.nn.LayerNorm(normalized_shape)`.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import ones, zeros
from odyssey.core.module import Module
from odyssey.core.normalization import layer_norm


struct LayerNorm[dtype: DType = DType.float32](Copyable, Module, Movable):
    """Layer normalization over the last (feature) dimension.

    Parameters:
        dtype: Data type for the learnable parameters (default: float32).

    Attributes:
        gamma: Learnable per-feature scale, shape (features,), initialized to 1.
        beta: Learnable per-feature shift, shape (features,), initialized to 0.
        num_features: Size of the normalized feature dimension.
        epsilon: Numerical-stability constant added to the variance.
    """

    var gamma: AnyTensor
    var beta: AnyTensor
    var num_features: Int
    var epsilon: Float64

    def __init__(out self, num_features: Int, epsilon: Float64 = 1e-5) raises:
        """Initialize the layer with gamma=1, beta=0.

        Args:
            num_features: Size of the normalized (last) dimension.
            epsilon: Small constant added to the variance (default: 1e-5,
                matching torch.nn.LayerNorm).

        Raises:
            Error: If num_features <= 0, or construction fails.

        Example:
            ```mojo
            var ln = LayerNorm(4)
            var y = ln.forward(x)   # x: [batch, 4] -> y: [batch, 4]
            ```
        """
        if num_features <= 0:
            raise Error("LayerNorm: num_features must be positive")
        self.num_features = num_features
        self.epsilon = epsilon
        self.gamma = ones([num_features], Self.dtype)
        self.beta = zeros([num_features], Self.dtype)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Normalize `input` over its feature dimension, then scale and shift.

        Args:
            input: Input tensor of shape (batch, num_features).

        Returns:
            Normalized tensor, same shape as `input`.

        Raises:
            Error: If tensor operations fail or the feature dimension mismatches.
        """
        # Guard the feature dimension the docstring promises: gamma/beta are
        # sized num_features, and the functional layer_norm indexes them by the
        # last dim, so a mismatch would misindex the affine params. This layer
        # covers the 1-D normalized_shape convention (2-D (batch, features)
        # input); the wrapped functional's 4-D per-position affine path is not
        # constructed here.
        var shape = input.shape()
        var last = shape[len(shape) - 1]
        if last != self.num_features:
            raise Error(
                "LayerNorm: input feature dimension ("
                + String(last)
                + ") does not match num_features ("
                + String(self.num_features)
                + ")"
            )
        return layer_norm(input, self.gamma, self.beta, self.epsilon)

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters.

        Returns:
            List of 2 tensors (gamma, beta).

        Raises:
            Error if tensor copying fails.
        """
        var params = List[AnyTensor]()
        params.append(self.gamma)
        params.append(self.beta)
        return params^

    def train(mut self):
        """Switch to training mode (no-op; layer norm is mode-independent)."""
        pass

    def eval(mut self):
        """Switch to inference mode (no-op; layer norm is mode-independent)."""
        pass
