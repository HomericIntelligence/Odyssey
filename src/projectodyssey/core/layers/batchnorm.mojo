"""BatchNorm2D (2D batch normalization) layer with parameter management.

This module provides a BatchNorm2dLayer wrapper class that manages gamma, beta,
and running statistics for 2D batch normalization. The layer wraps the pure
functional batch_norm2d function and maintains learnable scale/shift parameters
along with exponential moving averages of batch statistics.

Key components:
- BatchNorm2dLayer: 2D batch normalization layer with learnable parameters
  Implements: y = gamma * (x - mean) / sqrt(var + eps) + beta (training)
             y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta (inference)
"""

from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    zeros_like,
    ones_like,
)
from projectodyssey.core.normalization_simd import batch_norm2d_fused


struct BatchNorm2dLayer[dtype: DType = DType.float32](Copyable, Movable):
    """2D Batch Normalization layer.

    Normalizes activations across the batch dimension for each channel
    Maintains running statistics for use during inference

    Parameters:
        dtype: Data type for parameters and running statistics
            (default: float32).

    Attributes:
        gamma: Scale parameter of shape (channels,)
        beta: Shift parameter of shape (channels,)
        running_mean: Running mean of shape (channels,)
        running_var: Running variance of shape (channels,)
        num_channels: Number of channels to normalize
        momentum: Momentum for running statistics update
        eps: Small constant for numerical stability
    """

    var gamma: AnyTensor
    """Scale parameter of shape (channels,)."""
    var beta: AnyTensor
    """Shift parameter of shape (channels,)."""
    var running_mean: AnyTensor
    """Running mean of shape (channels,)."""
    var running_var: AnyTensor
    """Running variance of shape (channels,)."""
    var num_channels: Int
    """Number of channels to normalize."""
    var momentum: Float32
    """Momentum for exponential moving average."""
    var eps: Float32
    """Small constant for numerical stability."""

    def __init__(
        out self,
        num_channels: Int,
        momentum: Float32 = 0.1,
        eps: Float32 = 1e-5,
    ) raises:
        """Initialize BatchNorm2D layer with learnable parameters.

        Gamma (scale) is initialized to 1.0 for each channel.
        Beta (shift) is initialized to 0.0.
        Running mean is initialized to 0.0 and running variance to 1.0.

        Args:
            num_channels: Number of channels to normalize.
            momentum: Momentum for exponential moving average
                (default: 0.1).
            eps: Small constant for numerical stability
                (default: 1e-5).

        Raises:
            Error if tensor creation fails

        Example:
            ```mojo
            var bn = BatchNorm2dLayer(16, momentum=0.1)
            ```
        """
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        # Initialize gamma (scale) to 1.0 for each channel
        var gamma_shape = List[Int]()
        gamma_shape.append(num_channels)
        self.gamma = ones(gamma_shape, Self.dtype)

        # Initialize beta (shift) to 0.0
        var beta_shape = List[Int]()
        beta_shape.append(num_channels)
        self.beta = zeros(beta_shape, Self.dtype)

        # Initialize running_mean to 0.0
        var running_mean_shape = List[Int]()
        running_mean_shape.append(num_channels)
        self.running_mean = zeros(running_mean_shape, Self.dtype)

        # Initialize running_var to 1.0
        var running_var_shape = List[Int]()
        running_var_shape.append(num_channels)
        self.running_var = ones(running_var_shape, Self.dtype)

    def forward(
        mut self, input: AnyTensor, training: Bool = True
    ) raises -> AnyTensor:
        """Forward pass with batch normalization.

        In training mode: computes batch statistics and updates running stats
        In inference mode: uses running statistics for normalization

        Args:
            input: Input tensor of shape (batch, channels, height, width).
            training: If True, use batch statistics and update running stats
                     If False, use running statistics (default: True).

        Returns:
            Output tensor of shape (batch, channels, height, width).

        Raises:
            Error if tensor operations fail.

        Example:
            ```mojo
            var bn = BatchNorm2dLayer(16)
            var input_t = zeros([2, 16, 32, 32], DType.float32)
            var output = bn.forward(input_t, training=True)
            ```
        """
        var (output, new_running_mean, new_running_var) = batch_norm2d_fused(
            input,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            training,
            Float64(self.momentum),
            Float64(self.eps),
        )

        # Update running statistics if training
        if training:
            self.running_mean = new_running_mean^
            self.running_var = new_running_var^

        return output^

    def parameters(self) raises -> List[AnyTensor]:
        """Get list of trainable parameters.

        Returns gamma and beta as copies. No bitcast needed since fields
        are already AnyTensor with the correct dtype.

        Returns:
            List containing [gamma, beta] tensors that need gradients
            (Running statistics are not trainable parameters)

        Raises:
            Error if tensor copying fails

        Example:
            ```mojo
            var bn = BatchNorm2dLayer(16)
            var params = bn.parameters()
            # params[0] is gamma (scale), params[1] is beta (shift)
            ```
        """
        var params = List[AnyTensor]()
        params.append(self.gamma)
        params.append(self.beta)
        return params^

    def get_running_stats(self) raises -> Tuple[AnyTensor, AnyTensor]:
        """Get current running statistics.

        Returns:
            Tuple of (running_mean, running_var) for use during inference
            or checkpointing

        Raises:
            Error if tensor copying fails

        Example:
            ```mojo
            var bn = BatchNorm2dLayer(16)
            var (mean, var) = bn.get_running_stats()
            ```
        """
        return Tuple[AnyTensor, AnyTensor](self.running_mean, self.running_var)

    def set_running_stats(
        mut self, running_mean: AnyTensor, running_var: AnyTensor
    ) raises:
        """Set running statistics (for loading from checkpoint).

        Args:
            running_mean: Running mean to set, shape (channels,).
            running_var: Running variance to set, shape (channels,).

        Raises:
            Error if tensor shapes don't match.

        Example:
            ```mojo
            var bn = BatchNorm2dLayer(16)
            var (saved_mean, saved_var) = load_stats()
            bn.set_running_stats(saved_mean, saved_var)
            ```
        """
        var mean_size = running_mean.numel()
        var var_size = running_var.numel()

        if mean_size != self.running_mean.numel():
            raise Error("Running mean size mismatch")

        if var_size != self.running_var.numel():
            raise Error("Running variance size mismatch")

        self.running_mean = running_mean
        self.running_var = running_var
