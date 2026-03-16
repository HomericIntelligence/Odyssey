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

from ..extensor import ExTensor, zeros, ones, zeros_like, ones_like
from ..normalization_simd import batch_norm2d_fused


struct BatchNorm2dLayer(Copyable, Movable):
    """2D Batch Normalization layer.

    Normalizes activations across the batch dimension for each channel
    Maintains running statistics for use during inference

    Attributes:
        gamma: Scale parameter of shape (channels,)
        beta: Shift parameter of shape (channels,)
        running_mean: Running mean of shape (channels,)
        running_var: Running variance of shape (channels,)
        num_channels: Number of channels to normalize
        momentum: Momentum for running statistics update
        eps: Small constant for numerical stability
    """

    var gamma: ExTensor
    """Scale parameter of shape (channels,)."""
    var beta: ExTensor
    """Shift parameter of shape (channels,)."""
    var running_mean: ExTensor
    """Running mean of shape (channels,)."""
    var running_var: ExTensor
    """Running variance of shape (channels,)."""
    var num_channels: Int
    """Number of channels to normalize."""
    var momentum: Float32
    """Momentum for exponential moving average."""
    var eps: Float32
    """Small constant for numerical stability."""

    fn __init__(
        out self,
        num_channels: Int,
        momentum: Float32 = 0.1,
        eps: Float32 = 1e-5,
    ) raises:
        """Initialize BatchNorm2D layer with learnable parameters and running statistics.

        Gamma (scale) is initialized to 1.0 for each channel (identity transform).
        Beta (shift) is initialized to 0.0.
        Running mean is initialized to 0.0 and running variance to 1.0.

        Args:
            num_channels: Number of channels to normalize.
            momentum: Momentum for exponential moving average of running statistics
                     (default: 0.1). Higher values give more weight to current batch.
            eps: Small constant for numerical stability to avoid division by zero
                (default: 1e-5).

        Raises:
            Error if tensor creation fails

        Example:
            ```mojo
            # Normalize 16 channels from Conv2d output
            var bn = BatchNorm2dLayer(16, momentum=0.1)
            ```
        """
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        # Initialize gamma (scale) to 1.0 for each channel
        # Shape: (channels,)
        var gamma_shape = List[Int]()
        gamma_shape.append(num_channels)
        self.gamma = ones(gamma_shape, DType.float32)

        # Initialize beta (shift) to 0.0
        # Shape: (channels,)
        var beta_shape = List[Int]()
        beta_shape.append(num_channels)
        self.beta = zeros(beta_shape, DType.float32)

        # Initialize running_mean to 0.0
        # Shape: (channels,)
        var running_mean_shape = List[Int]()
        running_mean_shape.append(num_channels)
        self.running_mean = zeros(running_mean_shape, DType.float32)

        # Initialize running_var to 1.0
        # Shape: (channels,)
        var running_var_shape = List[Int]()
        running_var_shape.append(num_channels)
        self.running_var = ones(running_var_shape, DType.float32)

    fn forward(
        mut self, input: ExTensor, training: Bool = True
    ) raises -> ExTensor:
        """Forward pass with batch normalization.

        In training mode: computes batch statistics and updates running statistics
        In inference mode: uses running statistics for normalization

        Args:
            input: Input tensor of shape (batch, channels, height, width).
            training: If True, use batch statistics and update running stats
                     If False, use running statistics (default: True).

        Returns:
            Output tensor of shape (batch, channels, height, width).

        Raises:
            Error if tensor operations fail.

        Note:
            This method mutates self.running_mean and self.running_var
            when training=True to track exponential moving averages.

        Formula (training):
        ```
            mean = mean(x, axis=(0, 2, 3))  # Per channel
            var = var(x, axis=(0, 2, 3))
            x_norm = (x - mean) / sqrt(var + eps)
            output = gamma * x_norm + beta
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var
        ```

        Formula (inference):
        ```
            x_norm = (x - running_mean) / sqrt(running_var + eps)
            output = gamma * x_norm + beta
        ```

        Example:
            ```mojo
            var bn = BatchNorm2dLayer(16)
            var input = randn([2, 16, 32, 32], DType.float32)

            # Training mode: updates running statistics
            var output = bn.forward(input, training=True)

            # Inference mode: uses running statistics
            var output = bn.forward(input, training=False)
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

    fn parameters(self) raises -> List[ExTensor]:
        """Get list of trainable parameters.

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
        var params: List[ExTensor] = []

        # Create copies of gamma and beta tensors
        var gamma_copy = zeros_like(self.gamma)
        var beta_copy = zeros_like(self.beta)

        var gamma_size = self.gamma.numel()
        var beta_size = self.beta.numel()

        var gamma_src = self.gamma._data.bitcast[Float32]()
        var gamma_dst = gamma_copy._data.bitcast[Float32]()
        var beta_src = self.beta._data.bitcast[Float32]()
        var beta_dst = beta_copy._data.bitcast[Float32]()

        for i in range(gamma_size):
            gamma_dst[i] = gamma_src[i]

        for i in range(beta_size):
            beta_dst[i] = beta_src[i]

        params.append(gamma_copy^)
        params.append(beta_copy^)
        return params^

    fn get_running_stats(self) raises -> Tuple[ExTensor, ExTensor]:
        """Get current running statistics.

        Returns:
            Tuple of (running_mean, running_var) for use during inference
            or checkpointing

        Raises:
            Error if tensor copying fails

        Example:
            ```mojo
            var bn = BatchNorm2dLayer(16)
            # After training...
            var (mean, var) = bn.get_running_stats()
            ```
        """
        var mean_copy = zeros_like(self.running_mean)
        var var_copy = zeros_like(self.running_var)

        var mean_size = self.running_mean.numel()
        var var_size = self.running_var.numel()

        var mean_src = self.running_mean._data.bitcast[Float32]()
        var mean_dst = mean_copy._data.bitcast[Float32]()
        var var_src = self.running_var._data.bitcast[Float32]()
        var var_dst = var_copy._data.bitcast[Float32]()

        for i in range(mean_size):
            mean_dst[i] = mean_src[i]

        for i in range(var_size):
            var_dst[i] = var_src[i]

        return Tuple[ExTensor, ExTensor](mean_copy^, var_copy^)

    fn set_running_stats(
        mut self, running_mean: ExTensor, running_var: ExTensor
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
            # Load from checkpoint
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

        self.running_mean = zeros_like(self.running_mean)
        self.running_var = zeros_like(self.running_var)

        var mean_src = running_mean._data.bitcast[Float32]()
        var mean_dst = self.running_mean._data.bitcast[Float32]()
        var var_src = running_var._data.bitcast[Float32]()
        var var_dst = self.running_var._data.bitcast[Float32]()

        for i in range(mean_size):
            mean_dst[i] = mean_src[i]

        for i in range(var_size):
            var_dst[i] = var_src[i]
