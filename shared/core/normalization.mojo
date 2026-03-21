"""Functional normalization layers.

This module provides pure functional implementations of normalization operations
All operations are stateless - caller manages running statistics and parameters.
"""

from algorithm import parallelize

from .extensor import (
    ExTensor,
    zeros,
    zeros_like,
    ones_like,
    full_like,
)
from .arithmetic import subtract, add, multiply, divide, power
from .elementwise import sqrt
from .reduction import mean as reduce_mean, sum as reduce_sum
from .normalize_ops import normalize_rgb
from .parallel_utils import should_parallelize, parallel_for_batch
from .scalar_ops import (
    sqrt_scalar_f32,
    sqrt_scalar_f64,
    pow_scalar_f32,
    pow_scalar_f64,
)


fn batch_norm2d(
    x: ExTensor,
    gamma: ExTensor,
    beta: ExTensor,
    running_mean: ExTensor,
    running_var: ExTensor,
    training: Bool,
    momentum: Float64 = 0.1,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[ExTensor, ExTensor, ExTensor]:
    """Functional 2D batch normalization.

        Normalizes activations across the batch dimension for each channel.
        Returns updated running statistics (pure functional - caller must capture).

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            gamma: Scale parameter of shape (channels,).
            beta: Shift parameter of shape (channels,).
            running_mean: Running mean of shape (channels,).
            running_var: Running variance of shape (channels,).
            training: If True, use batch statistics and update running stats.
                     If False, use running statistics.
            momentum: Momentum for running statistics update (default: 0.1).
            epsilon: Small constant for numerical stability (default: 1e-5).

    Returns:
            Tuple of (output, new_running_mean, new_running_var):
                - output: Normalized tensor, shape (batch, channels, height, width).
                - new_running_mean: Updated running mean, shape (channels,).
                - new_running_var: Updated running variance, shape (channels,).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import batch_norm2d, zeros, ones

            var gamma = ones([channels])
            var beta = zeros([channels])
            var running_mean = zeros([channels])
            var running_var = ones([channels])

            # Training mode
            var (output, new_mean, new_var) = batch_norm2d(
                x, gamma, beta, running_mean, running_var,
                training=True, momentum=0.1
            )
            # Update running stats
            running_mean = new_mean
            running_var = new_var

            # Inference mode
            var (output, _, _) = batch_norm2d(
                x, gamma, beta, running_mean, running_var,
                training=False
            )
            ```

        Formula (training):
            mean = mean(x, axis=(0, 2, 3))  # Per channel
            var = var(x, axis=(0, 2, 3))
            x_norm = (x - mean) / sqrt(var + epsilon)
            output = gamma * x_norm + beta
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var

        Formula (inference):
            x_norm = (x - running_mean) / sqrt(running_var + epsilon)
            output = gamma * x_norm + beta

    Note:
            Pure functional: caller must capture and manage all three return values.
            Running statistics are updated only during training mode.
    """
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error(
            "batch_norm2d requires 4D input (batch, channels, height, width)"
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    if training:
        # Training mode: compute batch statistics
        # Compute mean and variance per channel across (batch, height, width)

        var batch_mean = zeros([channels], x.dtype())
        var batch_var = zeros([channels], x.dtype())

        var batch_mean_ptr = batch_mean._data
        var batch_var_ptr = batch_var._data
        var x_ptr = x._data

        var spatial_size = batch * height * width

        # Compute mean per channel
        if x.dtype() == DType.float32:
            for c in range(channels):
                var sum_val = Float32(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                batch_mean[c] = Float32(sum_val / Float32(spatial_size))

            # Compute variance per channel
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                var sum_sq_diff = Float32(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                batch_var[c] = Float32(sum_sq_diff / Float32(spatial_size))

        elif x.dtype() == DType.float64:
            for c in range(channels):
                var sum_val = Float64(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                batch_mean.set(c, sum_val / Float64(spatial_size))

            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                var sum_sq_diff = Float64(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                batch_var.set(c, sum_sq_diff / Float64(spatial_size))
        else:
            raise Error("batch_norm2d: only float32/64 dtypes supported")

        # Normalize
        var output = zeros_like(x)
        var output_ptr = output._data
        var gamma_ptr = gamma._data
        var beta_ptr = beta._data

        if x.dtype() == DType.float32:
            if should_parallelize(batch):
                # Parallel normalization over batch dimension
                @parameter
                fn normalize_batch_f32(b: Int) capturing:
                    for c in range(channels):
                        var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                        var var_val = batch_var_ptr.bitcast[Float32]()[c]
                        var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                        var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                        var beta_val = beta_ptr.bitcast[Float32]()[c]

                        for h in range(height):
                            for w in range(width):
                                var idx = (
                                    b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w
                                )
                                var x_val = x_ptr.bitcast[Float32]()[idx]
                                var x_norm = (x_val - mean_val) / std
                                output._set_float64(idx, Float64(gamma_val * x_norm + beta_val))

                parallelize[normalize_batch_f32](batch)
            else:
                # Sequential normalization for small batches
                for b in range(batch):
                    for c in range(channels):
                        var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                        var var_val = batch_var_ptr.bitcast[Float32]()[c]
                        var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                        var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                        var beta_val = beta_ptr.bitcast[Float32]()[c]

                        for h in range(height):
                            for w in range(width):
                                var idx = (
                                    b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w
                                )
                                var x_val = x_ptr.bitcast[Float32]()[idx]
                                var x_norm = (x_val - mean_val) / std
                                output[idx] = Float32(
                                    gamma_val * x_norm + beta_val
                                )

        elif x.dtype() == DType.float64:
            if should_parallelize(batch):
                # Parallel normalization over batch dimension
                @parameter
                fn normalize_batch_f64(b: Int) capturing:
                    for c in range(channels):
                        var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                        var var_val = batch_var_ptr.bitcast[Float64]()[c]
                        var std = sqrt_scalar_f64(var_val + epsilon)
                        var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                        var beta_val = beta_ptr.bitcast[Float64]()[c]

                        for h in range(height):
                            for w in range(width):
                                var idx = (
                                    b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w
                                )
                                var x_val = x_ptr.bitcast[Float64]()[idx]
                                var x_norm = (x_val - mean_val) / std
                                output._set_float64(idx, gamma_val * x_norm + beta_val)

                parallelize[normalize_batch_f64](batch)
            else:
                # Sequential normalization for small batches
                for b in range(batch):
                    for c in range(channels):
                        var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                        var var_val = batch_var_ptr.bitcast[Float64]()[c]
                        var std = sqrt_scalar_f64(var_val + epsilon)
                        var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                        var beta_val = beta_ptr.bitcast[Float64]()[c]

                        for h in range(height):
                            for w in range(width):
                                var idx = (
                                    b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w
                                )
                                var x_val = x_ptr.bitcast[Float64]()[idx]
                                var x_norm = (x_val - mean_val) / std
                                output.set(idx, gamma_val * x_norm + beta_val)

        # Update running statistics
        var new_running_mean = zeros_like(running_mean)
        var new_running_var = zeros_like(running_var)
        var rm_ptr = running_mean._data
        var rv_ptr = running_var._data
        var nrm_ptr = new_running_mean._data
        var nrv_ptr = new_running_var._data

        if x.dtype() == DType.float32:
            for c in range(channels):
                var rm_val = rm_ptr.bitcast[Float32]()[c]
                var rv_val = rv_ptr.bitcast[Float32]()[c]
                var bm_val = batch_mean_ptr.bitcast[Float32]()[c]
                var bv_val = batch_var_ptr.bitcast[Float32]()[c]

                new_running_mean[c] = Float32(
                    Float32(1.0 - momentum) * rm_val
                    + Float32(momentum) * bm_val
                )
                new_running_var[c] = Float32(
                    Float32(1.0 - momentum) * rv_val
                    + Float32(momentum) * bv_val
                )

        elif x.dtype() == DType.float64:
            for c in range(channels):
                var rm_val = rm_ptr.bitcast[Float64]()[c]
                var rv_val = rv_ptr.bitcast[Float64]()[c]
                var bm_val = batch_mean_ptr.bitcast[Float64]()[c]
                var bv_val = batch_var_ptr.bitcast[Float64]()[c]

                new_running_mean.set(
                    c,
                    (
                        1.0 - momentum
                    ) * rm_val + momentum * bm_val
                )
                new_running_var.set(
                    c,
                    (
                        1.0 - momentum
                    ) * rv_val + momentum * bv_val
                )

        return (output, new_running_mean, new_running_var)

    else:
        # Inference mode: use running statistics
        var output = zeros_like(x)
        var output_ptr = output._data
        var x_ptr = x._data
        var gamma_ptr = gamma._data
        var beta_ptr = beta._data
        var rm_ptr = running_mean._data
        var rv_ptr = running_var._data

        if x.dtype() == DType.float32:
            if should_parallelize(batch):
                # Parallel normalization over batch dimension
                @parameter
                fn normalize_batch_infer_f32(b: Int) capturing:
                    for c in range(channels):
                        var mean_val = rm_ptr.bitcast[Float32]()[c]
                        var var_val = rv_ptr.bitcast[Float32]()[c]
                        var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                        var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                        var beta_val = beta_ptr.bitcast[Float32]()[c]

                        for h in range(height):
                            for w in range(width):
                                var idx = (
                                    b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w
                                )
                                var x_val = x_ptr.bitcast[Float32]()[idx]
                                var x_norm = (x_val - mean_val) / std
                                output._set_float64(idx, Float64(gamma_val * x_norm + beta_val))

                parallelize[normalize_batch_infer_f32](batch)
            else:
                # Sequential normalization for small batches
                for b in range(batch):
                    for c in range(channels):
                        var mean_val = rm_ptr.bitcast[Float32]()[c]
                        var var_val = rv_ptr.bitcast[Float32]()[c]
                        var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                        var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                        var beta_val = beta_ptr.bitcast[Float32]()[c]

                        for h in range(height):
                            for w in range(width):
                                var idx = (
                                    b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w
                                )
                                var x_val = x_ptr.bitcast[Float32]()[idx]
                                var x_norm = (x_val - mean_val) / std
                                output[idx] = Float32(
                                    gamma_val * x_norm + beta_val
                                )

        elif x.dtype() == DType.float64:
            if should_parallelize(batch):
                # Parallel normalization over batch dimension
                @parameter
                fn normalize_batch_infer_f64(b: Int) capturing:
                    for c in range(channels):
                        var mean_val = rm_ptr.bitcast[Float64]()[c]
                        var var_val = rv_ptr.bitcast[Float64]()[c]
                        var std = sqrt_scalar_f64(var_val + epsilon)
                        var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                        var beta_val = beta_ptr.bitcast[Float64]()[c]

                        for h in range(height):
                            for w in range(width):
                                var idx = (
                                    b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w
                                )
                                var x_val = x_ptr.bitcast[Float64]()[idx]
                                var x_norm = (x_val - mean_val) / std
                                output._set_float64(idx, gamma_val * x_norm + beta_val)

                parallelize[normalize_batch_infer_f64](batch)
            else:
                # Sequential normalization for small batches
                for b in range(batch):
                    for c in range(channels):
                        var mean_val = rm_ptr.bitcast[Float64]()[c]
                        var var_val = rv_ptr.bitcast[Float64]()[c]
                        var std = sqrt_scalar_f64(var_val + epsilon)
                        var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                        var beta_val = beta_ptr.bitcast[Float64]()[c]

                        for h in range(height):
                            for w in range(width):
                                var idx = (
                                    b * (channels * height * width)
                                    + c * (height * width)
                                    + h * width
                                    + w
                                )
                                var x_val = x_ptr.bitcast[Float64]()[idx]
                                var x_norm = (x_val - mean_val) / std
                                output.set(idx, gamma_val * x_norm + beta_val)

        # Running stats unchanged in inference mode
        return (output, running_mean, running_var)


fn batch_norm2d_backward(
    grad_output: ExTensor,
    x: ExTensor,
    gamma: ExTensor,
    running_mean: ExTensor,
    running_var: ExTensor,
    training: Bool,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[ExTensor, ExTensor, ExTensor]:
    """Backward pass for 2D batch normalization.

        Computes gradients with respect to input, gamma, and beta parameters.

    Args:
            grad_output: Gradient w.r.t. output (batch, channels, height, width).
            x: Original input tensor (batch, channels, height, width).
            gamma: Scale parameter (channels,).
            running_mean: Running mean (channels,) - used in inference mode.
            running_var: Running variance (channels,) - used in inference mode.
            training: Whether in training mode (affects gradient computation).
            epsilon: Small constant for numerical stability (default: 1e-5).

    Returns:
            Tuple of (grad_input, grad_gamma, grad_beta):
                - grad_input: Gradient w.r.t. input (batch, channels, height, width).
                - grad_gamma: Gradient w.r.t. gamma (channels,).
                - grad_beta: Gradient w.r.t. beta (channels,).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import batch_norm2d_backward

            # Forward pass (save x for backward)
            var (output, new_mean, new_var) = batch_norm2d(
                x, gamma, beta, running_mean, running_var, training=True
            )

            # ... compute loss and grad_output ...

            # Backward pass
            var (grad_x, grad_gamma, grad_beta) = batch_norm2d_backward(
                grad_output, x, gamma, running_mean, running_var, training=True
            )
            ```

        Mathematical Formulation (Training Mode):

            Forward pass computes:
                mean = E[x] over (batch, height, width) per channel
                var = Var[x] over (batch, height, width) per channel
                x_norm = (x - mean) / sqrt(var + eps)
                y = gamma * x_norm + beta

            Backward pass (chain rule):
                grad_beta = sum(grad_output) over (batch, height, width) per channel
                grad_gamma = sum(grad_output * x_norm) over (batch, height, width) per channel

                grad_x_norm = grad_output * gamma
                grad_var = sum(grad_x_norm * (x - mean) * -0.5 * (var + eps)^(-3/2))
                grad_mean = sum(grad_x_norm * -1/sqrt(var + eps)) +
                            grad_var * mean(-2(x - mean))

                grad_input = grad_x_norm / sqrt(var + eps) +
                             grad_var * 2(x - mean) / N +
                             grad_mean / N

            where N = batch * height * width (spatial size)

        Mathematical Formulation (Inference Mode):

            Forward pass uses fixed running statistics:
                x_norm = (x - running_mean) / sqrt(running_var + eps)
                y = gamma * x_norm + beta

            Backward pass (simpler):
                grad_beta = sum(grad_output)
                grad_gamma = sum(grad_output * x_norm)
                grad_input = grad_output * gamma / sqrt(running_var + eps)

        References:
            - Ioffe & Szegedy (2015). Batch Normalization: Accelerating Deep Network
              Training by Reducing Internal Covariate Shift. ICML 2015
              https://arxiv.org/abs/1502.03167

            - Gradient derivation:
              https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

    Note:
            Pure functional: returns new tensors, does not modify inputs.
            Training mode requires computing batch statistics from x.
            Inference mode uses precomputed running statistics.
    """
    var x_shape = x.shape()
    var grad_shape = grad_output.shape()

    if len(x_shape) != 4 or len(grad_shape) != 4:
        raise Error(
            "batch_norm2d_backward requires 4D inputs (batch, channels, height,"
            " width)"
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    var spatial_size = batch * height * width

    # Initialize output gradients
    var grad_input = zeros_like(x)
    var grad_gamma = zeros([channels], x.dtype())
    var grad_beta = zeros([channels], x.dtype())

    var grad_output_ptr = grad_output._data
    var x_ptr = x._data
    var gamma_ptr = gamma._data
    var grad_input_ptr = grad_input._data
    var grad_gamma_ptr = grad_gamma._data
    var grad_beta_ptr = grad_beta._data

    if training:
        # ========== TRAINING MODE: Complex gradients through batch statistics ==========

        # Step 1: Compute batch statistics (mean and variance) per channel
        var batch_mean = zeros([channels], x.dtype())
        var batch_var = zeros([channels], x.dtype())
        var batch_mean_ptr = batch_mean._data
        var batch_var_ptr = batch_var._data

        if x.dtype() == DType.float32:
            var N = Float32(spatial_size)

            # Compute mean per channel
            for c in range(channels):
                var sum_val = Float32(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                batch_mean[c] = Float32(sum_val / N)

            # Compute variance per channel
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                var sum_sq_diff = Float32(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                batch_var[c] = Float32(sum_sq_diff / N)

            # Step 2: Compute grad_beta and grad_gamma
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                var var_val = batch_var_ptr.bitcast[Float32]()[c]
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                var sum_grad_output = Float32(0.0)
                var sum_grad_output_x_norm = Float32(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            sum_grad_output += grad_out
                            sum_grad_output_x_norm += grad_out * x_norm

                grad_beta[c] = Float32(sum_grad_output)
                grad_gamma[c] = Float32(sum_grad_output_x_norm)

            # Step 3: Compute grad_input using Kratzert's three-term formula
            # Follows: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float32]()[c]
                var var_val = batch_var_ptr.bitcast[Float32]()[c]
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                var invstd = Float32(1.0) / std
                var gamma_val = gamma_ptr.bitcast[Float32]()[c]

                # TRAINING MODE: Three-term formula from Kratzert's blog
                # grad_input = gamma * invstd * (grad_output - grad_mean - grad_var * (x - mean))
                # where:
                #   grad_mean = (1/N) * sum(grad_output * (-1/std)) + grad_var * (-2/N) * sum(x - mean)
                #   grad_var = (1/2) * sum(grad_output * (x - mean)) * (-1/2) * (var + eps)^(-3/2)

                # PyTorch consolidated formula (training mode):
                # grad_input = (grad_output - k/N - x_norm * dotp/N) * gamma / std
                # where:
                #   k = sum(grad_output)
                #   dotp = sum(grad_output * x_norm)
                #   x_norm = (x - mean) / std

                # First pass: compute k and dotp
                var k = Float32(0.0)
                var dotp = Float32(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) * invstd

                            k += grad_out
                            dotp += grad_out * x_norm

                # Second pass: compute grad_input
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) * invstd

                            grad_input[idx] = Float32(
                                (grad_out - k / N - x_norm * dotp / N)
                                * gamma_val
                                * invstd
                            )

        elif x.dtype() == DType.float64:
            var N = Float64(spatial_size)

            # Compute mean per channel
            for c in range(channels):
                var sum_val = Float64(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                batch_mean.set(c, sum_val / N)

            # Compute variance per channel
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                var sum_sq_diff = Float64(0.0)
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                batch_var.set(c, sum_sq_diff / N)

            # Step 2: Compute grad_beta and grad_gamma
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                var var_val = batch_var_ptr.bitcast[Float64]()[c]
                var std = sqrt_scalar_f64(var_val + epsilon)

                var sum_grad_output = Float64(0.0)
                var sum_grad_output_x_norm = Float64(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            sum_grad_output += grad_out
                            sum_grad_output_x_norm += grad_out * x_norm

                grad_beta.set(c, sum_grad_output)
                grad_gamma.set(c, sum_grad_output_x_norm)

            # Step 3: Compute grad_input using Kratzert's three-term formula
            # Follows: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
            for c in range(channels):
                var mean_val = batch_mean_ptr.bitcast[Float64]()[c]
                var var_val = batch_var_ptr.bitcast[Float64]()[c]
                var std = sqrt_scalar_f64(var_val + epsilon)
                var invstd = Float64(1.0) / std
                var gamma_val = gamma_ptr.bitcast[Float64]()[c]

                # PyTorch consolidated formula (training mode):
                # grad_input = (grad_output - k/N - x_norm * dotp/N) * gamma / std
                # where:
                #   k = sum(grad_output)
                #   dotp = sum(grad_output * x_norm)
                #   x_norm = (x - mean) / std

                # First pass: compute k and dotp
                var k = Float64(0.0)
                var dotp = Float64(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) * invstd

                            k += grad_out
                            dotp += grad_out * x_norm

                # Second pass: compute grad_input
                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) * invstd

                            grad_input.set(
                                idx,
                                (
                                    (grad_out - k / N - x_norm * dotp / N)
                                    * gamma_val
                                    * invstd
                                )
                            )

        else:
            raise Error(
                "batch_norm2d_backward: only float32/64 dtypes supported"
            )

    else:
        # ========== INFERENCE MODE: Simpler gradients using running statistics ==========

        var rm_ptr = running_mean._data
        var rv_ptr = running_var._data

        if x.dtype() == DType.float32:
            # Compute grad_beta and grad_gamma
            for c in range(channels):
                var mean_val = rm_ptr.bitcast[Float32]()[c]
                var var_val = rv_ptr.bitcast[Float32]()[c]
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                var sum_grad_output = Float32(0.0)
                var sum_grad_output_x_norm = Float32(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            sum_grad_output += grad_out
                            sum_grad_output_x_norm += grad_out * x_norm

                grad_beta[c] = Float32(sum_grad_output)
                grad_gamma[c] = Float32(sum_grad_output_x_norm)

            # Compute grad_input (simple rescaling)
            for b in range(batch):
                for c in range(channels):
                    var var_val = rv_ptr.bitcast[Float32]()[c]
                    var std = sqrt_scalar_f32(var_val + Float32(epsilon))
                    var gamma_val = gamma_ptr.bitcast[Float32]()[c]

                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            grad_input[idx] = Float32(
                                grad_out * gamma_val / std
                            )

        elif x.dtype() == DType.float64:
            # Compute grad_beta and grad_gamma
            for c in range(channels):
                var mean_val = rm_ptr.bitcast[Float64]()[c]
                var var_val = rv_ptr.bitcast[Float64]()[c]
                var std = sqrt_scalar_f64(var_val + epsilon)

                var sum_grad_output = Float64(0.0)
                var sum_grad_output_x_norm = Float64(0.0)

                for b in range(batch):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            sum_grad_output += grad_out
                            sum_grad_output_x_norm += grad_out * x_norm

                grad_beta.set(c, sum_grad_output)
                grad_gamma.set(c, sum_grad_output_x_norm)

            # Compute grad_input (simple rescaling)
            for b in range(batch):
                for c in range(channels):
                    var var_val = rv_ptr.bitcast[Float64]()[c]
                    var std = sqrt_scalar_f64(var_val + epsilon)
                    var gamma_val = gamma_ptr.bitcast[Float64]()[c]

                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            grad_input.set(idx, grad_out * gamma_val / std)

        else:
            raise Error(
                "batch_norm2d_backward: only float32/64 dtypes supported"
            )

    return (grad_input, grad_gamma, grad_beta)


fn layer_norm(
    x: ExTensor, gamma: ExTensor, beta: ExTensor, epsilon: Float64 = 1e-5
) raises -> ExTensor:
    """Functional layer normalization.

        Normalizes activations across the feature dimension for each sample.
        Unlike batch norm, this doesn't require running statistics.

    Args:
            x: Input tensor of shape (batch, features) or (batch, channels, height, width).
            gamma: Scale parameter of shape matching last dim(s).
            beta: Shift parameter of shape matching last dim(s).
            epsilon: Small constant for numerical stability (default: 1e-5).

    Returns:
            Normalized tensor, same shape as input.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import layer_norm, zeros, ones

            # For 2D input (batch, features)
            var gamma = ones([features])
            var beta = zeros([features])
            var output = layer_norm(x, gamma, beta)

            # For 4D input (batch, channels, height, width)
            var gamma = ones([channels, height, width])
            var beta = zeros([channels, height, width])
            var output = layer_norm(x, gamma, beta)
            ```

        Formula:
            For each sample:
                mean = mean(x[i])  # Over all features
                var = var(x[i])
                x_norm[i] = (x[i] - mean) / sqrt(var + epsilon)
                output[i] = gamma * x_norm[i] + beta

    Note:
            - No running statistics needed (stateless).
            - Normalizes each sample independently.
            - Commonly used in transformers and RNNs.
    """
    var x_shape = x.shape()

    if len(x_shape) == 2:
        # 2D input: (batch, features)
        var batch = x_shape[0]
        var features = x_shape[1]

        var output = zeros_like(x)
        var x_ptr = x._data
        var output_ptr = output._data
        var gamma_ptr = gamma._data
        var beta_ptr = beta._data

        if x.dtype() == DType.float32:
            for b in range(batch):
                # Compute mean for this sample
                var sum_val = Float32(0.0)
                for f in range(features):
                    var idx = b * features + f
                    sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / Float32(features)

                # Compute variance
                var sum_sq_diff = Float32(0.0)
                for f in range(features):
                    var idx = b * features + f
                    var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                    sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / Float32(features)
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Normalize and scale
                for f in range(features):
                    var idx = b * features + f
                    var x_val = x_ptr.bitcast[Float32]()[idx]
                    var x_norm = (x_val - mean_val) / std
                    var gamma_val = gamma_ptr.bitcast[Float32]()[f]
                    var beta_val = beta_ptr.bitcast[Float32]()[f]
                    output[idx] = Float32(
                        gamma_val * x_norm + beta_val
                    )

        elif x.dtype() == DType.float64:
            for b in range(batch):
                var sum_val = Float64(0.0)
                for f in range(features):
                    var idx = b * features + f
                    sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / Float64(features)

                var sum_sq_diff = Float64(0.0)
                for f in range(features):
                    var idx = b * features + f
                    var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                    sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / Float64(features)
                var std = sqrt_scalar_f64(var_val + epsilon)

                for f in range(features):
                    var idx = b * features + f
                    var x_val = x_ptr.bitcast[Float64]()[idx]
                    var x_norm = (x_val - mean_val) / std
                    var gamma_val = gamma_ptr.bitcast[Float64]()[f]
                    var beta_val = beta_ptr.bitcast[Float64]()[f]
                    output.set(idx, gamma_val * x_norm + beta_val)
        else:
            raise Error("layer_norm: only float32/64 dtypes supported")

        return output

    elif len(x_shape) == 4:
        # 4D input: (batch, channels, height, width)
        # Normalize over (channels, height, width) for each sample
        var batch = x_shape[0]
        var channels = x_shape[1]
        var height = x_shape[2]
        var width = x_shape[3]
        var feature_size = channels * height * width

        var output = zeros_like(x)
        var x_ptr = x._data
        var output_ptr = output._data
        var gamma_ptr = gamma._data
        var beta_ptr = beta._data

        if x.dtype() == DType.float32:
            for b in range(batch):
                # Compute mean for this sample
                var sum_val = Float32(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / Float32(feature_size)

                # Compute variance
                var sum_sq_diff = Float32(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / Float32(feature_size)
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Normalize and scale
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var gamma_idx = c * (height * width) + h * width + w
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            var gamma_val = gamma_ptr.bitcast[Float32]()[
                                gamma_idx
                            ]
                            var beta_val = beta_ptr.bitcast[Float32]()[
                                gamma_idx
                            ]
                            output[idx] = Float32(
                                gamma_val * x_norm + beta_val
                            )

        elif x.dtype() == DType.float64:
            for b in range(batch):
                var sum_val = Float64(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / Float64(feature_size)

                var sum_sq_diff = Float64(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / Float64(feature_size)
                var std = sqrt_scalar_f64(var_val + epsilon)

                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var gamma_idx = c * (height * width) + h * width + w
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            var gamma_val = gamma_ptr.bitcast[Float64]()[
                                gamma_idx
                            ]
                            var beta_val = beta_ptr.bitcast[Float64]()[
                                gamma_idx
                            ]
                            output.set(idx, gamma_val * x_norm + beta_val)
        else:
            raise Error("layer_norm: only float32/64 dtypes supported")

        return output

    else:
        raise Error("layer_norm supports 2D or 4D inputs only")


fn layer_norm_backward(
    grad_output: ExTensor, x: ExTensor, gamma: ExTensor, epsilon: Float64 = 1e-5
) raises -> Tuple[ExTensor, ExTensor, ExTensor]:
    """Backward pass for layer normalization.

        Computes gradients with respect to input, gamma, and beta parameters.

    Args:
            grad_output: Gradient w.r.t. output, same shape as input.
            x: Original input tensor (batch, features) or (batch, channels, height, width).
            gamma: Scale parameter matching normalized dimensions.
            epsilon: Small constant for numerical stability (default: 1e-5).

    Returns:
            Tuple of (grad_input, grad_gamma, grad_beta):
                - grad_input: Gradient w.r.t. input (same shape as input).
                - grad_gamma: Gradient w.r.t. gamma (same shape as gamma).
                - grad_beta: Gradient w.r.t. beta (same shape as gamma).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import layer_norm, layer_norm_backward

            # Forward pass
            var output = layer_norm(x, gamma, beta)

            # ... compute loss and grad_output ...

            # Backward pass
            var (grad_x, grad_gamma, grad_beta) = layer_norm_backward(
                grad_output, x, gamma
            )
            ```

        Mathematical Formulation:

            Forward pass computes (per sample):
                mean = E[x] over features
                var = Var[x] over features
                x_norm = (x - mean) / sqrt(var + eps)
                y = gamma * x_norm + beta

            Backward pass (chain rule):
                grad_beta = sum(grad_output) over batch dimension
                grad_gamma = sum(grad_output * x_norm) over batch dimension

                grad_x_norm = grad_output * gamma
                grad_var = sum(grad_x_norm * (x - mean) * -0.5 * (var + eps)^(-3/2))
                grad_mean = sum(grad_x_norm * -1/sqrt(var + eps)) +
                            grad_var * mean(-2(x - mean))

                grad_input = grad_x_norm / sqrt(var + eps) +
                             grad_var * 2(x - mean) / N +
                             grad_mean / N

            where N = number of features being normalized

        References:
            - Ba et al. (2016). Layer Normalization
              https://arxiv.org/abs/1607.06450

    Note:
            Pure functional: returns new tensors, does not modify inputs.
            Unlike batch_norm, there are no running statistics - normalization is
            computed independently for each sample.
    """
    var x_shape = x.shape()
    var grad_shape = grad_output.shape()

    if len(x_shape) == 2:
        # ========== 2D Input: (batch, features) ==========
        var batch = x_shape[0]
        var features = x_shape[1]
        var N = Float32(features)

        # Initialize output gradients
        var grad_input = zeros_like(x)
        var grad_gamma = zeros_like(gamma)
        var grad_beta = zeros_like(gamma)

        var grad_output_ptr = grad_output._data
        var x_ptr = x._data
        var gamma_ptr = gamma._data
        var grad_input_ptr = grad_input._data
        var grad_gamma_ptr = grad_gamma._data
        var grad_beta_ptr = grad_beta._data

        if x.dtype() == DType.float32:
            # First pass: compute grad_gamma and grad_beta by accumulating over batch
            for b in range(batch):
                # Compute mean and variance for this sample
                var sum_val = Float32(0.0)
                for f in range(features):
                    var idx = b * features + f
                    sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float32(0.0)
                for f in range(features):
                    var idx = b * features + f
                    var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                    sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Accumulate grad_gamma and grad_beta
                for f in range(features):
                    var idx = b * features + f
                    var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                    var x_val = x_ptr.bitcast[Float32]()[idx]
                    var x_norm = (x_val - mean_val) / std

                    grad_beta[f] = Float32(grad_beta[f] + grad_out)
                    grad_gamma[f] = Float32(grad_gamma[f] + grad_out * x_norm)

            # Second pass: compute grad_input for each sample
            for b in range(batch):
                # Recompute mean and variance for this sample
                var sum_val = Float32(0.0)
                for f in range(features):
                    var idx = b * features + f
                    sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float32(0.0)
                for f in range(features):
                    var idx = b * features + f
                    var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                    sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Compute gradient through variance and mean
                var grad_var = Float32(0.0)
                var grad_mean = Float32(0.0)

                for f in range(features):
                    var idx = b * features + f
                    var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                    var x_val = x_ptr.bitcast[Float32]()[idx]
                    var x_minus_mean = x_val - mean_val
                    var gamma_val = gamma_ptr.bitcast[Float32]()[f]

                    var grad_x_norm = grad_out * gamma_val

                    # Accumulate for grad_var
                    grad_var += (
                        grad_x_norm
                        * x_minus_mean
                        * Float32(-0.5)
                        * pow_scalar_f32(
                            var_val + Float32(epsilon), Float32(-1.5)
                        )
                    )

                    # Accumulate for grad_mean
                    grad_mean += grad_x_norm * Float32(-1.0) / std

                # Compute grad_input for each feature
                for f in range(features):
                    var idx = b * features + f
                    var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                    var x_val = x_ptr.bitcast[Float32]()[idx]
                    var x_minus_mean = x_val - mean_val
                    var gamma_val = gamma_ptr.bitcast[Float32]()[f]

                    # Three terms in layer norm backward:
                    var grad_x_norm = grad_out * gamma_val
                    var term1 = grad_x_norm / std
                    var term2 = grad_var * Float32(2.0) * x_minus_mean / N
                    var term3 = grad_mean / N

                    grad_input[idx] = Float32(
                        term1 + term2 + term3
                    )

        elif x.dtype() == DType.float64:
            var N64 = Float64(features)

            for b in range(batch):
                var sum_val = Float64(0.0)
                for f in range(features):
                    var idx = b * features + f
                    sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N64

                var sum_sq_diff = Float64(0.0)
                for f in range(features):
                    var idx = b * features + f
                    var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                    sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N64
                var std = sqrt_scalar_f64(var_val + epsilon)

                for f in range(features):
                    var idx = b * features + f
                    var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                    var x_val = x_ptr.bitcast[Float64]()[idx]
                    var x_norm = (x_val - mean_val) / std

                    grad_beta.set(f, Float64(grad_beta[f]) + grad_out)
                    grad_gamma.set(f, Float64(grad_gamma[f]) + grad_out * x_norm)

            for b in range(batch):
                var sum_val = Float64(0.0)
                for f in range(features):
                    var idx = b * features + f
                    sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N64

                var sum_sq_diff = Float64(0.0)
                for f in range(features):
                    var idx = b * features + f
                    var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                    sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N64
                var std = sqrt_scalar_f64(var_val + epsilon)

                var grad_var = Float64(0.0)
                var grad_mean = Float64(0.0)

                for f in range(features):
                    var idx = b * features + f
                    var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                    var x_val = x_ptr.bitcast[Float64]()[idx]
                    var x_minus_mean = x_val - mean_val
                    var gamma_val = gamma_ptr.bitcast[Float64]()[f]

                    var grad_x_norm = grad_out * gamma_val
                    grad_var += (
                        grad_x_norm
                        * x_minus_mean
                        * Float64(-0.5)
                        * pow_scalar_f64(var_val + epsilon, Float64(-1.5))
                    )
                    grad_mean += grad_x_norm * Float64(-1.0) / std

                for f in range(features):
                    var idx = b * features + f
                    var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                    var x_val = x_ptr.bitcast[Float64]()[idx]
                    var x_minus_mean = x_val - mean_val
                    var gamma_val = gamma_ptr.bitcast[Float64]()[f]

                    var grad_x_norm = grad_out * gamma_val
                    var term1 = grad_x_norm / std
                    var term2 = grad_var * Float64(2.0) * x_minus_mean / N64
                    var term3 = grad_mean / N64

                    grad_input.set(idx, term1 + term2 + term3)

        else:
            raise Error("layer_norm_backward: only float32/64 dtypes supported")

        return (grad_input, grad_gamma, grad_beta)

    elif len(x_shape) == 4:
        # ========== 4D Input: (batch, channels, height, width) ==========
        # Normalize over (channels, height, width) for each sample
        var batch = x_shape[0]
        var channels = x_shape[1]
        var height = x_shape[2]
        var width = x_shape[3]
        var feature_size = channels * height * width
        var N = Float32(feature_size)

        # Initialize output gradients
        var grad_input = zeros_like(x)
        var grad_gamma = zeros_like(gamma)
        var grad_beta = zeros_like(gamma)

        var grad_output_ptr = grad_output._data
        var x_ptr = x._data
        var gamma_ptr = gamma._data
        var grad_input_ptr = grad_input._data
        var grad_gamma_ptr = grad_gamma._data
        var grad_beta_ptr = grad_beta._data

        if x.dtype() == DType.float32:
            # First pass: compute grad_gamma and grad_beta by accumulating over batch
            for b in range(batch):
                # Compute mean and variance for this sample
                var sum_val = Float32(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float32(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Accumulate grad_gamma and grad_beta
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var gamma_idx = c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            grad_beta[gamma_idx] = Float32(grad_beta[gamma_idx] + grad_out)
                            grad_gamma[gamma_idx] = Float32(grad_gamma[gamma_idx] + grad_out * x_norm)

            # Second pass: compute grad_input for each sample
            for b in range(batch):
                # Recompute mean and variance for this sample
                var sum_val = Float32(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float32(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Compute gradient through variance and mean
                var grad_var = Float32(0.0)
                var grad_mean = Float32(0.0)

                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var gamma_idx = c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_minus_mean = x_val - mean_val
                            var gamma_val = gamma_ptr.bitcast[Float32]()[
                                gamma_idx
                            ]

                            var grad_x_norm = grad_out * gamma_val
                            grad_var += (
                                grad_x_norm
                                * x_minus_mean
                                * Float32(-0.5)
                                * pow_scalar_f32(
                                    var_val + Float32(epsilon), Float32(-1.5)
                                )
                            )
                            grad_mean += grad_x_norm * Float32(-1.0) / std

                # Compute grad_input for each element
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var gamma_idx = c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_minus_mean = x_val - mean_val
                            var gamma_val = gamma_ptr.bitcast[Float32]()[
                                gamma_idx
                            ]

                            var grad_x_norm = grad_out * gamma_val
                            var term1 = grad_x_norm / std
                            var term2 = (
                                grad_var * Float32(2.0) * x_minus_mean / N
                            )
                            var term3 = grad_mean / N

                            grad_input[idx] = Float32(
                                term1 + term2 + term3
                            )

        elif x.dtype() == DType.float64:
            var N64 = Float64(feature_size)

            for b in range(batch):
                var sum_val = Float64(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N64

                var sum_sq_diff = Float64(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N64
                var std = sqrt_scalar_f64(var_val + epsilon)

                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var gamma_idx = c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            grad_beta.set(gamma_idx, Float64(grad_beta[gamma_idx]) + grad_out)
                            grad_gamma.set(gamma_idx, Float64(grad_gamma[gamma_idx]) + grad_out * x_norm)

            for b in range(batch):
                var sum_val = Float64(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N64

                var sum_sq_diff = Float64(0.0)
                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N64
                var std = sqrt_scalar_f64(var_val + epsilon)

                var grad_var = Float64(0.0)
                var grad_mean = Float64(0.0)

                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var gamma_idx = c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_minus_mean = x_val - mean_val
                            var gamma_val = gamma_ptr.bitcast[Float64]()[
                                gamma_idx
                            ]

                            var grad_x_norm = grad_out * gamma_val
                            grad_var += (
                                grad_x_norm
                                * x_minus_mean
                                * Float64(-0.5)
                                * pow_scalar_f64(
                                    var_val + epsilon, Float64(-1.5)
                                )
                            )
                            grad_mean += grad_x_norm * Float64(-1.0) / std

                for c in range(channels):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var gamma_idx = c * (height * width) + h * width + w
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_minus_mean = x_val - mean_val
                            var gamma_val = gamma_ptr.bitcast[Float64]()[
                                gamma_idx
                            ]

                            var grad_x_norm = grad_out * gamma_val
                            var term1 = grad_x_norm / std
                            var term2 = (
                                grad_var * Float64(2.0) * x_minus_mean / N64
                            )
                            var term3 = grad_mean / N64

                            grad_input.set(idx, term1 + term2 + term3)

        else:
            raise Error("layer_norm_backward: only float32/64 dtypes supported")

        return (grad_input, grad_gamma, grad_beta)

    else:
        raise Error("layer_norm_backward supports 2D or 4D inputs only")


fn group_norm(
    x: ExTensor,
    num_groups: Int,
    gamma: ExTensor,
    beta: ExTensor,
    epsilon: Float64 = 1e-5,
) raises -> ExTensor:
    """Functional group normalization.

        Normalizes activations by dividing channels into groups and normalizing
        within each group. Works well with small batch sizes where batch norm fails.

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            num_groups: Number of groups to divide channels into.
            gamma: Scale parameter of shape (channels,).
            beta: Shift parameter of shape (channels,).
            epsilon: Small constant for numerical stability (default: 1e-5).

    Returns:
            Normalized tensor, same shape as input.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import group_norm, zeros, ones

            # Divide 32 channels into 8 groups of 4 channels each
            var gamma = ones([32])
            var beta = zeros([32])
            var output = group_norm(x, num_groups=8, gamma=gamma, beta=beta)
            ```

        Formula:
            For each sample and each group:
                mean = mean(x[group]) over spatial and channel dims within group
                var = var(x[group])
                x_norm = (x - mean) / sqrt(var + epsilon)
                output = gamma * x_norm + beta

    Note:
            - Channels must be divisible by num_groups.
            - No running statistics needed (stateless).
            - Commonly used in detection and segmentation models.
    """
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error(
            "group_norm requires 4D input (batch, channels, height, width)"
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    if channels % num_groups != 0:
        raise Error("group_norm: channels must be divisible by num_groups")

    var channels_per_group = channels // num_groups
    var group_size = channels_per_group * height * width

    var output = zeros_like(x)
    var x_ptr = x._data
    var output_ptr = output._data
    var gamma_ptr = gamma._data
    var beta_ptr = beta._data

    if x.dtype() == DType.float32:
        var N = Float32(group_size)

        for b in range(batch):
            for g in range(num_groups):
                var c_start = g * channels_per_group
                var c_end = c_start + channels_per_group

                # Compute mean for this group
                var sum_val = Float32(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                # Compute variance for this group
                var sum_sq_diff = Float32(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Normalize and scale
                for c in range(c_start, c_end):
                    var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                    var beta_val = beta_ptr.bitcast[Float32]()[c]
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            output[idx] = Float32(
                                gamma_val * x_norm + beta_val
                            )

    elif x.dtype() == DType.float64:
        var N = Float64(group_size)

        for b in range(batch):
            for g in range(num_groups):
                var c_start = g * channels_per_group
                var c_end = c_start + channels_per_group

                var sum_val = Float64(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float64(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f64(var_val + epsilon)

                for c in range(c_start, c_end):
                    var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                    var beta_val = beta_ptr.bitcast[Float64]()[c]
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std
                            output.set(idx, gamma_val * x_norm + beta_val)
    else:
        raise Error("group_norm: only float32/64 dtypes supported")

    return output


fn group_norm_backward(
    grad_output: ExTensor,
    x: ExTensor,
    num_groups: Int,
    gamma: ExTensor,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[ExTensor, ExTensor, ExTensor]:
    """Backward pass for group normalization.

        Computes gradients with respect to input, gamma, and beta parameters.

    Args:
            grad_output: Gradient w.r.t. output (batch, channels, height, width).
            x: Original input tensor (batch, channels, height, width).
            num_groups: Number of groups channels were divided into.
            gamma: Scale parameter (channels,).
            epsilon: Small constant for numerical stability (default: 1e-5).

    Returns:
            Tuple of (grad_input, grad_gamma, grad_beta):
                - grad_input: Gradient w.r.t. input (batch, channels, height, width).
                - grad_gamma: Gradient w.r.t. gamma (channels,).
                - grad_beta: Gradient w.r.t. beta (channels,).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import group_norm, group_norm_backward

            # Forward pass
            var output = group_norm(x, num_groups=8, gamma=gamma, beta=beta)

            # ... compute loss and grad_output ...

            # Backward pass
            var (grad_x, grad_gamma, grad_beta) = group_norm_backward(
                grad_output, x, num_groups=8, gamma=gamma
            )
            ```

    Note:
            Pure functional: returns new tensors, does not modify inputs.
    """
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error(
            "group_norm_backward requires 4D input (batch, channels, height,"
            " width)"
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    if channels % num_groups != 0:
        raise Error(
            "group_norm_backward: channels must be divisible by num_groups"
        )

    var channels_per_group = channels // num_groups
    var group_size = channels_per_group * height * width

    # Initialize output gradients
    var grad_input = zeros_like(x)
    var grad_gamma = zeros([channels], x.dtype())
    var grad_beta = zeros([channels], x.dtype())

    var grad_output_ptr = grad_output._data
    var x_ptr = x._data
    var gamma_ptr = gamma._data
    var grad_input_ptr = grad_input._data
    var grad_gamma_ptr = grad_gamma._data
    var grad_beta_ptr = grad_beta._data

    if x.dtype() == DType.float32:
        var N = Float32(group_size)

        # First pass: compute grad_gamma and grad_beta
        for b in range(batch):
            for g in range(num_groups):
                var c_start = g * channels_per_group
                var c_end = c_start + channels_per_group

                # Compute mean and variance for this group
                var sum_val = Float32(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float32(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Accumulate grad_gamma and grad_beta
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            grad_beta[c] = Float32(grad_beta[c] + grad_out)
                            grad_gamma[c] = Float32(grad_gamma[c] + grad_out * x_norm)

        # Second pass: compute grad_input
        for b in range(batch):
            for g in range(num_groups):
                var c_start = g * channels_per_group
                var c_end = c_start + channels_per_group

                # Recompute mean and variance
                var sum_val = Float32(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float32(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Compute gradient accumulations
                var grad_var = Float32(0.0)
                var grad_mean = Float32(0.0)

                for c in range(c_start, c_end):
                    var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_minus_mean = x_val - mean_val

                            var grad_x_norm = grad_out * gamma_val
                            grad_var += (
                                grad_x_norm
                                * x_minus_mean
                                * Float32(-0.5)
                                * pow_scalar_f32(
                                    var_val + Float32(epsilon), Float32(-1.5)
                                )
                            )
                            grad_mean += grad_x_norm * Float32(-1.0) / std

                # Compute grad_input
                for c in range(c_start, c_end):
                    var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float32]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float32]()[idx]
                            var x_minus_mean = x_val - mean_val

                            var grad_x_norm = grad_out * gamma_val
                            var term1 = grad_x_norm / std
                            var term2 = (
                                grad_var * Float32(2.0) * x_minus_mean / N
                            )
                            var term3 = grad_mean / N

                            grad_input[idx] = Float32(
                                term1 + term2 + term3
                            )

    elif x.dtype() == DType.float64:
        var N = Float64(group_size)

        for b in range(batch):
            for g in range(num_groups):
                var c_start = g * channels_per_group
                var c_end = c_start + channels_per_group

                var sum_val = Float64(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float64(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f64(var_val + epsilon)

                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_norm = (x_val - mean_val) / std

                            grad_beta.set(c, Float64(grad_beta[c]) + grad_out)
                            grad_gamma.set(c, Float64(grad_gamma[c]) + grad_out * x_norm)

        for b in range(batch):
            for g in range(num_groups):
                var c_start = g * channels_per_group
                var c_end = c_start + channels_per_group

                var sum_val = Float64(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float64(0.0)
                for c in range(c_start, c_end):
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                            sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f64(var_val + epsilon)

                var grad_var = Float64(0.0)
                var grad_mean = Float64(0.0)

                for c in range(c_start, c_end):
                    var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_minus_mean = x_val - mean_val

                            var grad_x_norm = grad_out * gamma_val
                            grad_var += (
                                grad_x_norm
                                * x_minus_mean
                                * Float64(-0.5)
                                * pow_scalar_f64(
                                    var_val + epsilon, Float64(-1.5)
                                )
                            )
                            grad_mean += grad_x_norm * Float64(-1.0) / std

                for c in range(c_start, c_end):
                    var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                    for h in range(height):
                        for w in range(width):
                            var idx = (
                                b * (channels * height * width)
                                + c * (height * width)
                                + h * width
                                + w
                            )
                            var grad_out = grad_output_ptr.bitcast[Float64]()[
                                idx
                            ]
                            var x_val = x_ptr.bitcast[Float64]()[idx]
                            var x_minus_mean = x_val - mean_val

                            var grad_x_norm = grad_out * gamma_val
                            var term1 = grad_x_norm / std
                            var term2 = (
                                grad_var * Float64(2.0) * x_minus_mean / N
                            )
                            var term3 = grad_mean / N

                            grad_input.set(idx, term1 + term2 + term3)

    else:
        raise Error("group_norm_backward: only float32/64 dtypes supported")

    return (grad_input, grad_gamma, grad_beta)


fn instance_norm(
    x: ExTensor,
    gamma: ExTensor,
    beta: ExTensor,
    epsilon: Float64 = 1e-5,
) raises -> ExTensor:
    """Functional instance normalization.

        Normalizes each sample independently across spatial dimensions for each channel.
        Used in style transfer and image generation models.

    Args:
            x: Input tensor of shape (batch, channels, height, width).
            gamma: Scale parameter of shape (channels,).
            beta: Shift parameter of shape (channels,).
            epsilon: Small constant for numerical stability (default: 1e-5).

    Returns:
            Normalized tensor, same shape as input.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import instance_norm, zeros, ones

            var gamma = ones([channels])
            var beta = zeros([channels])
            var output = instance_norm(x, gamma=gamma, beta=beta)
            ```

        Formula:
            For each sample b and channel c:
                mean = mean(x[b, c, :, :]) over spatial dims (H, W)
                var = var(x[b, c, :, :])
                x_norm = (x - mean) / sqrt(var + epsilon)
                output = gamma * x_norm + beta

    Note:
            - No batch statistics needed (each sample normalized independently).
            - No running statistics (stateless).
            - Commonly used in style transfer, GANs, and image generation.
    """
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error(
            "instance_norm requires 4D input (batch, channels, height, width)"
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]
    var spatial_size = height * width

    var output = zeros_like(x)
    var x_ptr = x._data
    var output_ptr = output._data
    var gamma_ptr = gamma._data
    var beta_ptr = beta._data

    if x.dtype() == DType.float32:
        var N = Float32(spatial_size)

        for b in range(batch):
            for c in range(channels):
                # Compute mean for this instance (sample, channel)
                var sum_val = Float32(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                # Compute variance
                var sum_sq_diff = Float32(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                        sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Normalize and scale
                var gamma_val = gamma_ptr.bitcast[Float32]()[c]
                var beta_val = beta_ptr.bitcast[Float32]()[c]
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var x_val = x_ptr.bitcast[Float32]()[idx]
                        var x_norm = (x_val - mean_val) / std
                        output[idx] = Float32(
                            gamma_val * x_norm + beta_val
                        )

    elif x.dtype() == DType.float64:
        var N = Float64(spatial_size)

        for b in range(batch):
            for c in range(channels):
                var sum_val = Float64(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float64(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                        sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f64(var_val + epsilon)

                var gamma_val = gamma_ptr.bitcast[Float64]()[c]
                var beta_val = beta_ptr.bitcast[Float64]()[c]
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var x_val = x_ptr.bitcast[Float64]()[idx]
                        var x_norm = (x_val - mean_val) / std
                        output.set(idx, gamma_val * x_norm + beta_val)
    else:
        raise Error("instance_norm: only float32/64 dtypes supported")

    return output


fn instance_norm_backward(
    grad_output: ExTensor,
    x: ExTensor,
    gamma: ExTensor,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[ExTensor, ExTensor, ExTensor]:
    """Backward pass for instance normalization.

        Computes gradients with respect to input, gamma, and beta parameters.

    Args:
            grad_output: Gradient w.r.t. output (batch, channels, height, width).
            x: Original input tensor (batch, channels, height, width).
            gamma: Scale parameter (channels,).
            epsilon: Small constant for numerical stability (default: 1e-5).

    Returns:
            Tuple of (grad_input, grad_gamma, grad_beta):
                - grad_input: Gradient w.r.t. input (batch, channels, height, width).
                - grad_gamma: Gradient w.r.t. gamma (channels,).
                - grad_beta: Gradient w.r.t. beta (channels,).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import instance_norm, instance_norm_backward

            # Forward pass
            var output = instance_norm(x, gamma=gamma, beta=beta)

            # ... compute loss and grad_output ...

            # Backward pass
            var (grad_x, grad_gamma, grad_beta) = instance_norm_backward(
                grad_output, x, gamma=gamma
            )
            ```

    Note:
            Pure functional: returns new tensors, does not modify inputs.
    """
    var x_shape = x.shape()
    if len(x_shape) != 4:
        raise Error(
            "instance_norm_backward requires 4D input (batch, channels, height,"
            " width)"
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]
    var spatial_size = height * width

    # Initialize output gradients
    var grad_input = zeros_like(x)
    var grad_gamma = zeros([channels], x.dtype())
    var grad_beta = zeros([channels], x.dtype())

    var grad_output_ptr = grad_output._data
    var x_ptr = x._data
    var gamma_ptr = gamma._data
    var grad_input_ptr = grad_input._data
    var grad_gamma_ptr = grad_gamma._data
    var grad_beta_ptr = grad_beta._data

    if x.dtype() == DType.float32:
        var N = Float32(spatial_size)

        # First pass: compute grad_gamma and grad_beta
        for b in range(batch):
            for c in range(channels):
                # Compute mean and variance for this instance
                var sum_val = Float32(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float32(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                        sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                # Accumulate grad_gamma and grad_beta
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                        var x_val = x_ptr.bitcast[Float32]()[idx]
                        var x_norm = (x_val - mean_val) / std

                        grad_beta[c] = Float32(grad_beta[c] + grad_out)
                        grad_gamma[c] = Float32(grad_gamma[c] + grad_out * x_norm)

        # Second pass: compute grad_input
        for b in range(batch):
            for c in range(channels):
                # Recompute mean and variance
                var sum_val = Float32(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        sum_val += x_ptr.bitcast[Float32]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float32(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var diff = x_ptr.bitcast[Float32]()[idx] - mean_val
                        sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f32(var_val + Float32(epsilon))

                var gamma_val = gamma_ptr.bitcast[Float32]()[c]

                # Compute gradient accumulations
                var grad_var = Float32(0.0)
                var grad_mean = Float32(0.0)

                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                        var x_val = x_ptr.bitcast[Float32]()[idx]
                        var x_minus_mean = x_val - mean_val

                        var grad_x_norm = grad_out * gamma_val
                        grad_var += (
                            grad_x_norm
                            * x_minus_mean
                            * Float32(-0.5)
                            * pow_scalar_f32(
                                var_val + Float32(epsilon), Float32(-1.5)
                            )
                        )
                        grad_mean += grad_x_norm * Float32(-1.0) / std

                # Compute grad_input
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var grad_out = grad_output_ptr.bitcast[Float32]()[idx]
                        var x_val = x_ptr.bitcast[Float32]()[idx]
                        var x_minus_mean = x_val - mean_val

                        var grad_x_norm = grad_out * gamma_val
                        var term1 = grad_x_norm / std
                        var term2 = grad_var * Float32(2.0) * x_minus_mean / N
                        var term3 = grad_mean / N

                        grad_input[idx] = Float32(
                            term1 + term2 + term3
                        )

    elif x.dtype() == DType.float64:
        var N = Float64(spatial_size)

        # First pass: compute grad_gamma and grad_beta
        for b in range(batch):
            for c in range(channels):
                var sum_val = Float64(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float64(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                        sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f64(var_val + epsilon)

                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                        var x_val = x_ptr.bitcast[Float64]()[idx]
                        var x_norm = (x_val - mean_val) / std

                        grad_beta.set(c, Float64(grad_beta[c]) + grad_out)
                        grad_gamma.set(c, Float64(grad_gamma[c]) + grad_out * x_norm)

        # Second pass: compute grad_input
        for b in range(batch):
            for c in range(channels):
                var sum_val = Float64(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        sum_val += x_ptr.bitcast[Float64]()[idx]
                var mean_val = sum_val / N

                var sum_sq_diff = Float64(0.0)
                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var diff = x_ptr.bitcast[Float64]()[idx] - mean_val
                        sum_sq_diff += diff * diff
                var var_val = sum_sq_diff / N
                var std = sqrt_scalar_f64(var_val + epsilon)

                var gamma_val = gamma_ptr.bitcast[Float64]()[c]

                var grad_var = Float64(0.0)
                var grad_mean = Float64(0.0)

                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                        var x_val = x_ptr.bitcast[Float64]()[idx]
                        var x_minus_mean = x_val - mean_val

                        var grad_x_norm = grad_out * gamma_val
                        grad_var += (
                            grad_x_norm
                            * x_minus_mean
                            * Float64(-0.5)
                            * pow_scalar_f64(var_val + epsilon, Float64(-1.5))
                        )
                        grad_mean += grad_x_norm * Float64(-1.0) / std

                for h in range(height):
                    for w in range(width):
                        var idx = (
                            b * (channels * height * width)
                            + c * (height * width)
                            + h * width
                            + w
                        )
                        var grad_out = grad_output_ptr.bitcast[Float64]()[idx]
                        var x_val = x_ptr.bitcast[Float64]()[idx]
                        var x_minus_mean = x_val - mean_val

                        var grad_x_norm = grad_out * gamma_val
                        var term1 = grad_x_norm / std
                        var term2 = grad_var * Float64(2.0) * x_minus_mean / N
                        var term3 = grad_mean / N

                        grad_input.set(idx, term1 + term2 + term3)

    else:
        raise Error("instance_norm_backward: only float32/64 dtypes supported")

    return (grad_input, grad_gamma, grad_beta)
