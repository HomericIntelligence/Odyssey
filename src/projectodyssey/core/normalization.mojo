"""Functional normalization layers.

This module provides pure functional implementations of normalization operations
All operations are stateless - caller manages running statistics and parameters.
"""

from std.algorithm import parallelize

from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    zeros_like,
    ones_like,
    full_like,
)
from projectodyssey.core.parallel_utils import should_parallelize
from projectodyssey.core.dtype_dispatch import dispatch_float3


# ============================================================================
# Internal: Compute 4D flat index for (b, c, h, w) layout
# ============================================================================


@always_inline
def _idx4d(
    b: Int, c: Int, h: Int, w: Int, channels: Int, height: Int, width: Int
) -> Int:
    return (
        b * (channels * height * width) + c * (height * width) + h * width + w
    )


# ============================================================================
# Internal: Typed sqrt and pow helpers
# ============================================================================


@always_inline
def _sqrt_typed[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Compute square root for any float dtype using Scalar[dtype]."""
    return x**0.5


@always_inline
def _pow_typed[
    dtype: DType
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    """Compute x^y for any float dtype using Scalar[dtype]."""
    return x**y


# ============================================================================
# batch_norm2d: Parametric implementation
# ============================================================================


def _batch_norm2d_compute_stats[
    dtype: DType
](
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    batch_mean: AnyTensor,
    batch_var: AnyTensor,
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    spatial_size: Int,
) raises:
    """Compute per-channel mean and variance for batch normalization."""
    var typed_x = x_ptr.bitcast[Scalar[dtype]]()
    var mean_ptr = batch_mean._data.bitcast[Scalar[dtype]]()
    var var_ptr = batch_var._data.bitcast[Scalar[dtype]]()
    var N = Scalar[dtype](spatial_size)

    for c in range(channels):
        var sum_val = Scalar[dtype](0.0)
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    sum_val += typed_x[idx]
        mean_ptr[c] = sum_val / N

    for c in range(channels):
        var mean_val = mean_ptr[c]
        var sum_sq_diff = Scalar[dtype](0.0)
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var diff = typed_x[idx] - mean_val
                    sum_sq_diff += diff * diff
        var_ptr[c] = sum_sq_diff / N


def _batch_norm2d_normalize[
    dtype: DType
](
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    output: AnyTensor,
    mean_ptr_raw: UnsafePointer[UInt8, MutAnyOrigin],
    var_ptr_raw: UnsafePointer[UInt8, MutAnyOrigin],
    gamma_ptr_raw: UnsafePointer[UInt8, MutAnyOrigin],
    beta_ptr_raw: UnsafePointer[UInt8, MutAnyOrigin],
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    epsilon: Float64,
) raises:
    """Normalize using given mean/variance with optional parallelization."""
    var typed_x = x_ptr.bitcast[Scalar[dtype]]()
    var out_ptr = output._data.bitcast[Scalar[dtype]]()
    var mean_ptr = mean_ptr_raw.bitcast[Scalar[dtype]]()
    var var_ptr = var_ptr_raw.bitcast[Scalar[dtype]]()
    var gamma_ptr = gamma_ptr_raw.bitcast[Scalar[dtype]]()
    var beta_ptr = beta_ptr_raw.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)

    if should_parallelize(batch):

        @parameter
        def normalize_batch(b: Int) capturing:
            for c in range(channels):
                var mean_val = mean_ptr[c]
                var std = _sqrt_typed[dtype](var_ptr[c] + eps)
                var gamma_val = gamma_ptr[c]
                var beta_val = beta_ptr[c]
                for h in range(height):
                    for w in range(width):
                        var idx = _idx4d(b, c, h, w, channels, height, width)
                        var x_norm = (typed_x[idx] - mean_val) / std
                        out_ptr[idx] = gamma_val * x_norm + beta_val

        parallelize[normalize_batch](batch)
    else:
        for b in range(batch):
            for c in range(channels):
                var mean_val = mean_ptr[c]
                var std = _sqrt_typed[dtype](var_ptr[c] + eps)
                var gamma_val = gamma_ptr[c]
                var beta_val = beta_ptr[c]
                for h in range(height):
                    for w in range(width):
                        var idx = _idx4d(b, c, h, w, channels, height, width)
                        var x_norm = (typed_x[idx] - mean_val) / std
                        out_ptr[idx] = gamma_val * x_norm + beta_val


def _batch_norm2d_update_running_stats[
    dtype: DType
](
    running_mean: AnyTensor,
    running_var: AnyTensor,
    batch_mean: AnyTensor,
    batch_var: AnyTensor,
    new_running_mean: AnyTensor,
    new_running_var: AnyTensor,
    channels: Int,
    momentum: Float64,
) raises:
    """Update running statistics with EMA."""
    var rm_ptr = running_mean._data.bitcast[Scalar[dtype]]()
    var rv_ptr = running_var._data.bitcast[Scalar[dtype]]()
    var bm_ptr = batch_mean._data.bitcast[Scalar[dtype]]()
    var bv_ptr = batch_var._data.bitcast[Scalar[dtype]]()
    var nrm_ptr = new_running_mean._data.bitcast[Scalar[dtype]]()
    var nrv_ptr = new_running_var._data.bitcast[Scalar[dtype]]()
    var mom = Scalar[dtype](momentum)
    var one_minus_mom = Scalar[dtype](1.0 - momentum)

    for c in range(channels):
        nrm_ptr[c] = one_minus_mom * rm_ptr[c] + mom * bm_ptr[c]
        nrv_ptr[c] = one_minus_mom * rv_ptr[c] + mom * bv_ptr[c]


def batch_norm2d(
    x: AnyTensor,
    gamma: AnyTensor,
    beta: AnyTensor,
    running_mean: AnyTensor,
    running_var: AnyTensor,
    training: Bool,
    momentum: Float64 = 0.1,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
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
            from projectodyssey.core import batch_norm2d, zeros, ones

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

    if (
        x.dtype() != gamma.dtype()
        or x.dtype() != beta.dtype()
        or x.dtype() != running_mean.dtype()
        or x.dtype() != running_var.dtype()
    ):
        raise Error(
            "batch_norm2d: all tensors must have the same dtype. Got x: "
            + String(x.dtype())
            + ", gamma: "
            + String(gamma.dtype())
            + ", beta: "
            + String(beta.dtype())
        )

    var batch = x_shape[0]
    var channels = x_shape[1]
    var height = x_shape[2]
    var width = x_shape[3]

    if training:
        var spatial_size = batch * height * width
        var batch_mean = zeros([channels], x.dtype())
        var batch_var = zeros([channels], x.dtype())

        @parameter
        def _run_compute_stats[T: DType]() raises:
            _batch_norm2d_compute_stats[T](
                x._data,
                batch_mean,
                batch_var,
                batch,
                channels,
                height,
                width,
                spatial_size,
            )

        dispatch_float3[_run_compute_stats](x.dtype())

        var output = zeros_like(x)

        @parameter
        def _run_normalize_train[T: DType]() raises:
            _batch_norm2d_normalize[T](
                x._data,
                output,
                batch_mean._data,
                batch_var._data,
                gamma._data,
                beta._data,
                batch,
                channels,
                height,
                width,
                epsilon,
            )

        dispatch_float3[_run_normalize_train](x.dtype())

        var new_running_mean = zeros_like(running_mean)
        var new_running_var = zeros_like(running_var)

        @parameter
        def _run_update_stats[T: DType]() raises:
            _batch_norm2d_update_running_stats[T](
                running_mean,
                running_var,
                batch_mean,
                batch_var,
                new_running_mean,
                new_running_var,
                channels,
                momentum,
            )

        dispatch_float3[_run_update_stats](x.dtype())

        return (output, new_running_mean, new_running_var)

    else:
        # Inference mode: use running statistics
        var output = zeros_like(x)

        @parameter
        def _run_normalize_infer[T: DType]() raises:
            _batch_norm2d_normalize[T](
                x._data,
                output,
                running_mean._data,
                running_var._data,
                gamma._data,
                beta._data,
                batch,
                channels,
                height,
                width,
                epsilon,
            )

        dispatch_float3[_run_normalize_infer](x.dtype())

        # Running stats unchanged in inference mode
        return (output, running_mean, running_var)


# ============================================================================
# batch_norm2d_backward: Parametric implementation
# ============================================================================


def _batch_norm2d_backward_training[
    dtype: DType
](
    grad_output_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    grad_input: AnyTensor,
    grad_gamma: AnyTensor,
    grad_beta: AnyTensor,
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    spatial_size: Int,
    epsilon: Float64,
) raises:
    """Training-mode backward pass for batch normalization."""
    var go = grad_output_ptr.bitcast[Scalar[dtype]]()
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var gi_ptr = grad_input._data.bitcast[Scalar[dtype]]()
    var gg_ptr = grad_gamma._data.bitcast[Scalar[dtype]]()
    var gb_ptr = grad_beta._data.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var N = Scalar[dtype](spatial_size)

    # Step 1: Compute batch mean and variance per channel
    for c in range(channels):
        var sum_val = Scalar[dtype](0.0)
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    sum_val += xp[_idx4d(b, c, h, w, channels, height, width)]
        var mean_val = sum_val / N

        var sum_sq_diff = Scalar[dtype](0.0)
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    var diff = (
                        xp[_idx4d(b, c, h, w, channels, height, width)]
                        - mean_val
                    )
                    sum_sq_diff += diff * diff
        var var_val = sum_sq_diff / N
        var std = _sqrt_typed[dtype](var_val + eps)

        # Step 2: Compute grad_beta and grad_gamma
        var sum_grad_output = Scalar[dtype](0.0)
        var sum_grad_output_x_norm = Scalar[dtype](0.0)
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var grad_out = go[idx]
                    var x_norm = (xp[idx] - mean_val) / std
                    sum_grad_output += grad_out
                    sum_grad_output_x_norm += grad_out * x_norm
        gb_ptr[c] = sum_grad_output
        gg_ptr[c] = sum_grad_output_x_norm

        # Step 3: Compute grad_input using PyTorch consolidated formula
        var invstd = Scalar[dtype](1.0) / std
        var gamma_val = gp[c]
        var k = Scalar[dtype](0.0)
        var dotp = Scalar[dtype](0.0)

        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var grad_out = go[idx]
                    var x_norm = (xp[idx] - mean_val) * invstd
                    k += grad_out
                    dotp += grad_out * x_norm

        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var grad_out = go[idx]
                    var x_norm = (xp[idx] - mean_val) * invstd
                    gi_ptr[idx] = (
                        (grad_out - k / N - x_norm * dotp / N)
                        * gamma_val
                        * invstd
                    )


def _batch_norm2d_backward_inference[
    dtype: DType
](
    grad_output_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    rm_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    rv_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    grad_input: AnyTensor,
    grad_gamma: AnyTensor,
    grad_beta: AnyTensor,
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    epsilon: Float64,
) raises:
    """Inference-mode backward pass for batch normalization."""
    var go = grad_output_ptr.bitcast[Scalar[dtype]]()
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var rm = rm_ptr.bitcast[Scalar[dtype]]()
    var rv = rv_ptr.bitcast[Scalar[dtype]]()
    var gi_ptr = grad_input._data.bitcast[Scalar[dtype]]()
    var gg_ptr = grad_gamma._data.bitcast[Scalar[dtype]]()
    var gb_ptr = grad_beta._data.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)

    # Compute grad_beta and grad_gamma
    for c in range(channels):
        var mean_val = rm[c]
        var std = _sqrt_typed[dtype](rv[c] + eps)

        var sum_grad_output = Scalar[dtype](0.0)
        var sum_grad_output_x_norm = Scalar[dtype](0.0)

        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var grad_out = go[idx]
                    var x_norm = (xp[idx] - mean_val) / std
                    sum_grad_output += grad_out
                    sum_grad_output_x_norm += grad_out * x_norm

        gb_ptr[c] = sum_grad_output
        gg_ptr[c] = sum_grad_output_x_norm

    # Compute grad_input (simple rescaling)
    for b in range(batch):
        for c in range(channels):
            var std = _sqrt_typed[dtype](rv[c] + eps)
            var gamma_val = gp[c]
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    gi_ptr[idx] = go[idx] * gamma_val / std


def batch_norm2d_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    gamma: AnyTensor,
    running_mean: AnyTensor,
    running_var: AnyTensor,
    training: Bool,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
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
            from projectodyssey.core import batch_norm2d_backward

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

    var grad_input = zeros_like(x)
    var grad_gamma = zeros([channels], x.dtype())
    var grad_beta = zeros([channels], x.dtype())

    if training:

        @parameter
        def _run_backward_train[T: DType]() raises:
            _batch_norm2d_backward_training[T](
                grad_output._data,
                x._data,
                gamma._data,
                grad_input,
                grad_gamma,
                grad_beta,
                batch,
                channels,
                height,
                width,
                spatial_size,
                epsilon,
            )

        dispatch_float3[_run_backward_train](x.dtype())
    else:

        @parameter
        def _run_backward_infer[T: DType]() raises:
            _batch_norm2d_backward_inference[T](
                grad_output._data,
                x._data,
                gamma._data,
                running_mean._data,
                running_var._data,
                grad_input,
                grad_gamma,
                grad_beta,
                batch,
                channels,
                height,
                width,
                epsilon,
            )

        dispatch_float3[_run_backward_infer](x.dtype())

    return (grad_input, grad_gamma, grad_beta)


# ============================================================================
# layer_norm: Parametric implementation
# ============================================================================


def _layer_norm_2d[
    dtype: DType
](
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    output: AnyTensor,
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    beta_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    batch: Int,
    features: Int,
    epsilon: Float64,
) raises:
    """Layer norm for 2D input (batch, features)."""
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var out = output._data.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var bp = beta_ptr.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var N = Scalar[dtype](features)

    for b in range(batch):
        var sum_val = Scalar[dtype](0.0)
        for f in range(features):
            sum_val += xp[b * features + f]
        var mean_val = sum_val / N

        var sum_sq_diff = Scalar[dtype](0.0)
        for f in range(features):
            var diff = xp[b * features + f] - mean_val
            sum_sq_diff += diff * diff
        var std = _sqrt_typed[dtype](sum_sq_diff / N + eps)

        for f in range(features):
            var idx = b * features + f
            var x_norm = (xp[idx] - mean_val) / std
            out[idx] = gp[f] * x_norm + bp[f]


def _layer_norm_4d[
    dtype: DType
](
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    output: AnyTensor,
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    beta_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    epsilon: Float64,
) raises:
    """Layer norm for 4D input (batch, channels, height, width)."""
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var out = output._data.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var bp = beta_ptr.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var feature_size = channels * height * width
    var N = Scalar[dtype](feature_size)

    for b in range(batch):
        var sum_val = Scalar[dtype](0.0)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    sum_val += xp[_idx4d(b, c, h, w, channels, height, width)]
        var mean_val = sum_val / N

        var sum_sq_diff = Scalar[dtype](0.0)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    var diff = (
                        xp[_idx4d(b, c, h, w, channels, height, width)]
                        - mean_val
                    )
                    sum_sq_diff += diff * diff
        var std = _sqrt_typed[dtype](sum_sq_diff / N + eps)

        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var gamma_idx = c * (height * width) + h * width + w
                    var x_norm = (xp[idx] - mean_val) / std
                    out[idx] = gp[gamma_idx] * x_norm + bp[gamma_idx]


def layer_norm(
    x: AnyTensor, gamma: AnyTensor, beta: AnyTensor, epsilon: Float64 = 1e-5
) raises -> AnyTensor:
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
            from projectodyssey.core import layer_norm, zeros, ones

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
        var batch = x_shape[0]
        var features = x_shape[1]
        var output = zeros_like(x)

        @parameter
        def _run_layer_norm_2d[T: DType]() raises:
            _layer_norm_2d[T](
                x._data,
                output,
                gamma._data,
                beta._data,
                batch,
                features,
                epsilon,
            )

        dispatch_float3[_run_layer_norm_2d](x.dtype())
        return output

    elif len(x_shape) == 4:
        var batch = x_shape[0]
        var channels = x_shape[1]
        var height = x_shape[2]
        var width = x_shape[3]
        var output = zeros_like(x)

        @parameter
        def _run_layer_norm_4d[T: DType]() raises:
            _layer_norm_4d[T](
                x._data,
                output,
                gamma._data,
                beta._data,
                batch,
                channels,
                height,
                width,
                epsilon,
            )

        dispatch_float3[_run_layer_norm_4d](x.dtype())
        return output

    else:
        raise Error("layer_norm supports 2D or 4D inputs only")


# ============================================================================
# layer_norm_backward: Parametric implementation
# ============================================================================


def _layer_norm_backward_2d[
    dtype: DType
](
    grad_output_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    grad_input: AnyTensor,
    grad_gamma: AnyTensor,
    grad_beta: AnyTensor,
    batch: Int,
    features: Int,
    epsilon: Float64,
) raises:
    """Layer norm backward for 2D input."""
    var go = grad_output_ptr.bitcast[Scalar[dtype]]()
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var gi_ptr = grad_input._data.bitcast[Scalar[dtype]]()
    var gg_ptr = grad_gamma._data.bitcast[Scalar[dtype]]()
    var gb_ptr = grad_beta._data.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var N = Scalar[dtype](features)

    # First pass: compute grad_gamma and grad_beta by accumulating over batch
    for b in range(batch):
        var sum_val = Scalar[dtype](0.0)
        for f in range(features):
            sum_val += xp[b * features + f]
        var mean_val = sum_val / N

        var sum_sq_diff = Scalar[dtype](0.0)
        for f in range(features):
            var diff = xp[b * features + f] - mean_val
            sum_sq_diff += diff * diff
        var std = _sqrt_typed[dtype](sum_sq_diff / N + eps)
        var var_val = sum_sq_diff / N

        for f in range(features):
            var idx = b * features + f
            var grad_out = go[idx]
            var x_norm = (xp[idx] - mean_val) / std
            gb_ptr[f] = gb_ptr[f] + grad_out
            gg_ptr[f] = gg_ptr[f] + grad_out * x_norm

    # Second pass: compute grad_input for each sample
    for b in range(batch):
        var sum_val = Scalar[dtype](0.0)
        for f in range(features):
            sum_val += xp[b * features + f]
        var mean_val = sum_val / N

        var sum_sq_diff = Scalar[dtype](0.0)
        for f in range(features):
            var diff = xp[b * features + f] - mean_val
            sum_sq_diff += diff * diff
        var var_val = sum_sq_diff / N
        var std = _sqrt_typed[dtype](var_val + eps)

        var grad_var = Scalar[dtype](0.0)
        var grad_mean = Scalar[dtype](0.0)

        for f in range(features):
            var idx = b * features + f
            var x_minus_mean = xp[idx] - mean_val
            var grad_x_norm = go[idx] * gp[f]
            grad_var += (
                grad_x_norm
                * x_minus_mean
                * Scalar[dtype](-0.5)
                * _pow_typed[dtype](var_val + eps, Scalar[dtype](-1.5))
            )
            grad_mean += grad_x_norm * Scalar[dtype](-1.0) / std

        for f in range(features):
            var idx = b * features + f
            var x_minus_mean = xp[idx] - mean_val
            var grad_x_norm = go[idx] * gp[f]
            var term1 = grad_x_norm / std
            var term2 = grad_var * Scalar[dtype](2.0) * x_minus_mean / N
            var term3 = grad_mean / N
            gi_ptr[idx] = term1 + term2 + term3


def _layer_norm_backward_4d[
    dtype: DType
](
    grad_output_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    grad_input: AnyTensor,
    grad_gamma: AnyTensor,
    grad_beta: AnyTensor,
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    epsilon: Float64,
) raises:
    """Layer norm backward for 4D input."""
    var go = grad_output_ptr.bitcast[Scalar[dtype]]()
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var gi_ptr = grad_input._data.bitcast[Scalar[dtype]]()
    var gg_ptr = grad_gamma._data.bitcast[Scalar[dtype]]()
    var gb_ptr = grad_beta._data.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var feature_size = channels * height * width
    var N = Scalar[dtype](feature_size)

    # First pass: compute grad_gamma and grad_beta by accumulating over batch
    for b in range(batch):
        var sum_val = Scalar[dtype](0.0)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    sum_val += xp[_idx4d(b, c, h, w, channels, height, width)]
        var mean_val = sum_val / N

        var sum_sq_diff = Scalar[dtype](0.0)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    var diff = (
                        xp[_idx4d(b, c, h, w, channels, height, width)]
                        - mean_val
                    )
                    sum_sq_diff += diff * diff
        var std = _sqrt_typed[dtype](sum_sq_diff / N + eps)

        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var gamma_idx = c * (height * width) + h * width + w
                    var grad_out = go[idx]
                    var x_norm = (xp[idx] - mean_val) / std
                    gb_ptr[gamma_idx] = gb_ptr[gamma_idx] + grad_out
                    gg_ptr[gamma_idx] = gg_ptr[gamma_idx] + grad_out * x_norm

    # Second pass: compute grad_input for each sample
    for b in range(batch):
        var sum_val = Scalar[dtype](0.0)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    sum_val += xp[_idx4d(b, c, h, w, channels, height, width)]
        var mean_val = sum_val / N

        var sum_sq_diff = Scalar[dtype](0.0)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    var diff = (
                        xp[_idx4d(b, c, h, w, channels, height, width)]
                        - mean_val
                    )
                    sum_sq_diff += diff * diff
        var var_val = sum_sq_diff / N
        var std = _sqrt_typed[dtype](var_val + eps)

        var grad_var = Scalar[dtype](0.0)
        var grad_mean = Scalar[dtype](0.0)

        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var gamma_idx = c * (height * width) + h * width + w
                    var x_minus_mean = xp[idx] - mean_val
                    var grad_x_norm = go[idx] * gp[gamma_idx]
                    grad_var += (
                        grad_x_norm
                        * x_minus_mean
                        * Scalar[dtype](-0.5)
                        * _pow_typed[dtype](var_val + eps, Scalar[dtype](-1.5))
                    )
                    grad_mean += grad_x_norm * Scalar[dtype](-1.0) / std

        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var gamma_idx = c * (height * width) + h * width + w
                    var x_minus_mean = xp[idx] - mean_val
                    var grad_x_norm = go[idx] * gp[gamma_idx]
                    var term1 = grad_x_norm / std
                    var term2 = grad_var * Scalar[dtype](2.0) * x_minus_mean / N
                    var term3 = grad_mean / N
                    gi_ptr[idx] = term1 + term2 + term3


def layer_norm_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    gamma: AnyTensor,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
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
            from projectodyssey.core import layer_norm, layer_norm_backward

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
        var batch = x_shape[0]
        var features = x_shape[1]

        var grad_input = zeros_like(x)
        var grad_gamma = zeros_like(gamma)
        var grad_beta = zeros_like(gamma)

        @parameter
        def _run_layer_norm_backward_2d[T: DType]() raises:
            _layer_norm_backward_2d[T](
                grad_output._data,
                x._data,
                gamma._data,
                grad_input,
                grad_gamma,
                grad_beta,
                batch,
                features,
                epsilon,
            )

        dispatch_float3[_run_layer_norm_backward_2d](x.dtype())

        return (grad_input, grad_gamma, grad_beta)

    elif len(x_shape) == 4:
        var batch = x_shape[0]
        var channels = x_shape[1]
        var height = x_shape[2]
        var width = x_shape[3]

        var grad_input = zeros_like(x)
        var grad_gamma = zeros_like(gamma)
        var grad_beta = zeros_like(gamma)

        @parameter
        def _run_layer_norm_backward_4d[T: DType]() raises:
            _layer_norm_backward_4d[T](
                grad_output._data,
                x._data,
                gamma._data,
                grad_input,
                grad_gamma,
                grad_beta,
                batch,
                channels,
                height,
                width,
                epsilon,
            )

        dispatch_float3[_run_layer_norm_backward_4d](x.dtype())

        return (grad_input, grad_gamma, grad_beta)

    else:
        raise Error("layer_norm_backward supports 2D or 4D inputs only")


# ============================================================================
# group_norm: Parametric implementation
# ============================================================================


def _group_norm_impl[
    dtype: DType
](
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    output: AnyTensor,
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    beta_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    num_groups: Int,
    channels_per_group: Int,
    group_size: Int,
    epsilon: Float64,
) raises:
    """Parametric group norm forward pass."""
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var out = output._data.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var bp = beta_ptr.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var N = Scalar[dtype](group_size)

    for b in range(batch):
        for g in range(num_groups):
            var c_start = g * channels_per_group
            var c_end = c_start + channels_per_group

            var sum_val = Scalar[dtype](0.0)
            for c in range(c_start, c_end):
                for h in range(height):
                    for w in range(width):
                        sum_val += xp[
                            _idx4d(b, c, h, w, channels, height, width)
                        ]
            var mean_val = sum_val / N

            var sum_sq_diff = Scalar[dtype](0.0)
            for c in range(c_start, c_end):
                for h in range(height):
                    for w in range(width):
                        var diff = (
                            xp[_idx4d(b, c, h, w, channels, height, width)]
                            - mean_val
                        )
                        sum_sq_diff += diff * diff
            var std = _sqrt_typed[dtype](sum_sq_diff / N + eps)

            for c in range(c_start, c_end):
                var gamma_val = gp[c]
                var beta_val = bp[c]
                for h in range(height):
                    for w in range(width):
                        var idx = _idx4d(b, c, h, w, channels, height, width)
                        var x_norm = (xp[idx] - mean_val) / std
                        out[idx] = gamma_val * x_norm + beta_val


def group_norm(
    x: AnyTensor,
    num_groups: Int,
    gamma: AnyTensor,
    beta: AnyTensor,
    epsilon: Float64 = 1e-5,
) raises -> AnyTensor:
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
            from projectodyssey.core import group_norm, zeros, ones

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

    @parameter
    def _run_group_norm[T: DType]() raises:
        _group_norm_impl[T](
            x._data,
            output,
            gamma._data,
            beta._data,
            batch,
            channels,
            height,
            width,
            num_groups,
            channels_per_group,
            group_size,
            epsilon,
        )

    dispatch_float3[_run_group_norm](x.dtype())

    return output


# ============================================================================
# group_norm_backward: Parametric implementation
# ============================================================================


def _group_norm_backward_impl[
    dtype: DType
](
    grad_output_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    grad_input: AnyTensor,
    grad_gamma: AnyTensor,
    grad_beta: AnyTensor,
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    num_groups: Int,
    channels_per_group: Int,
    group_size: Int,
    epsilon: Float64,
) raises:
    """Parametric group norm backward pass."""
    var go = grad_output_ptr.bitcast[Scalar[dtype]]()
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var gi_ptr = grad_input._data.bitcast[Scalar[dtype]]()
    var gg_ptr = grad_gamma._data.bitcast[Scalar[dtype]]()
    var gb_ptr = grad_beta._data.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var N = Scalar[dtype](group_size)

    # First pass: compute grad_gamma and grad_beta
    for b in range(batch):
        for g in range(num_groups):
            var c_start = g * channels_per_group
            var c_end = c_start + channels_per_group

            var sum_val = Scalar[dtype](0.0)
            for c in range(c_start, c_end):
                for h in range(height):
                    for w in range(width):
                        sum_val += xp[
                            _idx4d(b, c, h, w, channels, height, width)
                        ]
            var mean_val = sum_val / N

            var sum_sq_diff = Scalar[dtype](0.0)
            for c in range(c_start, c_end):
                for h in range(height):
                    for w in range(width):
                        var diff = (
                            xp[_idx4d(b, c, h, w, channels, height, width)]
                            - mean_val
                        )
                        sum_sq_diff += diff * diff
            var std = _sqrt_typed[dtype](sum_sq_diff / N + eps)

            for c in range(c_start, c_end):
                for h in range(height):
                    for w in range(width):
                        var idx = _idx4d(b, c, h, w, channels, height, width)
                        var grad_out = go[idx]
                        var x_norm = (xp[idx] - mean_val) / std
                        gb_ptr[c] = gb_ptr[c] + grad_out
                        gg_ptr[c] = gg_ptr[c] + grad_out * x_norm

    # Second pass: compute grad_input
    for b in range(batch):
        for g in range(num_groups):
            var c_start = g * channels_per_group
            var c_end = c_start + channels_per_group

            var sum_val = Scalar[dtype](0.0)
            for c in range(c_start, c_end):
                for h in range(height):
                    for w in range(width):
                        sum_val += xp[
                            _idx4d(b, c, h, w, channels, height, width)
                        ]
            var mean_val = sum_val / N

            var sum_sq_diff = Scalar[dtype](0.0)
            for c in range(c_start, c_end):
                for h in range(height):
                    for w in range(width):
                        var diff = (
                            xp[_idx4d(b, c, h, w, channels, height, width)]
                            - mean_val
                        )
                        sum_sq_diff += diff * diff
            var var_val = sum_sq_diff / N
            var std = _sqrt_typed[dtype](var_val + eps)

            var grad_var = Scalar[dtype](0.0)
            var grad_mean = Scalar[dtype](0.0)

            for c in range(c_start, c_end):
                var gamma_val = gp[c]
                for h in range(height):
                    for w in range(width):
                        var idx = _idx4d(b, c, h, w, channels, height, width)
                        var x_minus_mean = xp[idx] - mean_val
                        var grad_x_norm = go[idx] * gamma_val
                        grad_var += (
                            grad_x_norm
                            * x_minus_mean
                            * Scalar[dtype](-0.5)
                            * _pow_typed[dtype](
                                var_val + eps, Scalar[dtype](-1.5)
                            )
                        )
                        grad_mean += grad_x_norm * Scalar[dtype](-1.0) / std

            for c in range(c_start, c_end):
                var gamma_val = gp[c]
                for h in range(height):
                    for w in range(width):
                        var idx = _idx4d(b, c, h, w, channels, height, width)
                        var x_minus_mean = xp[idx] - mean_val
                        var grad_x_norm = go[idx] * gamma_val
                        var term1 = grad_x_norm / std
                        var term2 = (
                            grad_var * Scalar[dtype](2.0) * x_minus_mean / N
                        )
                        var term3 = grad_mean / N
                        gi_ptr[idx] = term1 + term2 + term3


def group_norm_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    num_groups: Int,
    gamma: AnyTensor,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
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
            from projectodyssey.core import group_norm, group_norm_backward

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

    var grad_input = zeros_like(x)
    var grad_gamma = zeros([channels], x.dtype())
    var grad_beta = zeros([channels], x.dtype())

    @parameter
    def _run_group_norm_backward[T: DType]() raises:
        _group_norm_backward_impl[T](
            grad_output._data,
            x._data,
            gamma._data,
            grad_input,
            grad_gamma,
            grad_beta,
            batch,
            channels,
            height,
            width,
            num_groups,
            channels_per_group,
            group_size,
            epsilon,
        )

    dispatch_float3[_run_group_norm_backward](x.dtype())

    return (grad_input, grad_gamma, grad_beta)


# ============================================================================
# instance_norm: Parametric implementation
# ============================================================================


def _instance_norm_impl[
    dtype: DType
](
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    output: AnyTensor,
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    beta_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    epsilon: Float64,
) raises:
    """Parametric instance norm forward pass."""
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var out = output._data.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var bp = beta_ptr.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var N = Scalar[dtype](height * width)

    for b in range(batch):
        for c in range(channels):
            var sum_val = Scalar[dtype](0.0)
            for h in range(height):
                for w in range(width):
                    sum_val += xp[_idx4d(b, c, h, w, channels, height, width)]
            var mean_val = sum_val / N

            var sum_sq_diff = Scalar[dtype](0.0)
            for h in range(height):
                for w in range(width):
                    var diff = (
                        xp[_idx4d(b, c, h, w, channels, height, width)]
                        - mean_val
                    )
                    sum_sq_diff += diff * diff
            var std = _sqrt_typed[dtype](sum_sq_diff / N + eps)

            var gamma_val = gp[c]
            var beta_val = bp[c]
            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var x_norm = (xp[idx] - mean_val) / std
                    out[idx] = gamma_val * x_norm + beta_val


def instance_norm(
    x: AnyTensor,
    gamma: AnyTensor,
    beta: AnyTensor,
    epsilon: Float64 = 1e-5,
) raises -> AnyTensor:
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
            from projectodyssey.core import instance_norm, zeros, ones

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
    var output = zeros_like(x)

    @parameter
    def _run_instance_norm[T: DType]() raises:
        _instance_norm_impl[T](
            x._data,
            output,
            gamma._data,
            beta._data,
            batch,
            channels,
            height,
            width,
            epsilon,
        )

    dispatch_float3[_run_instance_norm](x.dtype())

    return output


# ============================================================================
# instance_norm_backward: Parametric implementation
# ============================================================================


def _instance_norm_backward_impl[
    dtype: DType
](
    grad_output_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    x_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    gamma_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    grad_input: AnyTensor,
    grad_gamma: AnyTensor,
    grad_beta: AnyTensor,
    batch: Int,
    channels: Int,
    height: Int,
    width: Int,
    epsilon: Float64,
) raises:
    """Parametric instance norm backward pass."""
    var go = grad_output_ptr.bitcast[Scalar[dtype]]()
    var xp = x_ptr.bitcast[Scalar[dtype]]()
    var gp = gamma_ptr.bitcast[Scalar[dtype]]()
    var gi_ptr = grad_input._data.bitcast[Scalar[dtype]]()
    var gg_ptr = grad_gamma._data.bitcast[Scalar[dtype]]()
    var gb_ptr = grad_beta._data.bitcast[Scalar[dtype]]()
    var eps = Scalar[dtype](epsilon)
    var spatial_size = height * width
    var N = Scalar[dtype](spatial_size)

    # First pass: compute grad_gamma and grad_beta
    for b in range(batch):
        for c in range(channels):
            var sum_val = Scalar[dtype](0.0)
            for h in range(height):
                for w in range(width):
                    sum_val += xp[_idx4d(b, c, h, w, channels, height, width)]
            var mean_val = sum_val / N

            var sum_sq_diff = Scalar[dtype](0.0)
            for h in range(height):
                for w in range(width):
                    var diff = (
                        xp[_idx4d(b, c, h, w, channels, height, width)]
                        - mean_val
                    )
                    sum_sq_diff += diff * diff
            var std = _sqrt_typed[dtype](sum_sq_diff / N + eps)

            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var grad_out = go[idx]
                    var x_norm = (xp[idx] - mean_val) / std
                    gb_ptr[c] = gb_ptr[c] + grad_out
                    gg_ptr[c] = gg_ptr[c] + grad_out * x_norm

    # Second pass: compute grad_input
    for b in range(batch):
        for c in range(channels):
            var sum_val = Scalar[dtype](0.0)
            for h in range(height):
                for w in range(width):
                    sum_val += xp[_idx4d(b, c, h, w, channels, height, width)]
            var mean_val = sum_val / N

            var sum_sq_diff = Scalar[dtype](0.0)
            for h in range(height):
                for w in range(width):
                    var diff = (
                        xp[_idx4d(b, c, h, w, channels, height, width)]
                        - mean_val
                    )
                    sum_sq_diff += diff * diff
            var var_val = sum_sq_diff / N
            var std = _sqrt_typed[dtype](var_val + eps)

            var gamma_val = gp[c]
            var grad_var = Scalar[dtype](0.0)
            var grad_mean = Scalar[dtype](0.0)

            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var x_minus_mean = xp[idx] - mean_val
                    var grad_x_norm = go[idx] * gamma_val
                    grad_var += (
                        grad_x_norm
                        * x_minus_mean
                        * Scalar[dtype](-0.5)
                        * _pow_typed[dtype](var_val + eps, Scalar[dtype](-1.5))
                    )
                    grad_mean += grad_x_norm * Scalar[dtype](-1.0) / std

            for h in range(height):
                for w in range(width):
                    var idx = _idx4d(b, c, h, w, channels, height, width)
                    var x_minus_mean = xp[idx] - mean_val
                    var grad_x_norm = go[idx] * gamma_val
                    var term1 = grad_x_norm / std
                    var term2 = grad_var * Scalar[dtype](2.0) * x_minus_mean / N
                    var term3 = grad_mean / N
                    gi_ptr[idx] = term1 + term2 + term3


def instance_norm_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    gamma: AnyTensor,
    epsilon: Float64 = 1e-5,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
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
            from projectodyssey.core import instance_norm, instance_norm_backward

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

    var grad_input = zeros_like(x)
    var grad_gamma = zeros([channels], x.dtype())
    var grad_beta = zeros([channels], x.dtype())

    @parameter
    def _run_instance_norm_backward[T: DType]() raises:
        _instance_norm_backward_impl[T](
            grad_output._data,
            x._data,
            gamma._data,
            grad_input,
            grad_gamma,
            grad_beta,
            batch,
            channels,
            height,
            width,
            epsilon,
        )

    dispatch_float3[_run_instance_norm_backward](x.dtype())

    return (grad_input, grad_gamma, grad_beta)
