"""RMSprop optimizer (Root Mean Square Propagation).

This module provides the RMSprop optimizer for updating model parameters
during training using adaptive learning rates.

RMSprop is an adaptive learning rate method that uses a moving average
of squared gradients to normalize the gradient. It's particularly effective
for non-stationary objectives and recurrent neural networks.

Standard RMSprop update rule:
    square_avg = alpha * square_avg + (1 - alpha) * gradients^2
    params = params - learning_rate * gradients / (sqrt(square_avg) + epsilon)

With momentum (optional):
    square_avg = alpha * square_avg + (1 - alpha) * gradients^2
    buf = momentum * buf + gradients / (sqrt(square_avg) + epsilon)
    params = params - learning_rate * buf

Reference:
    Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient
    by a running average of its recent magnitude. COURSERA: Neural networks for
    machine learning, 4(2), 26-31
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full_like, zeros_like
from odyssey.core.arithmetic import (
    subtract,
    multiply,
    add,
    divide,
    power,
)
from odyssey.core.elementwise import sqrt


def rmsprop_step(
    params: AnyTensor,
    gradients: AnyTensor,
    square_avg: AnyTensor,
    t: Int,
    learning_rate: Float64,
    alpha: Float64 = 0.99,
    epsilon: Float64 = 1e-8,
    weight_decay: Float64 = 0.0,
    momentum: Float64 = 0.0,
    buf: Optional[AnyTensor] = None,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single RMSprop optimization step - pure functional.

        Returns new parameters, new square average, and new momentum buffer
        Caller manages all state including timestep tracking.

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            square_avg: Running average of squared gradients.
            t: Current timestep (starts at 1, increments each step).
            learning_rate: Step size for parameter updates.
            alpha: Smoothing constant for running average (default: 0.99).
            epsilon: Small constant for numerical stability (default: 1e-8).
            weight_decay: L2 regularization factor (default: 0.0, no regularization).
            momentum: Momentum factor (default: 0.0, no momentum).
            buf: Momentum buffer (only used if momentum > 0).

    Returns:
            Tuple of (new_params, new_square_avg, new_buf).

    Example (basic RMSprop):
        ```mojo
        from odyssey.core import AnyTensor, zeros_like
        from odyssey.training.optimizers import rmsprop_step

        var W = xavier_uniform(784, 128, DType.float32)
        var square_avg = zeros_like(W)
        var buf = zeros([0], DType.float32)  # Empty tensor (no momentum)
        var t = 1

        # Training loop
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            (W, square_avg, buf) = rmsprop_step(W, grad_W, square_avg, t, lr=0.01)
            t += 1
        ```

    Example (RMSprop with momentum):
        ```mojo
        var W = xavier_uniform(784, 128, DType.float32)
        var square_avg = zeros_like(W)
        var buf = zeros_like(W)  # Momentum buffer
        var t = 1

        # Training loop with momentum
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            (W, square_avg, buf) = rmsprop_step(
                W, grad_W, square_avg, t,
                lr=0.01, momentum=0.9, buf=buf
            )
            t += 1
        ```

    Note:
            This is a pure function - it returns new state rather than mutating.
            Caller must capture all three return values and update their variables.
            Timestep t must be tracked by caller and incremented after each step.

    Raises:
            Error: If operation fails.
    """
    if params.shape() != gradients.shape():
        raise Error("Parameters and gradients must have the same shape")

    if params.dtype() != gradients.dtype():
        raise Error("Parameters and gradients must have the same dtype")

    if square_avg.numel() == 0:
        raise Error("square_avg must be initialized (use zeros_like(params))")

    if t <= 0:
        raise Error("Timestep t must be positive (starts at 1)")

    # Initialize buf if not provided
    var initialized_buf: AnyTensor
    if buf:
        initialized_buf = buf.value()
    else:
        initialized_buf = zeros([0], DType.float32)

    var effective_gradients = gradients

    # Apply weight decay (L2 regularization) if specified
    if weight_decay > 0.0:
        # grad = grad + weight_decay * params
        var wd_tensor = full_like(params, weight_decay)
        var weight_penalty = multiply(wd_tensor, params)
        effective_gradients = add(effective_gradients, weight_penalty)

    # Update running average of squared gradients
    # square_avg = alpha * square_avg + (1 - alpha) * grad^2
    var alpha_tensor = full_like(square_avg, alpha)
    var one_minus_alpha = full_like(square_avg, 1.0 - alpha)

    var exponent_tensor = full_like(effective_gradients, 2.0)
    var grad_squared = power(effective_gradients, exponent_tensor)
    var avg_term1 = multiply(alpha_tensor, square_avg)
    var avg_term2 = multiply(one_minus_alpha, grad_squared)
    var new_square_avg = add(avg_term1, avg_term2)

    # Compute normalized gradient: grad / (sqrt(square_avg) + epsilon)
    var eps_tensor = full_like(new_square_avg, epsilon)
    var denom = add(sqrt(new_square_avg), eps_tensor)
    var normalized_grad = divide(effective_gradients, denom)

    # Apply momentum if specified
    var new_buf = initialized_buf
    var update = normalized_grad

    if momentum > 0.0:
        if initialized_buf.numel() == 0:
            # Initialize buffer if not provided
            new_buf = zeros_like(params)

        # buf = momentum * buf + normalized_grad
        var momentum_tensor = full_like(params, momentum)
        var buf_term = multiply(momentum_tensor, initialized_buf)
        new_buf = add(buf_term, normalized_grad)
        update = new_buf

    # Update parameters: params = params - lr * update
    var lr_tensor = full_like(params, learning_rate)
    var param_update = multiply(lr_tensor, update)
    var new_params = subtract(params, param_update)

    return (new_params, new_square_avg, new_buf)


def rmsprop_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    square_avg: AnyTensor,
    learning_rate: Float64,
    alpha: Float64 = 0.99,
    epsilon: Float64 = 1e-8,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified RMSprop step without weight decay, momentum, or timestep.

        This is a convenience function for basic RMSprop updates.

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            square_avg: Running average of squared gradients.
            learning_rate: Step size for parameter updates.
            alpha: Smoothing constant for running average (default: 0.99).
            epsilon: Small constant for numerical stability (default: 1e-8).

    Returns:
            Tuple of (new_params, new_square_avg).

    Example:
        ```mojo
        from odyssey.core import AnyTensor, zeros_like
        from odyssey.training.optimizers import rmsprop_step_simple

        var W = xavier_uniform(784, 128, DType.float32)
        var square_avg = zeros_like(W)

        # Training loop
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            (W, square_avg) = rmsprop_step_simple(W, grad_W, square_avg, lr=0.01)
        ```

    Raises:
            Error: If operation fails.
    """
    var (new_params, new_square_avg, _) = rmsprop_step(
        params,
        gradients,
        square_avg,
        1,  # t=1 (not used without momentum/wd)
        learning_rate,
        alpha,
        epsilon,
        weight_decay=0.0,
        momentum=0.0,
        buf=None,
    )

    return (new_params, new_square_avg)


def init_rmsprop_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the rmsprop optimizer.

    Returns a `List[List[AnyTensor]]` with outer length == `len(params_list)` (one entry per parameter) and inner length == 2 (one entry per state buffer the optimizer threads across calls).

    Each state buffer starts at zero with the parameter's shape. The dtype matches the parameter, or float64 if `force_f64=True`.

    Args:
        params_list: Model parameters.
        force_f64: Up-cast all state buffers to float64 regardless of param dtype.

    Returns:
        A list of state buffer lists in the same order as `params_list`.

    Raises:
        Error: If shape contracts cannot be honored.
    """
    from odyssey.tensor.tensor_creation import zeros

    var all_states: List[List[AnyTensor]] = []
    for i in range(len(params_list)):
        var p = params_list[i]
        var d = p.dtype() if not force_f64 else DType.float64
        var per: List[AnyTensor] = []
        for _ in range(2):
            per.append(zeros(p.shape(), d))
        all_states.append(per^)
    return all_states^
