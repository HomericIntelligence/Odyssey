"""Adam optimizer (Adaptive Moment Estimation).

This module provides the Adam optimizer for updating model parameters
during training using adaptive learning rates.

Adam combines the benefits of:
- Momentum: Using exponentially decaying average of past gradients.
- RMSprop: Using exponentially decaying average of past squared gradients.

Standard Adam update rule:
    ```
    m = beta1 * m + (1 - beta1) * gradients
    v = beta2 * v + (1 - beta2) * gradients^2
    m_hat = m / (1 - beta1^t)  # Bias correction
    v_hat = v / (1 - beta2^t)  # Bias correction
    params = params - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    ```

Reference:
    Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization
    arXiv preprint arXiv:1412.6980
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.core.arithmetic import (
    subtract,
    multiply,
    add,
    divide,
    power,
)
from projectodyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from projectodyssey.core.elementwise import sqrt
from projectodyssey.tensor.tensor_creation import full_like, ones_like


def adam_step(
    params: AnyTensor,
    gradients: AnyTensor,
    m: AnyTensor,
    v: AnyTensor,
    t: Int,
    learning_rate: Float64,
    beta1: Float64 = 0.9,
    beta2: Float64 = 0.999,
    epsilon: Float64 = 1e-8,
    weight_decay: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single Adam optimization step - pure functional.

        Returns new parameters, new first moment (m), and new second moment (v).
        Caller manages all state including timestep tracking.

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            m: First moment estimates (exponential moving average of gradients).
            v: Second moment estimates (exponential moving average of squared gradients).
            t: Current timestep (starts at 1, increments each step).
            learning_rate: Step size for parameter updates.
            beta1: Exponential decay rate for first moment (default: 0.9).
            beta2: Exponential decay rate for second moment (default: 0.999).
            epsilon: Small constant for numerical stability (default: 1e-8).
            weight_decay: L2 regularization factor (default: 0.0, no regularization).

    Returns:
            Tuple of (new_params, new_m, new_v).

    Example (basic Adam):
        ```mojo
        from projectodyssey.core import AnyTensor, zeros_like
        from projectodyssey.training.optimizers import adam_step

        var W = xavier_uniform(784, 128, DType.float32)
        var m = zeros_like(W)
        var v = zeros_like(W)
        var t = 1

        # Training loop
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            (W, m, v) = adam_step(W, grad_W, m, v, t, lr=0.001)
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

    if m.numel() == 0 or v.numel() == 0:
        raise Error(
            "Moment buffers (m and v) must be initialized (use"
            " zeros_like(params))"
        )

    if t <= 0:
        raise Error("Timestep t must be positive (starts at 1)")

    var effective_gradients = gradients

    # Apply weight decay (L2 regularization) if specified
    if weight_decay > 0.0:
        # grad = grad + weight_decay * params (SIMD optimized)
        var wd_tensor = full_like(params, weight_decay)
        var decay_term = multiply_simd(wd_tensor, params)
        effective_gradients = add_simd(gradients, decay_term)

    # Update biased first moment estimate (SIMD optimized)
    # m = beta1 * m + (1 - beta1) * grad
    var beta1_tensor = full_like(m, beta1)
    var one_minus_beta1 = full_like(m, 1.0 - beta1)

    var m_decay = multiply_simd(beta1_tensor, m)
    var grad_term = multiply_simd(one_minus_beta1, effective_gradients)
    var new_m = add_simd(m_decay, grad_term)

    # Update biased second moment estimate (SIMD optimized)
    # v = beta2 * v + (1 - beta2) * grad^2
    var beta2_tensor = full_like(v, beta2)
    var one_minus_beta2 = full_like(v, 1.0 - beta2)

    var v_decay = multiply_simd(beta2_tensor, v)
    var grad_squared = multiply_simd(effective_gradients, effective_gradients)
    var grad_squared_term = multiply_simd(one_minus_beta2, grad_squared)
    var new_v = add_simd(v_decay, grad_squared_term)

    # Compute bias-corrected first moment (SIMD optimized)
    # m_hat = m / (1 - beta1^t)
    var bias_correction1 = 1.0 - pow(beta1, Float64(t))
    var bc1_tensor = full_like(new_m, bias_correction1)
    var m_hat = divide_simd(new_m, bc1_tensor)

    # Compute bias-corrected second moment (SIMD optimized)
    # v_hat = v / (1 - beta2^t)
    var bias_correction2 = 1.0 - pow(beta2, Float64(t))
    var bc2_tensor = full_like(new_v, bias_correction2)
    var v_hat = divide_simd(new_v, bc2_tensor)

    # Compute parameter update (SIMD optimized)
    # params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
    var v_hat_sqrt = sqrt(v_hat)
    var epsilon_tensor = full_like(v_hat_sqrt, epsilon)
    var denominator = add_simd(v_hat_sqrt, epsilon_tensor)
    var update_direction = divide_simd(m_hat, denominator)

    var lr_tensor = full_like(params, learning_rate)
    var update = multiply_simd(lr_tensor, update_direction)
    var new_params = subtract_simd(params, update)

    # Return new state (pure functional)
    return (new_params, new_m, new_v)


def adam_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    m: AnyTensor,
    v: AnyTensor,
    t: Int,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Simplified Adam step with default hyperparameters.

        This is a convenience function for basic Adam optimization.

    Formula:
    ```
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad^2
        m_hat = m / (1 - 0.9^t)
        v_hat = v / (1 - 0.999^t)
        params = params - lr * m_hat / (sqrt(v_hat) + 1e-8)
    ```

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            m: First moment estimate.
            v: Second moment estimate.
            t: Current timestep (starts at 1).
            learning_rate: Step size for parameter updates.

    Returns:
            Tuple of (new_params, new_m, new_v).

        Example:
            ```mojo
            var W = xavier_uniform(784, 128, shape, DType.float32)
            var m = zeros_like(W)
            var v = zeros_like(W)
            var t = 1

            for epoch in range(100):
                var grad_W = ... # Computed gradients
                (W, m, v) = adam_step_simple(W, grad_W, m, v, t, 0.001)
                t += 1
            ```

    Raises:
        Error: If operation fails.
    """
    return adam_step(
        params,
        gradients,
        m,
        v,
        t,
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
    )
