"""Layer-wise Adaptive Rate Scaling (LARS) optimizer.

This module provides the LARS optimizer for large-batch training on distributed systems

LARS adapts the learning rate based on the ratio of parameter norm to gradient norm,
enabling stable training with very large batch sizes.

Standard LARS update rule:
    param_norm = ||params||
    grad_norm = ||gradients||
    trust_ratio = trust_coefficient * param_norm / (grad_norm + weight_decay * param_norm + epsilon)
    velocity = momentum * velocity + trust_ratio * (gradients + weight_decay * params)
    params = params - learning_rate * velocity

Reference:
    You, Y., Li, J., Reddi, S., et al. (2019). LARS: Layer-wise Adaptive Rate Scaling
    for Massive Batch Training. arXiv preprint arXiv:1708.03888
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.core.arithmetic import subtract, multiply, add
from projectodyssey.core.arithmetic_simd import subtract_simd, multiply_simd, add_simd
from projectodyssey.tensor.any_tensor import full_like
from projectodyssey.core.numerical_safety import compute_tensor_l2_norm


def lars_step(
    params: AnyTensor,
    gradients: AnyTensor,
    velocity: AnyTensor,
    learning_rate: Float64,
    momentum: Float64 = 0.9,
    weight_decay: Float64 = 0.0001,
    trust_coefficient: Float64 = 0.001,
    epsilon: Float64 = 1e-8,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single LARS optimization step - pure functional.

        Returns new parameters and new velocity. Caller manages all state.

        LARS (Layer-wise Adaptive Rate Scaling) adapts the learning rate based on
        the ratio of parameter norm to gradient norm, enabling stable large-batch
        training on distributed systems.

    Args:
        params: Model parameters to update.
        gradients: Gradients of loss with respect to params.
        velocity: Momentum buffer (use zeros_like(params) if no momentum).
        learning_rate: Base learning rate (will be scaled by trust_ratio).
        momentum: Momentum factor (default: 0.9).
        weight_decay: L2 regularization factor (default: 0.0001).
        trust_coefficient: Trust coefficient for adaptive scaling (default: 0.001).
        epsilon: Small constant for numerical stability (default: 1e-8).

    Returns:
            Tuple of (new_params, new_velocity).

    Example (LARS with momentum):
        ```mojo
        from projectodyssey.core import AnyTensor, zeros_like
        from projectodyssey.training.optimizers import lars_step

        var W = xavier_uniform([784, 128], DType.float32)
        var W_vel = zeros_like(W)

        # Training loop
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            # Returns updated params AND updated velocity
            (W, W_vel) = lars_step(
                W, grad_W, W_vel,
                learning_rate=0.1,
                momentum=0.9,
                weight_decay=0.0001,
                trust_coefficient=0.001
            )
        ```

    Note:
            This is a pure function - it returns new state rather than mutating.
            Caller must capture both return values and update their variables.

            LARS is particularly useful for large-batch training where the learning
            rate must be carefully scaled to prevent divergence.

    Raises:
            Error: If operation fails.
    """
    if params.shape() != gradients.shape():
        raise Error("Parameters and gradients must have the same shape")

    if params.dtype() != gradients.dtype():
        raise Error("Parameters and gradients must have the same dtype")

    if velocity.numel() == 0:
        raise Error(
            "Velocity buffer required for LARS (use zeros_like(params))"
        )

    # Compute norms for adaptive learning rate scaling
    # param_norm = ||params||
    var param_norm = compute_tensor_l2_norm(params)

    # grad_norm = ||gradients||
    var grad_norm = compute_tensor_l2_norm(gradients)

    # Compute effective gradient with weight decay
    # grad_eff = grad + weight_decay * params
    var effective_gradients = gradients

    if weight_decay > 0.0:
        # grad_eff = grad + weight_decay * params (SIMD optimized)
        var wd_tensor = full_like(params, weight_decay)
        var decay_term = multiply_simd(wd_tensor, params)
        effective_gradients = add_simd(gradients, decay_term)

    # Compute trust ratio for adaptive scaling
    # trust_ratio = trust_coefficient * param_norm / (grad_norm + weight_decay * param_norm + epsilon)
    var denominator = grad_norm + weight_decay * param_norm + epsilon
    var trust_ratio = trust_coefficient * param_norm / denominator

    # Create tensor filled with trust_ratio for SIMD operations
    var trust_ratio_tensor = full_like(params, trust_ratio)

    # Apply trust ratio scaling to effective gradients
    # scaled_grad = trust_ratio * grad_eff (SIMD optimized)
    var scaled_gradients = multiply_simd(
        trust_ratio_tensor, effective_gradients
    )

    # Update velocity with momentum
    # velocity = momentum * velocity + scaled_grad (SIMD optimized)
    var momentum_tensor = full_like(velocity, momentum)
    var scaled_velocity = multiply_simd(momentum_tensor, velocity)
    var new_velocity = add_simd(scaled_velocity, scaled_gradients)

    # Standard gradient descent with momentum
    # params = params - lr * velocity (SIMD optimized)
    var lr_tensor = full_like(params, learning_rate)
    var update = multiply_simd(lr_tensor, new_velocity)
    var new_params = subtract_simd(params, update)

    # Return both new params and new velocity (pure functional)
    return (new_params, new_velocity)


def lars_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    velocity: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified LARS step with default hyperparameters.

    This is a convenience function for basic LARS optimization.

    Formula:
    ```
        param_norm = ||params||
        grad_norm = ||gradients||
        trust_ratio = 0.001 * param_norm / (grad_norm + 0.0001 * param_norm + 1e-8)
        velocity = 0.9 * velocity + trust_ratio * (gradients + 0.0001 * params)
        params = params - learning_rate * velocity
    ```

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            velocity: Momentum buffer (use zeros_like(params)).
            learning_rate: Base learning rate.

    Returns:
            Tuple of (new_params, new_velocity).

    Example:
        ```mojo
        var W = xavier_uniform([784, 128], DType.float32)
        var W_vel = zeros_like(W)

        for epoch in range(100):
            var grad_W = ...  # Computed gradients
            (W, W_vel) = lars_step_simple(W, grad_W, W_vel, 0.1)
        ```

    Raises:
            Error: If operation fails.
    """
    return lars_step(
        params,
        gradients,
        velocity,
        learning_rate=learning_rate,
        momentum=0.9,
        weight_decay=0.0001,
        trust_coefficient=0.001,
        epsilon=1e-8,
    )
