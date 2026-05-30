"""Stochastic Gradient Descent (SGD) optimizer.

This module provides the SGD optimizer for updating model parameters
during training using gradient descent.

Standard SGD update rule:
    params = params - learning_rate * gradients

With momentum:
    velocity = momentum * velocity + gradients
    params = params - learning_rate * velocity

With weight decay (L2 regularization):
    params = params - learning_rate * (gradients + weight_decay * params)
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.core.arithmetic import subtract, multiply, add
from projectodyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
)
from projectodyssey.tensor.tensor_creation import full_like


def sgd_step(
    params: AnyTensor,
    gradients: AnyTensor,
    velocity: AnyTensor,
    learning_rate: Float64,
    momentum: Float64 = 0.0,
    weight_decay: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single SGD optimization step - pure functional.

        Returns new parameters and new velocity. Caller manages all state.

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            velocity: Momentum buffer (use zeros_like(params) if no momentum).
            learning_rate: Step size for parameter updates.
            momentum: Momentum factor (default: 0.0, no momentum).
            weight_decay: L2 regularization factor (default: 0.0, no regularization).

    Returns:
            Tuple of (new_params, new_velocity)

    Raises:
            Error: If operation fails.

        Example (basic SGD without momentum):
            ```mojo
            from projectodyssey.core import AnyTensor, zeros_like
            from projectodyssey.training.optimizers import sgd_step

            var W = xavier_uniform(784, 128, DType.float32)
            var W_vel = zeros_like(W)  # Not used, but required
            var grad_W = ...  # Computed via backpropagation

            # Returns new state
            (W, W_vel) = sgd_step(W, grad_W, W_vel, learning_rate=0.01)
            ```

        Example (SGD with momentum):
            ```mojo
            var W = xavier_uniform(784, 128, DType.float32)
            var W_vel = zeros_like(W)

            # Training loop
            for epoch in range(100):
                var grad_W = ...  # Compute gradients
                # Returns updated params AND updated velocity
                (W, W_vel) = sgd_step(W, grad_W, W_vel, lr=0.01, momentum=0.9)
            ```

    Note:
            This is a pure function - it returns new state rather than mutating.
            Caller must capture both return values and update their variables.
    """
    if params.shape() != gradients.shape():
        raise Error("Parameters and gradients must have the same shape")

    if params.dtype() != gradients.dtype():
        raise Error("Parameters and gradients must have the same dtype")

    var effective_gradients = gradients

    # Apply weight decay (L2 regularization) if specified
    if weight_decay > 0.0:
        # grad = grad + weight_decay * params (SIMD optimized)
        var wd_tensor = full_like(params, weight_decay)
        var decay_term = multiply_simd(wd_tensor, params)
        effective_gradients = add_simd(gradients, decay_term)

    var new_velocity = velocity  # Default: no change to velocity

    # Apply momentum if specified
    if momentum > 0.0:
        if velocity.numel() == 0:
            raise Error(
                "Velocity buffer required when using momentum (use"
                " zeros_like(params))"
            )

        # velocity = momentum * velocity + gradients (SIMD optimized)
        var momentum_tensor = full_like(velocity, momentum)
        var scaled_velocity = multiply_simd(momentum_tensor, velocity)
        new_velocity = add_simd(scaled_velocity, effective_gradients)

        # Use velocity for update
        effective_gradients = new_velocity

    # Standard SGD update: params = params - lr * gradients (SIMD optimized)
    var lr_tensor = full_like(params, learning_rate)
    var update = multiply_simd(lr_tensor, effective_gradients)
    var new_params = subtract_simd(params, update)

    # Return both new params and new velocity (pure functional)
    return (new_params, new_velocity)


def sgd_step_simple(
    params: AnyTensor, gradients: AnyTensor, learning_rate: Float64
) raises -> AnyTensor:
    """Simplified SGD step without momentum or weight decay.

        This is a convenience function for basic gradient descent.

        Formula:
            params = params - learning_rate * gradients.

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            learning_rate: Step size for parameter updates.

    Returns:
            Updated parameters

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var W1 = xavier_uniform(784, 128, shape, DType.float32)
            var grad_W1 = ... # Computed gradients
            W1 = sgd_step_simple(W1, grad_W1, 0.01)
            ```
    """
    if params.shape() != gradients.shape():
        raise Error("Parameters and gradients must have the same shape")

    if params.dtype() != gradients.dtype():
        raise Error("Parameters and gradients must have the same dtype")

    # params = params - lr * gradients (SIMD optimized)
    var lr_tensor = full_like(params, learning_rate)
    var update = multiply_simd(lr_tensor, gradients)

    return subtract_simd(params, update)


def sgd_momentum_update_inplace(
    mut param: AnyTensor,
    grad: AnyTensor,
    mut velocity: AnyTensor,
    lr: Float64,
    momentum: Float64,
) raises:
    """SGD parameter update with momentum (in-place mutation).

        This function performs in-place updates for efficiency in training loops.
        Mutates both param and velocity tensors directly.

        Formula:
            `velocity = momentum * velocity - lr * grad`
            `param = param + velocity`

    Args:
            param: Parameter tensor to update (modified in-place).
            grad: Gradient tensor.
            velocity: Momentum velocity tensor (modified in-place).
            lr: Learning rate.
            momentum: Momentum coefficient (typically 0.9).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from projectodyssey.core import AnyTensor, zeros_like
            from projectodyssey.training.optimizers import sgd_momentum_update_inplace

            var W = xavier_uniform([784, 128], DType.float32)
            var W_vel = zeros_like(W)

            # Training loop
            for epoch in range(100):
                var grad_W = ...  # Compute gradients
                sgd_momentum_update_inplace(W, grad_W, W_vel, lr=0.01, momentum=0.9)
            ```

    Note:
            - In-place mutation for efficiency.
            - Velocity tensor must be pre-allocated (use zeros_like).
            - Both param and velocity are modified directly.
            - This is the AlexNet/ResNet standard momentum formulation.
            - Supports float32 and float64 dtypes.
    """
    var numel = param.numel()

    if param.shape() != grad.shape():
        raise Error("Parameter and gradient must have the same shape")

    if param.shape() != velocity.shape():
        raise Error("Parameter and velocity must have the same shape")

    # Dispatch based on dtype
    if param.dtype() == DType.float32:
        var lr_f32 = Float32(lr)
        var momentum_f32 = Float32(momentum)

        # Update velocity and parameters in-place
        for i in range(numel):
            # velocity = momentum * velocity - lr * grad
            var v = momentum_f32 * velocity.load[DType.float32](
                i
            ) - lr_f32 * grad.load[DType.float32](i)
            velocity.store[DType.float32](i, v)
            # param = param + velocity
            param.store[DType.float32](i, param.load[DType.float32](i) + v)
    elif param.dtype() == DType.float64:
        # Update velocity and parameters in-place
        for i in range(numel):
            # velocity = momentum * velocity - lr * grad
            var v = momentum * velocity.load[DType.float64](i) - lr * grad.load[
                DType.float64
            ](i)
            velocity.store[DType.float64](i, v)
            # param = param + velocity
            param.store[DType.float64](i, param.load[DType.float64](i) + v)
    else:
        raise Error(
            "sgd_momentum_update_inplace only supports float32 and float64"
        )


# ============================================================================
# Velocity Initialization Utilities
# ============================================================================


def initialize_velocities(
    param_shapes: List[List[Int]], dtype: DType = DType.float32
) raises -> List[AnyTensor]:
    """Create zero-initialized velocity tensors for SGD with momentum.

        This utility function creates momentum buffers for all parameters in a model
        Each velocity tensor is zero-initialized with the same shape as its
        corresponding parameter.

    Args:
            param_shapes: List of parameter shapes to create velocities for.
            dtype: Data type for velocity tensors (default: float32).

    Returns:
            List of zero-initialized tensors matching parameter shapes.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from projectodyssey.training.optimizers import initialize_velocities

            # Get shapes from model parameters
            var shapes = List[List[Int]]()
            shapes.append(model.conv1_kernel.shape())
            shapes.append(model.conv1_bias.shape())
            shapes.append(model.fc1_weights.shape())
            shapes.append(model.fc1_bias.shape())

            var velocities = initialize_velocities(shapes)
            # velocities[0] matches conv1_kernel shape
            # velocities[1] matches conv1_bias shape, etc.
            ```

    Note:
            The order of velocities matches the order of shapes provided.
            Ensure you use the same ordering when calling sgd_momentum_update_inplace.
    """
    from projectodyssey.tensor.tensor_creation import zeros

    var velocities: List[AnyTensor] = []

    for i in range(len(param_shapes)):
        # Copy the shape since List[Int] is not ImplicitlyCopyable
        var shape = List[Int]()
        for j in range(len(param_shapes[i])):
            shape.append(param_shapes[i][j])
        velocities.append(zeros(shape, dtype))

    return velocities^


def initialize_velocities_from_params(
    params: List[AnyTensor],
) raises -> List[AnyTensor]:
    """Create zero-initialized velocity tensors matching a list of parameters.

        Convenience function that extracts shapes from existing parameter tensors
        and creates matching velocity buffers.

    Args:
            params: List of parameter tensors.

    Returns:
            List of zero-initialized velocity tensors with matching shapes and dtypes.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from projectodyssey.training.optimizers import initialize_velocities_from_params

            # Collect all model parameters
            var params : List[AnyTensor] = []
            params.append(model.conv1_kernel)
            params.append(model.conv1_bias)
            params.append(model.fc1_weights)
            params.append(model.fc1_bias)

            var velocities = initialize_velocities_from_params(params)
            ```
    """
    from projectodyssey.tensor.tensor_creation import zeros

    var velocities: List[AnyTensor] = []

    for i in range(len(params)):
        var param = params[i]
        velocities.append(zeros(param.shape(), param.dtype()))

    return velocities^
