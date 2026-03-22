"""Base optimizer functionality for gradient-based parameter updates.

This module provides the OptimizerBase struct that encapsulates common functionality
shared across all optimizer implementations (SGD, Adam, AdaGrad, RMSprop).

The base class handles:
- Learning rate management (get/set)
- Gradient zeroing (clearing the tape)
- Gradient clipping by global norm
- Common parameter validation

This reduces code duplication and provides a consistent interface across all optimizers.

Design Note:
    This is designed for the class-based autograd optimizer API.
    For the modern functional optimizer API, see shared/training/optimizers/.
"""

from shared.autograd.tape import GradientTape
from shared.autograd.variable import Variable
from math import sqrt


trait Optimizer:
    """Base trait for all optimizers.

    Defines the minimal interface that all optimizers must implement.
    Each optimizer provides its own step() implementation with specific
    update logic (momentum, adaptive rates, etc.).
    """

    fn step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        """Update parameters using their gradients.

        This method must be implemented by each optimizer with its specific
        update algorithm (SGD with momentum, Adam, etc.).

        Args:
            parameters: List of Variables to update (model parameters).
            tape: The gradient tape containing computed gradients.

        Raises:
            Error: If operation fails or gradients have incompatible shapes.
        """
        ...

    fn zero_grad(self, mut tape: GradientTape):
        """Reset all gradients in the tape.

        Should be called after each optimizer step to clear gradients before
        the next backward pass.

        Args:
            tape: The gradient tape to clear.
        """
        ...

    fn get_lr(self) -> Float64:
        """Get the current learning rate.

        Returns:
            Current learning rate value.
        """
        ...

    fn set_lr(mut self, lr: Float64) raises:
        """Set the learning rate.

        Args:
            lr: New learning rate value (must be positive).

        Raises:
            Error: If lr is non-positive.
        """
        ...


fn zero_grad_impl(mut tape: GradientTape):
    """Shared implementation of gradient zeroing.

    Clears all gradients in the tape's registry. This is the same logic
    used by all optimizers and is extracted here to avoid duplication.

    Args:
        tape: The gradient tape to clear.

    Example:
        ```mojo
        # In optimizer's zero_grad method:
        fn zero_grad(self, mut tape: GradientTape):
            zero_grad_impl(tape)
        ```
    """
    tape.registry.clear()


fn validate_learning_rate(lr: Float64) raises:
    """Validate that learning rate is positive.

    Args:
        lr: Learning rate to validate.

    Raises:
        Error: If lr <= 0.

    Example:
        ```mojo
        fn set_lr(mut self, lr: Float64) raises:
            validate_learning_rate(lr)
            self.learning_rate = lr
        ```
    """
    if lr <= 0.0:
        raise Error("Learning rate must be positive, got: " + String(lr))


fn clip_gradients_by_global_norm(
    mut parameters: List[Variable],
    mut tape: GradientTape,
    max_norm: Float64,
) raises -> Float64:
    """Clip gradients by global norm across all parameters.

    Computes the global L2 norm across all parameter gradients and scales
    them down if the global norm exceeds max_norm. This preserves the
    direction of the gradient vector while limiting its magnitude.

    Algorithm:
        1. Compute global_norm = sqrt(sum over all params of sum(grad^2))
        2. If global_norm > max_norm:
            scale = max_norm / global_norm
            For each parameter gradient:
                grad = grad * scale

    Args:
        parameters: List of Variables with gradients to clip.
        tape: The gradient tape containing gradients.
        max_norm: Maximum allowed global L2 norm.

    Returns:
        The original global norm before clipping.

    Raises:
        Error: If max_norm is negative.

    Example:
        ```mojo
        # Before optimizer step, clip gradients
        var original_norm = clip_gradients_by_global_norm(
            parameters, tape, max_norm=5.0
        )
        optimizer.step(parameters, tape)
        ```

    Note:
        This is particularly useful for training RNNs and prevents
        exploding gradients. It's recommended to clip before calling step().
    """
    if max_norm < 0.0:
        raise Error("max_norm must be non-negative, got: " + String(max_norm))

    # Compute global L2 norm across all parameter gradients
    var global_norm_squared = 0.0

    for i in range(len(parameters)):
        if not parameters[i].requires_grad:
            continue

        var param_id = parameters[i].id
        if not tape.registry.has_gradient(param_id):
            continue

        var grad = tape.registry.get_grad(param_id)

        # Accumulate squared norm
        for j in range(grad.numel()):
            var val = grad._get_float64(j)
            global_norm_squared += val * val

    var global_norm = sqrt(global_norm_squared)

    # If norm exceeds threshold, scale all gradients down proportionally
    if global_norm > max_norm and global_norm > 0.0:
        var scale_factor = max_norm / global_norm

        # Scale each gradient in place
        for i in range(len(parameters)):
            if not parameters[i].requires_grad:
                continue

            var param_id = parameters[i].id
            if not tape.registry.has_gradient(param_id):
                continue

            var grad = tape.registry.get_grad(param_id)

            # Create scaled gradient
            from shared.core.any_tensor import AnyTensor

            var scaled_grad = AnyTensor(grad.shape(), grad.dtype())

            for j in range(grad.numel()):
                var val = grad._get_float64(j)
                scaled_grad._set_float64(j, val * scale_factor)

            # Clear the has_grad flag so set_grad replaces instead of accumulates
            tape.registry.has_grad[param_id] = False

            # Update gradient in tape
            tape.registry.set_grad(param_id, scaled_grad^)

    return global_norm


fn count_parameters_with_gradients(
    parameters: List[Variable], tape: GradientTape
) -> Int:
    """Count how many parameters have gradients in the tape.

    Useful for debugging and validation. Parameters without gradients
    are skipped during optimization.

    Args:
        parameters: List of Variables to check.
        tape: The gradient tape containing gradients.

    Returns:
        Number of parameters that have gradients and require_grad=True.

    Example:
        ```mojo
        var num_params = count_parameters_with_gradients(parameters, tape)
        print("Optimizing", num_params, "parameters")
        ```
    """
    var count = 0

    for i in range(len(parameters)):
        if not parameters[i].requires_grad:
            continue

        var param_id = parameters[i].id
        if tape.registry.has_gradient(param_id):
            count += 1

    return count
