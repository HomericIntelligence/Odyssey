"""AdaGrad optimizer (Adaptive Gradient Algorithm).

This module provides the AdaGrad optimizer for updating model parameters
during training using per-parameter adaptive learning rates.

AdaGrad adapts the learning rate for each parameter based on the historical
sum of squared gradients. Parameters with large accumulated gradients receive
small effective learning rates; parameters with sparse gradients receive large
effective learning rates. This makes AdaGrad well-suited for sparse data
(e.g. NLP embeddings) but it can shrink the effective learning rate to
near-zero on dense, long-running training.

Standard AdaGrad update rule:
    accum_t = accum_{t-1} + grad_t^2
    params_t = params_{t-1} - learning_rate * grad_t / (sqrt(accum_t) + epsilon)

With weight decay (L2 regularization, applied to the gradient):
    grad = grad + weight_decay * params
    accum = accum + grad^2
    params = params - learning_rate * grad / (sqrt(accum) + epsilon)

Reference:
    Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods
    for Online Learning and Stochastic Optimization. JMLR 12:2121-2159.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from odyssey.core.elementwise import sqrt
from odyssey.tensor.tensor_creation import full_like


def adagrad_step(
    params: AnyTensor,
    gradients: AnyTensor,
    accum: AnyTensor,
    learning_rate: Float64,
    epsilon: Float64 = 1e-10,
    weight_decay: Float64 = 0.0,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single AdaGrad optimization step - pure functional.

        Returns new parameters and new accumulated-squared-gradient buffer.
        Caller manages all state.

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            accum: Accumulator of squared gradients (use `zeros_like(params)` at init).
            learning_rate: Step size for parameter updates.
            epsilon: Small constant for numerical stability (default: 1e-10).
            weight_decay: L2 regularization factor folded into the gradient
                          before accumulation (default: 0.0).

    Returns:
            Tuple of (new_params, new_accum).

    Example (basic AdaGrad):
        ```mojo
        from odyssey.core import AnyTensor, zeros_like
        from odyssey.training.optimizers import adagrad_step

        var W = xavier_uniform(784, 128, DType.float32)
        var accum = zeros_like(W)

        # Training loop
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            (W, accum) = adagrad_step(W, grad_W, accum, lr=0.01)
        ```

    Note:
            This is a pure function - it returns new state rather than mutating.
            Caller must capture both return values and update their variables.

    Raises:
            Error: If shape or dtype contracts cannot be honored.
    """
    if params.shape() != gradients.shape():
        raise Error("Parameters and gradients must have the same shape")
    if params.dtype() != gradients.dtype():
        raise Error("Parameters and gradients must have the same dtype")
    if accum.numel() == 0:
        raise Error("accumulator must be initialized (use zeros_like(params))")

    # accum_t = accum_{t-1} + grad_t^2  (NO weight decay fold — match legacy)
    var grad_squared = multiply_simd(gradients, gradients)
    var new_accum = add_simd(accum, grad_squared)

    # adaptive_grad = grad / (sqrt(accum) + epsilon)
    var eps_tensor = full_like(new_accum, epsilon)
    var denom = add_simd(sqrt(new_accum), eps_tensor)
    var adaptive_grad = divide_simd(gradients, denom)

    # update = lr * adaptive_grad + (wd * params if weight_decay > 0)
    # Weight decay is applied additively *outside* the adaptive scaling —
    # this matches the legacy AdaGrad semantics in
    # `odyssey.autograd.optimizers.AdaGrad` so refactors do not drift.
    var lr_tensor = full_like(params, learning_rate)
    var update = multiply_simd(lr_tensor, adaptive_grad)
    if weight_decay > 0.0:
        var wd_tensor = full_like(params, weight_decay)
        var decay_term = multiply_simd(wd_tensor, params)
        update = add_simd(update, decay_term)

    var new_params = subtract_simd(params, update)

    return (new_params, new_accum)


def adagrad_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    accum: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified AdaGrad step with default hyperparameters.

        This is a convenience function for basic AdaGrad updates.

    Formula:
    ```
        accum = accum + grad^2
        params = params - lr * grad / (sqrt(accum) + 1e-10)
    ```

    Args:
            params: Model parameters to update.
            gradients: Gradients of loss with respect to params.
            accum: Accumulator of squared gradients.
            learning_rate: Step size for parameter updates.

    Returns:
            Tuple of (new_params, new_accum).

    Example:
        ```mojo
        var W = xavier_uniform(784, 128, shape, DType.float32)
        var accum = zeros_like(W)

        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            (W, accum) = adagrad_step_simple(W, grad_W, accum, lr=0.01)
        ```

    Raises:
            Error: If operation fails.
    """
    return adagrad_step(
        params,
        gradients,
        accum,
        learning_rate=learning_rate,
        epsilon=1e-10,
        weight_decay=0.0,
    )


def init_adagrad_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the adagrad optimizer.

    Returns a `List[List[AnyTensor]]` with outer length == `len(params_list)`
    (one entry per parameter) and inner length == 1 (one accumulator per
    parameter). Each accumulator starts at zero with the parameter's shape;
    the dtype matches the parameter, or float64 if `force_f64=True`.

    Args:
        params_list: Model parameters.
        force_f64: Up-cast all state buffers to float64 regardless of param
                   dtype.

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
        for _ in range(1):
            per.append(zeros(p.shape(), d))
        all_states.append(per^)
    return all_states^
