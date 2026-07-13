"""Optimizer utilities and helper functions.

This module provides common utility functions used across optimizer implementations,
including parameter state initialization, norm computation, and scaling operations.

Utilities provided:
- initialize_optimizer_state: Create and initialize optimizer state tensors
- compute_weight_decay_term: Compute L2 regularization term
- apply_weight_decay: Apply L2 regularization to parameters
- scale_tensor: Multiply tensor by scalar value
- normalize_tensor: Normalize tensor to unit norm

Design Philosophy:
    These utilities are designed to be reusable across different optimizer
    implementations (SGD, Adam, AdamW, RMSprop, LARS, etc.) and support
    both pure functional and in-place mutation styles
"""

from std.math import sqrt
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros_like, full_like
from odyssey.core.arithmetic_simd import multiply_simd


def initialize_optimizer_state(
    param_shapes: List[List[Int]], num_states: Int, dtype: DType = DType.float32
) raises -> List[List[AnyTensor]]:
    """Initialize multiple state buffers for optimizer (e.g., momentum, moments).

        Creates num_states lists of zero-initialized tensors for each parameter shape.
        This is useful for initializing all required state tensors for an optimizer.

    Args:
            param_shapes: List of parameter shapes to create state buffers for.
            num_states: Number of state buffers to create per parameter.
            dtype: Data type for state tensors (default: float32).

    Returns:
            List of state buffer lists. Each element is a list of AnyTensor states
            for a single parameter.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            # For Adam, which needs first moment (m) and second moment (v)
            var shapes = List[List[Int]]()
            shapes.append([784, 128])  # Weight shape
            shapes.append([128])       # Bias shape.

            var states = initialize_optimizer_state(shapes, num_states=2)
            # states[0] = [m_weight, v_weight]
            # states[1] = [m_bias, v_bias]
            ```

    Note:
            For SGD with momentum, use num_states=1 (one velocity buffer per param).
            For Adam variants, use num_states=2 (m and v buffers per param).
    """
    from odyssey.tensor.tensor_creation import zeros

    var all_states = List[List[AnyTensor]]()

    for i in range(len(param_shapes)):
        var param_state: List[AnyTensor] = []

        for _ in range(num_states):
            # Copy the shape since List[Int] is not ImplicitlyCopyable
            var shape = List[Int]()
            for j in range(len(param_shapes[i])):
                shape.append(param_shapes[i][j])

            param_state.append(zeros(shape, dtype))

        all_states.append(param_state^)

    return all_states^


def initialize_optimizer_state_from_params(
    params: List[AnyTensor], num_states: Int
) raises -> List[List[AnyTensor]]:
    """Initialize multiple state buffers for optimizer from existing parameters.

        Convenience function that extracts shapes from parameter tensors and creates
        matching state buffers with the same dtype as each parameter.

    Args:
            params: List of parameter tensors to base state initialization on.
            num_states: Number of state buffers to create per parameter.

    Returns:
            List of state buffer lists with matching shapes and dtypes.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            # Collect all model parameters
            var params : List[AnyTensor] = []
            params.append(layer1_weight)
            params.append(layer1_bias)
            params.append(layer2_weight)

            # Initialize 2 states per parameter (for Adam: m and v)
            var states = initialize_optimizer_state_from_params(params, num_states=2)
            ```
    """
    from odyssey.tensor.tensor_creation import zeros

    var all_states = List[List[AnyTensor]]()

    for i in range(len(params)):
        var param = params[i]
        var param_state: List[AnyTensor] = []

        for _ in range(num_states):
            param_state.append(zeros(param.shape(), param.dtype()))

        all_states.append(param_state^)

    return all_states^


def compute_weight_decay_term(
    params: AnyTensor, weight_decay: Float64
) raises -> AnyTensor:
    """Compute L2 regularization term: weight_decay * params.

        Returns a tensor that represents the weight decay contribution to the
        effective gradient. This is used in coupled weight decay (as in Adam).

    Args:
            params: Model parameters.
            weight_decay: L2 regularization coefficient.

    Returns:
            Tensor of same shape as params containing weight_decay * params.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var params = xavier_uniform([784, 128], DType.float32)
            var wd_term = compute_weight_decay_term(params, weight_decay=0.0001)
            # wd_term = 0.0001 * params
            ```

    Note:
            This computes the "coupled" weight decay used in Adam.
            For "decoupled" weight decay (AdamW), apply decay directly to params
            after the gradient-based update.
    """
    var wd_tensor = full_like(params, weight_decay)
    return multiply_simd(wd_tensor, params)


def apply_weight_decay(mut params: AnyTensor, weight_decay: Float64) raises:
    """Apply L2 regularization directly to parameters (in-place).

        This performs decoupled weight decay: params = params * (1 - weight_decay).
        Used in AdamW and other decoupled weight decay schemes.

    Args:
            params: Model parameters to regularize (modified in-place).
            weight_decay: L2 regularization coefficient.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var params = xavier_uniform([784, 128], DType.float32)
            apply_weight_decay(params, weight_decay=0.01)
            # params *= (1 - 0.01) = 0.99
            ```

    Note:
            This modifies the params tensor in-place.
            Typically applied AFTER gradient-based updates.
    """
    if weight_decay < 0.0 or weight_decay > 1.0:
        raise Error(
            "weight_decay must be in [0, 1], got: " + String(weight_decay)
        )

    if weight_decay == 0.0:
        return  # No-op

    var scale = 1.0 - weight_decay
    var scale_tensor = full_like(params, scale)
    var decayed = multiply_simd(scale_tensor, params)

    # Copy result back to params (in-place)
    for i in range(params.numel()):
        params._set_float64(i, decayed._get_float64(i))


def scale_tensor(tensor: AnyTensor, scale: Float64) raises -> AnyTensor:
    """Multiply tensor by scalar value.

        Returns a new tensor containing tensor * scale.

    Args:
            tensor: Input tensor.
            scale: Scalar multiplier.

    Returns:
            New tensor of same shape and dtype as input, with all elements scaled.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var grad = zeros([100], DType.float32)
            var scaled = scale_tensor(grad, scale=0.1)
            # scaled = 0.1 * grad
            ```
    """
    var scale_tensor = full_like(tensor, scale)
    return multiply_simd(scale_tensor, tensor)


def scale_tensor_inplace(mut tensor: AnyTensor, scale: Float64) raises:
    """Multiply tensor by scalar value (in-place).

        Modifies the tensor in-place by multiplying all elements by scale.

    Args:
            tensor: Input tensor (modified in-place).
            scale: Scalar multiplier.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var grad = zeros([100], DType.float32)
            scale_tensor_inplace(grad, scale=0.5)
            # grad *= 0.5
            ```
    """
    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        tensor._set_float64(i, val * scale)


def compute_tensor_norm(tensor: AnyTensor) raises -> Float64:
    """Compute L2 norm (Euclidean norm) of a tensor.

        Returns sqrt(sum(tensor^2)).

    Args:
            tensor: Input tensor.

    Returns:
            L2 norm of the tensor as Float64.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var grad = full([100], 1.0, DType.float32)
            var norm = compute_tensor_norm(grad)
            # norm = sqrt(100) = 10.0
            ```

    Note:
            This is useful for gradient clipping and adaptive learning rates.
    """
    var norm_squared = 0.0

    for i in range(tensor.numel()):
        var val = tensor._get_float64(i)
        norm_squared += val * val

    return sqrt(norm_squared)


def compute_global_norm(tensors: List[AnyTensor]) raises -> Float64:
    """Compute global L2 norm across multiple tensors.

        Returns sqrt(sum over all tensors of sum(tensor^2)).
        This is useful for gradient clipping across all parameters.

    Args:
            tensors: List of input tensors.

    Returns:
            Global L2 norm as Float64.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var grad1 = full([100], 1.0, DType.float32)
            var grad2 = full([50], 1.0, DType.float32)
            var tensors : List[AnyTensor] = [grad1, grad2]

            var global_norm = compute_global_norm(tensors)
            # global_norm = sqrt(100 + 50) = sqrt(150) ≈ 12.25
            ```

    Note:
            For empty list, returns 0.0.
    """
    if len(tensors) == 0:
        return 0.0

    var total_norm_squared = 0.0

    for tensor_idx in range(len(tensors)):
        var tensor = tensors[tensor_idx]

        for elem_idx in range(tensor.numel()):
            var val = tensor._get_float64(elem_idx)
            total_norm_squared += val * val

    return sqrt(total_norm_squared)


def normalize_tensor_to_unit_norm(mut tensor: AnyTensor) raises:
    """Normalize tensor to unit L2 norm (in-place).

        Modifies the tensor in-place so that its L2 norm becomes 1.0.
        If the tensor has zero norm, it remains unchanged (no division by zero).

    Args:
            tensor: Input tensor (modified in-place).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            var grad = full([100], 2.0, DType.float32)
            normalize_tensor_to_unit_norm(grad)
            # grad now has L2 norm = 1.0
            ```

    Note:
            Safe against division by zero - if norm is 0, tensor is unchanged.
    """
    var norm = compute_tensor_norm(tensor)

    if norm > 0.0:
        var scale = 1.0 / norm
        scale_tensor_inplace(tensor, scale)


def clip_tensor_norm(
    mut tensor: AnyTensor, max_norm: Float64
) raises -> Float64:
    """Clip tensor norm if it exceeds max_norm (in-place).

        If the L2 norm of the tensor exceeds max_norm, scales all elements
        down proportionally to bring the norm to exactly max_norm.
        Preserves the direction of the tensor.

    Args:
            tensor: Input tensor (modified in-place if norm exceeds max_norm).
            max_norm: Maximum allowed L2 norm.

    Returns:
            Original L2 norm before clipping.

    Raises:
            Error: If max_norm is negative.

        Example:
            ```mojo
            var grad = full([100], 1.0, DType.float32)
            var original_norm = clip_tensor_norm(grad, max_norm=1.0)
            # original_norm ≈ 10.0
            # grad is now scaled to norm = 1.0
            ```

    Note:
            This is the standard approach for gradient clipping in RNNs.
    """
    if max_norm < 0.0:
        raise Error("max_norm must be non-negative, got: " + String(max_norm))

    var norm = compute_tensor_norm(tensor)

    if norm > max_norm and norm > 0.0:
        var scale_factor = max_norm / norm
        scale_tensor_inplace(tensor, scale_factor)

    return norm


def clip_global_norm(
    mut tensors: List[AnyTensor], max_norm: Float64
) raises -> Float64:
    """Clip global L2 norm across all tensors (in-place).

        If the global L2 norm (computed across all tensors) exceeds max_norm,
        scales all elements in all tensors down proportionally.

    Args:
            tensors: List of tensors to clip (modified in-place if needed).
            max_norm: Maximum allowed global L2 norm.

    Returns:
            Original global L2 norm before clipping.

    Raises:
            Error: If max_norm is negative or tensors list is empty.

        Example:
            ```mojo
            var grad1 = full([100], 1.0, DType.float32)
            var grad2 = full([50], 1.0, DType.float32)
            var grads : List[AnyTensor] = [grad1, grad2]

            var global_norm = clip_global_norm(grads, max_norm=5.0)
            # Both gradients scaled proportionally
            ```

    Note:
            This is recommended for gradient clipping in recurrent networks.
    """
    if max_norm < 0.0:
        raise Error("max_norm must be non-negative, got: " + String(max_norm))

    if len(tensors) == 0:
        raise Error("tensors list cannot be empty")

    var global_norm = compute_global_norm(tensors)

    if global_norm > max_norm and global_norm > 0.0:
        var scale_factor = max_norm / global_norm

        for tensor_idx in range(len(tensors)):
            scale_tensor_inplace(tensors[tensor_idx], scale_factor)

    return global_norm


def apply_bias_correction(
    estimate: AnyTensor, decay: Float64, timestep: Int
) raises -> AnyTensor:
    """Apply bias correction to exponential moving average.

        Used in adaptive optimizers (Adam, RMSprop, etc.) to correct for bias
        in the initial estimates when using exponential moving averages.

        Formula: corrected = estimate / (1 - decay^timestep).

    Args:
            estimate: The exponential moving average estimate.
            decay: Decay factor (beta for momentum, typically 0.9 or 0.999).
            timestep: Current timestep (starts at 1).

    Returns:
            Bias-corrected estimate with same shape and dtype as input.

    Raises:
            Error: If timestep <= 0 or decay not in [0, 1).

        Example:
            ```mojo
            var m = zeros_like(params)
            # After first Adam step:
            m = ... # updated first moment

            # Bias correction
            var m_corrected = apply_bias_correction(m, decay=0.9, timestep=1)
            ```

    Note:
            The correction factor (1 - decay^t) accounts for the bias in early
            timesteps. As t increases, the correction becomes negligible.
    """
    if timestep <= 0:
        raise Error("timestep must be positive, got: " + String(timestep))

    if decay < 0.0 or decay >= 1.0:
        raise Error("decay must be in [0, 1), got: " + String(decay))

    # Compute correction factor: 1 - decay^timestep
    var decay_power = 1.0
    for _ in range(timestep):
        decay_power *= decay

    var correction_factor = 1.0 - decay_power

    if correction_factor <= 0.0:
        raise Error(
            "Bias correction factor invalid: " + String(correction_factor)
        )

    # Correction: estimate / (1 - decay^t)
    var corrected = scale_tensor(estimate, 1.0 / correction_factor)

    return corrected


def validate_optimizer_state(
    params: List[AnyTensor], states: List[List[AnyTensor]]
) raises:
    """Validate that optimizer state matches parameter shapes.

        Checks that:
        - Number of state lists matches number of parameters.
        - Each state tensor shape matches corresponding parameter shape.
        - All tensors are non-empty.

    Args:
            params: List of parameter tensors.
            states: List of state buffer lists from optimizer.

    Raises:
            Error: If state dimensions don't match parameters.

        Example:
            ```mojo
            validate_optimizer_state(params, [m, v])  # Raises if mismatch
            ```
    """
    if len(params) != len(states):
        raise Error(
            "Parameter count ("
            + String(len(params))
            + ") does not match state count ("
            + String(len(states))
            + ")"
        )

    for i in range(len(params)):
        var param = params[i]

        for state_idx in range(len(states[i])):
            var state = states[i][state_idx]

            if param.shape() != state.shape():
                raise Error(
                    "State shape mismatch at parameter "
                    + String(i)
                    + ", state "
                    + String(state_idx)
                )

            if state.numel() == 0:
                raise Error(
                    "State tensor is empty at parameter "
                    + String(i)
                    + ", state "
                    + String(state_idx)
                )
