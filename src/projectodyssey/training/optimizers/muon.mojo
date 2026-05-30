"""Muon optimizer with Newton-Schulz orthogonalization of momentum.

This module provides the Muon optimizer for matrix-shaped parameters. Muon applies
Newton-Schulz iteration to orthogonalize the momentum buffer before each update,
which consistently outperforms AdamW on language and vision tasks with matrix-shaped
parameters (linear layer weights, convolutional kernels).

Key Concepts:
    Muon is specialized for matrix-shaped parameters (rank-2 tensors). It applies
    SGD-style heavy-ball momentum but orthogonalizes the update direction using
    Newton-Schulz iteration. This orthogonalization improves conditioning and
    convergence for weight matrices while maintaining computational efficiency.

    For non-matrix parameters (embeddings, biases, scalar params), Muon is not
    applicable — use AdamW instead. The `is_muon_eligible` predicate identifies
    which parameters should route through Muon in a hybrid optimizer.

Standard Muon update rule (matrix-shaped params only):
    m_new      = momentum_beta * m + grad             # heavy-ball momentum
    u          = grad + momentum_beta * m_new  (if nesterov)  else  m_new
    u_orth     = newton_schulz_orthogonalize(u, ns_steps)
    scale      = 0.2 * max(R, C) / sqrt(R * C)        # shape-invariant magnitude
    p_new      = p - lr * scale * u_orth
    if weight_decay > 0:
        p_new = p_new - lr * weight_decay * p

Key difference from AdamW:
    AdamW uses adaptive learning rates (element-wise scaling via v_hat).
    Muon uses a single scalar magnitude (scale) for all elements, applied after
    orthogonalization. Weight decay includes the learning_rate factor (lr * wd * p),
    differing from AdamW's decoupled form (wd * p). This is the Jordan et al. 2024
    recipe and improves stability in practice.

Reference:
    Jordan et al. 2024, "Muon: An optimizer for hidden layers in neural networks"
    https://kellerjordan.github.io/posts/muon/
    https://github.com/KellerJordan/Muon
"""

from std.math import sqrt as scalar_sqrt
from projectodyssey.tensor.any_tensor import AnyTensor, zeros_like, full_like
from projectodyssey.core.arithmetic_simd import (
    subtract_simd,
    multiply_simd,
    add_simd,
    divide_simd,
)
from projectodyssey.core.elementwise import sqrt
from projectodyssey.core.matrix import matmul, transpose
from projectodyssey.training.optimizers.optimizer_utils import (
    compute_tensor_norm,
)


def is_muon_eligible(params: AnyTensor) -> Bool:
    """Check if a parameter tensor is eligible for Muon optimization.

    Muon is designed for matrix-shaped parameters (rank-2 tensors). It is not
    applicable to embeddings, biases, or other scalar/vector parameters.

    A parameter is eligible if:
    - Rank is exactly 2 (a matrix)
    - Both dimensions >= 2 (avoids degenerate cases where one dimension is 1,
      which have only one nonzero singular value and would be equivalent to
      plain gradient normalization)

    Args:
        params: Tensor to check for Muon eligibility.

    Returns:
        True if params is a matrix with both dimensions >= 2, False otherwise.

    Example:
        ```mojo
        var weight = zeros([784, 128], DType.float32)  # Linear weight matrix
        assert is_muon_eligible(weight)  # True

        var bias = zeros([128], DType.float32)         # Bias vector
        assert not is_muon_eligible(bias)  # False

        var kernel = zeros([16, 8, 3, 3], DType.float32)  # Conv4D (not reshaped)
        assert not is_muon_eligible(kernel)  # False — needs reshape to [16, 72] first
        ```

    Note:
        Callers should reshape conv kernels from (out_channels, in_channels, kH, kW)
        to (out_channels, in_channels * kH * kW) before checking and passing to Muon.
        Embeddings and biases should always route through AdamW instead.
    """
    if params.ndim() != 2:
        return False

    var shape = params.shape()
    var rows = shape[0]
    var cols = shape[1]

    return rows >= 2 and cols >= 2


def newton_schulz_orthogonalize(
    X: AnyTensor, steps: Int = 5
) raises -> AnyTensor:
    """Apply Newton-Schulz iteration to orthogonalize a matrix.

    Newton-Schulz iteration converges to the orthogonal matrix nearest X in the
    Frobenius norm. It is a fixed-point iteration: at convergence, the result Y
    satisfies Y @ Y^T ≈ I_rows (if rows <= cols) or Y^T @ Y ≈ I_cols (if rows > cols).

    Algorithm (Jordan et al. 2024):
        1. Pre-normalize X by its Frobenius norm (spectral pre-normalization)
        2. If rows > cols, transpose X (always iterate on the shorter dimension)
        3. For each of 5 iterations:
            - Compute A = X @ X^T (Gram matrix on shorter dimension)
            - Apply quintic polynomial: B = b * A + c * (A @ A)
            - Update: X = a * X + B @ X
        4. Transpose back if needed

    The coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) are from the published
    recipe. They are optimized for fast convergence to orthogonality in 5 steps.
    See: https://github.com/KellerJordan/Muon/blob/main/muon/torch/optim.py

    Args:
        X: A rank-2 tensor (matrix) to orthogonalize.
        steps: Number of Newton-Schulz iterations (default: 5).
               5 iterations typically achieves singular values within 1e-5 of 1.0.

    Returns:
        Orthogonalized matrix Y with the same shape as X.

    Raises:
        Error: If X is not rank-2 or steps <= 0.

    Example:
        ```mojo
        var X = zeros([8, 16], DType.float32)  # Tall matrix (8 rows, 16 cols)
        # Fill X with random values...
        var Y = newton_schulz_orthogonalize(X, steps=5)
        # Y satisfies: Y @ Y^T ≈ I_8 (rows orthonormal)
        # Frobenius norm of (Y @ Y^T - I) < 1e-5 in fp32
        ```

    Note:
        Pre-normalization stabilizes convergence and ensures the iteration
        operates in a regime where the quintic polynomial converges quickly.
        The transpose-in/out logic is transparent: the function returns a result
        in the same orientation as the input.
    """
    if X.ndim() != 2:
        raise Error("newton_schulz_orthogonalize requires a rank-2 tensor")

    if steps <= 0:
        raise Error("steps must be positive, got: " + String(steps))

    var shape = X.shape()
    var rows = shape[0]
    var cols = shape[1]

    # Determine if we need to transpose: always iterate on the shorter dimension
    var transposed = rows > cols
    var Y = X

    if transposed:
        Y = transpose(
            X, None
        )  # Reverse all axes for rank-2: [rows, cols] -> [cols, rows]
        var tmp_shape = Y.shape()
        rows = tmp_shape[0]
        cols = tmp_shape[1]

    # Spectral pre-normalization: X / (frobenius_norm(X) + 1e-7)
    var norm = compute_tensor_norm(Y)
    var norm_safe = norm + 1e-7
    var norm_tensor = full_like(Y, norm_safe)
    Y = divide_simd(Y, norm_tensor)

    # Newton-Schulz quintic iteration coefficients (Jordan et al. 2024)
    var a = 3.4445
    var b = -4.7750
    var c = 2.0315

    # Iterate: X <- a*X + (b*A + c*A^2) @ X, where A = X @ X^T
    for _ in range(steps):
        # Compute A = Y @ Y^T (Gram matrix, shape: [rows, rows])
        var A = matmul(Y, transpose(Y, None))

        # Compute A^2 = A @ A
        var A_squared = matmul(A, A)

        # Compute B = b * A + c * A^2
        var b_tensor = full_like(A, b)
        var c_tensor = full_like(A_squared, c)
        var b_A = multiply_simd(b_tensor, A)
        var c_A2 = multiply_simd(c_tensor, A_squared)
        var B = add_simd(b_A, c_A2)

        # Compute B @ Y
        var B_Y = matmul(B, Y)

        # Update: Y = a * Y + B @ Y
        var a_tensor = full_like(Y, a)
        var a_Y = multiply_simd(a_tensor, Y)
        Y = add_simd(a_Y, B_Y)

    # Transpose back if we transposed in
    if transposed:
        Y = transpose(Y, None)

    return Y


def muon_step(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
    momentum_beta: Float64 = 0.95,
    weight_decay: Float64 = 0.01,
    ns_steps: Int = 5,
    nesterov: Bool = True,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single Muon optimization step - pure functional.

    Returns new parameters and new momentum buffer. Caller manages all state.

    Muon applies Newton-Schulz orthogonalization to the momentum buffer before
    updating parameters. It is specialized for matrix-shaped parameters and
    consistently outperforms AdamW on language and vision tasks.

    **Important: Signature differs from AdamW.**
    AdamW returns (params, m, v) as a 3-tuple (two state buffers: m and v).
    Muon returns (params, momentum) as a 2-tuple (one state buffer: momentum).
    When swapping optimizers in training loops, update the destructuring:
        Old: (p, m, v) = adamw_step(...)
        New: (p, m) = muon_step(...)
    See the README in optimizers/ for a hybrid-optimizer routing pattern.

    Args:
        params: Model parameters to update (must be rank-2 matrix).
        gradients: Gradients of loss with respect to params.
        momentum: Momentum buffer (use zeros_like(params) initially).
        learning_rate: Step size for parameter updates.
        momentum_beta: Momentum decay rate (default: 0.95, typical range: 0.9–0.99).
        weight_decay: L2 regularization factor (default: 0.01, Jordan recipe).
                      Note: includes learning_rate factor (lr * wd * p), differing
                      from AdamW's decoupled form (wd * p). See docstring above.
        ns_steps: Number of Newton-Schulz iterations (default: 5).
                  Higher values achieve better orthogonality but are slower.
        nesterov: If True, use Nesterov momentum (default: True).
                  If False, use heavy-ball momentum.

    Returns:
        Tuple of (new_params, new_momentum).

    Example (Muon on matrix weights):
        ```mojo
        from projectodyssey.core import AnyTensor, zeros_like
        from projectodyssey.training.optimizers import muon_step

        var W = xavier_uniform([784, 128], DType.float32)  # Linear weight
        var m = zeros_like(W)

        # Training loop
        for epoch in range(100):
            var grad_W = ...  # Compute gradients
            (W, m) = muon_step(W, grad_W, m, lr=0.01, momentum_beta=0.95)
        ```

    Note on hybrid optimization:
        Muon should be applied only to matrix-shaped parameters. For a model with
        mixed parameter types:
            1. Weights (linear/conv): use Muon (after reshaping conv kernels if needed)
            2. Biases, embeddings, norm scales: use AdamW
        See is_muon_eligible() and the README's "Hybrid Muon+AdamW" section.

    Raises:
        Error: If operation fails (shape mismatch, dtype mismatch, non-matrix params).
    """
    # Validation
    if params.shape() != gradients.shape():
        raise Error("Parameters and gradients must have the same shape")

    if params.dtype() != gradients.dtype():
        raise Error("Parameters and gradients must have the same dtype")

    if momentum.numel() == 0:
        raise Error(
            "Momentum buffer must be initialized (use zeros_like(params))"
        )

    if params.ndim() != 2:
        raise Error(
            "Muon requires rank-2 parameters (matrix). Use"
            " is_muon_eligible(params) to check. For non-matrix params"
            " (embeddings, biases), use AdamW instead. See"
            " projectodyssey/training/optimizers/README.md for hybrid routing."
        )

    # Update momentum: m_new = momentum_beta * m + grad
    var beta_tensor = full_like(momentum, momentum_beta)
    var m_scaled = multiply_simd(beta_tensor, momentum)
    var new_momentum = add_simd(m_scaled, gradients)

    # Compute update direction (with optional Nesterov acceleration)
    var update_direction = new_momentum
    if nesterov:
        # Nesterov: u = grad + momentum_beta * m_new
        var nesterov_term = multiply_simd(beta_tensor, new_momentum)
        update_direction = add_simd(gradients, nesterov_term)

    # Orthogonalize update direction via Newton-Schulz
    var u_orth = newton_schulz_orthogonalize(update_direction, steps=ns_steps)

    # Compute shape-invariant scaling: scale = 0.2 * max(R, C) / sqrt(R * C)
    var shape = params.shape()
    var R = Float64(shape[0])
    var C = Float64(shape[1])
    var max_dim = max(R, C)
    var sqrt_product = scalar_sqrt(R * C)
    var scale = 0.2 * max_dim / sqrt_product

    # Apply scaled update: p_new = p - lr * scale * u_orth
    var scale_tensor = full_like(u_orth, scale)
    var scaled_update = multiply_simd(scale_tensor, u_orth)
    var lr_tensor = full_like(scaled_update, learning_rate)
    var grad_update = multiply_simd(lr_tensor, scaled_update)
    var params_after_grad = subtract_simd(params, grad_update)

    # Apply weight decay: p_new = p_new - lr * wd * p (includes lr factor, Jordan recipe)
    var new_params = params_after_grad
    if weight_decay > 0.0:
        var wd_factor = learning_rate * weight_decay
        var wd_tensor = full_like(params, wd_factor)
        var decay_term = multiply_simd(wd_tensor, params)
        new_params = subtract_simd(params_after_grad, decay_term)

    # Return new state (pure functional)
    return (new_params, new_momentum)


def muon_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified Muon step with default hyperparameters.

    This is a convenience function for basic Muon optimization with
    commonly-used default parameters. Defaults match the Jordan et al. 2024
    recipe and are optimized for matrix-shaped parameters.

    Formula:
    ```
        m = 0.95 * m + grad              # SGD momentum
        u_orth = newton_schulz(grad + 0.95*m, steps=5)
        scale = 0.2 * max(R,C) / sqrt(R*C)
        p = p - lr * scale * u_orth - lr * 0.01 * p  (weight decay)
    ```

    Args:
        params: Model parameters to update (must be rank-2 matrix).
        gradients: Gradients of loss with respect to params.
        momentum: Momentum buffer.
        learning_rate: Step size for parameter updates.

    Returns:
        Tuple of (new_params, new_momentum).

    Example:
        ```mojo
        var W = xavier_uniform([784, 128], DType.float32)
        var m = zeros_like(W)

        for epoch in range(100):
            var grad_W = ... # Computed gradients
            (W, m) = muon_step_simple(W, grad_W, m, 0.01)
        ```

    Note:
        Default weight_decay=0.01 aligns with the Jordan et al. 2024 recipe.
        This differs from SGD/RMSProp (which default to 0.0) but matches the
        paper's recommended baseline. Adjust weight_decay if needed for your
        specific task (e.g., set to 0.0 for vision models on small datasets).

    Raises:
        Error: If operation fails.
    """
    return muon_step(
        params,
        gradients,
        momentum,
        learning_rate=learning_rate,
        momentum_beta=0.95,
        weight_decay=0.01,
        ns_steps=5,
        nesterov=True,
    )
