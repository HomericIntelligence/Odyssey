"""Muon Hyperball optimizer — norm-constrained Muon.

Wraps the Muon optimizer (Jordan et al., 2024) with a "hyperball" constraint that
clamps the Frobenius norm of BOTH the per-step update and the resulting weight
matrix to fixed constants. Constraining these norms improves learning-rate
transferability across model width and depth.

Update rule (per matrix parameter):

    (W_muon, m_new) = muon_step(W, grad, m, lr, ...)   # standard Muon
    dW = W_muon - W                                    # the Muon update
    dW = dW * min(1, update_norm_max / ||dW||_F)       # clamp update norm
    W_new = W + dW
    W_new = W_new * min(1, weight_norm_max / ||W_new||_F)  # clamp weight norm

Both clamps are one-sided projections onto Frobenius-norm balls: a norm at or below
the radius is left unchanged; a norm above it is scaled down to the radius. As with
Muon, this operates on rank-2 (matrix) parameters.

Reference:
    Muon core: Jordan, K., Jin, Y., Boza, V., et al. (2024). Muon: An optimizer for
    the hidden layers of neural networks. https://kellerjordan.github.io/posts/muon/
    Hyperball norm-constraint: constrains ||W||_F and ||dW||_F to fixed radii for
    width/depth-robust learning-rate transfer.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full_like
from odyssey.core.arithmetic_simd import multiply_simd, subtract_simd, add_simd
from odyssey.core.numerical_safety import compute_tensor_l2_norm
from odyssey.training.optimizers.muon import muon_step


def _project_to_ball(x: AnyTensor, radius: Float64) raises -> AnyTensor:
    """Project x onto the Frobenius-norm ball of the given radius.

    Returns x unchanged if ||x||_F <= radius (or radius <= 0, meaning "no
    constraint"); otherwise scales x down so that ||x||_F == radius. The
    Frobenius norm of a matrix equals the L2 norm of its flattened entries.

    Args:
        x: Tensor to project.
        radius: Ball radius. A non-positive radius disables the constraint.

    Returns:
        The projected tensor (a fresh tensor scaled from x, or x-equivalent).

    Raises:
        Error: If tensor operations fail.
    """
    if radius <= 0.0:
        return add_simd(x, full_like(x, 0.0))
    var norm = compute_tensor_l2_norm(x)
    if norm <= radius:
        return add_simd(x, full_like(x, 0.0))
    var scale = radius / norm
    return multiply_simd(full_like(x, scale), x)


def muon_hyperball_step(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
    weight_norm_max: Float64 = 1.0,
    update_norm_max: Float64 = 0.1,
    momentum_beta: Float64 = 0.95,
    weight_decay: Float64 = 0.01,
    ns_steps: Int = 5,
    nesterov: Bool = True,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single Muon Hyperball step — pure functional.

    Runs a standard Muon step, then applies the hyperball constraint: the per-step
    update is projected to `update_norm_max` and the resulting weight matrix to
    `weight_norm_max` (both Frobenius-norm balls). A non-positive radius disables
    the corresponding constraint. Returns new parameters and the new momentum
    buffer; the caller manages all state.

    Args:
        params: Model parameters to update (rank-2 matrix).
        gradients: Gradients of the loss with respect to params.
        momentum: Momentum buffer (use zeros_like(params) initially).
        learning_rate: Step size for the underlying Muon update.
        weight_norm_max: Frobenius-norm radius for the weight matrix
            (default 1.0; <= 0 disables the weight-norm constraint).
        update_norm_max: Frobenius-norm radius for the per-step update
            (default 0.1; <= 0 disables the update-norm constraint).
        momentum_beta: Muon momentum decay (default 0.95).
        weight_decay: Muon weight-decay factor (default 0.01, Jordan recipe).
        ns_steps: Newton-Schulz iterations for orthogonalization (default 5).
        nesterov: If True, use Nesterov momentum (default True).

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If operation fails (shape/dtype mismatch, non-matrix params).
    """
    # Standard Muon step.
    var muon_result = muon_step(
        params,
        gradients,
        momentum,
        learning_rate,
        momentum_beta,
        weight_decay,
        ns_steps,
        nesterov,
    )
    var w_muon = muon_result[0]
    var new_momentum = muon_result[1]

    # Clamp the update norm: dW = W_muon - W, project, re-apply.
    var update = subtract_simd(w_muon, params)
    var update_clamped = _project_to_ball(update, update_norm_max)
    var w_after_update = add_simd(params, update_clamped)

    # Clamp the weight norm.
    var new_params = _project_to_ball(w_after_update, weight_norm_max)

    return (new_params, new_momentum)


def muon_hyperball_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified Muon Hyperball step with default hyperparameters.

    Uses weight_norm_max=1.0, update_norm_max=0.1, and the Jordan Muon defaults
    (momentum_beta=0.95, weight_decay=0.01, ns_steps=5, nesterov=True).

    Args:
        params: Model parameters to update (rank-2 matrix).
        gradients: Gradients of the loss with respect to params.
        momentum: Momentum buffer.
        learning_rate: Step size for the underlying Muon update.

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If operation fails.
    """
    return muon_hyperball_step(params, gradients, momentum, learning_rate)


def init_muon_hyperball_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the muon_hyperball optimizer.

    Returns a `List[List[AnyTensor]]` with outer length == `len(params_list)` (one entry per parameter) and inner length == 1 (one entry per state buffer the optimizer threads across calls).

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
        for _ in range(1):
            per.append(zeros(p.shape(), d))
        all_states.append(per^)
    return all_states^
