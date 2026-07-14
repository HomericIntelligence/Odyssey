"""MGUP-Muon optimizer — Muon with selective (MGUP) updates.

Integrates Muon (Jordan et al., 2024) with a Maximum-Gradient-Utilization / selective-
update (MGUP) mechanism: after a standard Muon step, a fixed fraction of parameters
receives a larger effective step-size. The selection is deterministic — the entries
of the Muon update with the largest magnitude |ΔW| — so the same fraction of
"most-active" coordinates is amplified each step, concentrating learning capacity on
the coordinates Muon already moves the most.

Update rule (per matrix parameter):

    (W_muon, m_new) = muon_step(W, grad, m, lr, ...)   # standard Muon
    dW = W_muon - W                                     # the Muon update
    threshold = the (1 - selected_fraction) quantile of |dW|
    dW[|dW| >= threshold] *= select_scale               # amplify the selected top fraction
    W_new = W + dW

With `selected_fraction = 0` or `select_scale = 1` this reduces exactly to Muon.

Reference:
    Muon core: Jordan, K., Jin, Y., Boza, V., et al. (2024). Muon: An optimizer for
    the hidden layers of neural networks. https://kellerjordan.github.io/posts/muon/
    MGUP selective-update mechanism: applies larger step-sizes to a selected fixed
    proportion of parameters each iteration.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros_like
from odyssey.core.arithmetic_simd import subtract_simd, add_simd
from odyssey.training.optimizers.muon import muon_step


def _select_threshold(
    abs_update: AnyTensor, fraction: Float64
) raises -> Float64:
    """Return the |dW| cutoff so that a `fraction` of entries are at or above it.

    Computes the (1 - fraction) quantile of the update magnitudes by a simple
    counting scan: the smallest threshold t such that at least
    ceil(fraction * N) entries satisfy |dW| >= t. Uses an O(N^2) selection (N is a
    single weight matrix's element count) to stay dependency-free and deterministic.

    Args:
        abs_update: Tensor of per-entry update magnitudes |dW|.
        fraction: Fraction of entries to select (clamped to [0, 1]).

    Returns:
        The magnitude cutoff. Entries with |dW| >= cutoff are "selected".

    Raises:
        Error: If tensor access fails.
    """
    var n = abs_update.numel()
    if n == 0:
        return 0.0
    var frac = fraction
    if frac <= 0.0:
        # Nothing selected: a cutoff above every magnitude.
        var max_v = abs_update._get_float64(0)
        for i in range(1, n):
            var v = abs_update._get_float64(i)
            if v > max_v:
                max_v = v
        return max_v + 1.0
    if frac >= 1.0:
        return 0.0  # everything selected

    # Number of entries to select (at least 1).
    var k = Int(Float64(n) * frac)
    if k < 1:
        k = 1

    # The k-th largest magnitude is the threshold: for each candidate value, count
    # how many entries are >= it, and take the largest value whose count >= k.
    var threshold = 0.0
    var found = False
    for i in range(n):
        var cand = abs_update._get_float64(i)
        var count = 0
        for j in range(n):
            if abs_update._get_float64(j) >= cand:
                count += 1
        if count >= k:
            if not found or cand > threshold:
                threshold = cand
                found = True
    return threshold


def mgup_muon_step(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
    selected_fraction: Float64 = 0.25,
    select_scale: Float64 = 2.0,
    momentum_beta: Float64 = 0.95,
    weight_decay: Float64 = 0.01,
    ns_steps: Int = 5,
    nesterov: Bool = True,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single MGUP-Muon step — pure functional.

    Runs a standard Muon step, then amplifies the step-size of the selected fixed
    fraction of coordinates (those with the largest |ΔW|) by `select_scale`. With
    `selected_fraction == 0` or `select_scale == 1` this is exactly Muon. Returns
    new parameters and the new momentum buffer; the caller manages all state.

    Args:
        params: Model parameters to update (rank-2 matrix).
        gradients: Gradients of the loss with respect to params.
        momentum: Momentum buffer (use zeros_like(params) initially).
        learning_rate: Step size for the underlying Muon update.
        selected_fraction: Fraction of coordinates to amplify each step
            (default 0.25; clamped to [0, 1]).
        select_scale: Step-size multiplier applied to the selected coordinates
            (default 2.0).
        momentum_beta: Muon momentum decay (default 0.95).
        weight_decay: Muon weight-decay factor (default 0.01, Jordan recipe).
        ns_steps: Newton-Schulz iterations for orthogonalization (default 5).
        nesterov: If True, use Nesterov momentum (default True).

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If operation fails (shape/dtype mismatch, non-matrix params).
    """
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

    # The Muon update per coordinate.
    var update = subtract_simd(w_muon, params)

    # Fast exit: no selective amplification requested.
    if selected_fraction <= 0.0 or select_scale == 1.0:
        return (w_muon, new_momentum)

    # Magnitudes |dW|.
    var n = update.numel()
    var abs_update = zeros_like(update)
    for i in range(n):
        var v = update._get_float64(i)
        if v < 0:
            v = -v
        abs_update._set_float64(i, v)

    var threshold = _select_threshold(abs_update, selected_fraction)

    # Amplify the selected coordinates' step-size in place.
    var new_update = zeros_like(update)
    for i in range(n):
        var d = update._get_float64(i)
        if abs_update._get_float64(i) >= threshold:
            d = d * select_scale
        new_update._set_float64(i, d)

    var new_params = add_simd(params, new_update)
    return (new_params, new_momentum)


def mgup_muon_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified MGUP-Muon step with default hyperparameters.

    Uses selected_fraction=0.25, select_scale=2.0, and the Jordan Muon defaults.

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
    return mgup_muon_step(params, gradients, momentum, learning_rate)
