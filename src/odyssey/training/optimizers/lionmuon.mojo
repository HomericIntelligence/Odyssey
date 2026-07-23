"""LionMuon optimizer — alternating Lion / Muon on a fixed period.

Alternates the Lion update (Chen et al., 2023) and the Muon update (Jordan et al.,
2024) on a fixed period `P`, with SEPARATE per-rule momentum buffers. The expensive
Muon step (Newton-Schulz orthogonalization) runs once every `P` steps; the cheap
sign-based Lion step runs on the other `P - 1` steps. This cuts average per-step
compute versus pure Muon while retaining Muon's conditioning benefit at the
periodic refresh.

Schedule (0-indexed step):

    if step_index % period == 0:  Muon update  (reads/writes muon_momentum only)
    else:                         Lion update  (reads/writes lion_momentum only)

Each parent keeps its own momentum buffer, so each buffer retains its own rule's
semantics across the alternation: Lion's buffer is a gradient EMA
(lion_beta2 * m + (1 - lion_beta2) * g, which stays O(||g||)), while Muon's buffer
is a heavy-ball accumulator (muon_beta * m + g, which converges to
~||g|| / (1 - muon_beta) at steady state — ~20x ||g|| at the default beta). Sharing
one buffer between the two rules would let Muon's large-scale accumulator corrupt
Lion's sign(beta1 * m + (1 - beta1) * g) steps after every Muon step, so the
buffers are deliberately kept separate. The step that does not run passes its
buffer through unchanged.

[novel — unvalidated] The alternation itself is a novel combination, not a
published algorithm. The "conditioning benefit at the periodic refresh" is a
heuristic expectation, NOT an established result. Treat this optimizer as
experimental until an ablation validates that the alternation helps.

Reference:
    Lion: Chen, X., Liang, C., Huang, D., et al. (2023). Symbolic Discovery of
    Optimization Algorithms. arXiv:2302.06675.
    Muon: Jordan, K., Jin, Y., Boza, V., et al. (2024). Muon: An optimizer for the
    hidden layers of neural networks. https://kellerjordan.github.io/posts/muon/
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.training.optimizers.lion import lion_step
from odyssey.training.optimizers.muon import muon_step


def lionmuon_step(
    params: AnyTensor,
    gradients: AnyTensor,
    lion_momentum: AnyTensor,
    muon_momentum: AnyTensor,
    learning_rate: Float64,
    step_index: Int,
    period: Int = 4,
    lion_beta1: Float64 = 0.9,
    lion_beta2: Float64 = 0.99,
    muon_beta: Float64 = 0.95,
    weight_decay: Float64 = 0.0,
    ns_steps: Int = 5,
    nesterov: Bool = True,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Perform a single LionMuon step — pure functional.

    Dispatches to Muon on steps where `step_index % period == 0` and to Lion
    otherwise. Each branch reads and writes ONLY its own momentum buffer; the
    other buffer is returned unchanged. Returns new parameters and both new
    momentum buffers; the caller manages all state, including `step_index`
    (increment it once per call).

    Args:
        params: Model parameters to update (rank-2 matrix — required by the Muon
            branch; the Lion branch accepts any shape but the shared schedule means
            LionMuon is used on matrix parameters).
        gradients: Gradients of the loss with respect to params.
        lion_momentum: Lion's gradient-EMA momentum buffer (use
            zeros_like(params) initially). Only touched on Lion steps.
        muon_momentum: Muon's heavy-ball momentum buffer (use
            zeros_like(params) initially). Only touched on Muon steps.
        learning_rate: Step size (applies to whichever parent runs this step).
        step_index: 0-indexed global step counter (caller-incremented).
        period: Muon runs when step_index % period == 0; Lion otherwise
            (default 4 — Muon once every four steps).
        lion_beta1: Lion's momentum-update beta (default 0.9).
        lion_beta2: Lion's momentum-accumulation beta (default 0.99).
        muon_beta: Muon's momentum decay (default 0.95).
        weight_decay: Weight-decay factor passed to whichever parent runs
            (default 0.0). NOTE this default 0.0 is forwarded to the Muon branch
            too, overriding muon_step's own 0.01 default (the Jordan recipe) — so
            LionMuon applies NO weight decay unless you pass it explicitly, unlike
            muon_step_simple.
        ns_steps: Newton-Schulz iterations for the Muon branch (default 5).
        nesterov: Nesterov momentum for the Muon branch (default True).

    Returns:
        Tuple of (new_params, new_lion_momentum, new_muon_momentum). Exactly one
        of the two momentum buffers changes per step; the other is passed
        through unchanged.

    Raises:
        Error: If period <= 0, or if the underlying step fails (e.g. the Muon
            branch on a non-matrix parameter).
    """
    if period <= 0:
        raise Error("LionMuon: period must be positive")

    if step_index % period == 0:
        var (new_params, new_muon_m) = muon_step(
            params,
            gradients,
            muon_momentum,
            learning_rate,
            muon_beta,
            weight_decay,
            ns_steps,
            nesterov,
        )
        return (new_params, lion_momentum, new_muon_m)
    else:
        var (new_params, new_lion_m) = lion_step(
            params,
            gradients,
            lion_momentum,
            learning_rate,
            lion_beta1,
            lion_beta2,
            weight_decay,
        )
        return (new_params, new_lion_m, muon_momentum)


def lionmuon_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    lion_momentum: AnyTensor,
    muon_momentum: AnyTensor,
    learning_rate: Float64,
    step_index: Int,
) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    """Simplified LionMuon step with default hyperparameters.

    Uses period=4 and both parents' default betas/weight-decay.

    Args:
        params: Model parameters to update (rank-2 matrix).
        gradients: Gradients of the loss with respect to params.
        lion_momentum: Lion's gradient-EMA momentum buffer.
        muon_momentum: Muon's heavy-ball momentum buffer.
        learning_rate: Step size.
        step_index: 0-indexed global step counter (caller-incremented).

    Returns:
        Tuple of (new_params, new_lion_momentum, new_muon_momentum).

    Raises:
        Error: If operation fails.
    """
    return lionmuon_step(
        params,
        gradients,
        lion_momentum,
        muon_momentum,
        learning_rate,
        step_index,
    )


def init_lionmuon_state(
    params_list: List[AnyTensor],
    *,
    force_f64: Bool = False,
) raises -> List[List[AnyTensor]]:
    """Allocate per-parameter state buffers for the lionmuon optimizer.

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
