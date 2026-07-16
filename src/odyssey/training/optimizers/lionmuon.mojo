"""LionMuon optimizer — alternating Lion / Muon on a fixed period.

Alternates the Lion update (Chen et al., 2023) and the Muon update (Jordan et al.,
2024) on a fixed period `P`, sharing a SINGLE momentum buffer between them. The
expensive Muon step (Newton-Schulz orthogonalization) runs once every `P` steps; the
cheap sign-based Lion step runs on the other `P - 1` steps. This cuts average
per-step compute versus pure Muon while retaining Muon's conditioning benefit at the
periodic refresh.

Schedule (0-indexed step):

    if step_index % period == 0:  Muon update
    else:                         Lion update

Both parents consume and return the same one-tensor momentum buffer, so the buffer
carries continuously across the alternation with no extra state.

[novel — unvalidated] The alternation itself and the SHARED momentum buffer are a
novel combination, not a published algorithm. Note the two branches write the buffer
under different laws — Muon uses a heavy-ball accumulator (muon_beta*m + g) while Lion
uses an EMA (lion_beta2*m + (1-lion_beta2)*g) — so at each alternation boundary the
buffer's scale/meaning is reinterpreted by the other rule. The "conditioning benefit
at the periodic refresh" is a heuristic expectation, NOT an established result; the
shared-buffer semantics are unbenchmarked here. Treat this optimizer as experimental
until an ablation validates that the shared buffer helps rather than harms.

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
    momentum: AnyTensor,
    learning_rate: Float64,
    step_index: Int,
    period: Int = 4,
    lion_beta1: Float64 = 0.9,
    lion_beta2: Float64 = 0.99,
    muon_beta: Float64 = 0.95,
    weight_decay: Float64 = 0.0,
    ns_steps: Int = 5,
    nesterov: Bool = True,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Perform a single LionMuon step — pure functional.

    Dispatches to Muon on steps where `step_index % period == 0` and to Lion
    otherwise, threading the shared momentum buffer through either parent. Returns
    new parameters and the new momentum buffer; the caller manages all state,
    including `step_index` (increment it once per call).

    Args:
        params: Model parameters to update (rank-2 matrix — required by the Muon
            branch; the Lion branch accepts any shape but the shared schedule means
            LionMuon is used on matrix parameters).
        gradients: Gradients of the loss with respect to params.
        momentum: Shared momentum buffer (use zeros_like(params) initially).
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
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If period <= 0, or if the underlying step fails (e.g. the Muon
            branch on a non-matrix parameter).
    """
    if period <= 0:
        raise Error("LionMuon: period must be positive")

    if step_index % period == 0:
        return muon_step(
            params,
            gradients,
            momentum,
            learning_rate,
            muon_beta,
            weight_decay,
            ns_steps,
            nesterov,
        )
    else:
        return lion_step(
            params,
            gradients,
            momentum,
            learning_rate,
            lion_beta1,
            lion_beta2,
            weight_decay,
        )


def lionmuon_step_simple(
    params: AnyTensor,
    gradients: AnyTensor,
    momentum: AnyTensor,
    learning_rate: Float64,
    step_index: Int,
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Simplified LionMuon step with default hyperparameters.

    Uses period=4 and both parents' default betas/weight-decay.

    Args:
        params: Model parameters to update (rank-2 matrix).
        gradients: Gradients of the loss with respect to params.
        momentum: Shared momentum buffer.
        learning_rate: Step size.
        step_index: 0-indexed global step counter (caller-incremented).

    Returns:
        Tuple of (new_params, new_momentum).

    Raises:
        Error: If operation fails.
    """
    return lionmuon_step(params, gradients, momentum, learning_rate, step_index)
