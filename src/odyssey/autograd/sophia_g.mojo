"""Experimental diagonal-curvature estimator contract for Sophia.

The contract is derivative-context aware: callers provide parameter
``Variable`` objects, a scalar objective ``Variable`` built from those
parameters, a dedicated ``GradientTape`` that recorded that objective, the
objective's batch size, and one requested result dtype per parameter.
Detached ``AnyTensor`` values are insufficient because they do not retain the
graph needed to differentiate with respect to the parameters.

Sophia has two distinct estimator families:

* Sophia-G uses the Gauss-Newton-Bartlett (GNB) estimator.  The caller builds
  the sampled-label negative-log-likelihood objective on the supplied tape;
  the estimator takes one ordinary backward pass and returns the batch-scaled
  squared parameter gradients.  Sophia-G does not require an HVP or JVP.
* Sophia-H uses Hutchinson's estimator and does require a Hessian-vector
  product.  That backend remains dependent on higher-order autograd support.

Phase 1, PR #5719, exposes only this experimental trait and an inner-module
placeholder.  It does not wire an estimator into the Sophia optimizer and
does not close the end-to-end work in #5683.
"""

from std.collections import List

from odyssey.autograd.tape import GradientTape
from odyssey.autograd.variable import Variable
from odyssey.tensor.any_tensor import AnyTensor


def _validate_estimator_context(
    parameters: List[Variable],
    objective: Variable,
    estimator_tape: GradientTape,
    batch_size: Int,
    result_dtypes: List[DType],
) raises:
    """Validate the Phase-1 preconditions observable through today's tape API.
    """
    if batch_size <= 0:
        raise Error("batch_size must be positive")
    if len(result_dtypes) != len(parameters):
        raise Error("result_dtypes length must match parameters")
    if objective.data.numel() != 1:
        raise Error("objective must be scalar")

    for i in range(len(parameters)):
        ref parameter = parameters[i]
        if not parameter.requires_grad:
            raise Error("parameter[" + String(i) + "] must require gradients")
        if (
            parameter.id < 0
            or parameter.id >= estimator_tape.registry.next_id
            or not estimator_tape.registry.requires_grad[parameter.id]
        ):
            raise Error(
                "parameter["
                + String(i)
                + "] is not registered on estimator_tape"
            )

    if not objective.requires_grad:
        raise Error("objective must require gradients")
    if (
        objective.id < 0
        or objective.id >= estimator_tape.registry.next_id
        or not estimator_tape.registry.requires_grad[objective.id]
    ):
        raise Error("objective is not registered on estimator_tape")

    for variable_id in range(estimator_tape.registry.next_id):
        if estimator_tape.registry.has_gradient(variable_id):
            raise Error(
                "dedicated estimator tape must not contain pre-existing "
                "gradients"
            )


trait DiagonalHessianEstimator:
    """Estimate one diagonal curvature tensor for each parameter.

    The input context preserves the derivative relationship, scaling, result
    precision, and gradient lifecycle that a concrete implementation needs:

    * ``parameters`` contains the differentiable ``Variable`` objects.
    * ``objective`` is a scalar ``Variable`` derived from those parameters.
    * ``estimator_tape`` is a dedicated tape that registered the parameters
      and recorded the operations producing ``objective``.
    * ``batch_size`` is the number of samples reduced into ``objective``.
    * ``result_dtypes`` has one requested output dtype per parameter.  It is
      normally ``DType.float32`` or the exact dtype of the corresponding
      Sophia Hessian-moment buffer, including ``DType.float64`` when
      ``force_f64`` state is active.

    Preconditions:

    1. ``batch_size > 0``.
    2. ``objective.data.numel() == 1``.
    3. Every parameter and the objective has ``requires_grad=True``.
    4. Every parameter and the objective was registered on
       ``estimator_tape``.
    5. ``len(result_dtypes) == len(parameters)``.
    6. The dedicated tape has no pre-existing gradients.  It must not be the
       ordinary training tape.

    ``Variable`` currently stores a numeric registry ID but no owning-tape
    identity.  Runtime validation can therefore check that each ID exists in
    the supplied registry and agrees on ``requires_grad``, but cannot detect a
    colliding ID from another tape.  Constructing all supplied Variables on
    ``estimator_tape`` remains a caller precondition until tape ownership is
    represented in the API.

    For Sophia-G, ``objective`` is the mean negative log-likelihood formed
    with labels sampled from the model distribution.  A concrete GNB
    implementation may run ``objective.backward(estimator_tape)``, read each
    parameter gradient from the tape, square it element-wise, and apply the
    batch-size scale ``batch_size * gradient^2`` from the Sophia paper.  For a
    future Sophia-H implementation, ``objective`` is the ordinary mini-batch
    loss and the tape must support the required Hessian-vector product.

    Result invariants:

    1. ``len(result) == len(parameters)``.
    2. ``result[i].shape() == parameters[i].data.shape()``.
    3. ``result[i].dtype() == result_dtypes[i]``.
    4. Every result value is finite.
    5. Implementations do not mutate parameter or objective data.  They may
       mutate their own estimator state and the tape's gradient registry.
    6. At return, the dedicated registry contains only gradients produced by
       the estimator backward pass.  The caller consumes the estimates and
       then clears or discards this tape.  Ordinary training gradients live on
       a different tape, so they are neither accumulated over nor destroyed.

    This surface is experimental until a real #5683 estimator and optimizer
    wiring exercise it end to end.
    """

    def estimate_diag_hessian(
        mut self,
        parameters: List[Variable],
        objective: Variable,
        mut estimator_tape: GradientTape,
        batch_size: Int,
        result_dtypes: List[DType],
    ) raises -> List[AnyTensor]:
        """Return derivative-connected diagonal curvature estimates."""
        ...


struct PlaceholderDiagonalHessianEstimator(DiagonalHessianEstimator):
    """Inner-module-only Phase-1 stub that always fails when invoked.

    The stub demonstrates trait conformance and provides an executable
    diagnostic for explicit experiments.  Nothing in Odyssey's Sophia
    dispatch selects this type in Phase 1.
    """

    var estimator_kind: String

    def __init__(out self):
        self.estimator_kind = "PlaceholderDiagonalHessianEstimator"

    def estimate_diag_hessian(
        mut self,
        parameters: List[Variable],
        objective: Variable,
        mut estimator_tape: GradientTape,
        batch_size: Int,
        result_dtypes: List[DType],
    ) raises -> List[AnyTensor]:
        """Fail because Phase 1 contains no Sophia-G implementation."""
        _validate_estimator_context(
            parameters,
            objective,
            estimator_tape,
            batch_size,
            result_dtypes,
        )
        var tape_state = "enabled" if estimator_tape.enabled else "disabled"
        raise Error(
            self.estimator_kind
            + ".estimate_diag_hessian called, but Sophia-G is not implemented. "
            "PR #5719 exposes only an experimental derivative-context-aware "
            "trait; follow #5717 under parent #5683 and "
            "src/odyssey/autograd/TODO.md for sampled-label objective "
            "plumbing, the GNB gradient-square estimator, optimizer wiring, "
            "and end-to-end tests. (parameters="
            + String(len(parameters))
            + ", objective_id="
            + String(objective.id)
            + ", tape="
            + tape_state
            + ", batch_size="
            + String(batch_size)
            + ", result_dtypes="
            + String(len(result_dtypes))
            + ")"
        )

    def name(self) -> String:
        """Return the concrete placeholder identifier."""
        return self.estimator_kind
