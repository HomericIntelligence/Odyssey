"""Behavioral tests for the experimental Sophia curvature-estimator surface.

Phase 1 exposes the experimental trait from ``odyssey.autograd`` while keeping
the temporary fail-fast implementation at ``odyssey.autograd.sophia_g``.
The estimator receives derivative-connected ``Variable`` objects and the
dedicated ``GradientTape`` that recorded the scalar objective, plus its batch
size and requested result dtypes.
"""

from std.collections import List

from odyssey.autograd import (
    DiagonalHessianEstimator,
    GradientTape,
    Variable,
    variable_sum,
)
from odyssey.autograd.sophia_g import (
    PlaceholderDiagonalHessianEstimator,
)
from odyssey.tensor.tensor_creation import zeros


def _accept_as_estimator[
    Estimator: DiagonalHessianEstimator
](mut estimator: Estimator):
    """Require concrete conformance to the public trait at compile time."""
    _ = estimator


def test_public_trait_and_inner_placeholder_imports() raises:
    """The public trait and inner-only temporary stub use their intended paths.
    """
    var estimator = PlaceholderDiagonalHessianEstimator()
    _accept_as_estimator[PlaceholderDiagonalHessianEstimator](estimator)

    if estimator.name() != "PlaceholderDiagonalHessianEstimator":
        raise Error("unexpected placeholder estimator name")


def _one_result_dtype(dtype: DType) -> List[DType]:
    var result = List[DType]()
    result.append(dtype)
    return result^


def _expect_context_error(
    parameters: List[Variable],
    objective: Variable,
    mut estimator_tape: GradientTape,
    batch_size: Int,
    result_dtypes: List[DType],
    expected: String,
) raises:
    var estimator = PlaceholderDiagonalHessianEstimator()
    var raised = False
    try:
        _ = estimator.estimate_diag_hessian(
            parameters,
            objective,
            estimator_tape,
            batch_size,
            result_dtypes,
        )
    except error:
        raised = True
        var message = String(error)
        if expected not in message:
            raise Error(
                "expected placeholder error containing '"
                + expected
                + "'; got: "
                + message
            )
    if not raised:
        raise Error("placeholder estimate_diag_hessian must fail")


def test_placeholder_failure_is_observable() raises:
    """A valid context reaches the actionable not-implemented diagnostic."""
    var estimator_tape = GradientTape()
    estimator_tape.enable()

    var parameter = Variable(zeros([2], DType.float32), True, estimator_tape)
    var objective = variable_sum(parameter, estimator_tape)
    var parameters = List[Variable]()
    parameters.append(parameter^)
    # The requested result precision is independent of parameter precision.
    var result_dtypes = _one_result_dtype(DType.float64)

    var estimator = PlaceholderDiagonalHessianEstimator()
    var raised = False
    try:
        _ = estimator.estimate_diag_hessian(
            parameters,
            objective,
            estimator_tape,
            batch_size=2,
            result_dtypes=result_dtypes,
        )
    except error:
        raised = True
        var message = String(error)
        if (
            "Sophia-G" not in message
            or "not implemented" not in message
            or "#5717" not in message
            or "autograd/TODO.md" not in message
        ):
            raise Error(
                "placeholder error must identify the unavailable Sophia-G "
                "implementation and its TODO; got: "
                + message
            )

    if not raised:
        raise Error("placeholder estimate_diag_hessian must fail")


def test_batch_size_must_be_positive() raises:
    var tape = GradientTape()
    tape.enable()
    var parameter = Variable(zeros([1], DType.float32), True, tape)
    var objective = variable_sum(parameter, tape)
    var parameters = List[Variable]()
    parameters.append(parameter^)
    _expect_context_error(
        parameters,
        objective,
        tape,
        0,
        _one_result_dtype(DType.float32),
        "batch_size must be positive",
    )


def test_result_dtype_count_must_match_parameters() raises:
    var tape = GradientTape()
    tape.enable()
    var parameter = Variable(zeros([1], DType.float32), True, tape)
    var objective = variable_sum(parameter, tape)
    var parameters = List[Variable]()
    parameters.append(parameter^)
    _expect_context_error(
        parameters,
        objective,
        tape,
        1,
        List[DType](),
        "result_dtypes length must match parameters",
    )


def test_objective_must_be_scalar() raises:
    var tape = GradientTape()
    tape.enable()
    var parameter = Variable(zeros([2], DType.float32), True, tape)
    var objective = Variable(zeros([2], DType.float32), True, tape)
    var parameters = List[Variable]()
    parameters.append(parameter^)
    _expect_context_error(
        parameters,
        objective,
        tape,
        2,
        _one_result_dtype(DType.float32),
        "objective must be scalar",
    )


def test_parameters_must_require_grad() raises:
    var tape = GradientTape()
    tape.enable()
    var parameter = Variable(zeros([1], DType.float32), False, tape)
    var objective = variable_sum(parameter, tape)
    var parameters = List[Variable]()
    parameters.append(parameter^)
    _expect_context_error(
        parameters,
        objective,
        tape,
        1,
        _one_result_dtype(DType.float32),
        "parameter[0] must require gradients",
    )


def test_objective_must_require_grad() raises:
    var tape = GradientTape()
    tape.enable()
    var parameter = Variable(zeros([1], DType.float32), True, tape)
    var objective = Variable(zeros([1], DType.float32), False, tape)
    var parameters = List[Variable]()
    parameters.append(parameter^)
    _expect_context_error(
        parameters,
        objective,
        tape,
        1,
        _one_result_dtype(DType.float32),
        "objective must require gradients",
    )


def test_estimator_tape_must_have_clean_gradient_registry() raises:
    var tape = GradientTape()
    tape.enable()
    var parameter = Variable(zeros([1], DType.float32), True, tape)
    var objective = variable_sum(parameter, tape)
    tape.registry.set_grad(parameter.id, zeros([1], DType.float32))
    var parameters = List[Variable]()
    parameters.append(parameter^)
    _expect_context_error(
        parameters,
        objective,
        tape,
        1,
        _one_result_dtype(DType.float32),
        "dedicated estimator tape must not contain pre-existing gradients",
    )


def test_parameters_must_be_registered_on_estimator_tape() raises:
    var estimator_tape = GradientTape()
    estimator_tape.enable()
    var local_parameter = Variable(
        zeros([1], DType.float32), True, estimator_tape
    )
    var objective = variable_sum(local_parameter, estimator_tape)

    var foreign_tape = GradientTape()
    _ = foreign_tape.register_variable(True)
    _ = foreign_tape.register_variable(True)
    var foreign_parameter = Variable(
        zeros([1], DType.float32), True, foreign_tape
    )
    var parameters = List[Variable]()
    parameters.append(foreign_parameter^)
    _expect_context_error(
        parameters,
        objective,
        estimator_tape,
        1,
        _one_result_dtype(DType.float32),
        "parameter[0] is not registered on estimator_tape",
    )


def test_objective_must_be_registered_on_estimator_tape() raises:
    var estimator_tape = GradientTape()
    estimator_tape.enable()
    var parameter = Variable(zeros([1], DType.float32), True, estimator_tape)

    var foreign_tape = GradientTape()
    _ = foreign_tape.register_variable(True)
    var foreign_parameter = Variable(
        zeros([1], DType.float32), True, foreign_tape
    )
    var foreign_objective = variable_sum(foreign_parameter, foreign_tape)

    var parameters = List[Variable]()
    parameters.append(parameter^)
    _expect_context_error(
        parameters,
        foreign_objective,
        estimator_tape,
        1,
        _one_result_dtype(DType.float32),
        "objective is not registered on estimator_tape",
    )


def main() raises:
    test_public_trait_and_inner_placeholder_imports()
    test_placeholder_failure_is_observable()
    test_batch_size_must_be_positive()
    test_result_dtype_count_must_match_parameters()
    test_objective_must_be_scalar()
    test_parameters_must_require_grad()
    test_objective_must_require_grad()
    test_estimator_tape_must_have_clean_gradient_registry()
    test_parameters_must_be_registered_on_estimator_tape()
    test_objective_must_be_registered_on_estimator_tape()
    print("Sophia curvature-estimator Phase-1 tests PASSED")
