"""Gradient-check tests for the `variable_batch_norm` autograd op.

Verifies that the tape-recorded backward for batch normalization matches
finite-difference numerical gradients for grad_x and grad_gamma, and the
analytic grad_beta, on a small 4D input in training mode.

The loss is `sum(y * w)` with a NON-UNIFORM weight `w`: a uniform (ones)
grad_output makes the batch-norm input gradient collapse to ~0 by symmetry
(the per-channel mean-subtraction terms cancel), which would make a numerical
check vacuous. A non-uniform weight breaks that symmetry so grad_x is a real
signal to validate.
"""

from projectodyssey.autograd import Variable, GradientTape
from projectodyssey.autograd.variable import (
    variable_batch_norm,
    variable_multiply,
    variable_sum,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.core.normalization import batch_norm2d
from projectodyssey.core.arithmetic import multiply
from projectodyssey.testing.gradient_checker import (
    compute_numerical_gradient,
    assert_gradients_close,
    NumericalForward,
)


def _make_tensor(shape: List[Int], values: List[Float64]) raises -> AnyTensor:
    var t = zeros(shape, DType.float32)
    for i in range(len(values)):
        t._set_float64(i, values[i])
    return t^


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


# ============================================================================
# NumericalForward: loss(x) = sum(batch_norm2d(x, gamma, beta) * w)
# (used to compute the numerical gradient wrt whichever tensor is perturbed —
#  x when checking grad_x, gamma when checking grad_gamma)
# ============================================================================
@fieldwise_init
struct _BNInputForward(NumericalForward):
    """loss(x) = sum(batch_norm2d(x, gamma, beta, ...) * w). Perturbs x."""

    var gamma: AnyTensor
    var beta: AnyTensor
    var running_mean: AnyTensor
    var running_var: AnyTensor
    var w: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            inp,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            training=True,
            epsilon=1e-5,
        )
        return multiply(res[0], self.w)


@fieldwise_init
struct _BNGammaForward(NumericalForward):
    """loss(gamma) = sum(batch_norm2d(x, gamma, beta, ...) * w). Perturbs gamma.
    """

    var x: AnyTensor
    var beta: AnyTensor
    var running_mean: AnyTensor
    var running_var: AnyTensor
    var w: AnyTensor

    def __call__(self, g: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            self.x,
            g,
            self.beta,
            self.running_mean,
            self.running_var,
            training=True,
            epsilon=1e-5,
        )
        return multiply(res[0], self.w)


def test_batch_norm_grad_check() raises:
    print("[test] variable_batch_norm gradient check ...")

    # Small 4D input: (batch=2, C=2, H=2, W=2) = 16 elements; params are (C,).
    var shape: List[Int] = [2, 2, 2, 2]
    var x_vals: List[Float64] = [
        0.5,
        -1.2,
        0.3,
        0.8,
        -0.4,
        1.1,
        0.2,
        -0.7,
        0.9,
        -0.1,
        1.4,
        -0.6,
        0.05,
        0.7,
        -1.3,
        0.35,
    ]
    var gamma_vals: List[Float64] = [1.3, 0.7]
    var beta_vals: List[Float64] = [0.2, -0.4]
    # Non-uniform loss weights (break the ones-vector symmetry).
    var w_vals: List[Float64] = [
        1.0,
        0.3,
        -0.5,
        0.8,
        0.2,
        -0.9,
        1.1,
        0.4,
        -0.3,
        0.6,
        0.9,
        -0.2,
        0.7,
        -0.6,
        0.15,
        1.2,
    ]
    var c_shape: List[Int] = [2]
    var rm_vals: List[Float64] = [0.0, 0.0]
    var rv_vals: List[Float64] = [1.0, 1.0]

    var tape = GradientTape()
    tape.enable()

    var x = Variable(_make_tensor(shape, x_vals), True, tape)
    var gamma = Variable(_make_tensor(c_shape, gamma_vals), True, tape)
    var beta = Variable(_make_tensor(c_shape, beta_vals), True, tape)
    var running_mean = _make_tensor(c_shape, rm_vals)
    var running_var = _make_tensor(c_shape, rv_vals)
    var w = Variable(_make_tensor(shape, w_vals), False, tape)

    var bn = variable_batch_norm(
        x, gamma, beta, running_mean, running_var, tape, training=True
    )
    var y = bn[0].copy()
    var new_rm = bn[1].copy()

    # Output shape matches input.
    var y_shape = y.data.shape()
    if len(y_shape) != 4 or y_shape[0] != 2 or y_shape[1] != 2:
        raise Error("variable_batch_norm output shape mismatch")

    # Running stats were updated in training mode (should differ from inputs).
    var rm_changed = (
        _abs(new_rm._get_float64(0) - 0.0) > 1e-9
        or _abs(new_rm._get_float64(1) - 0.0) > 1e-9
    )
    if not rm_changed:
        raise Error("running_mean was not updated in training mode")

    # loss = sum(y * w)  (non-uniform w breaks BN symmetry)
    var weighted = variable_multiply(y, w, tape)
    var loss = variable_sum(weighted, tape, axis=-1)
    loss.backward(tape)

    var grad_x = tape.get_grad(x.id)
    var grad_gamma = tape.get_grad(gamma.id)
    var grad_beta = tape.get_grad(beta.id)

    # ---- grad_beta is analytic: d/dbeta_c sum(y*w) = sum over (n,h,w) of w ----
    # For each channel c, grad_beta[c] = sum of w over all elements in channel c.
    var w_t = _make_tensor(shape, w_vals)
    var expected_beta0 = Float64(0.0)
    var expected_beta1 = Float64(0.0)
    # layout (n, c, h, w): channel index is the 2nd dim (stride H*W=4 within n).
    for n in range(2):
        for h in range(2):
            for ww in range(2):
                var base = n * (2 * 2 * 2)
                var idx0 = base + 0 * (2 * 2) + h * 2 + ww
                var idx1 = base + 1 * (2 * 2) + h * 2 + ww
                expected_beta0 += w_t._get_float64(idx0)
                expected_beta1 += w_t._get_float64(idx1)
    var gb0 = Float64(grad_beta._get_float64(0))
    var gb1 = Float64(grad_beta._get_float64(1))
    if _abs(gb0 - expected_beta0) > 1e-3 or _abs(gb1 - expected_beta1) > 1e-3:
        raise Error(
            "grad_beta mismatch: got ("
            + String(gb0)
            + ", "
            + String(gb1)
            + "), expected ("
            + String(expected_beta0)
            + ", "
            + String(expected_beta1)
            + ")"
        )

    # ---- grad_x: finite-difference check ----
    var num_grad_x = compute_numerical_gradient(
        _BNInputForward(
            _make_tensor(c_shape, gamma_vals),
            _make_tensor(c_shape, beta_vals),
            _make_tensor(c_shape, rm_vals),
            _make_tensor(c_shape, rv_vals),
            _make_tensor(shape, w_vals),
        ),
        _make_tensor(shape, x_vals),
        epsilon=1e-3,
    )
    assert_gradients_close(
        grad_x,
        num_grad_x,
        rtol=5e-2,
        atol=5e-4,
        message="variable_batch_norm grad_x",
    )

    # ---- grad_gamma: finite-difference check ----
    var num_grad_gamma = compute_numerical_gradient(
        _BNGammaForward(
            _make_tensor(shape, x_vals),
            _make_tensor(c_shape, beta_vals),
            _make_tensor(c_shape, rm_vals),
            _make_tensor(c_shape, rv_vals),
            _make_tensor(shape, w_vals),
        ),
        _make_tensor(c_shape, gamma_vals),
        epsilon=1e-3,
    )
    assert_gradients_close(
        grad_gamma,
        num_grad_gamma,
        rtol=5e-2,
        atol=5e-4,
        message="variable_batch_norm grad_gamma",
    )

    print("       OK")


@fieldwise_init
struct _BNInferForward(NumericalForward):
    """Inference-mode loss(x) = sum(batch_norm2d(x, ..., training=False) * w).
    """

    var gamma: AnyTensor
    var beta: AnyTensor
    var running_mean: AnyTensor
    var running_var: AnyTensor
    var w: AnyTensor

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        var res = batch_norm2d(
            inp,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            training=False,
            epsilon=1e-5,
        )
        return multiply(res[0], self.w)


def test_batch_norm_inference_grad_check() raises:
    """Verify the training=False path: BN uses running stats, and the taped
    backward still matches finite differences for grad_x."""
    print("[test] variable_batch_norm inference-mode gradient check ...")

    var shape: List[Int] = [2, 2, 2, 2]
    var x_vals: List[Float64] = [
        0.5,
        -1.2,
        0.3,
        0.8,
        -0.4,
        1.1,
        0.2,
        -0.7,
        0.9,
        -0.1,
        1.4,
        -0.6,
        0.05,
        0.7,
        -1.3,
        0.35,
    ]
    var gamma_vals: List[Float64] = [1.3, 0.7]
    var beta_vals: List[Float64] = [0.2, -0.4]
    var w_vals: List[Float64] = [
        1.0,
        0.3,
        -0.5,
        0.8,
        0.2,
        -0.9,
        1.1,
        0.4,
        -0.3,
        0.6,
        0.9,
        -0.2,
        0.7,
        -0.6,
        0.15,
        1.2,
    ]
    var c_shape: List[Int] = [2]
    # Non-trivial running stats (inference uses these, NOT batch stats).
    var rm_vals: List[Float64] = [0.1, -0.2]
    var rv_vals: List[Float64] = [1.5, 0.8]

    var tape = GradientTape()
    tape.enable()

    var x = Variable(_make_tensor(shape, x_vals), True, tape)
    var gamma = Variable(_make_tensor(c_shape, gamma_vals), True, tape)
    var beta = Variable(_make_tensor(c_shape, beta_vals), True, tape)
    var running_mean = _make_tensor(c_shape, rm_vals)
    var running_var = _make_tensor(c_shape, rv_vals)
    var w = Variable(_make_tensor(shape, w_vals), False, tape)

    var bn = variable_batch_norm(
        x, gamma, beta, running_mean, running_var, tape, training=False
    )
    var y = bn[0].copy()

    var weighted = variable_multiply(y, w, tape)
    var loss = variable_sum(weighted, tape, axis=-1)
    loss.backward(tape)

    var grad_x = tape.get_grad(x.id)
    var num_grad_x = compute_numerical_gradient(
        _BNInferForward(
            _make_tensor(c_shape, gamma_vals),
            _make_tensor(c_shape, beta_vals),
            _make_tensor(c_shape, rm_vals),
            _make_tensor(c_shape, rv_vals),
            _make_tensor(shape, w_vals),
        ),
        _make_tensor(shape, x_vals),
        epsilon=1e-3,
    )
    assert_gradients_close(
        grad_x,
        num_grad_x,
        rtol=5e-2,
        atol=5e-4,
        message="variable_batch_norm inference grad_x",
    )
    print("       OK")


def main() raises:
    test_batch_norm_grad_check()
    test_batch_norm_inference_grad_check()
    print("\nvariable_batch_norm gradient check PASS")
