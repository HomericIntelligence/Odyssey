"""Gradient-check tests for the `variable_depthwise_conv2d` autograd op.

Verifies that the tape-recorded backward for depthwise convolution matches
finite-difference numerical gradients for grad_x and grad_weights, and the
analytic grad_bias, on a small multi-channel 4D input.

Depthwise conv applies one filter per input channel (no cross-channel mixing),
so a correct backward must keep each channel's gradient independent. The loss
is `sum(y * w)` with a NON-UNIFORM weight `w` so the numerical check exercises a
real (non-degenerate) gradient signal.
"""

from odyssey.autograd import Variable, GradientTape
from odyssey.autograd.variable import (
    variable_depthwise_conv2d,
    variable_multiply,
    variable_sum,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.conv import depthwise_conv2d
from odyssey.core.arithmetic import multiply
from odyssey.testing.gradient_checker import (
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
# NumericalForward closures: loss(perturbed) = sum(depthwise_conv2d(...) * w)
# ============================================================================
@fieldwise_init
struct _DWInputForward(NumericalForward):
    """loss(x) = sum(depthwise_conv2d(x, kernel, bias) * w). Perturbs x."""

    var kernel: AnyTensor
    var bias: AnyTensor
    var w: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, inp: AnyTensor) raises -> AnyTensor:
        var y = depthwise_conv2d(
            inp, self.kernel, self.bias, self.stride, self.padding
        )
        return multiply(y, self.w)


@fieldwise_init
struct _DWKernelForward(NumericalForward):
    """loss(kernel) = sum(depthwise_conv2d(x, kernel, bias) * w). Perturbs kernel.
    """

    var x: AnyTensor
    var bias: AnyTensor
    var w: AnyTensor
    var stride: Int
    var padding: Int

    def __call__(self, k: AnyTensor) raises -> AnyTensor:
        var y = depthwise_conv2d(
            self.x, k, self.bias, self.stride, self.padding
        )
        return multiply(y, self.w)


def test_depthwise_conv2d_grad_check() raises:
    print("[test] variable_depthwise_conv2d gradient check ...")

    # Small input: (batch=1, C=2, H=3, W=3); kernel (C=2, 1, 2, 2); stride 1 pad 0
    # => output (1, 2, 2, 2) = 8 elements.
    var x_shape: List[Int] = [1, 2, 3, 3]
    var k_shape: List[Int] = [2, 1, 2, 2]
    var b_shape: List[Int] = [2]
    var y_shape: List[Int] = [1, 2, 2, 2]
    var stride = 1
    var padding = 0

    var x_vals: List[Float64] = [
        0.5,
        -1.2,
        0.3,
        0.8,
        -0.4,
        1.1,
        0.2,
        -0.7,
        0.9,  # channel 0 (3x3)
        -0.1,
        1.4,
        -0.6,
        0.05,
        0.7,
        -1.3,
        0.35,
        0.6,
        -0.2,  # channel 1 (3x3)
    ]
    var k_vals: List[Float64] = [
        0.4,
        -0.3,
        0.7,
        0.1,  # channel-0 filter (2x2)
        -0.5,
        0.9,
        0.2,
        -0.8,  # channel-1 filter (2x2)
    ]
    var b_vals: List[Float64] = [0.15, -0.25]
    # Non-uniform loss weights over the 8-element output.
    var w_vals: List[Float64] = [1.0, 0.3, -0.5, 0.8, 0.2, -0.9, 1.1, 0.4]

    var tape = GradientTape()
    tape.enable()

    var x = Variable(_make_tensor(x_shape, x_vals), True, tape)
    var kernel = Variable(_make_tensor(k_shape, k_vals), True, tape)
    var bias = Variable(_make_tensor(b_shape, b_vals), True, tape)
    var w = Variable(_make_tensor(y_shape, w_vals), False, tape)

    var y = variable_depthwise_conv2d(x, kernel, bias, tape, stride, padding)

    # Output shape sanity.
    var ys = y.data.shape()
    if len(ys) != 4 or ys[0] != 1 or ys[1] != 2 or ys[2] != 2 or ys[3] != 2:
        raise Error("variable_depthwise_conv2d output shape mismatch")

    var weighted = variable_multiply(y, w, tape)
    var loss = variable_sum(weighted, tape, axis=-1)
    loss.backward(tape)

    var grad_x = tape.get_grad(x.id)
    var grad_k = tape.get_grad(kernel.id)
    var grad_b = tape.get_grad(bias.id)

    # ---- grad_bias is analytic: d/dbias_c sum(y*w) = sum of w over channel c ----
    # Output layout (n=1, c, h, w): channel c occupies elements [c*4 .. c*4+4).
    var w_t = _make_tensor(y_shape, w_vals)
    var expected_b0 = Float64(0.0)
    var expected_b1 = Float64(0.0)
    for i in range(4):
        expected_b0 += w_t._get_float64(0 * 4 + i)
        expected_b1 += w_t._get_float64(1 * 4 + i)
    var gb0 = Float64(grad_b._get_float64(0))
    var gb1 = Float64(grad_b._get_float64(1))
    if _abs(gb0 - expected_b0) > 1e-3 or _abs(gb1 - expected_b1) > 1e-3:
        raise Error(
            "grad_bias mismatch: got ("
            + String(gb0)
            + ", "
            + String(gb1)
            + "), expected ("
            + String(expected_b0)
            + ", "
            + String(expected_b1)
            + ")"
        )

    # ---- grad_x: finite-difference check ----
    var num_grad_x = compute_numerical_gradient(
        _DWInputForward(
            _make_tensor(k_shape, k_vals),
            _make_tensor(b_shape, b_vals),
            _make_tensor(y_shape, w_vals),
            stride,
            padding,
        ),
        _make_tensor(x_shape, x_vals),
        epsilon=1e-3,
    )
    assert_gradients_close(
        grad_x,
        num_grad_x,
        rtol=5e-2,
        atol=5e-4,
        message="variable_depthwise_conv2d grad_x",
    )

    # ---- grad_weights: finite-difference check ----
    var num_grad_k = compute_numerical_gradient(
        _DWKernelForward(
            _make_tensor(x_shape, x_vals),
            _make_tensor(b_shape, b_vals),
            _make_tensor(y_shape, w_vals),
            stride,
            padding,
        ),
        _make_tensor(k_shape, k_vals),
        epsilon=1e-3,
    )
    assert_gradients_close(
        grad_k,
        num_grad_k,
        rtol=5e-2,
        atol=5e-4,
        message="variable_depthwise_conv2d grad_weights",
    )

    print("       OK")


def main() raises:
    test_depthwise_conv2d_grad_check()
    print("\nvariable_depthwise_conv2d gradient check PASS")
