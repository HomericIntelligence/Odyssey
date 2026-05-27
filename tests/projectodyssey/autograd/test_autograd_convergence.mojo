"""Convergence tests for the tape-based autograd substrate.

The point of autograd is *training* — backward + step + writeback must
actually reduce loss across iterations. The existing `test_variable_layers`
suite checks that gradients exist and match analytical values on tiny
tensors, but it does NOT verify that a full forward → backward → step
loop reduces loss across iterations.

This file adds convergence-style tests that would have caught the LeNet
autograd port silently training to chance accuracy:

  test_linear_softmax_converges       — tiny MLP + cross-entropy
  test_conv2d_softmax_converges       — conv + flatten + linear + CE
  test_lenet_shape_converges          — full LeNet-shaped network on
                                        a 2-sample synthetic batch

Each test asserts loss decreases monotonically (or at least by >=50%
across N steps), guaranteeing the whole tape→registry→optimizer pipeline
moves parameters in a direction that reduces loss for that op stack.

Run with: pixi run mojo run -I src tests/projectodyssey/autograd/test_autograd_convergence.mojo
"""

from projectodyssey.autograd import (
    Variable,
    GradientTape,
    SGD,
    variable_conv2d,
    variable_relu,
    variable_maxpool2d,
    variable_flatten,
    variable_linear,
    variable_cross_entropy,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros


def _make_tensor(shape: List[Int], fill: Float64) raises -> AnyTensor:
    var t = zeros(shape, DType.float32)
    var n = 1
    for d in shape:
        n *= d
    for i in range(n):
        t._set_float64(i, fill)
    return t^


def _make_asymmetric(
    shape: List[Int], base: Float64, slope: Float64
) raises -> AnyTensor:
    """Build a tensor with `t[i] = base + i * slope`.

    Symmetric (uniform) weight init creates a mathematical pathology where
    every output of a linear/conv layer is identical, the softmax distribution
    is uniform, and the gradient back to the input is algebraically zero
    (catastrophic-cancellation noise only). Tests that exercise a multi-layer
    backward path must use asymmetric init to avoid masking real bugs with
    this dead-symmetry mode.
    """
    var t = zeros(shape, DType.float32)
    var n = 1
    for d in shape:
        n *= d
    for i in range(n):
        t._set_float64(i, base + Float64(i) * slope)
    return t^


def _set_value(mut t: AnyTensor, idx: Int, val: Float64) raises:
    t._set_float64(idx, val)


def _assert_loss_decreased(
    name: String, losses: List[Float32], min_relative_drop: Float64
) raises:
    var first = Float64(losses[0])
    var last = Float64(losses[len(losses) - 1])
    var drop = (first - last) / first
    if drop < min_relative_drop:
        var msg = String(name) + ": loss did not decrease enough. first="
        msg += String(first) + " last=" + String(last)
        msg += (
            " drop=" + String(drop) + " required=" + String(min_relative_drop)
        )
        msg += "\n  full curve: ["
        for i in range(len(losses)):
            if i > 0:
                msg += ", "
            msg += String(losses[i])
        msg += "]"
        raise Error(msg)


def _assert_parameter_changed(
    name: String, before: Float64, after: Float64, min_abs_delta: Float64
) raises:
    var delta = before - after if before > after else after - before
    if delta < min_abs_delta:
        raise Error(
            String(name)
            + ": parameter did not change. before="
            + String(before)
            + " after="
            + String(after)
            + " delta="
            + String(delta)
            + " required>="
            + String(min_abs_delta)
        )


# ============================================================================
# Test 1: Tiny MLP convergence — sanity check on the substrate end-to-end.
#
# Architecture: x (2x3) -> linear (3->4) -> cross_entropy
# Steps: 20, lr=1.0 (huge so any update is obvious)
# Expectation: loss drops by >= 90% across 20 steps. Anything less means
# the writeback or optimizer is silently dropping updates.
# ============================================================================
def test_linear_softmax_converges() raises:
    print("[test] linear+softmax convergence (20 steps) ...")
    var optimizer = SGD(learning_rate=1.0)

    # Persistent weights across steps — initialize once.
    var w_data = _make_tensor([4, 3], 0.1)
    var b_data = _make_tensor([4], 0.0)

    var losses = List[Float32]()
    var w_before_0 = w_data._get_float64(0)

    for step_i in range(20):
        var tape = GradientTape()
        tape.enable()

        var x_data = _make_tensor([2, 3], 0.5)
        var x = Variable(x_data^, False, tape)

        var w = Variable(w_data, True, tape)
        var b = Variable(b_data, True, tape)

        # One-hot labels: both samples are class 0.
        var labels_data = zeros([2, 4], DType.float32)
        _set_value(labels_data, 0, 1.0)
        _set_value(labels_data, 4, 1.0)
        var labels = Variable(labels_data^, False, tape)

        var logits = variable_linear(x, w, b, tape)
        var loss = variable_cross_entropy(logits, labels, tape)
        loss.backward(tape)
        losses.append(loss.data._data.bitcast[Float32]()[0])

        var params: List[Variable] = []
        params.append(w^)
        params.append(b^)
        optimizer.step(params, tape)

        # Writeback for next iteration.
        w_data = params[0].data
        b_data = params[1].data
        optimizer.zero_grad(tape)
        _ = step_i  # silence unused warning

    _assert_loss_decreased("linear+softmax", losses, 0.90)
    _assert_parameter_changed(
        "linear+softmax w[0]", w_before_0, w_data._get_float64(0), 0.01
    )
    print("       OK (loss[0]=", losses[0], " loss[19]=", losses[19], ")")


# ============================================================================
# Test 2: Conv-then-linear convergence — exercises conv2d backward at a
# nontrivial shape with multiple output channels and verifies its gradient
# actually moves the loss.
#
# Architecture:
#   x (2,1,6,6) -> conv2d (1->2 ch, 3x3) -> relu -> flatten ->
#   linear (2*4*4 -> 3) -> cross_entropy
# Steps: 30, lr=0.5
# Expectation: loss drops >= 70%.
# ============================================================================
def test_conv_linear_softmax_converges() raises:
    print("[test] conv+linear+softmax convergence (30 steps) ...")
    var optimizer = SGD(learning_rate=0.5)

    # Persistent params with asymmetric init (see _make_asymmetric docstring).
    var c1w_data = _make_asymmetric([2, 1, 3, 3], 0.05, 0.02)
    var c1b_data = _make_tensor([2], 0.0)
    var fcw_data = _make_asymmetric([3, 32], 0.01, 0.001)  # 2*4*4 = 32
    var fcb_data = _make_tensor([3], 0.0)

    var losses = List[Float32]()
    var c1w_before_0 = c1w_data._get_float64(0)

    for step_i in range(30):
        var tape = GradientTape()
        tape.enable()

        var x_data = _make_asymmetric([2, 1, 6, 6], 0.1, 0.005)
        var x = Variable(x_data^, False, tape)

        var c1w = Variable(c1w_data, True, tape)
        var c1b = Variable(c1b_data, True, tape)
        var fcw = Variable(fcw_data, True, tape)
        var fcb = Variable(fcb_data, True, tape)

        # Labels: sample 0 = class 0, sample 1 = class 1.
        var labels_data = zeros([2, 3], DType.float32)
        _set_value(labels_data, 0, 1.0)  # sample 0, class 0
        _set_value(labels_data, 4, 1.0)  # sample 1, class 1
        var labels = Variable(labels_data^, False, tape)

        var c1_out = variable_conv2d(x, c1w, c1b, tape, stride=1, padding=0)
        # c1_out shape: (2, 2, 4, 4)
        var relu_out = variable_relu(c1_out, tape)
        var flat = variable_flatten(relu_out, tape)
        var logits = variable_linear(flat, fcw, fcb, tape)
        var loss = variable_cross_entropy(logits, labels, tape)
        loss.backward(tape)
        losses.append(loss.data._data.bitcast[Float32]()[0])

        var params: List[Variable] = []
        params.append(c1w^)
        params.append(c1b^)
        params.append(fcw^)
        params.append(fcb^)
        optimizer.step(params, tape)

        c1w_data = params[0].data
        c1b_data = params[1].data
        fcw_data = params[2].data
        fcb_data = params[3].data
        optimizer.zero_grad(tape)
        _ = step_i

    _assert_loss_decreased("conv+linear+softmax", losses, 0.70)
    _assert_parameter_changed(
        "conv kernel c1w[0]", c1w_before_0, c1w_data._get_float64(0), 0.001
    )
    print("       OK (loss[0]=", losses[0], " loss[29]=", losses[29], ")")


# ============================================================================
# Test 3: LeNet-shaped convergence — the exact same op stack used in
# examples/lenet_emnist/train_autograd.mojo, at small batch + small input
# (so the test runs in seconds) but with all the real op kinds in place.
#
# Architecture (mirrors LeNet-5):
#   x (4,1,28,28) -> conv2d (1->6 ch, 5x5) -> relu -> maxpool2d (2,2) ->
#   conv2d (6->16 ch, 5x5) -> relu -> maxpool2d (2,2) -> flatten ->
#   linear (16*4*4 -> 120) -> relu -> linear (120 -> 84) -> relu ->
#   linear (84 -> 10) -> cross_entropy
#
# Steps: 50, lr=0.01
# Expectation: loss drops >= 30% across 50 steps. This is the test that
# would have caught the recently-merged train_autograd.mojo going to chance.
# ============================================================================
def test_lenet_shape_converges() raises:
    print("[test] LeNet-shape convergence (50 steps, batch=4, 10 classes) ...")
    var optimizer = SGD(learning_rate=0.01)

    # Persistent params with asymmetric init (see _make_asymmetric docstring).
    var c1w = _make_asymmetric([6, 1, 5, 5], 0.05, 0.003)
    var c1b = _make_tensor([6], 0.0)
    var c2w = _make_asymmetric([16, 6, 5, 5], 0.02, 0.001)
    var c2b = _make_tensor([16], 0.0)
    var fc1w = _make_asymmetric([120, 16 * 4 * 4], 0.005, 0.00005)
    var fc1b = _make_tensor([120], 0.0)
    var fc2w = _make_asymmetric([84, 120], 0.005, 0.00005)
    var fc2b = _make_tensor([84], 0.0)
    var fc3w = _make_asymmetric([10, 84], 0.005, 0.0005)
    var fc3b = _make_tensor([10], 0.0)

    var losses = List[Float32]()
    var c1w_before_0 = c1w._get_float64(0)

    for step_i in range(50):
        var tape = GradientTape()
        tape.enable()

        var x_data = _make_asymmetric([4, 1, 28, 28], 0.05, 0.0002)
        var x = Variable(x_data^, False, tape)

        var v_c1w = Variable(c1w, True, tape)
        var v_c1b = Variable(c1b, True, tape)
        var v_c2w = Variable(c2w, True, tape)
        var v_c2b = Variable(c2b, True, tape)
        var v_fc1w = Variable(fc1w, True, tape)
        var v_fc1b = Variable(fc1b, True, tape)
        var v_fc2w = Variable(fc2w, True, tape)
        var v_fc2b = Variable(fc2b, True, tape)
        var v_fc3w = Variable(fc3w, True, tape)
        var v_fc3b = Variable(fc3b, True, tape)

        # Labels: sample i -> class (i mod 10) one-hot.
        var labels_data = zeros([4, 10], DType.float32)
        for i in range(4):
            _set_value(labels_data, i * 10 + (i % 10), 1.0)
        var labels = Variable(labels_data^, False, tape)

        var c1 = variable_conv2d(x, v_c1w, v_c1b, tape, stride=1, padding=0)
        var r1 = variable_relu(c1, tape)
        var p1 = variable_maxpool2d(
            r1, tape, kernel_size=2, stride=2, padding=0
        )
        var c2 = variable_conv2d(p1, v_c2w, v_c2b, tape, stride=1, padding=0)
        var r2 = variable_relu(c2, tape)
        var p2 = variable_maxpool2d(
            r2, tape, kernel_size=2, stride=2, padding=0
        )
        var flat = variable_flatten(p2, tape)
        var fc1_o = variable_linear(flat, v_fc1w, v_fc1b, tape)
        var r3 = variable_relu(fc1_o, tape)
        var fc2_o = variable_linear(r3, v_fc2w, v_fc2b, tape)
        var r4 = variable_relu(fc2_o, tape)
        var logits = variable_linear(r4, v_fc3w, v_fc3b, tape)

        var loss = variable_cross_entropy(logits, labels, tape)
        loss.backward(tape)
        losses.append(loss.data._data.bitcast[Float32]()[0])

        var params: List[Variable] = []
        params.append(v_c1w^)
        params.append(v_c1b^)
        params.append(v_c2w^)
        params.append(v_c2b^)
        params.append(v_fc1w^)
        params.append(v_fc1b^)
        params.append(v_fc2w^)
        params.append(v_fc2b^)
        params.append(v_fc3w^)
        params.append(v_fc3b^)
        optimizer.step(params, tape)

        c1w = params[0].data
        c1b = params[1].data
        c2w = params[2].data
        c2b = params[3].data
        fc1w = params[4].data
        fc1b = params[5].data
        fc2w = params[6].data
        fc2b = params[7].data
        fc3w = params[8].data
        fc3b = params[9].data
        optimizer.zero_grad(tape)
        _ = step_i

    _assert_loss_decreased("LeNet-shape", losses, 0.30)
    _assert_parameter_changed(
        "LeNet conv1 kernel[0]", c1w_before_0, c1w._get_float64(0), 0.0001
    )
    print("       OK (loss[0]=", losses[0], " loss[49]=", losses[49], ")")


def main() raises:
    print("\n=== Autograd convergence tests ===\n")
    test_linear_softmax_converges()
    test_conv_linear_softmax_converges()
    test_lenet_shape_converges()
    print("\nAll autograd convergence checks PASS")
