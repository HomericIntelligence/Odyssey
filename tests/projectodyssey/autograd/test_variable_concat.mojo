"""Gradient-check tests for the `variable_concat` autograd op.

Concat is a pure data-routing op: forward stacks inputs along an axis, backward
slices the output gradient back into each input's contiguous range. For a loss
`sum(concat(a, b, ...) * w)`, the gradient wrt each input is EXACTLY the slice of
`w` covering that input's range — so the check is analytic and exact (no
finite-difference tolerance needed).

The critical case is the CHANNEL axis (axis=1) of an NCHW tensor: a channel-range
slice of grad_output is a strided (non-contiguous) view, and the gradient
registry reads grads by flat index, so `backward_concat` must materialize each
slice contiguous. This test concatenates two tensors with DIFFERENT channel
counts along axis=1 to exercise that path.
"""

from projectodyssey.autograd import Variable, GradientTape
from projectodyssey.autograd.variable import (
    variable_concat,
    variable_multiply,
    variable_sum,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros


def _make_tensor(shape: List[Int], values: List[Float64]) raises -> AnyTensor:
    var t = zeros(shape, DType.float32)
    for i in range(len(values)):
        t._set_float64(i, values[i])
    return t^


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _nchw_index(n: Int, c: Int, h: Int, w: Int, C: Int, H: Int, W: Int) -> Int:
    return ((n * C + c) * H + h) * W + w


def test_concat_channel_axis_grad_check() raises:
    print("[test] variable_concat channel-axis gradient check ...")

    # a: (batch=2, Ca=2, H=2, W=2); b: (batch=2, Cb=3, H=2, W=2)
    # concat along axis=1 -> (2, 5, 2, 2). Exercises unequal channel counts and
    # the strided channel-slice backward path.
    var H = 2
    var W = 2
    var N = 2
    var Ca = 2
    var Cb = 3
    var Cout = Ca + Cb

    var a_shape: List[Int] = [N, Ca, H, W]
    var b_shape: List[Int] = [N, Cb, H, W]
    var out_shape: List[Int] = [N, Cout, H, W]

    # Distinct, non-uniform values so a mis-routed slice would be detected.
    var a_vals = List[Float64]()
    for i in range(N * Ca * H * W):
        a_vals.append(Float64(i) * 0.1 - 0.5)
    var b_vals = List[Float64]()
    for i in range(N * Cb * H * W):
        b_vals.append(Float64(i) * -0.07 + 0.3)
    # Non-uniform loss weights over the concatenated output.
    var w_vals = List[Float64]()
    for i in range(N * Cout * H * W):
        w_vals.append(Float64((i * 7) % 11) * 0.13 - 0.4)

    var tape = GradientTape()
    tape.enable()

    var a = Variable(_make_tensor(a_shape, a_vals), True, tape)
    var b = Variable(_make_tensor(b_shape, b_vals), True, tape)
    var w = Variable(_make_tensor(out_shape, w_vals), False, tape)

    # Variable is not implicitly copyable; concat callers build the input list
    # with explicit .copy() (the Variable's tape id is preserved, so gradients
    # still route to the originals a/b).
    var inputs = List[Variable]()
    inputs.append(a.copy())
    inputs.append(b.copy())
    var y = variable_concat(inputs, tape, axis=1)

    # Output shape sanity: (2, 5, 2, 2).
    var ys = y.data.shape()
    if len(ys) != 4 or ys[0] != N or ys[1] != Cout or ys[2] != H or ys[3] != W:
        raise Error("variable_concat output shape mismatch")

    # loss = sum(concat(a, b) * w)
    var weighted = variable_multiply(y, w, tape)
    var loss = variable_sum(weighted, tape, axis=-1)
    loss.backward(tape)

    var grad_a = tape.get_grad(a.id)
    var grad_b = tape.get_grad(b.id)

    # Analytic: grad_a[n,c,h,w] == w[n, c,        h, w]  for c in [0, Ca)
    #           grad_b[n,c,h,w] == w[n, Ca + c,   h, w]  for c in [0, Cb)
    var w_t = _make_tensor(out_shape, w_vals)

    for n in range(N):
        for c in range(Ca):
            for h in range(H):
                for ww in range(W):
                    var ga = grad_a._get_float64(
                        _nchw_index(n, c, h, ww, Ca, H, W)
                    )
                    var expected = w_t._get_float64(
                        _nchw_index(n, c, h, ww, Cout, H, W)
                    )
                    if _abs(ga - expected) > 1e-5:
                        raise Error(
                            "grad_a mismatch at ("
                            + String(n)
                            + ","
                            + String(c)
                            + "): got "
                            + String(ga)
                            + " expected "
                            + String(expected)
                        )

    for n in range(N):
        for c in range(Cb):
            for h in range(H):
                for ww in range(W):
                    var gb = grad_b._get_float64(
                        _nchw_index(n, c, h, ww, Cb, H, W)
                    )
                    var expected = w_t._get_float64(
                        _nchw_index(n, Ca + c, h, ww, Cout, H, W)
                    )
                    if _abs(gb - expected) > 1e-5:
                        raise Error(
                            "grad_b mismatch at ("
                            + String(n)
                            + ","
                            + String(c)
                            + "): got "
                            + String(gb)
                            + " expected "
                            + String(expected)
                        )

    print("       OK")


def main() raises:
    test_concat_channel_axis_grad_check()
    print("\nvariable_concat gradient check PASS")
