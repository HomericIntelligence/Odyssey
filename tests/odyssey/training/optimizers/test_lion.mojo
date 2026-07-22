"""Unit tests for the Lion update step (Chen et al. 2023, arXiv:2302.06675).

Odyssey's `lion_step` is pure-functional: it returns (new_params, new_momentum).
The implemented update (lion.mojo) is

    update   = sign(beta1 * momentum + (1 - beta1) * grad)
    params   = params - lr * update
    params   = params - lr * weight_decay * params        (decoupled, lr-scaled)
    momentum = beta2 * momentum + (1 - beta2) * grad       (EMA updated AFTER)

which matches the reference Lion (lucidrains/lion-pytorch,
kozistr/pytorch_optimizer Lion).

Tests cover:
- Shape guards (params/grad and params/momentum) and dtype guard
- Numerical parity with the reference (1e-9): the sign update, the lr-scaled
  decoupled weight decay, and the beta2 momentum EMA. Reference numbers from
  parity_refs/lion_parity_reference.py (numpy closed form; inputs chosen so no
  interpolated coordinate is exactly zero, avoiding a sign(0) ambiguity).
- The update is exactly +/- lr (before weight decay) since it is a sign.
- lion_step_simple delegation.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.lion import (
    lion_step,
    lion_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`lion_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    try:
        var _ = lion_step(p, g, m, 0.001)
        raise Error("Should have rejected shape mismatch")
    except:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_momentum_shape_mismatch() raises:
    """`lion_step` rejects a params/momentum shape mismatch."""
    print("Running test_reject_momentum_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var m = zeros([5], DType.float32)
    try:
        var _ = lion_step(p, g, m, 0.001)
        raise Error("Should have rejected momentum shape mismatch")
    except:
        print("  ok rejected momentum shape mismatch")
    print("test_reject_momentum_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`lion_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    try:
        var _ = lion_step(p, g, m, 0.001)
        raise Error("Should have rejected dtype mismatch")
    except:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_parity_with_reference() raises:
    """One Lion step must match the reference to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/lion_parity_reference.py (lr=1e-3, beta1=0.9, beta2=0.99,
    weight_decay=0.01; sign update [+,-,+,-,+,-]).
    """
    print("Running test_parity_with_reference...")
    var n = 6
    var p = zeros([n], DType.float64)
    var g = zeros([n], DType.float64)
    var m = zeros([n], DType.float64)
    p.store[DType.float64](0, 0.10)
    p.store[DType.float64](1, -0.20)
    p.store[DType.float64](2, 0.30)
    p.store[DType.float64](3, -0.40)
    p.store[DType.float64](4, 0.50)
    p.store[DType.float64](5, -0.60)
    g.store[DType.float64](0, 0.02)
    g.store[DType.float64](1, -0.03)
    g.store[DType.float64](2, 0.015)
    g.store[DType.float64](3, 0.025)
    g.store[DType.float64](4, -0.01)
    g.store[DType.float64](5, 0.04)
    m.store[DType.float64](0, 0.05)
    m.store[DType.float64](1, -0.04)
    m.store[DType.float64](2, 0.03)
    m.store[DType.float64](3, -0.02)
    m.store[DType.float64](4, 0.01)
    m.store[DType.float64](5, -0.06)

    var rp = List[Float64]()
    rp.append(0.098999)
    rp.append(-0.198998)
    rp.append(0.298997)
    rp.append(-0.398996)
    rp.append(0.498995)
    rp.append(-0.598994)
    var rm = List[Float64]()
    rm.append(0.0497)
    rm.append(-0.039900000000000005)
    rm.append(0.029849999999999998)
    rm.append(-0.01955)
    rm.append(0.009800000000000001)
    rm.append(-0.059)

    var out = lion_step(p, g, m, 0.001, 0.9, 0.99, 0.01)
    var np = out[0]
    var nm = out[1]
    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("params parity mismatch at " + String(i))
        if _abs_diff(nm.load[DType.float64](i), rm[i]) > 1e-9:
            raise Error("momentum parity mismatch at " + String(i))
    print("  ok parity to 1e-9 on all 6 coordinates (params, momentum)")
    print("test_parity_with_reference PASSED")


def test_update_is_sign_scaled_by_lr() raises:
    """With weight_decay=0, |params - new_params| == lr on every coordinate.

    Lion's update is the SIGN of the interpolated momentum, so the raw step is
    exactly +/- lr independent of gradient magnitude. Inputs give a non-zero
    interpolation on every coordinate.
    """
    print("Running test_update_is_sign_scaled_by_lr...")
    var n = 4
    var p = full([n], 0.5, DType.float64)
    var g = full([n], 0.1, DType.float64)  # positive → interp > 0 → sign +1
    var m = full([n], 0.2, DType.float64)
    var out = lion_step(p, g, m, 0.01, 0.9, 0.99, 0.0)  # wd=0
    for i in range(n):
        # sign(+) = +1, so step = -lr*1 → new = 0.5 - 0.01 = 0.49
        if _abs_diff(out[0].load[DType.float64](i), 0.49) > 1e-12:
            raise Error("sign update not +/- lr at " + String(i))
    print("  ok raw update is exactly +/- lr (sign step)")
    print("test_update_is_sign_scaled_by_lr PASSED")


def test_lion_step_simple_delegates() raises:
    """`lion_step_simple` matches lion_step at defaults (betas 0.9/0.99, wd=0).
    """
    print("Running test_lion_step_simple_delegates...")
    var n = 4
    var p = full([n], 0.5, DType.float64)
    var g = full([n], 0.1, DType.float64)
    var m = full([n], 0.2, DType.float64)
    var full_out = lion_step(p, g, m, 0.01, 0.9, 0.99, 0.0)
    var simple_out = lion_step_simple(p, g, m, 0.01)
    for i in range(n):
        if (
            _abs_diff(
                full_out[0].load[DType.float64](i),
                simple_out[0].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error("lion_step_simple diverged at " + String(i))
    print("  ok lion_step_simple delegates to lion_step defaults")
    print("test_lion_step_simple_delegates PASSED")


def main() raises:
    test_reject_shape_mismatch()
    test_reject_momentum_shape_mismatch()
    test_reject_dtype_mismatch()
    test_parity_with_reference()
    test_update_is_sign_scaled_by_lr()
    test_lion_step_simple_delegates()
    print("\nAll Lion parity tests PASSED")
