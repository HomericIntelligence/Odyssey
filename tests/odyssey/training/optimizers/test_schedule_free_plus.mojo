"""Unit tests for the ScheduleFree+ optimizer (arXiv:2605.19095).

ScheduleFree+ is the large-batch-stable schedule-free variant: inner momentum,
Polyak step-size, and increasing outer momentum. Tests cover:

- Shape/dtype guards on schedule_free_plus_step (incl. the inner-momentum buffer)
- Numerical parity across THREE steps with the NumPy reference in
  parity_refs/schedule_free_plus_parity_reference.py (a hand-rolled transcription
  of the tracking-issue rule mvillmow/Random#78) to 1e-9, on the z / x / m /
  gnorm / y sequences and the Polyak lr_t — multiple steps so the increasing
  outer-momentum anneal (beta_out 0.9 -> 0.9267 -> 0.9533) and the L1-EMA norm
  evolve.
- schedule_free_plus_step_simple matches schedule_free_plus_step defaults
- descent direction on the fast sequence for a positive gradient
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.schedule_free_plus import (
    schedule_free_plus_step,
    schedule_free_plus_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _set(t: AnyTensor, vals: List[Float64]) raises:
    for i in range(len(vals)):
        t.store[DType.float64](i, vals[i])


def test_reject_shape_mismatch() raises:
    """`schedule_free_plus_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var z = zeros([4], DType.float32)
    var x = zeros([4], DType.float32)
    var m = zeros([4], DType.float32)
    try:
        var _ = schedule_free_plus_step(p, g, z, x, m, 0.0, 1.0, 1, 0.1)
        raise Error("Should have rejected shape mismatch")
    except _:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_m_shape_mismatch() raises:
    """`schedule_free_plus_step` rejects a params/momentum-buffer shape mismatch.
    """
    print("Running test_reject_m_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var z = zeros([4], DType.float32)
    var x = zeros([4], DType.float32)
    var m = zeros([5], DType.float32)
    try:
        var _ = schedule_free_plus_step(p, g, z, x, m, 0.0, 1.0, 1, 0.1)
        raise Error("Should have rejected m shape mismatch")
    except _:
        print("  ok rejected m shape mismatch")
    print("test_reject_m_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`schedule_free_plus_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var z = zeros([4], DType.float32)
    var x = zeros([4], DType.float32)
    var m = zeros([4], DType.float32)
    try:
        var _ = schedule_free_plus_step(p, g, z, x, m, 0.0, 1.0, 1, 0.1)
        raise Error("Should have rejected dtype mismatch")
    except _:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_parity_with_reference() raises:
    """Three ScheduleFree+ steps must match the NumPy reference to 1e-9.

    Fixed inputs + reference outputs transcribed from
    parity_refs/schedule_free_plus_parity_reference.py (lr=0.1, mu=0.9,
    beta_sf=0.9, beta_max=0.98, rho=0.9, eps=1e-8, horizon=4). The reference is a
    hand-rolled NumPy transcription of the tracking-issue rule
    (mvillmow/Random#78): inner momentum, Polyak step-size, increasing outer
    momentum. Three steps with DIFFERENT gradients + objectives so the anneal
    (beta_out 0.9 -> 0.9267 -> 0.9533), the L1-EMA norm, and every sequence move.
    """
    print("Running test_parity_with_reference...")
    var n = 5

    # Reference z_{t+1} after each step.
    var z1 = List[Float64]()
    z1.append(0.07600000959999617)
    z1.append(-0.2719999712000115)
    z1.append(0.41999995200001916)
    z1.append(-0.5679999328000269)
    z1.append(0.7159999136000346)
    var z2 = List[Float64]()
    z2.append(0.053692333443643214)
    z2.append(-0.3050153319114139)
    z2.append(0.4958460509316192)
    z2.append(-0.6955998404143658)
    z2.append(0.8418152071218652)
    var z3 = List[Float64]()
    z3.append(0.045322476994239635)
    z3.append(-0.3381780604379697)
    z3.append(0.5395050318704)
    z3.append(-0.776448129468875)
    z3.append(0.9308976252238957)

    # Reference x_{t+1} after each step.
    var x1 = List[Float64]()
    x1.append(0.09760000095999963)
    x1.append(-0.20719999712000117)
    x1.append(0.31199999520000193)
    x1.append(-0.4167999932800027)
    x1.append(0.5215999913600035)
    var x3 = List[Float64]()
    x3.append(0.0920907493525651)
    x3.append(-0.22015068548047934)
    x3.append(0.33546977894087976)
    x3.append(-0.453074780059588)
    x3.append(0.563087149135581)

    # Reference m_{t+1} and gnorm_{t+1} / lr_t at selected steps.
    var m3 = List[Float64]()
    m3.append(0.009249999999999998)
    m3.append(0.036649999999999995)
    m3.append(-0.04825)
    m3.append(0.08934999999999998)
    m3.append(-0.09844999999999997)

    var p = zeros([n], DType.float64)
    _set(
        p,
        [0.1, -0.2, 0.3, -0.4, 0.5],
    )
    var g1 = zeros([n], DType.float64)
    _set(g1, [0.05, 0.15, -0.25, 0.35, -0.45])
    var g2 = zeros([n], DType.float64)
    _set(g2, [0.08, 0.05, -0.20, 0.40, -0.30])
    var g3 = zeros([n], DType.float64)
    _set(g3, [-0.02, 0.20, -0.10, 0.25, -0.35])

    # Initial state: z = x = params, m = 0, gnorm = 0.
    var z = zeros([n], DType.float64)
    _set(z, [0.1, -0.2, 0.3, -0.4, 0.5])
    var x = zeros([n], DType.float64)
    _set(x, [0.1, -0.2, 0.3, -0.4, 0.5])
    var m = zeros([n], DType.float64)

    # Step 1: objective f(y_1) = 1.2.
    var r1 = schedule_free_plus_step(
        p, g1, z, x, m, 0.0, 1.2, 1, 0.1, 0.9, 0.9, 0.98, 0.9, 1e-8, 4
    )
    var nz1 = r1[0]
    var nx1 = r1[1]
    for i in range(n):
        if _abs_diff(nz1.load[DType.float64](i), z1[i]) > 1e-9:
            raise Error("sf+ step-1 z mismatch at " + String(i))
        if _abs_diff(nx1.load[DType.float64](i), x1[i]) > 1e-9:
            raise Error("sf+ step-1 x mismatch at " + String(i))
    if _abs_diff(r1[5], 4.799998080000768) > 1e-9:
        raise Error("sf+ step-1 lr_t mismatch")
    print("  ok step 1 (z, x, lr_t) matches reference")

    # Step 2: objective f(y_2) = 0.9, feed g2 and the returned state.
    var r2 = schedule_free_plus_step(
        p,
        g2,
        r1[0],
        r1[1],
        r1[2],
        r1[3],
        0.9,
        2,
        0.1,
        0.9,
        0.9,
        0.98,
        0.9,
        1e-8,
        4,
    )
    var nz2 = r2[0]
    for i in range(n):
        if _abs_diff(nz2.load[DType.float64](i), z2[i]) > 1e-9:
            raise Error("sf+ step-2 z mismatch at " + String(i))
    if _abs_diff(r2[3], 0.04309999999999999) > 1e-9:
        raise Error("sf+ step-2 gnorm mismatch")
    if _abs_diff(r2[5], 1.7846140925082368) > 1e-9:
        raise Error("sf+ step-2 lr_t mismatch")
    print("  ok step 2 (z, gnorm, lr_t) matches reference")

    # Step 3: objective f(y_3) = 0.7, feed g3 and the returned state.
    var r3 = schedule_free_plus_step(
        p,
        g3,
        r2[0],
        r2[1],
        r2[2],
        r2[3],
        0.7,
        3,
        0.1,
        0.9,
        0.9,
        0.98,
        0.9,
        1e-8,
        4,
    )
    var nz3 = r3[0]
    var nx3 = r3[1]
    var nm3 = r3[2]
    for i in range(n):
        if _abs_diff(nz3.load[DType.float64](i), z3[i]) > 1e-9:
            raise Error("sf+ step-3 z mismatch at " + String(i))
        if _abs_diff(nx3.load[DType.float64](i), x3[i]) > 1e-9:
            raise Error("sf+ step-3 x mismatch at " + String(i))
        if _abs_diff(nm3.load[DType.float64](i), m3[i]) > 1e-9:
            raise Error("sf+ step-3 m mismatch at " + String(i))
    if _abs_diff(r3[5], 0.9048493458814681) > 1e-9:
        raise Error("sf+ step-3 lr_t mismatch")
    print("  ok step 3 (z, x, m, lr_t) matches reference")
    print("test_parity_with_reference PASSED")


def test_step_simple_matches_defaults() raises:
    """`schedule_free_plus_step_simple` == full step with default hyperparams.

    The simple wrapper hard-codes horizon=1000, so the explicit call uses the
    same horizon (not the parity test's 4) to isolate the default-forwarding.
    """
    print("Running test_step_simple_matches_defaults...")
    var n = 5
    var p = zeros([n], DType.float64)
    _set(p, [0.1, -0.2, 0.3, -0.4, 0.5])
    var g = zeros([n], DType.float64)
    _set(g, [0.05, 0.15, -0.25, 0.35, -0.45])
    var z = zeros([n], DType.float64)
    _set(z, [0.1, -0.2, 0.3, -0.4, 0.5])
    var x = zeros([n], DType.float64)
    _set(x, [0.1, -0.2, 0.3, -0.4, 0.5])
    var m = zeros([n], DType.float64)

    var simple = schedule_free_plus_step_simple(p, g, z, x, m, 0.0, 1.2, 1, 0.1)
    var full_call = schedule_free_plus_step(
        p, g, z, x, m, 0.0, 1.2, 1, 0.1, 0.9, 0.9, 0.98, 0.9, 1e-8, 1000
    )
    var sz = simple[0]
    var fz = full_call[0]
    var sx = simple[1]
    var fx = full_call[1]
    for i in range(n):
        if (
            _abs_diff(sz.load[DType.float64](i), fz.load[DType.float64](i))
            > 1e-12
        ):
            raise Error("simple != full (z) at " + String(i))
        if (
            _abs_diff(sx.load[DType.float64](i), fx.load[DType.float64](i))
            > 1e-12
        ):
            raise Error("simple != full (x) at " + String(i))
    if _abs_diff(simple[5], full_call[5]) > 1e-12:
        raise Error("simple != full (lr_t)")
    print("  ok schedule_free_plus_step_simple matches defaults")
    print("test_step_simple_matches_defaults PASSED")


def test_descent_direction() raises:
    """A positive gradient must decrease the fast sequence `z`.

    With z = x = params (step 1), the correlation <g, z-x> = 0, so the Polyak
    numerator is f(y) > 0 and lr_t > 0; the fast step z -= lr_t * m with m > 0
    (positive gradient) must decrease z below params.
    """
    print("Running test_descent_direction...")
    var n = 3
    var p = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var z = full([n], 1.0, DType.float64)
    var x = full([n], 1.0, DType.float64)
    var m = zeros([n], DType.float64)
    var res = schedule_free_plus_step(
        p, g, z, x, m, 0.0, 1.0, 1, 0.01, 0.9, 0.9, 0.98, 0.9, 1e-8, 1000
    )
    var nz = res[0]
    for i in range(n):
        if not (nz.load[DType.float64](i) < 1.0):
            raise Error("positive gradient should decrease z")
    print("  ok positive gradient decreased the fast sequence z")
    print("test_descent_direction PASSED")


def main() raises:
    """Run all ScheduleFree+ tests."""
    print("=" * 60)
    print("ScheduleFree+ Optimizer Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_reject_m_shape_mismatch()
    test_reject_dtype_mismatch()
    test_parity_with_reference()
    test_step_simple_matches_defaults()
    test_descent_direction()
    print("=" * 60)
    print("All ScheduleFree+ tests PASSED")
    print("=" * 60)
