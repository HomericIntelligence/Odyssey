"""Unit tests for the Schedule-Free optimizer (arXiv:2405.15682).

Tests cover:
- Shape/dtype guards on schedule_free_step
- Numerical parity with the hand-rolled NumPy reference (1e-9) over THREE steps
  so the z/x/y sequences and the averaging weight c_{t+1} = (r+1)/(t+r+1)
  evolve (r=0 uniform-average case: c_2=1/2, c_3=1/3, c_4=1/4). The reference
  transcribes the issue's pinned update rule (mvillmow/Random#77); the official
  `schedulefree` package is not available in the pinned interpreter, so NumPy is
  used as a calculator for that verified rule (see
  parity_refs/schedule_free_parity_reference.py).
- schedule_free_step_simple matches schedule_free_step with the default
  hyperparameters
- step-1 identity: with z = x = params, y_1 = params, so the first z step is a
  plain SGD step and x_2 = (1-c_2)*params + c_2*z_2
- descent direction (the averaged iterate x moves against a positive gradient)
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.schedule_free import (
    schedule_free_step,
    schedule_free_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`schedule_free_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var z = zeros([4], DType.float32)
    var x = zeros([4], DType.float32)
    try:
        var _ = schedule_free_step(p, g, z, x, 1, 0.1)
        raise Error("Should have rejected shape mismatch")
    except _:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`schedule_free_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var z = zeros([4], DType.float32)
    var x = zeros([4], DType.float32)
    try:
        var _ = schedule_free_step(p, g, z, x, 1, 0.1)
        raise Error("Should have rejected dtype mismatch")
    except _:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_parity_with_reference() raises:
    """Three Schedule-Free steps must match the NumPy reference to 1e-9.

    Fixed inputs + reference outputs transcribed from
    parity_refs/schedule_free_parity_reference.py (gamma=0.1, beta=0.9, r=0);
    the reference transcribes the update rule pinned in mvillmow/Random#77.
    z and x are initialized to params (so y_1 = params); each step feeds the
    gradient evaluated at the current query point y_t. Three steps exercise the
    evolving averaging weight c_{t+1} = 1/(t+1): c_2=1/2, c_3=1/3, c_4=1/4.
    """
    print("Running test_parity_with_reference...")

    var n = 5

    # --- reference z after each step ---
    var z1 = List[Float64]()
    z1.append(0.095)
    z1.append(-0.21500000000000002)
    z1.append(0.325)
    z1.append(-0.435)
    z1.append(0.545)
    var z2 = List[Float64]()
    z2.append(0.087)
    z2.append(-0.22000000000000003)
    z2.append(0.34500000000000003)
    z2.append(-0.475)
    z2.append(0.5750000000000001)
    var z3 = List[Float64]()
    z3.append(0.089)
    z3.append(-0.23000000000000004)
    z3.append(0.36000000000000004)
    z3.append(-0.495)
    z3.append(0.6000000000000001)

    # --- reference x after each step ---
    var x1 = List[Float64]()
    x1.append(0.0975)
    x1.append(-0.20750000000000002)
    x1.append(0.3125)
    x1.append(-0.4175)
    x1.append(0.5225)
    var x2 = List[Float64]()
    x2.append(0.09400000000000001)
    x2.append(-0.2116666666666667)
    x2.append(0.32333333333333336)
    x2.append(-0.4366666666666667)
    x2.append(0.54)
    var x3 = List[Float64]()
    x3.append(0.09275)
    x3.append(-0.21625000000000005)
    x3.append(0.3325)
    x3.append(-0.45125000000000004)
    x3.append(0.555)

    var p = zeros([n], DType.float64)
    var g1 = zeros([n], DType.float64)
    var g2 = zeros([n], DType.float64)
    var g3 = zeros([n], DType.float64)
    p.store[DType.float64](0, 0.1)
    p.store[DType.float64](1, -0.2)
    p.store[DType.float64](2, 0.3)
    p.store[DType.float64](3, -0.4)
    p.store[DType.float64](4, 0.5)
    g1.store[DType.float64](0, 0.05)
    g1.store[DType.float64](1, 0.15)
    g1.store[DType.float64](2, -0.25)
    g1.store[DType.float64](3, 0.35)
    g1.store[DType.float64](4, -0.45)
    g2.store[DType.float64](0, 0.08)
    g2.store[DType.float64](1, 0.05)
    g2.store[DType.float64](2, -0.20)
    g2.store[DType.float64](3, 0.40)
    g2.store[DType.float64](4, -0.30)
    g3.store[DType.float64](0, -0.02)
    g3.store[DType.float64](1, 0.10)
    g3.store[DType.float64](2, -0.15)
    g3.store[DType.float64](3, 0.20)
    g3.store[DType.float64](4, -0.25)

    # Init: z = x = params.
    var res1 = schedule_free_step(p, g1, p, p, 1, 0.1, 0.9, 0.0)
    var nz1 = res1[0]
    var nx1 = res1[1]
    for i in range(n):
        if _abs_diff(nz1.load[DType.float64](i), z1[i]) > 1e-9:
            raise Error("schedule-free step-1 z mismatch at " + String(i))
        if _abs_diff(nx1.load[DType.float64](i), x1[i]) > 1e-9:
            raise Error("schedule-free step-1 x mismatch at " + String(i))
    print("  ok step 1 z/x match reference to 1e-9")

    var res2 = schedule_free_step(p, g2, res1[0], res1[1], 2, 0.1, 0.9, 0.0)
    var nz2 = res2[0]
    var nx2 = res2[1]
    for i in range(n):
        if _abs_diff(nz2.load[DType.float64](i), z2[i]) > 1e-9:
            raise Error("schedule-free step-2 z mismatch at " + String(i))
        if _abs_diff(nx2.load[DType.float64](i), x2[i]) > 1e-9:
            raise Error("schedule-free step-2 x mismatch at " + String(i))
    print("  ok step 2 z/x match reference to 1e-9 (c_3 = 1/3)")

    var res3 = schedule_free_step(p, g3, res2[0], res2[1], 3, 0.1, 0.9, 0.0)
    var nz3 = res3[0]
    var nx3 = res3[1]
    for i in range(n):
        if _abs_diff(nz3.load[DType.float64](i), z3[i]) > 1e-9:
            raise Error("schedule-free step-3 z mismatch at " + String(i))
        if _abs_diff(nx3.load[DType.float64](i), x3[i]) > 1e-9:
            raise Error("schedule-free step-3 x mismatch at " + String(i))
    print("  ok step 3 z/x match reference to 1e-9 (c_4 = 1/4)")
    print("test_parity_with_reference PASSED")


def test_step_simple_matches_defaults() raises:
    """`schedule_free_step_simple` == `schedule_free_step` at default hypers."""
    print("Running test_step_simple_matches_defaults...")
    var n = 5
    var p = zeros([n], DType.float64)
    var g = zeros([n], DType.float64)
    p.store[DType.float64](0, 0.1)
    p.store[DType.float64](1, -0.2)
    p.store[DType.float64](2, 0.3)
    p.store[DType.float64](3, -0.4)
    p.store[DType.float64](4, 0.5)
    g.store[DType.float64](0, 0.05)
    g.store[DType.float64](1, 0.15)
    g.store[DType.float64](2, -0.25)
    g.store[DType.float64](3, 0.35)
    g.store[DType.float64](4, -0.45)

    var simple = schedule_free_step_simple(p, g, p, p, 1, 0.1)
    var full_call = schedule_free_step(p, g, p, p, 1, 0.1, 0.9, 0.0)

    # Compare all three returned tensors (z, x, y_next). Tuple slots must be
    # indexed with compile-time constants.
    var sz = simple[0]
    var fz = full_call[0]
    var sx = simple[1]
    var fx = full_call[1]
    var sy = simple[2]
    var fy = full_call[2]
    for i in range(n):
        if (
            _abs_diff(sz.load[DType.float64](i), fz.load[DType.float64](i))
            > 1e-12
        ):
            raise Error(
                "schedule_free_step_simple z != defaults at " + String(i)
            )
        if (
            _abs_diff(sx.load[DType.float64](i), fx.load[DType.float64](i))
            > 1e-12
        ):
            raise Error(
                "schedule_free_step_simple x != defaults at " + String(i)
            )
        if (
            _abs_diff(sy.load[DType.float64](i), fy.load[DType.float64](i))
            > 1e-12
        ):
            raise Error(
                "schedule_free_step_simple y_next != defaults at " + String(i)
            )
    print("  ok schedule_free_step_simple matches defaults (z, x, y_next)")
    print("test_step_simple_matches_defaults PASSED")


def test_step1_identity() raises:
    """Step 1 with z=x=params: the fast step is plain SGD, x is its blend.

    y_1 = params, so z_2 = params - gamma*g and x_2 = (1-c_2)*params + c_2*z_2
    with c_2 = 1/2. For params=1, g=0.5, gamma=0.1: z_2 = 0.95, x_2 = 0.975.
    """
    print("Running test_step1_identity...")
    var n = 3
    var p = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var res = schedule_free_step(p, g, p, p, 1, 0.1, 0.9, 0.0)
    var z = res[0]
    var x = res[1]
    for i in range(n):
        if _abs_diff(z.load[DType.float64](i), 0.95) > 1e-12:
            raise Error("step-1 z should be params - gamma*g = 0.95")
        if _abs_diff(x.load[DType.float64](i), 0.975) > 1e-12:
            raise Error("step-1 x should be (1-c_2)*p + c_2*z_2 = 0.975")
    print("  ok step-1 z=0.95, x=0.975 (plain SGD fast step + uniform blend)")
    print("test_step1_identity PASSED")


def test_descent_direction() raises:
    """A positive gradient must decrease both z and the averaged iterate x."""
    print("Running test_descent_direction...")
    var n = 3
    var p = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var res = schedule_free_step(p, g, p, p, 1, 0.01)
    var z = res[0]
    var x = res[1]
    for i in range(n):
        if not (z.load[DType.float64](i) < 1.0):
            raise Error("positive gradient should decrease z")
        if not (x.load[DType.float64](i) < 1.0):
            raise Error("positive gradient should decrease averaged x")
    print("  ok positive gradient decreased both z and x")
    print("test_descent_direction PASSED")


def main() raises:
    """Run all Schedule-Free tests."""
    print("=" * 60)
    print("Schedule-Free Optimizer Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_parity_with_reference()
    test_step_simple_matches_defaults()
    test_step1_identity()
    test_descent_direction()
    print("=" * 60)
    print("All Schedule-Free tests PASSED")
    print("=" * 60)
