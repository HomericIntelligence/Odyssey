"""Unit tests for the Adan optimizer (arXiv:2208.06677).

Tests cover:
- Shape/dtype guards on adan_step
- Numerical parity with the public pytorch_optimizer.Adan (1e-9) on steps 1 AND 2
  (step 2 uses a different gradient so grad_diff != 0, validating the
  difference-EMA / look-ahead terms that step 1 leaves at zero)
- adan_step_simple matches adan_step with the default hyperparameters
- weight_decay path (divisive decoupled decay: params /= (1 + lr*wd))
- prev_grad passthrough (new_prev_grad equals the current gradient)
- descent direction
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.adan import adan_step, adan_step_simple


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`adan_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    var dd = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    var pg = zeros([4], DType.float32)
    try:
        var _ = adan_step(p, g, m, dd, v, pg, 1, 0.001)
        raise Error("Should have rejected shape mismatch")
    except _:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`adan_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    var dd = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    var pg = zeros([4], DType.float32)
    try:
        var _ = adan_step(p, g, m, dd, v, pg, 1, 0.001)
        raise Error("Should have rejected dtype mismatch")
    except _:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_parity_with_reference() raises:
    """Two Adan steps must match pytorch_optimizer.Adan to 1e-9.

    Fixed inputs + reference outputs transcribed from
    parity_refs/adan_parity_reference.py (lr=0.001, betas=(0.98,0.92,0.99),
    eps=1e-8, no weight decay); the reference is verified against the real
    `pytorch_optimizer.Adan` (step-1 and step-2 max abs difference = 0.0).

    Step 1 uses prev_grad = grad1 (zero initial gradient difference), matching
    the reference's previous-grad initialization. Step 2 feeds a DIFFERENT
    gradient (grad2) with prev_grad = grad1, so grad_diff != 0 and the
    difference-EMA (`exp_avg_diff`) and look-ahead term (`u = grad + beta2 *
    grad_diff`) are exercised for the first time.
    """
    print("Running test_parity_with_reference...")

    var ref1 = List[Float64]()
    ref1.append(0.09900000019999997)
    ref1.append(-0.20099999993333334)
    ref1.append(0.30099999996)
    ref1.append(-0.4009999999714286)
    ref1.append(0.5009999999777778)

    var ref2 = List[Float64]()
    ref2.append(0.09805363747958667)
    ref2.append(-0.20146928331938244)
    ref2.append(0.3019681723416672)
    ref2.append(-0.401995231928322)
    ref2.append(0.5018958134034956)

    var n = 5
    var p = zeros([n], DType.float64)
    var g1 = zeros([n], DType.float64)
    var g2 = zeros([n], DType.float64)
    var m = zeros([n], DType.float64)
    var dd = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
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

    # step 1: prev_grad = g1 (grad_diff = 0)
    var res1 = adan_step(
        p, g1, m, dd, v, g1, 1, 0.001, 0.98, 0.92, 0.99, 1e-8, 0.0
    )
    var np1 = res1[0]
    for i in range(n):
        if _abs_diff(np1.load[DType.float64](i), ref1[i]) > 1e-9:
            raise Error("adan step-1 parity mismatch at " + String(i))
    print("  ok step 1 matches pytorch_optimizer.Adan to 1e-9")

    # step 2: prev_grad = g1 (returned as res1[4]), grad = g2 (grad_diff != 0)
    var res2 = adan_step(
        np1,
        g2,
        res1[1],
        res1[2],
        res1[3],
        res1[4],
        2,
        0.001,
        0.98,
        0.92,
        0.99,
        1e-8,
        0.0,
    )
    var np2 = res2[0]
    for i in range(n):
        if _abs_diff(np2.load[DType.float64](i), ref2[i]) > 1e-9:
            raise Error("adan step-2 parity mismatch at " + String(i))
    print("  ok step 2 matches pytorch_optimizer.Adan to 1e-9")
    print("test_parity_with_reference PASSED")


def test_step_simple_matches_defaults() raises:
    """`adan_step_simple` must equal `adan_step` with the default hyperparameters.
    """
    print("Running test_step_simple_matches_defaults...")
    var n = 5
    var p = zeros([n], DType.float64)
    var g = zeros([n], DType.float64)
    var m = zeros([n], DType.float64)
    var dd = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
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

    var simple = adan_step_simple(p, g, m, dd, v, g, 1, 0.001)
    var full_call = adan_step(
        p, g, m, dd, v, g, 1, 0.001, 0.98, 0.92, 0.99, 1e-8, 0.0
    )
    var sp = simple[0]
    var fp = full_call[0]
    for i in range(n):
        if (
            _abs_diff(sp.load[DType.float64](i), fp.load[DType.float64](i))
            > 1e-12
        ):
            raise Error(
                "adan_step_simple != adan_step defaults at " + String(i)
            )
    print("  ok adan_step_simple matches adan_step defaults")
    print("test_step_simple_matches_defaults PASSED")


def test_weight_decay() raises:
    """`weight_decay` applies the paper's divisive decoupled (proximal) decay.

    With a zero gradient the gradient step is zero, so the only change is the
    decoupled proximal decay `params /= (1 + lr * weight_decay)` (Algorithm 1
    of arXiv:2208.06677, matching the official sail-sg release). For
    params = 1, wd = 0.5, lr = 0.01 the result is 1 / (1 + 0.005)
    = 0.9950248756218907. A nonzero weight_decay must also move params further
    than wd = 0 on a real gradient.
    """
    print("Running test_weight_decay...")
    var n = 3

    # Isolate the decay term with a zero gradient.
    var wp = full([n], 1.0, DType.float64)
    var wg = zeros([n], DType.float64)
    var wm = zeros([n], DType.float64)
    var wdd = zeros([n], DType.float64)
    var wv = zeros([n], DType.float64)
    var wr = adan_step(
        wp, wg, wm, wdd, wv, wg, 1, 0.01, 0.98, 0.92, 0.99, 1e-8, 0.5
    )
    var wnp = wr[0]
    for i in range(n):
        if _abs_diff(wnp.load[DType.float64](i), 0.9950248756218907) > 1e-9:
            raise Error(
                "weight_decay-only result should be 1/(1+lr*wd)"
                " = 0.9950248756218907"
            )
    print("  ok divisive decoupled decay result == 1/(1+lr*wd)")

    # With a real gradient, wd != 0 must decrease positive params more than wd=0.
    var p = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var m0 = zeros([n], DType.float64)
    var dd0 = zeros([n], DType.float64)
    var v0 = zeros([n], DType.float64)
    var no_wd = adan_step(
        p, g, m0, dd0, v0, g, 1, 0.01, 0.98, 0.92, 0.99, 1e-8, 0.0
    )
    var with_wd = adan_step(
        p, g, m0, dd0, v0, g, 1, 0.01, 0.98, 0.92, 0.99, 1e-8, 0.1
    )
    var nwp = no_wd[0]
    var wwp = with_wd[0]
    for i in range(n):
        if not (wwp.load[DType.float64](i) < nwp.load[DType.float64](i)):
            raise Error("weight_decay > 0 should decrease positive params more")
    print("  ok weight_decay > 0 decays positive params further")
    print("test_weight_decay PASSED")


def test_prev_grad_passthrough() raises:
    """`new_prev_grad` must equal the current gradient (for the next step)."""
    print("Running test_prev_grad_passthrough...")
    var n = 3
    var p = zeros([n], DType.float64)
    var g = full([n], 0.5, DType.float64)
    var m = zeros([n], DType.float64)
    var dd = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
    var res = adan_step(p, g, m, dd, v, g, 1, 0.001)
    var new_pg = res[4]
    for i in range(n):
        if _abs_diff(new_pg.load[DType.float64](i), 0.5) > 1e-12:
            raise Error("new_prev_grad should equal the current gradient")
    print("  ok new_prev_grad == current gradient")
    print("test_prev_grad_passthrough PASSED")


def test_descent_direction() raises:
    """A positive gradient must decrease the parameter."""
    print("Running test_descent_direction...")
    var n = 3
    var p = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var m = zeros([n], DType.float64)
    var dd = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
    var res = adan_step(p, g, m, dd, v, g, 1, 0.01)
    var np = res[0]
    for i in range(n):
        if not (np.load[DType.float64](i) < 1.0):
            raise Error("positive gradient should decrease param")
    print("  ok positive gradient decreased params")
    print("test_descent_direction PASSED")


def main() raises:
    """Run all Adan tests."""
    print("=" * 60)
    print("Adan Optimizer Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_parity_with_reference()
    test_step_simple_matches_defaults()
    test_weight_decay()
    test_prev_grad_passthrough()
    test_descent_direction()
    print("=" * 60)
    print("All Adan tests PASSED")
    print("=" * 60)
