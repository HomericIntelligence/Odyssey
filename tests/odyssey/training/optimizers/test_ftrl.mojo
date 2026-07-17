"""Unit tests for the FTRL-Proximal optimizer (McMahan et al. 2013).

Tests cover:
- Shape/dtype validation (rejects mismatched params/gradients/state)
- Numerical parity with an independent numpy transcription of McMahan Algorithm 1
  over a 2-step run (step 1 seeds state from zero; step 2 is a general step), 1e-6
- The L1 sparsity property: a coordinate whose |z| <= lambda1 is driven to EXACTLY
  zero (the feature that distinguishes FTRL from Adam/SGD)
- lambda1 = 0 produces a dense (no exact-zero-forcing) update via ftrl_step_simple
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.training.optimizers.ftrl import ftrl_step, ftrl_step_simple


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _seed(mut t: AnyTensor, vals: List[Float64]) raises:
    for i in range(len(vals)):
        t.store[DType.float64](i, vals[i])


def _lst(*vals: Float64) -> List[Float64]:
    """Build a List[Float64] from varargs (the List(a, b, ...) literal form
    does not parse on this Mojo build; append is the portable idiom)."""
    var out = List[Float64]()
    for v in vals:
        out.append(v)
    return out^


def test_reject_shape_mismatch() raises:
    """FTRL rejects mismatched params/gradients/state shapes."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([6], DType.float64)
    var g = zeros([5], DType.float64)
    var z = zeros([6], DType.float64)
    var n = zeros([6], DType.float64)
    try:
        var (_, _, _) = ftrl_step(p, g, z, n, 1.0)
        raise Error("should have rejected mismatched grad shape")
    except _:
        print("  ok rejected mismatched grad shape")
    var g2 = zeros([6], DType.float64)
    var z2 = zeros([5], DType.float64)
    try:
        var (_, _, _) = ftrl_step(p, g2, z2, n, 1.0)
        raise Error("should have rejected mismatched z shape")
    except _:
        print("  ok rejected mismatched z shape")
    print("test_reject_shape_mismatch PASSED")


def test_parity_two_step() raises:
    """Match the numpy transcription (McMahan Alg. 1) over 2 steps to 1e-6.

    Reference values from parity_refs/ftrl_parity_reference.py
    (N=6, alpha=0.1, beta=1.0, lambda1=0.02, lambda2=0.01, lr=1.0). Step 1 seeds
    (z, n) from zero; step 2 is a general step. Both are compared.
    """
    print("Running test_parity_two_step...")

    var params0 = _lst(0.10, -0.20, 0.30, -0.40, 0.50, -0.60)
    var grad_a = _lst(0.05, 0.15, -0.25, 0.35, -0.45, 0.55)
    var grad_b = _lst(-0.02, 0.08, 0.12, -0.18, 0.22, -0.28)

    var ref1 = _lst(
        -0.0,
        -0.0373588184,
        0.0783373301,
        -0.1280532939,
        0.1847002068,
        -0.2469374597,
    )
    var ref2 = _lst(
        0.0,
        -0.0441905861,
        0.0689499248,
        -0.1151461207,
        0.1700520865,
        -0.2296339729,
    )

    var w = zeros([6], DType.float64)
    _seed(w, params0)
    var g1 = zeros([6], DType.float64)
    _seed(g1, grad_a)
    var g2 = zeros([6], DType.float64)
    _seed(g2, grad_b)
    var z = zeros([6], DType.float64)
    var n = zeros([6], DType.float64)

    # step 1
    var r1 = ftrl_step(w, g1, z, n, 1.0, 0.1, 1.0, 0.02, 0.01)
    var w1 = r1[0]
    z = r1[1]
    n = r1[2]
    for i in range(6):
        if _abs_diff(w1.load[DType.float64](i), ref1[i]) > 1e-6:
            raise Error("FTRL step-1 mismatch at " + String(i))

    # step 2 (general step, from the seeded state)
    var r2 = ftrl_step(w1, g2, z, n, 1.0, 0.1, 1.0, 0.02, 0.01)
    var w2 = r2[0]
    for i in range(6):
        if _abs_diff(w2.load[DType.float64](i), ref2[i]) > 1e-6:
            raise Error("FTRL step-2 mismatch at " + String(i))

    print("  ok matches McMahan Alg.1 reference over 2 steps to 1e-6")
    print("test_parity_two_step PASSED")


def test_l1_sparsity() raises:
    """A coordinate with |z| <= lambda1 is driven to EXACTLY zero.

    Coordinate 0 (grad 0.05) has |z| below lambda1=0.02 after step 1, so FTRL's
    L1 shrinkage zeroes it exactly — the sparsity property. Assert it is bit-zero,
    not merely small.
    """
    print("Running test_l1_sparsity...")
    var params0 = _lst(0.10, -0.20, 0.30, -0.40, 0.50, -0.60)
    var grad_a = _lst(0.05, 0.15, -0.25, 0.35, -0.45, 0.55)

    var w = zeros([6], DType.float64)
    _seed(w, params0)
    var g = zeros([6], DType.float64)
    _seed(g, grad_a)
    var z = zeros([6], DType.float64)
    var n = zeros([6], DType.float64)

    var r = ftrl_step(w, g, z, n, 1.0, 0.1, 1.0, 0.02, 0.01)
    var w1 = r[0]
    if w1.load[DType.float64](0) != 0.0:
        raise Error("coordinate 0 should be exactly zero under L1 shrinkage")
    # a large-gradient coordinate must be nonzero
    if w1.load[DType.float64](5) == 0.0:
        raise Error("coordinate 5 (large grad) should be nonzero")
    print("  ok L1 drives small-|z| coordinate to exact zero")
    print("test_l1_sparsity PASSED")


def test_simple_no_reg_is_dense() raises:
    """The ftrl_step_simple (lambda1=0) produces a dense update — no forced zeros.

    With no L1 term the soft-threshold never clips to zero, so every coordinate
    with a nonzero accumulated z yields a nonzero weight (w = -z / denom).

    Input caveat: "dense" means no THRESHOLD-forced zeros, but z itself can be
    exactly zero by arithmetic cancellation: on step 1 from zero state,
    z = g - (|g|/alpha)*w0, which is exactly 0.0 whenever w0 == alpha*sign(g).
    The shared fixture (w0[0]=0.10 == alpha=0.1, g[0]=0.05 > 0) hits exactly
    that trap — w[0] = -0/denom = 0 is CORRECT dense behavior, not a bug. So
    this test uses w0[0]=0.15 to keep every z nonzero (numpy fp64 reference:
    z = [-0.025, 0.45, -1.0, 1.75, -2.7, 3.85], no zeros).
    """
    print("Running test_simple_no_reg_is_dense...")
    var params0 = _lst(0.15, -0.20, 0.30, -0.40, 0.50, -0.60)
    var grad_a = _lst(0.05, 0.15, -0.25, 0.35, -0.45, 0.55)

    var w = zeros([6], DType.float64)
    _seed(w, params0)
    var g = zeros([6], DType.float64)
    _seed(g, grad_a)
    var z = zeros([6], DType.float64)
    var n = zeros([6], DType.float64)

    var r = ftrl_step_simple(w, g, z, n, 1.0)
    var w1 = r[0]
    for i in range(6):
        if w1.load[DType.float64](i) == 0.0:
            raise Error(
                "no coordinate should be exactly zero without L1 (at "
                + String(i)
                + ")"
            )
    print("  ok no-regularization step is dense")
    print("test_simple_no_reg_is_dense PASSED")


def main() raises:
    """Run all FTRL tests."""
    print("=" * 60)
    print("FTRL-Proximal Optimizer Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_parity_two_step()
    test_l1_sparsity()
    test_simple_no_reg_is_dense()
    print("=" * 60)
    print("All FTRL tests PASSED")
    print("=" * 60)
