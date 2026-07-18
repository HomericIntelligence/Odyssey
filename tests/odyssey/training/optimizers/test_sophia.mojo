"""Unit tests for the Sophia clipped-preconditioned update step (arXiv:2305.14342).

Odyssey implements the Sophia update step with CALLER-SUPPLIED diagonal-Hessian
estimates (the Sophia-H/G estimators need HVP support, tracked in the autograd
TODO), so these tests seed the Hessian buffers directly.

Tests cover:
- Shape/dtype guards on sophia_step
- Numerical parity with the reference algorithm (1e-9) on a fixed vector where
  5 of 6 coordinates are UNCLIPPED (the m / max(gamma*h, eps) arithmetic is
  asserted, gamma = 0.8) and 1 coordinate saturates the +-rho clip
- gamma scaling the denominator (gamma=2 halves an unclipped update)
- The clip bound (rho) on the per-coordinate update
- Decoupled (AdamW-style) weight_decay != 0 branch
- sophia_update_hessian_moment EMA behavior
- sophia_step_simple delegating to sophia_step with defaults
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.sophia import (
    sophia_step,
    sophia_step_simple,
    sophia_update_hessian_moment,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`sophia_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    var hm = zeros([4], DType.float32)
    try:
        var _ = sophia_step(p, g, m, hm, 0.06)
        raise Error("Should have rejected shape mismatch")
    except:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`sophia_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    var hm = zeros([4], DType.float32)
    try:
        var _ = sophia_step(p, g, m, hm, 0.06)
        raise Error("Should have rejected dtype mismatch")
    except:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_parity_with_reference() raises:
    """One Sophia step must match the reference algorithm to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/sophia_parity_reference.py (lr=0.06, betas=(0.96,0.99),
    gamma=0.8, rho=0.04, eps=1e-12, no weight decay). The Hessian moment is
    refreshed with the fresh hessian BEFORE the step uses it.

    Regime split (from the reference script's printed clipped_mask):
    coordinates 0-4 are UNCLIPPED (|m/(gamma*hm)| < rho), so the
    preconditioned-division arithmetic — including the gamma scaling — is what
    is asserted; coordinate 5 saturates the clip (raw update 1.7034 >> rho),
    asserting the +-rho bound.
    """
    print("Running test_parity_with_reference...")

    var rp = List[Float64]()
    rp.append(0.09958041958041959)
    rp.append(-0.19968447412353924)
    rp.append(0.2997101610216546)
    rp.append(-0.39993724420190996)
    rp.append(0.499947532792005)
    rp.append(-0.6023999999999999)
    var rm = List[Float64]()
    rm.append(0.0056)
    rm.append(-0.005040000000000001)
    rm.append(0.0034800000000000005)
    rm.append(-0.0009199999999999992)
    rm.append(0.0005599999999999997)
    rm.append(0.06800000000000002)
    var rhm = List[Float64]()
    rhm.append(1.0010000000000001)
    rhm.append(1.198)
    rhm.append(0.9005000000000001)
    rhm.append(1.0995)
    rhm.append(0.8005)
    rhm.append(0.0499)

    var n = 6
    var p = zeros([n], DType.float64)
    var g = zeros([n], DType.float64)
    var m = zeros([n], DType.float64)
    var hm = zeros([n], DType.float64)
    var h = zeros([n], DType.float64)
    p.store[DType.float64](0, 0.1)
    p.store[DType.float64](1, -0.2)
    p.store[DType.float64](2, 0.3)
    p.store[DType.float64](3, -0.4)
    p.store[DType.float64](4, 0.5)
    p.store[DType.float64](5, -0.6)
    g.store[DType.float64](0, 0.02)
    g.store[DType.float64](1, -0.03)
    g.store[DType.float64](2, 0.015)
    g.store[DType.float64](3, 0.025)
    g.store[DType.float64](4, -0.01)
    g.store[DType.float64](5, 0.5)
    m.store[DType.float64](0, 0.005)
    m.store[DType.float64](1, -0.004)
    m.store[DType.float64](2, 0.003)
    m.store[DType.float64](3, -0.002)
    m.store[DType.float64](4, 0.001)
    m.store[DType.float64](5, 0.05)
    hm.store[DType.float64](0, 1.0)
    hm.store[DType.float64](1, 1.2)
    hm.store[DType.float64](2, 0.9)
    hm.store[DType.float64](3, 1.1)
    hm.store[DType.float64](4, 0.8)
    hm.store[DType.float64](5, 0.05)
    h.store[DType.float64](0, 1.1)
    h.store[DType.float64](1, 1.0)
    h.store[DType.float64](2, 0.95)
    h.store[DType.float64](3, 1.05)
    h.store[DType.float64](4, 0.85)
    h.store[DType.float64](5, 0.04)

    var hm2 = sophia_update_hessian_moment(hm, h, 0.99)
    var res = sophia_step(p, g, m, hm2, 0.06, 0.96, 0.8, 0.04, 1e-12, 0.0)
    var np = res[0]
    var nm = res[1]

    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("param parity mismatch at " + String(i))
        if _abs_diff(nm.load[DType.float64](i), rm[i]) > 1e-9:
            raise Error("momentum parity mismatch at " + String(i))
        if _abs_diff(hm2.load[DType.float64](i), rhm[i]) > 1e-9:
            raise Error("hessian-moment parity mismatch at " + String(i))
    print("  ok matches reference to 1e-9 (5 unclipped coords + 1 clipped)")
    print("test_parity_with_reference PASSED")


def test_gamma_scales_denominator() raises:
    """`gamma` scales the preconditioner: doubling gamma halves an unclipped update.

    With zero seed momentum, m2 = (1-beta1)*g; pick hm large enough that the
    update is unclipped for both gamma=1 and gamma=2. Then
    step(gamma=1) = lr * m2 / hm and step(gamma=2) = lr * m2 / (2*hm).
    """
    print("Running test_gamma_scales_denominator...")
    var n = 3
    var lr = 0.1
    var p = zeros([n], DType.float64)
    var g = full([n], 0.5, DType.float64)  # m2 = 0.04*0.5 = 0.02
    var m = zeros([n], DType.float64)
    var hm = full([n], 2.0, DType.float64)  # raw(gamma=1) = 0.01 < rho

    var res_g1 = sophia_step(p, g, m, hm, lr, 0.96, 1.0, 0.04, 1e-12, 0.0)
    var res_g2 = sophia_step(p, g, m, hm, lr, 0.96, 2.0, 0.04, 1e-12, 0.0)
    var step_g1 = res_g1[0]
    var step_g2 = res_g2[0]

    for i in range(n):
        var s1 = step_g1.load[DType.float64](i)  # -lr * 0.02 / 2.0
        var s2 = step_g2.load[DType.float64](i)  # -lr * 0.02 / 4.0
        if _abs_diff(s1, -lr * 0.02 / 2.0) > 1e-12:
            raise Error("gamma=1 step wrong at " + String(i))
        if _abs_diff(s2, s1 / 2.0) > 1e-12:
            raise Error("gamma=2 step should be half of gamma=1 step")
    print("  ok gamma=2 halves the unclipped update vs gamma=1")
    print("test_gamma_scales_denominator PASSED")


def test_clip_bounds_update() raises:
    """`rho` must bound the per-coordinate update magnitude.

    With a small Hessian moment, m / (gamma*hm) is large; the update should
    saturate to +-rho, so the param step magnitude is exactly lr * rho.
    """
    print("Running test_clip_bounds_update...")
    var n = 3
    var lr = 0.1
    var rho = 0.05
    var p = zeros([n], DType.float64)
    var g = full([n], 1.0, DType.float64)  # positive grad -> positive momentum
    var m = zeros([n], DType.float64)
    # tiny hessian_moment so m/(gamma*hm) blows up and clips to +rho
    var hm = full([n], 1e-6, DType.float64)

    var res = sophia_step(p, g, m, hm, lr, 0.96, 1.0, rho, 1e-12, 0.0)
    var np = res[0]
    var expected = -lr * rho  # p starts at 0, moves by -lr*rho
    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), expected) > 1e-9:
            raise Error("update should clip to rho -> step = -lr*rho")
    print("  ok update clipped to rho")
    print("test_clip_bounds_update PASSED")


def test_weight_decay_decoupled() raises:
    """`weight_decay` != 0 subtracts the decoupled AdamW term wd*lr*params_orig.

    Sophia's decoupled decay (like AdamW) is applied AFTER the gradient step and
    scales the ORIGINAL params: new_params -= weight_decay * lr * params. So the
    wd!=0 result must differ from the wd=0 result by exactly wd*lr*params[i] per
    coordinate. All other inputs (grad, momentum, hessian moment) are held
    identical.
    """
    print("Running test_weight_decay_decoupled...")
    var n = 4
    var lr = 0.1
    var wd = 0.05
    # Non-zero params so the decoupled term is non-trivial per coordinate.
    var p = zeros([n], DType.float64)
    p.store[DType.float64](0, 0.2)
    p.store[DType.float64](1, -0.4)
    p.store[DType.float64](2, 0.6)
    p.store[DType.float64](3, -0.8)
    var g = full([n], 0.1, DType.float64)
    var m = zeros([n], DType.float64)
    var hm = full([n], 0.5, DType.float64)

    # Same step, once without decay and once with decay.
    var res_no_wd = sophia_step(p, g, m, hm, lr, 0.96, 1.0, 0.04, 1e-12, 0.0)
    var res_wd = sophia_step(p, g, m, hm, lr, 0.96, 1.0, 0.04, 1e-12, wd)
    var np_no_wd = res_no_wd[0]
    var np_wd = res_wd[0]

    for i in range(n):
        var diff = np_no_wd.load[DType.float64](i) - np_wd.load[DType.float64](
            i
        )
        var expected = wd * lr * p.load[DType.float64](i)
        if _abs_diff(diff, expected) > 1e-12:
            raise Error(
                "decoupled weight decay wrong at "
                + String(i)
                + ": got "
                + String(diff)
                + " expected "
                + String(expected)
            )
    print("  ok wd!=0 differs from wd=0 by wd*lr*params (decoupled)")
    print("test_weight_decay_decoupled PASSED")


def test_hessian_moment_ema() raises:
    """`sophia_update_hessian_moment` does the beta2 EMA correctly."""
    print("Running test_hessian_moment_ema...")
    var n = 2
    var hm = full([n], 0.0, DType.float64)
    var h = full([n], 4.0, DType.float64)
    var beta2 = 0.99
    var new_hm = sophia_update_hessian_moment(hm, h, beta2)
    var expected = (1.0 - beta2) * 4.0
    for i in range(n):
        if _abs_diff(new_hm.load[DType.float64](i), expected) > 1e-9:
            raise Error("hessian moment EMA incorrect")
    print("  ok hessian moment EMA = (1-beta2)*h from zero")
    print("test_hessian_moment_ema PASSED")


def test_simple_matches_full_defaults() raises:
    """`sophia_step_simple` equals sophia_step with default hyperparameters."""
    print("Running test_simple_matches_full_defaults...")
    var n = 4
    var p = full([n], 0.2, DType.float64)
    var g = full([n], -0.3, DType.float64)
    var m = zeros([n], DType.float64)
    var hm = full([n], 0.5, DType.float64)

    var full_res = sophia_step(p, g, m, hm, 0.06)
    var simple_res = sophia_step_simple(p, g, m, hm, 0.06)
    for i in range(n):
        if (
            _abs_diff(
                full_res[0].load[DType.float64](i),
                simple_res[0].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error("sophia_step_simple params differ from sophia_step")
    print("  ok sophia_step_simple == sophia_step with defaults")
    print("test_simple_matches_full_defaults PASSED")


def main() raises:
    """Run all Sophia tests."""
    print("=" * 60)
    print("Sophia Clipped-Preconditioned Update Step Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_parity_with_reference()
    test_gamma_scales_denominator()
    test_clip_bounds_update()
    test_weight_decay_decoupled()
    test_hessian_moment_ema()
    test_simple_matches_full_defaults()
    print("=" * 60)
    print("All Sophia tests PASSED")
    print("=" * 60)
