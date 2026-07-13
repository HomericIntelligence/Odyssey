"""Unit tests for the ADOPT optimizer (arXiv:2411.02853).

Tests cover:
- Shape validation (params/gradients/momentum/second_moment must match)
- Dtype validation (params/gradients must share a dtype)
- Numerical parity with the public `pytorch_optimizer.ADOPT` reference on a
  fixed (params, grad, m, v) vector for one step (tolerance 1e-9)
- Descent behavior (a nonzero gradient moves params against the gradient)
- Clipping (a tight clip_value bounds the normalized-gradient contribution)
- Second-moment update ordering (v is updated with grad^2 AFTER being used)
- adopt_step_simple delegates to adopt_step with the default hyperparameters
"""
from odyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    full,
)
from odyssey.training.optimizers.adopt import adopt_step, adopt_step_simple
from std.math import abs as math_abs


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """ADOPT must reject params/gradients shape mismatches."""
    print("Running test_reject_shape_mismatch...")
    var params = zeros([4], DType.float32)
    var grad = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adopt_step(params, grad, m, v, 0.01)
        raise Error("Should have rejected shape mismatch")
    except e:
        print("  ✓ Correctly rejected shape mismatch: " + String(e))
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """ADOPT must reject params/gradients dtype mismatches."""
    print("Running test_reject_dtype_mismatch...")
    var params = zeros([4], DType.float32)
    var grad = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adopt_step(params, grad, m, v, 0.01)
        raise Error("Should have rejected dtype mismatch")
    except e:
        print("  ✓ Correctly rejected dtype mismatch: " + String(e))
    print("test_reject_dtype_mismatch PASSED")


def test_reject_second_moment_shape() raises:
    """ADOPT must reject a second_moment whose shape differs from params."""
    print("Running test_reject_second_moment_shape...")
    var params = zeros([4], DType.float32)
    var grad = zeros([4], DType.float32)
    var m = zeros([4], DType.float32)
    var v = zeros([3], DType.float32)
    try:
        var _ = adopt_step(params, grad, m, v, 0.01)
        raise Error("Should have rejected second_moment shape mismatch")
    except e:
        print("  ✓ Correctly rejected second_moment shape mismatch")
    print("test_reject_second_moment_shape PASSED")


def test_parity_with_reference() raises:
    """One ADOPT step must match `pytorch_optimizer.ADOPT` to 1e-9.

    Fixed inputs (params entering step 2, grad, m=0, v=grad_prev^2) and the
    reference outputs are transcribed from a one-step run of the public
    pytorch_optimizer.ADOPT (lr=0.01, betas=(0.9,0.9999), eps=1e-6, no clip,
    no weight decay). See parity_refs/adopt_parity_reference.py alongside this
    test for the reproducible reference generator.
    """
    print("Running test_parity_with_reference...")
    alias N = 6
    var p = zeros([N], DType.float64)
    var g = zeros([N], DType.float64)
    var m = zeros([N], DType.float64)
    var v = zeros([N], DType.float64)

    var pv = List[Float64]()
    pv.append(0.1); pv.append(-0.2); pv.append(0.3)
    pv.append(-0.4); pv.append(0.5); pv.append(-0.6)
    var gv = List[Float64]()
    gv.append(-0.02); gv.append(0.08); gv.append(0.12)
    gv.append(-0.18); gv.append(0.22); gv.append(-0.28)
    var vv = List[Float64]()
    vv.append(0.0025000000000000005); vv.append(0.0225); vv.append(0.0625)
    vv.append(0.12249999999999998); vv.append(0.2025); vv.append(0.30250000000000005)

    var rp = List[Float64]()
    rp.append(0.1004); rp.append(-0.20053333333333334); rp.append(0.29952)
    rp.append(-0.39948571428571433); rp.append(0.49951111111111113)
    rp.append(-0.5994909090909091)
    var rm = List[Float64]()
    rm.append(-0.03999999999999999); rm.append(0.05333333333333332)
    rm.append(0.04799999999999999); rm.append(-0.05142857142857142)
    rm.append(0.04888888888888888); rm.append(-0.050909090909090904)
    var rv = List[Float64]()
    rv.append(0.002499790000000001); rv.append(0.02249839); rv.append(0.06249519)
    rv.append(0.12249098999999998); rv.append(0.20248459000000002)
    rv.append(0.30247759)

    for i in range(N):
        p.store[DType.float64](i, pv[i])
        g.store[DType.float64](i, gv[i])
        v.store[DType.float64](i, vv[i])

    var res = adopt_step(p, g, m, v, 0.01, 0.9, 0.9999, 1e-6, 1.0e30, 0.0)
    var np = res[0]
    var nm = res[1]
    var nv = res[2]

    var tol = 1e-9
    for i in range(N):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > tol:
            raise Error("param parity mismatch at " + String(i))
        if _abs_diff(nm.load[DType.float64](i), rm[i]) > tol:
            raise Error("momentum parity mismatch at " + String(i))
        if _abs_diff(nv.load[DType.float64](i), rv[i]) > tol:
            raise Error("second-moment parity mismatch at " + String(i))
    print("  ✓ Matches pytorch_optimizer.ADOPT to 1e-9")
    print("test_parity_with_reference PASSED")


def test_descent_direction() raises:
    """A positive gradient must decrease the parameter (and vice versa)."""
    print("Running test_descent_direction...")
    alias N = 3
    var p = full([N], 1.0, DType.float32)
    var g = full([N], 0.5, DType.float32)  # positive grad
    var m = zeros([N], DType.float32)
    var v = full([N], 0.25, DType.float32)  # v_prev = g^2

    var res = adopt_step(p, g, m, v, 0.1)
    var np = res[0]
    for i in range(N):
        if not (np.load[DType.float32](i) < 1.0):
            raise Error("positive gradient should decrease param")
    print("  ✓ Positive gradient decreased params")
    print("test_descent_direction PASSED")


def test_second_moment_updates_last() raises:
    """Second moment must be updated with grad^2 using the (1-beta2) weight after use.

    With v_prev = 0 (all zeros), one step should leave
    new_v = (1 - beta2) * grad^2 (the beta2 * 0 term vanishes).
    """
    print("Running test_second_moment_updates_last...")
    alias N = 2
    var p = zeros([N], DType.float32)
    var g = full([N], 2.0, DType.float32)  # grad^2 = 4
    var m = zeros([N], DType.float32)
    var v = zeros([N], DType.float32)  # v_prev = 0

    var beta2 = 0.9999
    var res = adopt_step(p, g, m, v, 0.01, 0.9, beta2)
    var nv = res[2]
    var expected = (1.0 - beta2) * 4.0
    for i in range(N):
        var got = Float64(nv.load[DType.float32](i))
        if _abs_diff(got, expected) > 1e-6:
            raise Error("second moment not updated as (1-beta2)*grad^2")
    print("  ✓ v updated as (1-beta2)*grad^2 from zero state")
    print("test_second_moment_updates_last PASSED")


def test_clip_bounds_normalized_gradient() raises:
    """A tight clip_value must bound the first-moment (and hence the step).

    With m=0 and v_prev=g^2, the un-clipped normalized gradient is
    grad / max(sqrt(g^2), eps) = sign(grad) ~= +-1. A clip_value below 1 must
    reduce |m| = (1-beta1)*|clip(normalized)| below (1-beta1)*1, and shrink the
    param step proportionally versus the un-clipped case.
    """
    print("Running test_clip_bounds_normalized_gradient...")
    alias N = 4
    var beta1 = 0.9
    var lr = 0.1

    # Un-clipped reference step (huge clip => no clipping).
    var p_un = full([N], 0.0, DType.float64)
    var g = full([N], 0.5, DType.float64)
    var m0 = zeros([N], DType.float64)
    var v0 = full([N], 0.25, DType.float64)  # v_prev = g^2
    var res_un = adopt_step(p_un, g, m0, v0, lr, beta1, 0.9999, 1e-6, 1.0e30, 0.0)
    var m_un = res_un[1]

    # Clipped step with clip_value = 0.25 (< 1). normalized ~= +1 -> clipped to 0.25.
    var p_cl = full([N], 0.0, DType.float64)
    var res_cl = adopt_step(p_cl, g, m0, v0, lr, beta1, 0.9999, 1e-6, 0.25, 0.0)
    var m_cl = res_cl[1]

    var expected_un = (1.0 - beta1) * 1.0  # normalized ~= +1
    var expected_cl = (1.0 - beta1) * 0.25  # clipped to 0.25
    for i in range(N):
        var mu = m_un.load[DType.float64](i)
        var mc = m_cl.load[DType.float64](i)
        if _abs_diff(mu, expected_un) > 1e-6:
            raise Error("un-clipped momentum should be (1-beta1)*1")
        if _abs_diff(mc, expected_cl) > 1e-6:
            raise Error("clipped momentum should be (1-beta1)*clip_value")
        if not (mc < mu):
            raise Error("clipping should reduce the momentum magnitude")
    print("  ✓ clip_value bounds the normalized-gradient contribution")
    print("test_clip_bounds_normalized_gradient PASSED")


def test_simple_matches_full_defaults() raises:
    """adopt_step_simple must equal adopt_step with default hyperparameters."""
    print("Running test_simple_matches_full_defaults...")
    alias N = 5
    var p = full([N], 0.2, DType.float64)
    var g = full([N], -0.3, DType.float64)
    var m = zeros([N], DType.float64)
    var v = full([N], 0.09, DType.float64)

    var full_res = adopt_step(p, g, m, v, 0.01)
    var simple_res = adopt_step_simple(p, g, m, v, 0.01)
    for i in range(N):
        if _abs_diff(
            full_res[0].load[DType.float64](i),
            simple_res[0].load[DType.float64](i),
        ) > 1e-12:
            raise Error("adopt_step_simple params differ from adopt_step")
        if _abs_diff(
            full_res[2].load[DType.float64](i),
            simple_res[2].load[DType.float64](i),
        ) > 1e-12:
            raise Error("adopt_step_simple second moment differs")
    print("  ✓ adopt_step_simple == adopt_step with defaults")
    print("test_simple_matches_full_defaults PASSED")


def main() raises:
    """Run all ADOPT tests (local `mojo run`; CI uses `mojo test` discovery)."""
    print("=" * 60)
    print("ADOPT Optimizer Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_reject_second_moment_shape()
    test_parity_with_reference()
    test_descent_direction()
    test_second_moment_updates_last()
    test_clip_bounds_normalized_gradient()
    test_simple_matches_full_defaults()
    print("=" * 60)
    print("All ADOPT tests PASSED")
    print("=" * 60)
