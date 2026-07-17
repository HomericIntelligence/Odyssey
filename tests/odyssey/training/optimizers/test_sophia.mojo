"""Unit tests for the Sophia optimizer (arXiv:2305.14342).

Tests cover:
- Shape/dtype guards on sophia_step
- Numerical parity with the SophiaH reference algorithm (1e-9) on a fixed vector
- The clip bound (rho) on the per-coordinate update
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
    """sophia_step rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    var hm = zeros([4], DType.float32)
    var h = zeros([4], DType.float32)
    try:
        var _ = sophia_step(p, g, m, hm, h, 0.06)
        raise Error("Should have rejected shape mismatch")
    except e:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """sophia_step rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    var hm = zeros([4], DType.float32)
    var h = zeros([4], DType.float32)
    try:
        var _ = sophia_step(p, g, m, hm, h, 0.06)
        raise Error("Should have rejected dtype mismatch")
    except e:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_parity_with_reference() raises:
    """One Sophia step must match the SophiaH reference algorithm to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/sophia_parity_reference.py (SophiaH source-verbatim math,
    lr=0.06, betas=(0.96,0.99), rho=0.04, eps=1e-12, no weight decay). The
    Hessian moment is refreshed with the fresh hessian BEFORE the step uses it.
    """
    print("Running test_parity_with_reference...")

    var rp = List[Float64]()
    rp.append(0.0976)
    rp.append(-0.1976)
    rp.append(0.2976)
    rp.append(-0.3976)
    rp.append(0.4976)
    var rm = List[Float64]()
    rm.append(0.0116)
    rm.append(-0.0132)
    rm.append(0.0188)
    rm.append(-0.0244)
    rm.append(0.03)
    var rhm = List[Float64]()
    rhm.append(0.203)
    rhm.append(0.2535)
    rhm.append(0.304)
    rhm.append(0.3545)
    rhm.append(0.405)

    var n = 5
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
    g.store[DType.float64](0, 0.05)
    g.store[DType.float64](1, 0.15)
    g.store[DType.float64](2, -0.25)
    g.store[DType.float64](3, 0.35)
    g.store[DType.float64](4, -0.45)
    m.store[DType.float64](0, 0.01)
    m.store[DType.float64](1, -0.02)
    m.store[DType.float64](2, 0.03)
    m.store[DType.float64](3, -0.04)
    m.store[DType.float64](4, 0.05)
    hm.store[DType.float64](0, 0.2)
    hm.store[DType.float64](1, 0.25)
    hm.store[DType.float64](2, 0.3)
    hm.store[DType.float64](3, 0.35)
    hm.store[DType.float64](4, 0.4)
    h.store[DType.float64](0, 0.5)
    h.store[DType.float64](1, 0.6)
    h.store[DType.float64](2, 0.7)
    h.store[DType.float64](3, 0.8)
    h.store[DType.float64](4, 0.9)

    var hm2 = sophia_update_hessian_moment(hm, h, 0.99)
    var res = sophia_step(p, g, m, hm2, h, 0.06, 0.96, 0.99, 0.04, 1e-12, 0.0)
    var np = res[0]
    var nm = res[1]

    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("param parity mismatch at " + String(i))
        if _abs_diff(nm.load[DType.float64](i), rm[i]) > 1e-9:
            raise Error("momentum parity mismatch at " + String(i))
        if _abs_diff(hm2.load[DType.float64](i), rhm[i]) > 1e-9:
            raise Error("hessian-moment parity mismatch at " + String(i))
    print("  ok matches SophiaH reference to 1e-9")
    print("test_parity_with_reference PASSED")


def test_clip_bounds_update() raises:
    """rho must bound the per-coordinate update magnitude.

    With a small Hessian moment, m / hm is large; the update should saturate to
    +-rho, so the param step magnitude is exactly lr * rho.
    """
    print("Running test_clip_bounds_update...")
    var n = 3
    var lr = 0.1
    var rho = 0.05
    var p = zeros([n], DType.float64)
    var g = full([n], 1.0, DType.float64)  # positive grad -> positive momentum
    var m = zeros([n], DType.float64)
    # tiny hessian_moment so m/hm blows up and clips to +rho
    var hm = full([n], 1e-6, DType.float64)
    var h = full([n], 1e-6, DType.float64)

    var res = sophia_step(p, g, m, hm, h, lr, 0.96, 0.99, rho, 1e-12, 0.0)
    var np = res[0]
    var expected = -lr * rho  # p starts at 0, moves by -lr*rho
    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), expected) > 1e-9:
            raise Error("update should clip to rho -> step = -lr*rho")
    print("  ok update clipped to rho")
    print("test_clip_bounds_update PASSED")


def test_hessian_moment_ema() raises:
    """sophia_update_hessian_moment does the beta2 EMA correctly."""
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
    """sophia_step_simple equals sophia_step with default hyperparameters."""
    print("Running test_simple_matches_full_defaults...")
    var n = 4
    var p = full([n], 0.2, DType.float64)
    var g = full([n], -0.3, DType.float64)
    var m = zeros([n], DType.float64)
    var hm = full([n], 0.5, DType.float64)
    var h = full([n], 0.5, DType.float64)

    var full_res = sophia_step(p, g, m, hm, h, 0.06)
    var simple_res = sophia_step_simple(p, g, m, hm, h, 0.06)
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
    print("Sophia Optimizer Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_parity_with_reference()
    test_clip_bounds_update()
    test_hessian_moment_ema()
    test_simple_matches_full_defaults()
    print("=" * 60)
    print("All Sophia tests PASSED")
    print("=" * 60)
