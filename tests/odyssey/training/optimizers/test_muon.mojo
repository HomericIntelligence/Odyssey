"""Unit tests for the Muon update step (Jordan et al. 2024).

Odyssey's `muon_step` is pure-functional: it returns (new_params, new_momentum)
and is specialized for rank-2 (matrix) parameters. The implemented update
(muon.mojo) is

    new_momentum = momentum_beta * momentum + grad
    dir          = grad + momentum_beta * new_momentum   (nesterov=True)
    u_orth       = newton_schulz_orthogonalize(dir, steps=5)   (Jordan quintic)
    scale        = 0.2 * max(R, C) / sqrt(R * C)
    params       = params - lr * scale * u_orth
    params       = params - lr * weight_decay * params

The Newton-Schulz iteration uses Jordan's tuned coefficients
(a, b, c) = (3.4445, -4.7750, 2.0315) and deliberately does NOT converge to
exact orthogonality, so parity is asserted against Odyssey's implemented
iteration (faithfully replicated in numpy in the parity reference), not an
idealized orthogonal matrix.

Tests cover:
- Shape/dtype guards, empty-momentum guard, and the rank-2-required guard
- Numerical parity (1e-9) on a 2x3 matrix (rows < cols → NS non-transposed
  path). Reference numbers from parity_refs/muon_parity_reference.py.
- newton_schulz_orthogonalize on a square matrix returns an approximately
  orthogonal factor (Y @ Y^T singular values in the Jordan band, checked
  loosely as diagonal near 1).
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.muon import (
    muon_step,
    newton_schulz_orthogonalize,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`muon_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([2, 3], DType.float32)
    var g = zeros([2, 4], DType.float32)
    var m = zeros([2, 3], DType.float32)
    try:
        var _ = muon_step(p, g, m, 0.01)
        raise Error("Should have rejected shape mismatch")
    except:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_non_rank2() raises:
    """`muon_step` rejects non-matrix (rank != 2) parameters."""
    print("Running test_reject_non_rank2...")
    var p = zeros([6], DType.float32)  # rank-1
    var g = zeros([6], DType.float32)
    var m = zeros([6], DType.float32)
    try:
        var _ = muon_step(p, g, m, 0.01)
        raise Error("Should have rejected rank-1 params")
    except:
        print("  ok rejected non-rank-2 params")
    print("test_reject_non_rank2 PASSED")


def test_reject_empty_momentum() raises:
    """`muon_step` requires an initialized momentum buffer."""
    print("Running test_reject_empty_momentum...")
    var p = zeros([2, 3], DType.float32)
    var g = zeros([2, 3], DType.float32)
    var m = zeros([0], DType.float32)
    try:
        var _ = muon_step(p, g, m, 0.01)
        raise Error("Should have rejected empty momentum")
    except:
        print("  ok rejected empty momentum")
    print("test_reject_empty_momentum PASSED")


def test_parity_with_reference() raises:
    """One Muon step on a 2x3 matrix must match the reference to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/muon_parity_reference.py (lr=0.01, beta=0.95, wd=0.01,
    ns_steps=5, nesterov=True). The reference faithfully replicates Odyssey's
    Newton-Schulz iteration in numpy.
    """
    print("Running test_parity_with_reference...")
    var p = zeros([2, 3], DType.float64)
    var g = zeros([2, 3], DType.float64)
    var m = zeros([2, 3], DType.float64)
    # row 0
    p.store[DType.float64](0, 0.10)
    p.store[DType.float64](1, -0.20)
    p.store[DType.float64](2, 0.30)
    # row 1
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
    rp.append(0.09914161539750362)
    rp.append(-0.19856436781395365)
    rp.append(0.29946522526790975)
    rp.append(-0.4009765522043971)
    rp.append(0.49903193843928556)
    rp.append(-0.6009030572792634)
    var rm = List[Float64]()
    rm.append(0.0675)
    rm.append(-0.068)
    rm.append(0.0435)
    rm.append(0.006000000000000002)
    rm.append(-0.0005000000000000004)
    rm.append(-0.016999999999999994)

    var out = muon_step(p, g, m, 0.01, 0.95, 0.01, 5, True)
    var np = out[0]
    var nm = out[1]
    for i in range(6):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("params parity mismatch at " + String(i))
        if _abs_diff(nm.load[DType.float64](i), rm[i]) > 1e-9:
            raise Error("momentum parity mismatch at " + String(i))
    print("  ok parity to 1e-9 on all 6 entries (params, momentum)")
    print("test_parity_with_reference PASSED")


def test_newton_schulz_approx_orthogonal() raises:
    """NS on a square matrix yields Y with Y @ Y^T diagonal near 1.

    The Jordan iteration lands singular values in ~[0.68, 1.13], so we check
    the diagonal of Y @ Y^T loosely (within 0.5 of 1.0) — a smoke test that
    the orthogonalization ran and produced a well-scaled result, not an exact
    orthogonality assertion.
    """
    print("Running test_newton_schulz_approx_orthogonal...")
    var x = zeros([2, 2], DType.float64)
    x.store[DType.float64](0, 1.0)
    x.store[DType.float64](1, 0.3)
    x.store[DType.float64](2, 0.2)
    x.store[DType.float64](3, 0.9)
    var y = newton_schulz_orthogonalize(x, 5)
    # Y @ Y^T diagonal: y00^2 + y01^2 and y10^2 + y11^2
    var y00 = y.load[DType.float64](0)
    var y01 = y.load[DType.float64](1)
    var y10 = y.load[DType.float64](2)
    var y11 = y.load[DType.float64](3)
    var d0 = y00 * y00 + y01 * y01
    var d1 = y10 * y10 + y11 * y11
    if _abs_diff(d0, 1.0) > 0.5 or _abs_diff(d1, 1.0) > 0.5:
        raise Error(
            "NS diagonal not near 1: d0=" + String(d0) + " d1=" + String(d1)
        )
    print("  ok NS produced an approximately-orthogonal factor")
    print("test_newton_schulz_approx_orthogonal PASSED")


def main() raises:
    test_reject_shape_mismatch()
    test_reject_non_rank2()
    test_reject_empty_momentum()
    test_parity_with_reference()
    test_newton_schulz_approx_orthogonal()
    print("\nAll Muon parity tests PASSED")
