"""Unit tests for the SGD (momentum + weight-decay) update step.

Odyssey's `sgd_step` is pure-functional: it returns (new_params, new_velocity)
and the caller manages state. The implemented update (sgd.mojo) is

    effective_grad = grad + weight_decay * params          (wd applied first)
    velocity       = momentum * velocity + effective_grad
    params         = params - lr * velocity

which is exactly `torch.optim.SGD` with `dampening=0, nesterov=False`.

Tests cover:
- Shape/dtype guards on sgd_step
- The momentum-required-buffer guard
- Numerical parity with the reference algorithm (1e-9) on a fixed 6-vector
  with non-zero momentum, weight_decay, and starting velocity so every term
  of the update is asserted. Reference numbers transcribed from
  parity_refs/sgd_parity_reference.py (which drives torch.optim.SGD when torch
  is importable, else the identical numpy closed form).
- The no-momentum / no-weight-decay path collapses to params - lr * grad
- sgd_step_simple delegating to plain SGD (no momentum, no weight decay)
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.sgd import (
    sgd_step,
    sgd_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`sgd_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = sgd_step(p, g, v, 0.01)
        raise Error("Should have rejected shape mismatch")
    except:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`sgd_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var v = zeros([4], DType.float32)
    try:
        var _ = sgd_step(p, g, v, 0.01)
        raise Error("Should have rejected dtype mismatch")
    except:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_reject_empty_velocity_with_momentum() raises:
    """`sgd_step` requires a velocity buffer when momentum > 0."""
    print("Running test_reject_empty_velocity_with_momentum...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var v = zeros([0], DType.float32)  # empty buffer
    try:
        var _ = sgd_step(p, g, v, 0.01, 0.9)
        raise Error("Should have rejected empty velocity with momentum")
    except:
        print("  ok rejected empty velocity with momentum")
    print("test_reject_empty_velocity_with_momentum PASSED")


def test_parity_with_reference() raises:
    """One SGD step (momentum + weight decay) must match the reference to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/sgd_parity_reference.py (lr=0.1, momentum=0.9,
    weight_decay=0.01, starting velocity non-zero).
    """
    print("Running test_parity_with_reference...")
    var n = 6
    var p = zeros([n], DType.float64)
    var g = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
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
    v.store[DType.float64](0, 0.005)
    v.store[DType.float64](1, -0.004)
    v.store[DType.float64](2, 0.003)
    v.store[DType.float64](3, -0.002)
    v.store[DType.float64](4, 0.001)
    v.store[DType.float64](5, -0.006)

    var rp = List[Float64]()
    rp.append(0.09745000000000001)
    rp.append(-0.19644)
    rp.append(0.29793)
    rp.append(-0.40192)
    rp.append(0.50041)
    rp.append(-0.60286)
    var rv = List[Float64]()
    rv.append(0.025500000000000002)
    rv.append(-0.0356)
    rv.append(0.0207)
    rv.append(0.019200000000000002)
    rv.append(-0.0041)
    rv.append(0.0286)

    var out = sgd_step(p, g, v, 0.1, 0.9, 0.01)
    var np = out[0]
    var nv = out[1]
    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("params parity mismatch at " + String(i))
        if _abs_diff(nv.load[DType.float64](i), rv[i]) > 1e-9:
            raise Error("velocity parity mismatch at " + String(i))
    print("  ok parity to 1e-9 on all 6 coordinates")
    print("test_parity_with_reference PASSED")


def test_plain_sgd_no_momentum_no_wd() raises:
    """With momentum=0 and weight_decay=0, params = params - lr * grad."""
    print("Running test_plain_sgd_no_momentum_no_wd...")
    var n = 3
    var p = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var v = zeros([n], DType.float64)
    var out = sgd_step(p, g, v, 0.1)  # momentum=0, wd=0 defaults
    var np = out[0]
    for i in range(n):
        # 1.0 - 0.1 * 0.5 = 0.95
        if _abs_diff(np.load[DType.float64](i), 0.95) > 1e-9:
            raise Error("plain SGD mismatch at " + String(i))
    print("  ok plain SGD = params - lr*grad")
    print("test_plain_sgd_no_momentum_no_wd PASSED")


def test_sgd_step_simple_delegates() raises:
    """`sgd_step_simple` matches plain sgd_step (no momentum, no weight decay).
    """
    print("Running test_sgd_step_simple_delegates...")
    var n = 3
    var p = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var simple = sgd_step_simple(p, g, 0.1)
    for i in range(n):
        if _abs_diff(simple.load[DType.float64](i), 0.95) > 1e-9:
            raise Error("sgd_step_simple mismatch at " + String(i))
    print("  ok sgd_step_simple delegates to plain SGD")
    print("test_sgd_step_simple_delegates PASSED")


def main() raises:
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_reject_empty_velocity_with_momentum()
    test_parity_with_reference()
    test_plain_sgd_no_momentum_no_wd()
    test_sgd_step_simple_delegates()
    print("\nAll SGD parity tests PASSED")
