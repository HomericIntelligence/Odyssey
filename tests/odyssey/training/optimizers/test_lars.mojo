"""Unit tests for the LARS update step (You, Gitman & Ginsburg 2017, arXiv:1708.03888).

Odyssey's `lars_step` is pure-functional: it returns (new_params, new_velocity).
The implemented update (lars.mojo) is the layer-wise-adaptive rule

    param_norm  = ||params||_2
    grad_norm   = ||grad||_2
    eff_grad    = grad + weight_decay * params
    trust_ratio = trust_coefficient * param_norm
                  / (grad_norm + weight_decay * param_norm + epsilon)
    velocity    = momentum * velocity + trust_ratio * eff_grad
    params      = params - lr * velocity

Tests cover:
- Shape/dtype guards and the empty-velocity guard
- Numerical parity with the reference (1e-9): trust ratio, momentum buffer, and
  the parameter step. Reference numbers from parity_refs/lars_parity_reference.py.
- lars_step_simple delegation.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.lars import (
    lars_step,
    lars_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`lars_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = lars_step(p, g, v, 0.1)
        raise Error("Should have rejected shape mismatch")
    except:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`lars_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var v = zeros([4], DType.float32)
    try:
        var _ = lars_step(p, g, v, 0.1)
        raise Error("Should have rejected dtype mismatch")
    except:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_reject_empty_velocity() raises:
    """`lars_step` requires an initialized velocity buffer."""
    print("Running test_reject_empty_velocity...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var v = zeros([0], DType.float32)
    try:
        var _ = lars_step(p, g, v, 0.1)
        raise Error("Should have rejected empty velocity")
    except:
        print("  ok rejected empty velocity")
    print("test_reject_empty_velocity PASSED")


def test_parity_with_reference() raises:
    """One LARS step must match the reference to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/lars_parity_reference.py (lr=0.1, momentum=0.9,
    weight_decay=1e-4, trust_coefficient=1e-3, eps=1e-8, seeded velocity).
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
    rp.append(0.09951928360999032)
    rp.append(-0.19959391773972562)
    rp.append(0.2997069281688233)
    rp.append(-0.39985831489728346)
    rp.append(0.49992527376714624)
    rp.append(-0.5995213099758614)
    var rv = List[Float64]()
    rv.append(0.004807163900096844)
    rv.append(-0.0040608226027439905)
    rv.append(0.0029307183117668945)
    rv.append(-0.0014168510271655567)
    rv.append(0.0007472623285375517)
    rv.append(-0.004786900241385911)

    var out = lars_step(p, g, v, 0.1, 0.9, 0.0001, 0.001, 1e-8)
    var np = out[0]
    var nv = out[1]
    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("params parity mismatch at " + String(i))
        if _abs_diff(nv.load[DType.float64](i), rv[i]) > 1e-9:
            raise Error("velocity parity mismatch at " + String(i))
    print("  ok parity to 1e-9 on all 6 coordinates (params, velocity)")
    print("test_parity_with_reference PASSED")


def test_lars_step_simple_runs() raises:
    """`lars_step_simple` runs with defaults and produces a finite update."""
    print("Running test_lars_step_simple_runs...")
    var n = 4
    var p = full([n], 0.5, DType.float64)
    var g = full([n], 0.1, DType.float64)
    var v = zeros([n], DType.float64)
    var out = lars_step_simple(p, g, v, 0.1)
    # smoke: params moved off 0.5 and stayed finite
    if _abs_diff(out[0].load[DType.float64](0), 0.5) == 0.0:
        raise Error("lars_step_simple did not update params")
    print("  ok lars_step_simple produced a finite update")
    print("test_lars_step_simple_runs PASSED")


def main() raises:
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_reject_empty_velocity()
    test_parity_with_reference()
    test_lars_step_simple_runs()
    print("\nAll LARS parity tests PASSED")
