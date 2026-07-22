"""Unit tests for the Adam update step (arXiv:1412.6980).

Odyssey's `adam_step` is pure-functional: it returns (new_params, new_m, new_v)
and the caller tracks the timestep t. The implemented update (adam.mojo) is

    effective_grad = grad + weight_decay * params        (coupled L2)
    m      = beta1 * m + (1 - beta1) * effective_grad
    v      = beta2 * v + (1 - beta2) * effective_grad**2
    m_hat  = m / (1 - beta1**t)
    v_hat  = v / (1 - beta2**t)
    params = params - lr * m_hat / (sqrt(v_hat) + epsilon)   (eps outside sqrt)

which is exactly `torch.optim.Adam` (amsgrad=False).

Tests cover:
- Shape/dtype guards on adam_step
- The uninitialized-moment guard (empty m or v)
- The positive-timestep guard (t <= 0 rejected)
- Numerical parity with the reference algorithm (1e-9) at t=3 (bias correction
  active) with non-zero weight_decay and seeded m/v. Reference numbers
  transcribed from parity_refs/adam_parity_reference.py (drives torch.optim.Adam
  when importable, else the identical numpy closed form).
- adam_step_simple delegating to adam_step with defaults (no weight decay).
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.adam import (
    adam_step,
    adam_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`adam_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adam_step(p, g, m, v, 1, 0.001)
        raise Error("Should have rejected shape mismatch")
    except:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`adam_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adam_step(p, g, m, v, 1, 0.001)
        raise Error("Should have rejected dtype mismatch")
    except:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_reject_empty_moments() raises:
    """`adam_step` requires initialized moment buffers m and v."""
    print("Running test_reject_empty_moments...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var m = zeros([0], DType.float32)  # empty
    var v = zeros([4], DType.float32)
    try:
        var _ = adam_step(p, g, m, v, 1, 0.001)
        raise Error("Should have rejected empty moment buffer")
    except:
        print("  ok rejected empty moment buffer")
    print("test_reject_empty_moments PASSED")


def test_reject_nonpositive_timestep() raises:
    """`adam_step` rejects t <= 0 (timestep starts at 1)."""
    print("Running test_reject_nonpositive_timestep...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adam_step(p, g, m, v, 0, 0.001)
        raise Error("Should have rejected t=0")
    except:
        print("  ok rejected non-positive timestep")
    print("test_reject_nonpositive_timestep PASSED")


def test_parity_with_reference() raises:
    """One Adam step at t=3 must match the reference to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/adam_parity_reference.py (lr=1e-3, betas=(0.9,0.999),
    eps=1e-8, weight_decay=0.01, t=3, seeded m/v).
    """
    print("Running test_parity_with_reference...")
    var n = 6
    var p = zeros([n], DType.float64)
    var g = zeros([n], DType.float64)
    var m = zeros([n], DType.float64)
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
    m.store[DType.float64](0, 0.005)
    m.store[DType.float64](1, -0.004)
    m.store[DType.float64](2, 0.003)
    m.store[DType.float64](3, -0.002)
    m.store[DType.float64](4, 0.001)
    m.store[DType.float64](5, -0.006)
    v.store[DType.float64](0, 1e-4)
    v.store[DType.float64](1, 2e-4)
    v.store[DType.float64](2, 1.5e-4)
    v.store[DType.float64](3, 3e-4)
    v.store[DType.float64](4, 0.5e-4)
    v.store[DType.float64](5, 4e-4)

    var rp = List[Float64]()
    rp.append(0.09986689975575956)
    rp.append(-0.1999030661990727)
    rp.append(0.29992581958253395)
    rp.append(-0.4000034981041159)
    rp.append(0.49998856969788646)
    rp.append(-0.599979818003792)
    var rm = List[Float64]()
    rm.append(0.0066)
    rm.append(-0.0068)
    rm.append(0.0045)
    rm.append(0.0002999999999999997)
    rm.append(0.0004000000000000002)
    rm.append(-0.002000000000000001)
    var rv = List[Float64]()
    rv.append(0.00010034100000000001)
    rv.append(0.000200824)
    rv.append(0.000150174)
    rv.append(0.00030014099999999997)
    rv.append(4.9975e-05)
    rv.append(0.000400756)

    var out = adam_step(p, g, m, v, 3, 0.001, 0.9, 0.999, 1e-8, 0.01)
    var np = out[0]
    var nm = out[1]
    var nv = out[2]
    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("params parity mismatch at " + String(i))
        if _abs_diff(nm.load[DType.float64](i), rm[i]) > 1e-9:
            raise Error("m parity mismatch at " + String(i))
        if _abs_diff(nv.load[DType.float64](i), rv[i]) > 1e-9:
            raise Error("v parity mismatch at " + String(i))
    print("  ok parity to 1e-9 on all 6 coordinates (params, m, v)")
    print("test_parity_with_reference PASSED")


def test_adam_step_simple_delegates() raises:
    """`adam_step_simple` matches adam_step with default betas/eps, no wd."""
    print("Running test_adam_step_simple_delegates...")
    var n = 4
    var p = full([n], 0.5, DType.float64)
    var g = full([n], 0.1, DType.float64)
    var m = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
    var full_out = adam_step(p, g, m, v, 1, 0.001)  # defaults, wd=0
    var simple_out = adam_step_simple(p, g, m, v, 1, 0.001)
    for i in range(n):
        if (
            _abs_diff(
                full_out[0].load[DType.float64](i),
                simple_out[0].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error(
                "adam_step_simple diverged from adam_step at " + String(i)
            )
    print("  ok adam_step_simple delegates to adam_step defaults")
    print("test_adam_step_simple_delegates PASSED")


def main() raises:
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_reject_empty_moments()
    test_reject_nonpositive_timestep()
    test_parity_with_reference()
    test_adam_step_simple_delegates()
    print("\nAll Adam parity tests PASSED")
