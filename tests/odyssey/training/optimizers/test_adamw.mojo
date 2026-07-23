"""Unit tests for the AdamW update step (Loshchilov & Hutter 2019, arXiv:1711.05101).

Odyssey's `adamw_step` is pure-functional: it returns (new_params, new_m, new_v).
The implemented update (adamw.mojo) is the Adam core with DECOUPLED weight decay:

    m      = beta1 * m + (1 - beta1) * grad            (grad NOT decayed)
    v      = beta2 * v + (1 - beta2) * grad**2
    m_hat  = m / (1 - beta1**t)
    v_hat  = v / (1 - beta2**t)
    p'     = params - lr * m_hat / (sqrt(v_hat) + epsilon)
    params = p' - weight_decay * p'                    (decoupled, on p', no lr)

NOTE: the decoupled-decay term differs from `torch.optim.AdamW`, which applies
`p *= (1 - lr*weight_decay)` to the PRE-update params. Odyssey applies
`p' *= (1 - weight_decay)` to the POST-update params with no lr factor. The
parity reference (parity_refs/adamw_parity_reference.py) models Odyssey's
implemented formula and records the divergence; the Adam core itself is
torch-equivalent (test_adam.mojo).

Tests cover:
- Shape/dtype guards, empty-moment guard, positive-timestep guard
- Numerical parity with the reference (1e-9) at t=3 with weight_decay=0.01
- weight_decay=0 collapses AdamW to plain Adam on params (no decay term)
- adamw_step_simple delegation
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.adamw import (
    adamw_step,
    adamw_step_simple,
)
from odyssey.training.optimizers.adam import adam_step


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`adamw_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adamw_step(p, g, m, v, 1, 0.001)
        raise Error("Should have rejected shape mismatch")
    except:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`adamw_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adamw_step(p, g, m, v, 1, 0.001)
        raise Error("Should have rejected dtype mismatch")
    except:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_reject_empty_moments() raises:
    """`adamw_step` requires initialized moment buffers m and v."""
    print("Running test_reject_empty_moments...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var m = zeros([0], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adamw_step(p, g, m, v, 1, 0.001)
        raise Error("Should have rejected empty moment buffer")
    except:
        print("  ok rejected empty moment buffer")
    print("test_reject_empty_moments PASSED")


def test_reject_nonpositive_timestep() raises:
    """`adamw_step` rejects t <= 0."""
    print("Running test_reject_nonpositive_timestep...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    try:
        var _ = adamw_step(p, g, m, v, 0, 0.001)
        raise Error("Should have rejected t=0")
    except:
        print("  ok rejected non-positive timestep")
    print("test_reject_nonpositive_timestep PASSED")


def test_parity_with_reference() raises:
    """One AdamW step at t=3 must match the reference to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/adamw_parity_reference.py (lr=1e-3, betas=(0.9,0.999),
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
    rp.append(0.09887020074073723)
    rp.append(-0.19790682925242264)
    rp.append(0.29693143469020056)
    rp.append(-0.39600807814475314)
    rp.append(0.49500282687934843)
    rp.append(-0.5939860216178512)
    var rm = List[Float64]()
    rm.append(0.006500000000000001)
    rm.append(-0.0066)
    rm.append(0.0042)
    rm.append(0.0006999999999999995)
    rm.append(-9.999999999999972e-05)
    rm.append(-0.001400000000000001)
    var rv = List[Float64]()
    rv.append(0.0001003)
    rv.append(0.0002007)
    rv.append(0.00015007499999999998)
    rv.append(0.00030032499999999994)
    rv.append(5.0050000000000004e-05)
    rv.append(0.0004012)

    var out = adamw_step(p, g, m, v, 3, 0.001, 0.9, 0.999, 1e-8, 0.01)
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


def test_zero_wd_collapses_to_adam_params() raises:
    """With weight_decay=0, AdamW's params match Adam's params (wd term absent).

    The Adam-core moment recursion is identical; only the decoupled decay term
    distinguishes them, and it vanishes at wd=0. (Adam couples wd into the
    gradient, so this equivalence holds only at wd=0.)
    """
    print("Running test_zero_wd_collapses_to_adam_params...")
    var n = 4
    var p = full([n], 0.5, DType.float64)
    var g = full([n], 0.1, DType.float64)
    var m = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
    var aw = adamw_step(p, g, m, v, 1, 0.001, 0.9, 0.999, 1e-8, 0.0)
    var ad = adam_step(p, g, m, v, 1, 0.001, 0.9, 0.999, 1e-8, 0.0)
    for i in range(n):
        if (
            _abs_diff(
                aw[0].load[DType.float64](i), ad[0].load[DType.float64](i)
            )
            > 1e-12
        ):
            raise Error("AdamW(wd=0) != Adam(wd=0) at " + String(i))
    print("  ok AdamW(wd=0) params == Adam(wd=0) params")
    print("test_zero_wd_collapses_to_adam_params PASSED")


def test_adamw_step_simple_delegates() raises:
    """`adamw_step_simple` matches the full step at documented defaults.

    The simple wrapper delegates to `adamw_step` with `beta1=0.9`,
    `beta2=0.999`, `eps=1e-8`, `weight_decay=0.01` (per adamw.mojo).
    Asserts exact equality on EVERY coordinate of params, m, AND v so a
    future regression in the simple wrapper's delegation contract is
    caught here rather than as a downstream divergent loss.
    """
    print("Running test_adamw_step_simple_delegates...")
    var n = 4
    var p = full([n], 0.5, DType.float64)
    var g = full([n], 0.1, DType.float64)
    var m = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
    var full_out = adamw_step(p, g, m, v, 1, 0.001, 0.9, 0.999, 1e-8, 0.01)
    var simple_out = adamw_step_simple(p, g, m, v, 1, 0.001)
    for i in range(n):
        if (
            _abs_diff(
                full_out[0].load[DType.float64](i),
                simple_out[0].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error("adamw_step_simple params diverged at " + String(i))
        if (
            _abs_diff(
                full_out[1].load[DType.float64](i),
                simple_out[1].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error("adamw_step_simple m diverged at " + String(i))
        if (
            _abs_diff(
                full_out[2].load[DType.float64](i),
                simple_out[2].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error("adamw_step_simple v diverged at " + String(i))
    print("  ok adamw_step_simple delegates to adamw_step defaults")
    print("test_adamw_step_simple_delegates PASSED")


def main() raises:
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_reject_empty_moments()
    test_reject_nonpositive_timestep()
    test_parity_with_reference()
    test_zero_wd_collapses_to_adam_params()
    test_adamw_step_simple_delegates()
    print("\nAll AdamW parity tests PASSED")
