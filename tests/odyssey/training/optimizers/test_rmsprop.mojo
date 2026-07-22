"""Unit tests for the RMSprop update step (Tieleman & Hinton 2012).

Odyssey's `rmsprop_step` is pure-functional: it returns
(new_params, new_square_avg, new_buf). The implemented update (rmsprop.mojo)
for the centered=False, momentum=0 path is

    effective_grad = grad + weight_decay * params
    square_avg     = alpha * square_avg + (1 - alpha) * effective_grad**2
    params         = params - lr * effective_grad / (sqrt(square_avg) + epsilon)

which matches `torch.optim.RMSprop(centered=False, momentum=0)`.

Tests cover:
- Shape/dtype guards, empty square_avg guard, positive-timestep guard
- Numerical parity with the reference (1e-9) at alpha=0.99, wd=0, momentum=0.
  Reference numbers from parity_refs/rmsprop_parity_reference.py (drives
  torch.optim.RMSprop when importable, else the identical numpy closed form).
- rmsprop_step_simple delegation.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.rmsprop import (
    rmsprop_step,
    rmsprop_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`rmsprop_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var sq = zeros([4], DType.float32)
    try:
        var _ = rmsprop_step(p, g, sq, 1, 0.01)
        raise Error("Should have rejected shape mismatch")
    except:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`rmsprop_step` rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var sq = zeros([4], DType.float32)
    try:
        var _ = rmsprop_step(p, g, sq, 1, 0.01)
        raise Error("Should have rejected dtype mismatch")
    except:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_reject_empty_square_avg() raises:
    """`rmsprop_step` requires an initialized square_avg buffer."""
    print("Running test_reject_empty_square_avg...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var sq = zeros([0], DType.float32)
    try:
        var _ = rmsprop_step(p, g, sq, 1, 0.01)
        raise Error("Should have rejected empty square_avg")
    except:
        print("  ok rejected empty square_avg")
    print("test_reject_empty_square_avg PASSED")


def test_reject_nonpositive_timestep() raises:
    """`rmsprop_step` rejects t <= 0."""
    print("Running test_reject_nonpositive_timestep...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var sq = zeros([4], DType.float32)
    try:
        var _ = rmsprop_step(p, g, sq, 0, 0.01)
        raise Error("Should have rejected t=0")
    except:
        print("  ok rejected non-positive timestep")
    print("test_reject_nonpositive_timestep PASSED")


def test_parity_with_reference() raises:
    """One RMSprop step must match the reference to 1e-9.

    Fixed inputs and reference outputs transcribed from
    parity_refs/rmsprop_parity_reference.py (lr=0.01, alpha=0.99, eps=1e-8,
    wd=0, momentum=0, seeded square_avg).
    """
    print("Running test_parity_with_reference...")
    var n = 6
    var p = zeros([n], DType.float64)
    var g = zeros([n], DType.float64)
    var sq = zeros([n], DType.float64)
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
    sq.store[DType.float64](0, 1e-4)
    sq.store[DType.float64](1, 2e-4)
    sq.store[DType.float64](2, 1.5e-4)
    sq.store[DType.float64](3, 3e-4)
    sq.store[DType.float64](4, 0.5e-4)
    sq.store[DType.float64](5, 4e-4)

    var rp = List[Float64]()
    rp.append(0.08029343385417073)
    rp.append(-0.17914857308703608)
    rp.append(0.28778306551461014)
    rp.append(-0.41435619519148703)
    rp.append(0.5140719310926535)
    rp.append(-0.6197065758545528)
    var rsq = List[Float64]()
    rsq.append(0.00010300000000000001)
    rsq.append(0.00020700000000000002)
    rsq.append(0.00015074999999999998)
    rsq.append(0.00030324999999999997)
    rsq.append(5.050000000000001e-05)
    rsq.append(0.00041200000000000004)

    var out = rmsprop_step(p, g, sq, 1, 0.01, 0.99, 1e-8, 0.0, 0.0)
    var np = out[0]
    var nsq = out[1]
    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("params parity mismatch at " + String(i))
        if _abs_diff(nsq.load[DType.float64](i), rsq[i]) > 1e-9:
            raise Error("square_avg parity mismatch at " + String(i))
    print("  ok parity to 1e-9 on all 6 coordinates (params, square_avg)")
    print("test_parity_with_reference PASSED")


def test_rmsprop_step_simple_delegates() raises:
    """`rmsprop_step_simple` matches the full step at defaults (wd=0, mom=0)."""
    print("Running test_rmsprop_step_simple_delegates...")
    var n = 4
    var p = full([n], 0.5, DType.float64)
    var g = full([n], 0.1, DType.float64)
    var sq = full([n], 1e-4, DType.float64)
    var full_out = rmsprop_step(p, g, sq, 1, 0.01, 0.99, 1e-8, 0.0, 0.0)
    var simple_out = rmsprop_step_simple(p, g, sq, 0.01, 0.99, 1e-8)
    for i in range(n):
        if (
            _abs_diff(
                full_out[0].load[DType.float64](i),
                simple_out[0].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error("rmsprop_step_simple diverged at " + String(i))
    print("  ok rmsprop_step_simple delegates to the full step")
    print("test_rmsprop_step_simple_delegates PASSED")


def main() raises:
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_reject_empty_square_avg()
    test_reject_nonpositive_timestep()
    test_parity_with_reference()
    test_rmsprop_step_simple_delegates()
    print("\nAll RMSprop parity tests PASSED")
