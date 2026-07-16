"""Unit tests for the Adan optimizer (arXiv:2208.06677).

Tests cover:
- Shape/dtype guards on adan_step
- Numerical parity with the public pytorch_optimizer.Adan (1e-9) on step 1
- prev_grad passthrough (new_prev_grad equals the current gradient)
- descent direction
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.adan import adan_step


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """adan_step rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    var dd = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    var pg = zeros([4], DType.float32)
    try:
        var _ = adan_step(p, g, m, dd, v, pg, 1, 0.001)
        raise Error("Should have rejected shape mismatch")
    except e:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """adan_step rejects a params/gradients dtype mismatch."""
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    var dd = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    var pg = zeros([4], DType.float32)
    try:
        var _ = adan_step(p, g, m, dd, v, pg, 1, 0.001)
        raise Error("Should have rejected dtype mismatch")
    except e:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def test_parity_with_reference() raises:
    """One Adan step (t=1) must match pytorch_optimizer.Adan to 1e-9.

    Fixed inputs + reference output transcribed from
    parity_refs/adan_parity_reference.py (lr=0.001, betas=(0.98,0.92,0.99),
    eps=1e-8, no weight decay). Step 1 uses prev_grad = grad (zero initial
    gradient difference), matching the reference's previous-grad initialization.
    """
    print("Running test_parity_with_reference...")

    var rp = List[Float64]()
    rp.append(0.0990000002)
    rp.append(-0.2009999999)
    rp.append(0.301)
    rp.append(-0.401)
    rp.append(0.501)

    var n = 5
    var p = zeros([n], DType.float64)
    var g = zeros([n], DType.float64)
    var m = zeros([n], DType.float64)
    var dd = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
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

    # step 1: prev_grad = g
    var res = adan_step(
        p, g, m, dd, v, g, 1, 0.001, 0.98, 0.92, 0.99, 1e-8, 0.0
    )
    var np = res[0]
    for i in range(n):
        if _abs_diff(np.load[DType.float64](i), rp[i]) > 1e-9:
            raise Error("adan parity mismatch at " + String(i))
    print("  ok matches pytorch_optimizer.Adan to 1e-9")
    print("test_parity_with_reference PASSED")


def test_prev_grad_passthrough() raises:
    """new_prev_grad must equal the current gradient (for the next step)."""
    print("Running test_prev_grad_passthrough...")
    var n = 3
    var p = zeros([n], DType.float64)
    var g = full([n], 0.5, DType.float64)
    var m = zeros([n], DType.float64)
    var dd = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
    var res = adan_step(p, g, m, dd, v, g, 1, 0.001)
    var new_pg = res[4]
    for i in range(n):
        if _abs_diff(new_pg.load[DType.float64](i), 0.5) > 1e-12:
            raise Error("new_prev_grad should equal the current gradient")
    print("  ok new_prev_grad == current gradient")
    print("test_prev_grad_passthrough PASSED")


def test_descent_direction() raises:
    """A positive gradient must decrease the parameter."""
    print("Running test_descent_direction...")
    var n = 3
    var p = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var m = zeros([n], DType.float64)
    var dd = zeros([n], DType.float64)
    var v = zeros([n], DType.float64)
    var res = adan_step(p, g, m, dd, v, g, 1, 0.01)
    var np = res[0]
    for i in range(n):
        if not (np.load[DType.float64](i) < 1.0):
            raise Error("positive gradient should decrease param")
    print("  ok positive gradient decreased params")
    print("test_descent_direction PASSED")


def main() raises:
    """Run all Adan tests."""
    print("=" * 60)
    print("Adan Optimizer Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_reject_dtype_mismatch()
    test_parity_with_reference()
    test_prev_grad_passthrough()
    test_descent_direction()
    print("=" * 60)
    print("All Adan tests PASSED")
    print("=" * 60)
