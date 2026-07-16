"""Unit tests for the GRU cell (torch.nn.GRUCell convention).

Tests cover:
- Construction + validation (positive input_size/hidden_size)
- step() shape (batch, input_size) x (batch, hidden_size) -> (batch, hidden_size)
- parameter collection (12 tensors: ir/iz/in_/hr/hz/hn weight+bias)
- Numerical parity with torch.nn.GRUCell on fixed ramp weights (1e-5)
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.gru import GRUCell


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_shape() raises:
    """step maps (batch, input) x (batch, hidden) -> (batch, hidden)."""
    print("Running test_shape...")
    var cell = GRUCell[DType.float32](3, 4)
    var x = zeros([2, 3], DType.float32)
    var h = zeros([2, 4], DType.float32)
    var out = cell.step(x, h)
    if out.shape()[0] != 2 or out.shape()[1] != 4:
        raise Error("GRU step must return (batch, hidden)")
    print("  ok (2, 4)")
    print("test_shape PASSED")


def test_reject_bad_sizes() raises:
    """Non-positive sizes must raise."""
    print("Running test_reject_bad_sizes...")
    try:
        var _ = GRUCell[DType.float32](0, 4)
        raise Error("Should have rejected input_size = 0")
    except e:
        print("  ok rejected input_size = 0")
    try:
        var _ = GRUCell[DType.float32](3, 0)
        raise Error("Should have rejected hidden_size = 0")
    except e:
        print("  ok rejected hidden_size = 0")
    print("test_reject_bad_sizes PASSED")


def test_parameter_count() raises:
    """parameters() returns 12 tensors (weight+bias of six projections)."""
    print("Running test_parameter_count...")
    var cell = GRUCell[DType.float32](3, 4)
    if len(cell.parameters()) != 12:
        raise Error("GRU cell should expose 12 parameter tensors")
    print("  ok 12 parameter tensors")
    print("test_parameter_count PASSED")


def _seed_ramp(mut lin: AnyTensor, count: Int, scale: Float64, off: Float64) raises:
    """Seed a tensor's flat buffer with value[i] = i*scale + off."""
    for i in range(count):
        lin.store[DType.float64](i, Float64(i) * scale + off)


def test_parity_with_pytorch() raises:
    """step must match torch.nn.GRUCell on fixed ramp weights to 1e-5.

    Reference values from parity_refs/gru_parity_reference.py (input=3,
    hidden=4, batch=2). Odyssey Linear uses W (in, out); the reference sets
    torch's packed (3H, in)/(3H, H) weight to the per-gate transpose.
    """
    print("Running test_parity_with_pytorch...")

    var ref = List[Float64]()
    ref.append(-0.052972)
    ref.append(-0.0248613)
    ref.append(0.0035376)
    ref.append(0.0322239)
    ref.append(0.0472359)
    ref.append(0.0817735)
    ref.append(0.1168354)
    ref.append(0.1524132)

    var cell = GRUCell[DType.float64](3, 4)
    # input-to-hidden projections (weight: 3*4=12 elems, bias: 4 elems each)
    _seed_ramp(cell.ir.weight, 12, 0.01, -0.05)
    _seed_ramp(cell.ir.bias, 4, 0.01, -0.02)
    _seed_ramp(cell.iz.weight, 12, 0.011, -0.06)
    _seed_ramp(cell.iz.bias, 4, 0.012, -0.03)
    _seed_ramp(cell.in_.weight, 12, 0.009, -0.04)
    _seed_ramp(cell.in_.bias, 4, 0.008, -0.01)
    # hidden-to-hidden projections (weight: 4*4=16 elems, bias: 4 elems each)
    _seed_ramp(cell.hr.weight, 16, 0.007, -0.03)
    _seed_ramp(cell.hr.bias, 4, 0.006, -0.02)
    _seed_ramp(cell.hz.weight, 16, 0.008, -0.04)
    _seed_ramp(cell.hz.bias, 4, 0.005, -0.01)
    _seed_ramp(cell.hn.weight, 16, 0.006, -0.02)
    _seed_ramp(cell.hn.bias, 4, 0.004, -0.015)

    var x = zeros([2, 3], DType.float64)
    _seed_ramp(x, 6, 0.1, -0.2)
    var h = zeros([2, 4], DType.float64)
    _seed_ramp(h, 8, 0.05, -0.1)

    var out = cell.step(x, h)
    for i in range(8):
        if _abs_diff(out.load[DType.float64](i), ref[i]) > 1e-5:
            raise Error("GRU parity mismatch at " + String(i))
    print("  ok matches torch.nn.GRUCell to 1e-5")
    print("test_parity_with_pytorch PASSED")


def test_forward_equals_zero_state_step() raises:
    """forward(x) must equal step(x, zeros) even with nonzero h->h biases.

    The Module `forward` is documented as a step from a zero initial hidden
    state. A zero hidden zeros only the hidden-to-hidden WEIGHT terms; the
    hidden-to-hidden BIASES (b_hr, b_hz, b_hn) still contribute. This asserts
    `forward` includes them (regression guard: an earlier shortcut dropped
    them, so forward diverged from step once the biases were trained nonzero).
    """
    print("Running test_forward_equals_zero_state_step...")
    var cell = GRUCell[DType.float64](3, 4)
    # Seed the hidden-to-hidden biases nonzero (the terms a shortcut would drop).
    for i in range(4):
        cell.hr.bias.store[DType.float64](i, Float64(i) * 0.01 + 0.05)
        cell.hz.bias.store[DType.float64](i, Float64(i) * 0.02 - 0.03)
        cell.hn.bias.store[DType.float64](i, Float64(i) * 0.015 + 0.02)
    var x = zeros([2, 3], DType.float64)
    for i in range(6):
        x.store[DType.float64](i, Float64(i) * 0.1 - 0.2)
    var h0 = zeros([2, 4], DType.float64)

    var f = cell.forward(x)
    var s = cell.step(x, h0)
    for i in range(8):
        if _abs_diff(f.load[DType.float64](i), s.load[DType.float64](i)) > 1e-12:
            raise Error("forward != step(x, zeros) at " + String(i))
    print("  ok forward(x) == step(x, zeros) with nonzero h->h biases")
    print("test_forward_equals_zero_state_step PASSED")


def main() raises:
    """Run all GRU tests."""
    print("=" * 60)
    print("GRU Cell Test Suite")
    print("=" * 60)
    test_shape()
    test_reject_bad_sizes()
    test_parameter_count()
    test_parity_with_pytorch()
    test_forward_equals_zero_state_step()
    print("=" * 60)
    print("All GRU tests PASSED")
    print("=" * 60)
