"""Unit tests for the LSTM cell (torch.nn.LSTMCell convention).

Tests cover:
- Construction + validation (positive input_size/hidden_size)
- step() shapes: (batch, input) x (batch, hidden) x (batch, hidden)
  -> ((batch, hidden), (batch, hidden))
- parameter collection (16 tensors: ii/if_/ig/io/hi/hf/hg/ho weight+bias)
- Numerical parity with torch.nn.LSTMCell on fixed ramp weights (1e-5), for
  both the new hidden state h' and the new cell state c'
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.lstm import LSTMCell


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _seed_ramp(
    mut t: AnyTensor, count: Int, scale: Float64, off: Float64
) raises:
    """Seed a tensor's flat buffer with value[i] = i*scale + off."""
    for i in range(count):
        t.store[DType.float64](i, Float64(i) * scale + off)


def test_shape() raises:
    """step maps (batch, in) x (batch, hid) x (batch, hid) -> two (batch, hid).
    """
    print("Running test_shape...")
    var cell = LSTMCell[DType.float32](3, 4)
    var x = zeros([2, 3], DType.float32)
    var h = zeros([2, 4], DType.float32)
    var c = zeros([2, 4], DType.float32)
    var hc = cell.step(x, h, c)
    if hc[0].shape()[0] != 2 or hc[0].shape()[1] != 4:
        raise Error("LSTM step must return hidden (batch, hidden)")
    if hc[1].shape()[0] != 2 or hc[1].shape()[1] != 4:
        raise Error("LSTM step must return cell (batch, hidden)")
    print("  ok ((2, 4), (2, 4))")
    print("test_shape PASSED")


def test_reject_bad_sizes() raises:
    """Non-positive sizes must raise."""
    print("Running test_reject_bad_sizes...")
    try:
        var _ = LSTMCell[DType.float32](0, 4)
        raise Error("Should have rejected input_size = 0")
    except e:
        print("  ok rejected input_size = 0")
    try:
        var _ = LSTMCell[DType.float32](3, 0)
        raise Error("Should have rejected hidden_size = 0")
    except e:
        print("  ok rejected hidden_size = 0")
    print("test_reject_bad_sizes PASSED")


def test_parameter_count() raises:
    """parameters() returns 16 tensors (weight+bias of eight projections)."""
    print("Running test_parameter_count...")
    var cell = LSTMCell[DType.float32](3, 4)
    if len(cell.parameters()) != 16:
        raise Error("LSTM cell should expose 16 parameter tensors")
    print("  ok 16 parameter tensors")
    print("test_parameter_count PASSED")


def test_parity_with_pytorch() raises:
    """step must match torch.nn.LSTMCell on fixed ramp weights to 1e-5.

    Reference values from parity_refs/lstm_parity_reference.py (input=3,
    hidden=4, batch=2). Odyssey Linear uses W (in, out); the reference sets
    torch's packed (4H, in)/(4H, H) weight to the per-gate transpose. Both the
    new hidden state h' and the new cell state c' are checked.
    """
    print("Running test_parity_with_pytorch...")

    var ref_h = List[Float64]()
    ref_h.append(-0.017694965)
    ref_h.append(-0.008305559)
    ref_h.append(0.00130057)
    ref_h.append(0.011117769)
    ref_h.append(0.013930657)
    ref_h.append(0.027275374)
    ref_h.append(0.0412858)
    ref_h.append(0.055944425)

    var ref_c = List[Float64]()
    ref_c.append(-0.035740954)
    ref_c.append(-0.016693206)
    ref_c.append(0.002601797)
    ref_c.append(0.022143402)
    ref_c.append(0.028108978)
    ref_c.append(0.05436392)
    ref_c.append(0.08133847)
    ref_c.append(0.109021286)

    var cell = LSTMCell[DType.float64](3, 4)
    # input-to-hidden projections (weight: 3*4=12 elems, bias: 4 elems each)
    _seed_ramp(cell.ii.weight, 12, 0.01, -0.05)
    _seed_ramp(cell.ii.bias, 4, 0.01, -0.02)
    _seed_ramp(cell.if_.weight, 12, 0.011, -0.06)
    _seed_ramp(cell.if_.bias, 4, 0.012, -0.03)
    _seed_ramp(cell.ig.weight, 12, 0.009, -0.04)
    _seed_ramp(cell.ig.bias, 4, 0.008, -0.01)
    _seed_ramp(cell.io.weight, 12, 0.012, -0.055)
    _seed_ramp(cell.io.bias, 4, 0.009, -0.025)
    # hidden-to-hidden projections (weight: 4*4=16 elems, bias: 4 elems each)
    _seed_ramp(cell.hi.weight, 16, 0.007, -0.03)
    _seed_ramp(cell.hi.bias, 4, 0.006, -0.02)
    _seed_ramp(cell.hf.weight, 16, 0.008, -0.04)
    _seed_ramp(cell.hf.bias, 4, 0.005, -0.01)
    _seed_ramp(cell.hg.weight, 16, 0.006, -0.02)
    _seed_ramp(cell.hg.bias, 4, 0.004, -0.015)
    _seed_ramp(cell.ho.weight, 16, 0.0075, -0.035)
    _seed_ramp(cell.ho.bias, 4, 0.0045, -0.012)

    var x = zeros([2, 3], DType.float64)
    _seed_ramp(x, 6, 0.1, -0.2)
    var h = zeros([2, 4], DType.float64)
    _seed_ramp(h, 8, 0.05, -0.1)
    var c = zeros([2, 4], DType.float64)
    _seed_ramp(c, 8, 0.03, -0.06)

    var hc = cell.step(x, h, c)
    for i in range(8):
        if _abs_diff(hc[0].load[DType.float64](i), ref_h[i]) > 1e-5:
            raise Error("LSTM hidden parity mismatch at " + String(i))
        if _abs_diff(hc[1].load[DType.float64](i), ref_c[i]) > 1e-5:
            raise Error("LSTM cell parity mismatch at " + String(i))
    print("  ok matches torch.nn.LSTMCell (h and c) to 1e-5")
    print("test_parity_with_pytorch PASSED")


def test_forward_equals_zero_state_step() raises:
    """forward(x) must equal step(x, zeros, zeros)[0] with nonzero h->h biases.

    A zero initial (hidden, cell) zeros only the hidden-to-hidden WEIGHT terms,
    not the biases b_hi/b_hf/b_hg/b_ho. This asserts forward() includes them
    (regression guard: an earlier shortcut dropped the h->h biases, so forward
    diverged from a zero-state step once those biases were trained nonzero).
    """
    print("Running test_forward_equals_zero_state_step...")
    var cell = LSTMCell[DType.float64](3, 4)
    for i in range(4):
        cell.hi.bias.store[DType.float64](i, Float64(i) * 0.01 + 0.05)
        cell.hf.bias.store[DType.float64](i, Float64(i) * 0.02 - 0.03)
        cell.hg.bias.store[DType.float64](i, Float64(i) * 0.015 + 0.02)
        cell.ho.bias.store[DType.float64](i, Float64(i) * 0.01 - 0.01)
    var x = zeros([2, 3], DType.float64)
    for i in range(6):
        x.store[DType.float64](i, Float64(i) * 0.1 - 0.2)
    var h0 = zeros([2, 4], DType.float64)
    var c0 = zeros([2, 4], DType.float64)

    var f = cell.forward(x)
    var hc = cell.step(x, h0, c0)
    for i in range(8):
        if (
            _abs_diff(f.load[DType.float64](i), hc[0].load[DType.float64](i))
            > 1e-12
        ):
            raise Error("forward != step(x, zeros, zeros)[0] at " + String(i))
    print(
        "  ok forward(x) == step(x, zeros, zeros)[0] with nonzero h->h biases"
    )
    print("test_forward_equals_zero_state_step PASSED")


def main() raises:
    """Run all LSTM tests."""
    print("=" * 60)
    print("LSTM Cell Test Suite")
    print("=" * 60)
    test_shape()
    test_reject_bad_sizes()
    test_parameter_count()
    test_parity_with_pytorch()
    test_forward_equals_zero_state_step()
    print("=" * 60)
    print("All LSTM tests PASSED")
    print("=" * 60)
