"""Unit tests for the vanilla (Elman) RNN cell.

Tests cover:
- Construction + validation (positive input_size/hidden_size)
- step() shape (batch, input_size) x (batch, hidden_size) -> (batch, hidden_size)
- parameter collection (4 tensors: ih.weight/bias, hh.weight/bias)
- Numerical parity with torch.nn.RNNCell (tanh) on fixed ramp weights (1e-5)
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.rnn import RNNCell


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_shape() raises:
    """step maps (batch, input) x (batch, hidden) -> (batch, hidden)."""
    print("Running test_shape...")
    var cell = RNNCell[DType.float32](3, 4)
    var x = zeros([2, 3], DType.float32)
    var h = zeros([2, 4], DType.float32)
    var out = cell.step(x, h)
    if out.shape()[0] != 2 or out.shape()[1] != 4:
        raise Error("RNN step must return (batch, hidden)")
    print("  ok (2, 4)")
    print("test_shape PASSED")


def test_reject_bad_sizes() raises:
    """Non-positive sizes must raise."""
    print("Running test_reject_bad_sizes...")
    try:
        var _ = RNNCell[DType.float32](0, 4)
        raise Error("Should have rejected input_size = 0")
    except e:
        print("  ok rejected input_size = 0")
    try:
        var _ = RNNCell[DType.float32](3, 0)
        raise Error("Should have rejected hidden_size = 0")
    except e:
        print("  ok rejected hidden_size = 0")
    print("test_reject_bad_sizes PASSED")


def test_parameter_count() raises:
    """parameters() returns 4 tensors."""
    print("Running test_parameter_count...")
    var cell = RNNCell[DType.float32](3, 4)
    if len(cell.parameters()) != 4:
        raise Error("RNN cell should expose 4 parameter tensors")
    print("  ok 4 parameter tensors")
    print("test_parameter_count PASSED")


def test_parity_with_pytorch() raises:
    """step must match torch.nn.RNNCell(tanh) on fixed ramp weights to 1e-5.

    Reference values from parity_refs/rnn_parity_reference.py (input=3,
    hidden=4, batch=2). Odyssey Linear uses W (in, out); the reference sets
    torch's (out, in) weight to the transpose.
    """
    print("Running test_parity_with_pytorch...")

    var ref = List[Float64]()
    ref.append(-0.0479632)
    ref.append(-0.005)
    ref.append(0.0379817)
    ref.append(0.0808233)
    ref.append(-0.0659043)
    ref.append(0.003)
    ref.append(0.0718758)
    ref.append(0.140073)

    var cell = RNNCell[DType.float64](3, 4)
    for i in range(12):
        cell.ih.weight.store[DType.float64](i, Float64(i) * 0.02 - 0.1)
    for i in range(4):
        cell.ih.bias.store[DType.float64](i, Float64(i) * 0.03 - 0.05)
    for i in range(16):
        cell.hh.weight.store[DType.float64](i, Float64(i) * 0.01 - 0.06)
    for i in range(4):
        cell.hh.bias.store[DType.float64](i, Float64(i) * 0.02 - 0.03)
    var x = zeros([2, 3], DType.float64)
    for i in range(6):
        x.store[DType.float64](i, Float64(i) * 0.1 - 0.2)
    var h = zeros([2, 4], DType.float64)
    for i in range(8):
        h.store[DType.float64](i, Float64(i) * 0.05 - 0.1)

    var out = cell.step(x, h)
    for i in range(8):
        if _abs_diff(out.load[DType.float64](i), ref[i]) > 1e-5:
            raise Error("RNN parity mismatch at " + String(i))
    print("  ok matches torch.nn.RNNCell to 1e-5")
    print("test_parity_with_pytorch PASSED")


def test_forward_equals_zero_state_step() raises:
    """forward(x) must equal step(x, zeros) even with a nonzero b_hh.

    A zero initial hidden zeros only the hidden-to-hidden WEIGHT term, not its
    bias b_hh. This asserts forward() includes b_hh (regression guard: an earlier
    shortcut computed tanh(ih.forward(x)) and dropped b_hh, so forward diverged
    from step once b_hh was trained nonzero).
    """
    print("Running test_forward_equals_zero_state_step...")
    var cell = RNNCell[DType.float64](3, 4)
    for i in range(4):
        cell.hh.bias.store[DType.float64](i, Float64(i) * 0.02 + 0.05)
    var x = zeros([2, 3], DType.float64)
    for i in range(6):
        x.store[DType.float64](i, Float64(i) * 0.1 - 0.2)
    var h0 = zeros([2, 4], DType.float64)

    var f = cell.forward(x)
    var s = cell.step(x, h0)
    for i in range(8):
        if _abs_diff(f.load[DType.float64](i), s.load[DType.float64](i)) > 1e-12:
            raise Error("forward != step(x, zeros) at " + String(i))
    print("  ok forward(x) == step(x, zeros) with nonzero b_hh")
    print("test_forward_equals_zero_state_step PASSED")


def test_parameter_order() raises:
    """parameters() must return [ih.weight, ih.bias, hh.weight, hh.bias] in order.

    A consumer that threads per-parameter state (e.g. the PC error chain) relies
    on this positional contract, so assert identity by numel, not just count.
    """
    print("Running test_parameter_order...")
    var cell = RNNCell[DType.float64](3, 4)
    var params = cell.parameters()
    # ih.weight: 3x4=12, ih.bias: 4, hh.weight: 4x4=16, hh.bias: 4.
    if params[0].numel() != 12:
        raise Error("params[0] should be ih.weight (12)")
    if params[1].numel() != 4:
        raise Error("params[1] should be ih.bias (4)")
    if params[2].numel() != 16:
        raise Error("params[2] should be hh.weight (16)")
    if params[3].numel() != 4:
        raise Error("params[3] should be hh.bias (4)")
    print("  ok parameter order ih.weight/ih.bias/hh.weight/hh.bias")
    print("test_parameter_order PASSED")


def main() raises:
    """Run all RNN tests."""
    print("=" * 60)
    print("Vanilla RNN Cell Test Suite")
    print("=" * 60)
    test_shape()
    test_reject_bad_sizes()
    test_parameter_count()
    test_parity_with_pytorch()
    test_forward_equals_zero_state_step()
    test_parameter_order()
    print("=" * 60)
    print("All RNN tests PASSED")
    print("=" * 60)
