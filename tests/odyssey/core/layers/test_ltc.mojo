"""Unit tests for the LTC cell (Liquid Time-constant Networks, arXiv:2006.04439).

Tests cover:
- Construction + validation (positive input/hidden/solver_steps, elapsed > 0)
- step() shape (batch, input) x (batch, hidden) -> (batch, hidden)
- parameter collection (6 tensors: wi weight+bias, wh weight+bias, tau, A)
- Numerical parity with a hand-rolled numpy fused-solver reference (1e-5), for
  BOTH a single step AND a 3-step sequence with hidden carried across steps.
- forward(x) == step(x, zeros)

Reference values from parity_refs/ltc_parity_reference.py (input=3, hidden=4,
batch=2, solver_steps=6, elapsed=1.0). Odyssey Linear uses W (in, out) applied as
x @ W + b, so the reference sets the weights in Odyssey layout directly (no
transpose). The recurrent projection bias is zero (single gating bias mu on wi).
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.ltc import LTCCell

# Package-path import test: LTCCell must be reachable via the package __init__
# export (a sibling took a MAJOR for a docstring-without-export mismatch).
from odyssey.core.layers import LTCCell as LTCCellFromPackage


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
    """`step` maps (batch, input) x (batch, hidden) -> (batch, hidden)."""
    print("Running test_shape...")
    var cell = LTCCell[DType.float32](3, 4)
    var x = zeros([2, 3], DType.float32)
    var h = zeros([2, 4], DType.float32)
    var out = cell.step(x, h)
    if out.shape()[0] != 2 or out.shape()[1] != 4:
        raise Error("LTC step must return (batch, hidden)")
    print("  ok (2, 4)")
    print("test_shape PASSED")


def test_package_export() raises:
    """LTCCell must be importable from the package __init__ export."""
    print("Running test_package_export...")
    var cell = LTCCellFromPackage[DType.float32](3, 4)
    if len(cell.parameters()) != 6:
        raise Error("package-exported LTCCell should expose 6 parameters")
    print("  ok importable from odyssey.core.layers")
    print("test_package_export PASSED")


def test_reject_bad_args() raises:
    """Non-positive sizes / solver_steps / elapsed must raise."""
    print("Running test_reject_bad_args...")
    try:
        var _ = LTCCell[DType.float32](0, 4)
        raise Error("Should have rejected input_size = 0")
    except _:
        print("  ok rejected input_size = 0")
    try:
        var _ = LTCCell[DType.float32](3, 0)
        raise Error("Should have rejected hidden_size = 0")
    except _:
        print("  ok rejected hidden_size = 0")
    try:
        var _ = LTCCell[DType.float32](3, 4, 0)
        raise Error("Should have rejected solver_steps = 0")
    except _:
        print("  ok rejected solver_steps = 0")
    try:
        var _ = LTCCell[DType.float32](3, 4, 6, 0.0)
        raise Error("Should have rejected elapsed = 0")
    except _:
        print("  ok rejected elapsed = 0")
    print("test_reject_bad_args PASSED")


def test_parameter_count() raises:
    """`parameters()` returns 6 tensors (wi w+b, wh w+b, tau, A)."""
    print("Running test_parameter_count...")
    var cell = LTCCell[DType.float32](3, 4)
    if len(cell.parameters()) != 6:
        raise Error("LTC cell should expose 6 parameter tensors")
    print("  ok 6 parameter tensors")
    print("test_parameter_count PASSED")


def _seed_cell(mut cell: LTCCell[DType.float64]) raises:
    """Seed the cell's parameters to the parity-reference ramp values."""
    # wi: (in=3, out=4) weight = 12 elems, bias (mu) = 4 elems
    _seed_ramp(cell.wi.weight, 12, 0.01, -0.05)
    _seed_ramp(cell.wi.bias, 4, 0.01, -0.02)
    # wh: (hid=4, hid=4) weight = 16 elems; bias stays zero (single mu)
    _seed_ramp(cell.wh.weight, 16, 0.007, -0.03)
    # tau (all > 0) and A
    _seed_ramp(cell.tau, 4, 0.05, 0.8)
    _seed_ramp(cell.a, 4, 0.03, -0.05)


def test_parity_single_step() raises:
    """`step` must match the numpy fused-solver reference to 1e-5 (single step).
    """
    print("Running test_parity_single_step...")

    var ref_vals = List[Float64]()
    ref_vals.append(-0.03211515605880544)
    ref_vals.append(-0.017069226776112698)
    ref_vals.append(4.45448348569822e-05)
    ref_vals.append(0.0190608543619833)
    ref_vals.append(0.032552336550379285)
    ref_vals.append(0.05060515594569358)
    ref_vals.append(0.07046213445006379)
    ref_vals.append(0.09194570841741521)

    var cell = LTCCell[DType.float64](3, 4)
    _seed_cell(cell)

    var x = zeros([2, 3], DType.float64)
    _seed_ramp(x, 6, 0.1, -0.2)
    var h = zeros([2, 4], DType.float64)
    _seed_ramp(h, 8, 0.05, -0.1)

    var out = cell.step(x, h)
    for i in range(8):
        if _abs_diff(out.load[DType.float64](i), ref_vals[i]) > 1e-5:
            raise Error("LTC single-step parity mismatch at " + String(i))
    print("  ok matches numpy fused-solver reference to 1e-5")
    print("test_parity_single_step PASSED")


def test_parity_sequence() raises:
    """`step` must match the reference over a 3-step sequence with state carry.
    """
    print("Running test_parity_sequence...")

    # seq_out_0, seq_out_1, seq_out_2 from the parity reference (batch*hidden=8).
    var ref0 = List[Float64]()
    ref0.append(0.0002450985278981918)
    ref0.append(2.2324311132908162e-05)
    ref0.append(2.875746017229805e-05)
    ref0.append(0.0002819203196524359)
    ref0.append(0.000491837419013875)
    ref0.append(2.218368033114824e-05)
    ref0.append(8.01990746286676e-05)
    ref0.append(0.0006992201117434761)

    var ref1 = List[Float64]()
    ref1.append(0.0006408885434935317)
    ref1.append(7.475885211279207e-05)
    ref1.append(5.918413426727501e-05)
    ref1.append(0.0006428492013192681)
    ref1.append(0.0008460916752725523)
    ref1.append(7.448646882423489e-05)
    ref1.append(0.00010329100241930304)
    ref1.append(0.0010039025783440544)

    var ref2 = List[Float64]()
    ref2.append(0.0009652042720773147)
    ref2.append(0.0001650575775818208)
    ref2.append(3.701693111233088e-05)
    ref2.append(0.000657838486083935)
    ref2.append(0.0009570095909097236)
    ref2.append(0.00016486872283381785)
    ref2.append(3.752388257614683e-05)
    ref2.append(0.0006689456495354824)

    var cell = LTCCell[DType.float64](3, 4)
    _seed_cell(cell)

    # Three input frames (batch=2, input=3), ramp params from the reference.
    var x0 = zeros([2, 3], DType.float64)
    _seed_ramp(x0, 6, 0.1, -0.2)
    var x1 = zeros([2, 3], DType.float64)
    _seed_ramp(x1, 6, 0.05, 0.1)
    var x2 = zeros([2, 3], DType.float64)
    _seed_ramp(x2, 6, -0.03, 0.2)

    var h = zeros([2, 4], DType.float64)  # zero initial state
    h = cell.step(x0, h)
    for i in range(8):
        if _abs_diff(h.load[DType.float64](i), ref0[i]) > 1e-5:
            raise Error("LTC seq step 0 mismatch at " + String(i))
    h = cell.step(x1, h)
    for i in range(8):
        if _abs_diff(h.load[DType.float64](i), ref1[i]) > 1e-5:
            raise Error("LTC seq step 1 mismatch at " + String(i))
    h = cell.step(x2, h)
    for i in range(8):
        if _abs_diff(h.load[DType.float64](i), ref2[i]) > 1e-5:
            raise Error("LTC seq step 2 mismatch at " + String(i))
    print("  ok matches reference over 3-step sequence with state carry")
    print("test_parity_sequence PASSED")


def test_forward_equals_zero_state_step() raises:
    """`forward(x)` must equal step(x, zeros)."""
    print("Running test_forward_equals_zero_state_step...")
    var cell = LTCCell[DType.float64](3, 4)
    _seed_cell(cell)
    var x = zeros([2, 3], DType.float64)
    _seed_ramp(x, 6, 0.1, -0.2)
    var h0 = zeros([2, 4], DType.float64)

    var f = cell.forward(x)
    var s = cell.step(x, h0)
    for i in range(8):
        if (
            _abs_diff(f.load[DType.float64](i), s.load[DType.float64](i))
            > 1e-12
        ):
            raise Error("forward != step(x, zeros) at " + String(i))
    print("  ok forward(x) == step(x, zeros)")
    print("test_forward_equals_zero_state_step PASSED")


def main() raises:
    """Run all LTC tests."""
    print("=" * 60)
    print("LTC Cell Test Suite")
    print("=" * 60)
    test_shape()
    test_package_export()
    test_reject_bad_args()
    test_parameter_count()
    test_parity_single_step()
    test_parity_sequence()
    test_forward_equals_zero_state_step()
    print("=" * 60)
    print("All LTC tests PASSED")
    print("=" * 60)
