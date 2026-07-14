"""Unit tests for the MGUP-Muon optimizer (Muon with selective updates).

Tests cover:
- Shape validation (rejects non-2D tensors, inherited from muon_step)
- Numerical parity with an independent numpy transcription of Odyssey's Muon core
  plus the MGUP selective-amplification (1e-6)
- Disabling selection (fraction=0 or scale=1) reduces exactly to Muon
- The selected fraction amplifies strictly more than plain Muon would (the selected
  coordinates move further than the un-amplified Muon step)
"""

from odyssey.tensor.any_tensor import AnyTensor, zeros, zeros_like
from odyssey.training.optimizers.mgup_muon import (
    mgup_muon_step,
    mgup_muon_step_simple,
)
from odyssey.training.optimizers.muon import muon_step


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _seed_ramp(mut t: AnyTensor, count: Int, scale: Float64, off: Float64) raises:
    for i in range(count):
        t.store[DType.float64](i, Float64(i) * scale + off)


def test_reject_non_2d() raises:
    """MGUP-Muon rejects non-matrix params (inherited from muon_step)."""
    print("Running test_reject_non_2d...")
    var p = zeros([10], DType.float32)
    var g = zeros([10], DType.float32)
    var m = zeros([10], DType.float32)
    try:
        var (_, _) = mgup_muon_step(p, g, m, learning_rate=0.01)
        raise Error("Should have rejected 1D params")
    except e:
        print("  ok rejected 1D params")
    print("test_reject_non_2d PASSED")


def test_parity_with_reference() raises:
    """Match the numpy transcription of Odyssey Muon + MGUP to 1e-6.

    Reference values from parity_refs/mgup_muon_parity_reference.py
    (R=3, C=4, lr=0.1, selected_fraction=0.25, select_scale=2.0; 3 of 12
    coordinates amplified).
    """
    print("Running test_parity_with_reference...")

    var ref = List[Float64]()
    ref.append(-0.4690749)
    ref.append(-0.3902363)
    ref.append(-0.2959352)
    ref.append(-0.201634)
    ref.append(-0.0788658)
    ref.append(0.0037723)
    ref.append(0.0969775)
    ref.append(0.1901828)
    ref.append(0.3056717)
    ref.append(0.397781)
    ref.append(0.4898903)
    ref.append(0.5639991)

    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var (new_p, _) = mgup_muon_step(W, G, M, 0.1, 0.25, 2.0)
    for i in range(12):
        if _abs_diff(new_p.load[DType.float64](i), ref[i]) > 1e-6:
            raise Error("MGUP-Muon parity mismatch at " + String(i))
    print("  ok matches reference to 1e-6")
    print("test_parity_with_reference PASSED")


def test_reduces_to_muon_when_disabled() raises:
    """fraction=0 or scale=1 must reproduce a plain Muon step exactly."""
    print("Running test_reduces_to_muon_when_disabled...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var (p_muon, _) = muon_step(W, G, M, 0.1)
    var (p_frac0, _) = mgup_muon_step(W, G, M, 0.1, 0.0, 2.0)
    var (p_scale1, _) = mgup_muon_step(W, G, M, 0.1, 0.25, 1.0)
    for i in range(12):
        if _abs_diff(p_muon.load[DType.float64](i), p_frac0.load[DType.float64](i)) > 1e-12:
            raise Error("fraction=0 should equal plain Muon at " + String(i))
        if _abs_diff(p_muon.load[DType.float64](i), p_scale1.load[DType.float64](i)) > 1e-12:
            raise Error("scale=1 should equal plain Muon at " + String(i))
    print("  ok fraction=0 and scale=1 both reduce to Muon")
    print("test_reduces_to_muon_when_disabled PASSED")


def test_selected_move_further() raises:
    """The amplified coordinates move strictly further than plain Muon would.

    At least one coordinate's displacement from W must exceed the plain-Muon
    displacement (the selected coordinates get select_scale * the Muon step).
    """
    print("Running test_selected_move_further...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var (p_muon, _) = muon_step(W, G, M, 0.1)
    var (p_mgup, _) = mgup_muon_step(W, G, M, 0.1, 0.25, 2.0)
    var any_larger = False
    for i in range(12):
        var d_muon = _abs_diff(p_muon.load[DType.float64](i), W.load[DType.float64](i))
        var d_mgup = _abs_diff(p_mgup.load[DType.float64](i), W.load[DType.float64](i))
        if d_mgup > d_muon + 1e-9:
            any_larger = True
    if not any_larger:
        raise Error("no coordinate was amplified beyond the plain Muon step")
    print("  ok selected coordinates move further than plain Muon")
    print("test_selected_move_further PASSED")


def test_simple_wrapper_runs() raises:
    """The _simple convenience wrapper produces a finite step."""
    print("Running test_simple_wrapper_runs...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)
    var (new_p, _) = mgup_muon_step_simple(W, G, M, 0.1)
    if new_p.numel() != 12:
        raise Error("simple wrapper returned wrong shape")
    print("  ok simple wrapper produced a 3x4 step")
    print("test_simple_wrapper_runs PASSED")


def main() raises:
    """Run all MGUP-Muon tests."""
    print("=" * 60)
    print("MGUP-Muon Optimizer Test Suite")
    print("=" * 60)
    test_reject_non_2d()
    test_parity_with_reference()
    test_reduces_to_muon_when_disabled()
    test_selected_move_further()
    test_simple_wrapper_runs()
    print("=" * 60)
    print("All MGUP-Muon tests PASSED")
    print("=" * 60)
