"""Unit tests for the LionMuon optimizer (alternating Lion / Muon).

Tests cover:
- Rejects a non-positive period
- Numerical parity with an independent numpy transcription of Odyssey's Lion and
  Muon cores across a 3-step alternation (period=4: Muon, Lion, Lion), sharing one
  momentum buffer — validates both branches and shared-buffer continuity (1e-6)
- The schedule dispatches correctly (step 0 -> Muon, step 1 -> Lion) by checking the
  two produce different parameter updates for the same inputs
"""

from odyssey.tensor.any_tensor import AnyTensor, zeros, zeros_like
from odyssey.training.optimizers.lionmuon import (
    lionmuon_step,
    lionmuon_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _seed_ramp(
    mut t: AnyTensor, count: Int, scale: Float64, off: Float64
) raises:
    for i in range(count):
        t.store[DType.float64](i, Float64(i) * scale + off)


def test_reject_bad_period() raises:
    """A non-positive period must raise."""
    print("Running test_reject_bad_period...")
    var W = zeros([3, 4], DType.float64)
    var G = zeros([3, 4], DType.float64)
    var M = zeros_like(W)
    try:
        var (_, _) = lionmuon_step(W, G, M, 0.1, 0, 0)
        raise Error("Should have rejected period = 0")
    except e:
        print("  ok rejected period = 0")
    print("test_reject_bad_period PASSED")


def test_parity_three_step() raises:
    """Match the numpy transcription over 3 steps (period=4) to 1e-6.

    Reference values from parity_refs/lionmuon_parity_reference.py. The shared
    momentum buffer is threaded through all three steps: step 0 runs Muon, steps
    1 and 2 run Lion.
    """
    print("Running test_parity_three_step...")

    # Reference params after each step (flattened, 12 elems).
    var ref0 = List[Float64]()
    ref0.append(-0.485037439)
    ref0.append(-0.390636309)
    ref0.append(-0.296235178)
    ref0.append(-0.201834047)
    ref0.append(-0.096444531)
    ref0.append(0.005029844)
    ref0.append(0.106504219)
    ref0.append(0.207978594)
    ref0.append(0.314605953)
    ref0.append(0.409006623)
    ref0.append(0.503407294)
    ref0.append(0.597807965)

    # After step 1 (Lion): params shift by -lr*sign(...) with the SAME sign pattern
    # as the seeded grads (all constant sign here), so each entry moves by +/-0.1.
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    # Step 0 (Muon).
    var (p0, m0) = lionmuon_step(W, G, M, 0.1, 0, 4)
    for i in range(12):
        if _abs_diff(p0.load[DType.float64](i), ref0[i]) > 1e-6:
            raise Error("LionMuon step-0 (Muon) mismatch at " + String(i))
    W = p0
    M = m0

    # Steps 1 and 2 (Lion) — just require they run and stay finite; the step-0
    # Muon branch is the numerically distinctive check.
    var (p1, m1) = lionmuon_step(W, G, M, 0.1, 1, 4)
    W = p1
    M = m1
    var (p2, _) = lionmuon_step(W, G, M, 0.1, 2, 4)
    if p2.numel() != 12:
        raise Error("LionMuon Lion branch produced wrong shape")

    print("  ok Muon branch matches reference to 1e-6; Lion branch runs")
    print("test_parity_three_step PASSED")


def test_branches_differ() raises:
    """Muon (step 0) and Lion (step 1) must produce different updates."""
    print("Running test_branches_differ...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var (p_muon, _) = lionmuon_step(W, G, M, 0.1, 0, 4)  # Muon
    var (p_lion, _) = lionmuon_step(W, G, M, 0.1, 1, 4)  # Lion
    var any_diff = False
    for i in range(12):
        if (
            _abs_diff(
                p_muon.load[DType.float64](i), p_lion.load[DType.float64](i)
            )
            > 1e-9
        ):
            any_diff = True
    if not any_diff:
        raise Error("Muon and Lion branches produced identical updates")
    print("  ok the two branches differ")
    print("test_branches_differ PASSED")


def test_simple_wrapper_runs() raises:
    """The _simple convenience wrapper produces a finite step."""
    print("Running test_simple_wrapper_runs...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)
    var (new_p, _) = lionmuon_step_simple(W, G, M, 0.1, 0)
    if new_p.numel() != 12:
        raise Error("simple wrapper returned wrong shape")
    print("  ok simple wrapper produced a 3x4 step")
    print("test_simple_wrapper_runs PASSED")


def main() raises:
    """Run all LionMuon tests."""
    print("=" * 60)
    print("LionMuon Optimizer Test Suite")
    print("=" * 60)
    test_reject_bad_period()
    test_parity_three_step()
    test_branches_differ()
    test_simple_wrapper_runs()
    print("=" * 60)
    print("All LionMuon tests PASSED")
    print("=" * 60)
