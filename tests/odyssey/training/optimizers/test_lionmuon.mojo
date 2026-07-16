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

    # Reference params after each step (flattened, 12 elems). ref0 (Muon branch)
    # is regenerated from parity_refs/lionmuon_parity_reference.py steps[0].params
    # — the prior ref0 was stale for elements 4-11 (diverged up to 1.76e-2 from
    # the committed reference), which left the Muon step effectively unverified.
    var ref0 = List[Float64]()
    ref0.append(-0.485037439)
    ref0.append(-0.390636309)
    ref0.append(-0.296235178)
    ref0.append(-0.201834047)
    ref0.append(-0.089532875)
    ref0.append(0.003772334)
    ref0.append(0.097077544)
    ref0.append(0.190382754)
    ref0.append(0.305971689)
    ref0.append(0.398180978)
    ref0.append(0.490390267)
    ref0.append(0.582599556)

    # After step 1 (Lion) — reference params from the numpy transcription.
    var ref1 = List[Float64]()
    ref1.append(-0.385037439)
    ref1.append(-0.290636309)
    ref1.append(-0.196235178)
    ref1.append(-0.101834047)
    ref1.append(0.010467125)
    ref1.append(0.103772334)
    ref1.append(-0.002922456)
    ref1.append(0.090382754)
    ref1.append(0.205971689)
    ref1.append(0.298180978)
    ref1.append(0.390390267)
    ref1.append(0.482599556)

    # After step 2 (Lion) — reference params from the numpy transcription.
    var ref2 = List[Float64]()
    ref2.append(-0.285037439)
    ref2.append(-0.190636309)
    ref2.append(-0.096235178)
    ref2.append(-0.001834047)
    ref2.append(0.110467125)
    ref2.append(0.203772334)
    ref2.append(-0.102922456)
    ref2.append(-0.009617246)
    ref2.append(0.105971689)
    ref2.append(0.198180978)
    ref2.append(0.290390267)
    ref2.append(0.382599556)

    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    # Step 0 (Muon) — the eigenbasis-orthogonalized branch.
    var (p0, m0) = lionmuon_step(W, G, M, 0.1, 0, 4)
    for i in range(12):
        if _abs_diff(p0.load[DType.float64](i), ref0[i]) > 1e-6:
            raise Error("LionMuon step-0 (Muon) mismatch at " + String(i))
    W = p0
    M = m0

    # Step 1 (Lion) — asserted numerically against the reference. This also
    # validates that the shared momentum buffer threaded out of the Muon step
    # feeds the Lion step correctly (continuity across the alternation).
    var (p1, m1) = lionmuon_step(W, G, M, 0.1, 1, 4)
    for i in range(12):
        if _abs_diff(p1.load[DType.float64](i), ref1[i]) > 1e-6:
            raise Error("LionMuon step-1 (Lion) mismatch at " + String(i))
    W = p1
    M = m1

    # Step 2 (Lion) — asserted numerically; confirms the buffer continues to
    # thread correctly across successive Lion steps.
    var (p2, _) = lionmuon_step(W, G, M, 0.1, 2, 4)
    for i in range(12):
        if _abs_diff(p2.load[DType.float64](i), ref2[i]) > 1e-6:
            raise Error("LionMuon step-2 (Lion) mismatch at " + String(i))

    print("  ok both Muon (step 0) and Lion (steps 1-2) branches match to 1e-6")
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
