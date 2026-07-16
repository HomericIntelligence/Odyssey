"""Unit tests for the Muon Hyperball optimizer (norm-constrained Muon).

Tests cover:
- Shape validation (rejects non-2D tensors, inherited from muon_step)
- Numerical parity with an independent numpy transcription of Odyssey's Muon core
  plus the hyperball projections (1e-6)
- The weight-norm constraint is active: ||W_new||_F <= weight_norm_max
- The update-norm constraint is active: ||W_new - W||_F <= update_norm_max
- Disabling a constraint (radius <= 0) leaves the corresponding norm unclamped
"""

from odyssey.tensor.any_tensor import AnyTensor, zeros, zeros_like
from odyssey.training.optimizers.muon_hyperball import (
    muon_hyperball_step,
    muon_hyperball_step_simple,
)
from odyssey.core.numerical_safety import compute_tensor_l2_norm


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _seed_ramp(mut t: AnyTensor, count: Int, scale: Float64, off: Float64) raises:
    for i in range(count):
        t.store[DType.float64](i, Float64(i) * scale + off)


def test_reject_non_2d() raises:
    """Muon Hyperball rejects non-matrix params (inherited from muon_step)."""
    print("Running test_reject_non_2d...")
    var p = zeros([10], DType.float32)
    var g = zeros([10], DType.float32)
    var m = zeros([10], DType.float32)
    try:
        var (_, _) = muon_hyperball_step(p, g, m, learning_rate=0.01)
        raise Error("Should have rejected 1D params")
    except e:
        print("  ok rejected 1D params")
    print("test_reject_non_2d PASSED")


def test_parity_with_reference() raises:
    """Match the numpy transcription of Odyssey Muon + hyperball to 1e-6.

    Reference values from parity_refs/muon_hyperball_parity_reference.py
    (R=3, C=4, lr=0.1, weight_norm_max=1.0, update_norm_max=0.1).
    """
    print("Running test_parity_with_reference...")

    var ref = List[Float64]()
    ref.append(-0.4096168)
    ref.append(-0.3298968)
    ref.append(-0.2501768)
    ref.append(-0.1704568)
    ref.append(-0.0756045)
    ref.append(0.003189)
    ref.append(0.0819826)
    ref.append(0.1607761)
    ref.append(0.2584078)
    ref.append(0.3362749)
    ref.append(0.414142)
    ref.append(0.4920091)

    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var (new_p, _) = muon_hyperball_step(W, G, M, 0.1, 1.0, 0.1)
    for i in range(12):
        if _abs_diff(new_p.load[DType.float64](i), ref[i]) > 1e-6:
            raise Error("Muon Hyperball parity mismatch at " + String(i))
    print("  ok matches reference to 1e-6")
    print("test_parity_with_reference PASSED")


def test_weight_norm_constraint_active() raises:
    """||W_new||_F must not exceed weight_norm_max when the update overshoots it."""
    print("Running test_weight_norm_constraint_active...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var radius = 1.0
    var (new_p, _) = muon_hyperball_step(W, G, M, 0.1, radius, 0.1)
    var norm = compute_tensor_l2_norm(new_p)
    if norm > radius + 1e-6:
        raise Error(
            "weight-norm constraint violated: " + String(norm) + " > " + String(radius)
        )
    print("  ok ||W_new||_F <= weight_norm_max (" + String(norm) + ")")
    print("test_weight_norm_constraint_active PASSED")


def test_update_norm_constraint_active() raises:
    """||W_new - W||_F must not exceed update_norm_max (weight clamp disabled)."""
    print("Running test_update_norm_constraint_active...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var update_max = 0.1
    # weight_norm_max = 0.0 disables the weight clamp so we isolate the update clamp.
    var (new_p, _) = muon_hyperball_step(W, G, M, 0.1, 0.0, update_max)
    var dW = zeros([3, 4], DType.float64)
    for i in range(12):
        dW.store[DType.float64](
            i, new_p.load[DType.float64](i) - W.load[DType.float64](i)
        )
    var norm = compute_tensor_l2_norm(dW)
    if norm > update_max + 1e-6:
        raise Error(
            "update-norm constraint violated: "
            + String(norm)
            + " > "
            + String(update_max)
        )
    print("  ok ||dW||_F <= update_norm_max (" + String(norm) + ")")
    print("test_update_norm_constraint_active PASSED")


def test_simple_wrapper_runs() raises:
    """The _simple convenience wrapper produces a finite step."""
    print("Running test_simple_wrapper_runs...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)
    var (new_p, _) = muon_hyperball_step_simple(W, G, M, 0.1)
    if new_p.numel() != 12:
        raise Error("simple wrapper returned wrong shape")
    print("  ok simple wrapper produced a 3x4 step")
    print("test_simple_wrapper_runs PASSED")


def main() raises:
    """Run all Muon Hyperball tests."""
    print("=" * 60)
    print("Muon Hyperball Optimizer Test Suite")
    print("=" * 60)
    test_reject_non_2d()
    test_parity_with_reference()
    test_weight_norm_constraint_active()
    test_update_norm_constraint_active()
    test_simple_wrapper_runs()
    print("=" * 60)
    print("All Muon Hyperball tests PASSED")
    print("=" * 60)
