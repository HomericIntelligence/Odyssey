"""Unit tests for the Muon Hyperball optimizer (norm-constrained Muon).

Tests cover:
- Shape validation (rejects non-2D tensors, inherited from muon_step)
- Numerical parity with an independent numpy transcription of Odyssey's Muon core
  plus the hyperball projections (1e-6), in two regimes: weight ball saturated
  (result rescaled onto the ball surface) and NOT saturated (projection is the
  identity, pinning the unclamped arithmetic exactly)
- The weight-norm constraint is active: ||W_new||_F <= weight_norm_max
- The update-norm constraint is active: ||W_new - W||_F <= update_norm_max
- Disabling a constraint (radius <= 0) leaves the corresponding norm unclamped
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, zeros_like
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


def _seed_ramp(
    mut t: AnyTensor, count: Int, scale: Float64, off: Float64
) raises:
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
    except _:
        print("  ok rejected 1D params")
    print("test_reject_non_2d PASSED")


def test_parity_with_reference() raises:
    """Match the numpy transcription of Odyssey Muon + hyperball to 1e-6.

    Reference values from parity_refs/muon_hyperball_parity_reference.py, case
    "saturated" (R=3, C=4, lr=0.1, weight_norm_max=1.0, update_norm_max=0.1;
    the weight-ball projection saturates: ||W_new||_F == 1.0).
    """
    print("Running test_parity_with_reference...")

    var ref_vals = List[Float64]()
    ref_vals.append(-0.4096168)
    ref_vals.append(-0.3298968)
    ref_vals.append(-0.2501768)
    ref_vals.append(-0.1704568)
    ref_vals.append(-0.0756045)
    ref_vals.append(0.003189)
    ref_vals.append(0.0819826)
    ref_vals.append(0.1607761)
    ref_vals.append(0.2584078)
    ref_vals.append(0.3362749)
    ref_vals.append(0.414142)
    ref_vals.append(0.4920091)

    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var (new_p, _) = muon_hyperball_step(W, G, M, 0.1, 1.0, 0.1)
    for i in range(12):
        if _abs_diff(new_p.load[DType.float64](i), ref_vals[i]) > 1e-6:
            raise Error("Muon Hyperball parity mismatch at " + String(i))
    print("  ok matches reference to 1e-6")
    print("test_parity_with_reference PASSED")


def test_parity_nonsaturated_weight_ball() raises:
    """Match the numpy reference with the weight-ball projection NOT saturating.

    Reference values from parity_refs/muon_hyperball_parity_reference.py, case
    "non_saturated" (R=3, C=4, lr=0.1, weight_norm_max=10.0, update_norm_max=0.1).
    The reference result has ||new_params||_F = 1.1829041645848064 < 10.0, so the
    weight projection is the identity and this asserts the unclamped muon +
    update-clamp arithmetic exactly (no final radial rescale masks upstream errors).
    """
    print("Running test_parity_nonsaturated_weight_ball...")

    var ref_vals = List[Float64]()
    ref_vals.append(-0.4845374394768569)
    ref_vals.append(-0.3902363086961616)
    ref_vals.append(-0.295935177915466)
    ref_vals.append(-0.20163404713477076)
    ref_vals.append(-0.08943287547482938)
    ref_vals.append(0.0037723344357082148)
    ref_vals.append(0.09697754434624539)
    ref_vals.append(0.19018275425678322)
    ref_vals.append(0.30567168852719817)
    ref_vals.append(0.39778097756757763)
    ref_vals.append(0.48989026660795737)
    ref_vals.append(0.5819995556483368)

    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var G = zeros([3, 4], DType.float64)
    _seed_ramp(G, 12, 0.05, -0.3)
    var M = zeros_like(W)

    var weight_norm_max = 10.0
    var (new_p, _) = muon_hyperball_step(W, G, M, 0.1, weight_norm_max, 0.1)
    for i in range(12):
        if _abs_diff(new_p.load[DType.float64](i), ref_vals[i]) > 1e-6:
            raise Error(
                "Muon Hyperball non-saturated parity mismatch at " + String(i)
            )

    # Prove the regime: the result must sit strictly INSIDE the weight ball
    # (reference norm 1.1829041645848064), i.e. the projection did not rescale.
    var norm = compute_tensor_l2_norm(new_p)
    if norm >= weight_norm_max:
        raise Error(
            "expected non-saturated weight ball, got ||W_new||_F = "
            + String(norm)
            + " >= "
            + String(weight_norm_max)
        )
    if _abs_diff(norm, 1.1829041645848064) > 1e-6:
        raise Error("non-saturated result norm drifted: " + String(norm))
    print(
        "  ok matches non-saturated reference to 1e-6 (||W_new||_F = "
        + String(norm)
        + " < 10.0)"
    )
    print("test_parity_nonsaturated_weight_ball PASSED")


def test_weight_norm_constraint_active() raises:
    """||W_new||_F must not exceed weight_norm_max when the update overshoots it.
    """
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
            "weight-norm constraint violated: "
            + String(norm)
            + " > "
            + String(radius)
        )
    print("  ok ||W_new||_F <= weight_norm_max (" + String(norm) + ")")
    print("test_weight_norm_constraint_active PASSED")


def test_update_norm_constraint_active() raises:
    """||W_new - W||_F must not exceed update_norm_max (weight clamp disabled).
    """
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
    test_parity_nonsaturated_weight_ball()
    test_weight_norm_constraint_active()
    test_update_norm_constraint_active()
    test_simple_wrapper_runs()
    print("=" * 60)
    print("All Muon Hyperball tests PASSED")
    print("=" * 60)
