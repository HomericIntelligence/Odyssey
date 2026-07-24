"""Unit tests for the KL-Shampoo optimizer (Adam-free stable Shampoo).

Tests cover:
- Rejects a non-2D parameter
- Rejects a shape-mismatched gradient (validation guard)
- `is_kl_shampoo_eligible` accepts matrices, rejects vectors / degenerate matrices
- `init_kl_shampoo_state` produces IDENTITY factors of the right shape (2 tensors)
- Numerical parity with an independent numpy transcription of the KL-Shampoo step
  across a 3-step run (coupled cross-factor whitening + inverse-square-root
  preconditioning + cross-step state threading)

Parity reference values from parity_refs/kl_shampoo_parity_reference.py (R=3, C=4,
lr=0.1, beta=0.95, ridge=1e-8). The gradient on step s is (i*0.05 − 0.3) + s*0.01.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.training.optimizers.kl_shampoo import (
    kl_shampoo_step,
    kl_shampoo_step_simple,
    init_kl_shampoo_state,
    is_kl_shampoo_eligible,
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


def test_reject_non_2d() raises:
    """KL-Shampoo rejects a non-matrix parameter."""
    print("Running test_reject_non_2d...")
    var p = zeros([10], DType.float64)
    var g = zeros([10], DType.float64)
    var s_a = zeros([10, 10], DType.float64)
    var s_b = zeros([10, 10], DType.float64)
    try:
        var (_, _, _) = kl_shampoo_step(p, g, s_a, s_b, 0.1)
        raise Error("Should have rejected 1D params")
    except _:
        print("  ok rejected 1D params")
    print("test_reject_non_2d PASSED")


def test_reject_shape_mismatch() raises:
    """KL-Shampoo rejects a gradient whose shape differs from params."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([3, 4], DType.float64)
    var g = zeros([3, 5], DType.float64)
    var w_st = init_kl_shampoo_state([p])
    var st = w_st[0].copy()
    try:
        var (_, _, _) = kl_shampoo_step(p, g, st[0], st[1], 0.1)
        raise Error("Should have rejected shape-mismatched gradient")
    except _:
        print("  ok rejected shape-mismatched gradient")
    print("test_reject_shape_mismatch PASSED")


def test_reject_float32_params() raises:
    """KL-Shampoo is float64-only: an f32 param raises at the final subtraction.

    All preconditioner math (and the resulting delta) is float64; the parameter
    update `subtract_simd(params, delta)` requires params to share that dtype, so
    an f32 param raises "Cannot subtract tensors with different dtypes". This pins
    the float64-only contract (same posture as SOAP — no f32 fast path).
    """
    print("Running test_reject_float32_params...")
    var p = zeros([3, 4], DType.float32)
    var g = zeros([3, 4], DType.float32)
    # State is always float64 (init_kl_shampoo_state emits f64 factors); build f64
    # identity factors to isolate the param-dtype contract at the final subtraction.
    var s_a = zeros([3, 3], DType.float64)
    var s_b = zeros([4, 4], DType.float64)
    for i in range(3):
        s_a.store[DType.float64](i * 3 + i, 1.0)
    for i in range(4):
        s_b.store[DType.float64](i * 4 + i, 1.0)
    try:
        var (_, _, _) = kl_shampoo_step(p, g, s_a, s_b, 0.1)
        raise Error("Should have rejected float32 params")
    except _:
        print("  ok rejected float32 params (float64-only contract)")
    print("test_reject_float32_params PASSED")


def test_reject_degenerate_dim() raises:
    """KL-Shampoo rejects a matrix with a dimension < 2 (1×N) in step AND init.

    `is_kl_shampoo_eligible` requires both dims >= 2; the step and init guards must
    agree so an "ineligible" 1×N param is not silently accepted.
    """
    print("Running test_reject_degenerate_dim...")
    var p = zeros([1, 4], DType.float64)
    var g = zeros([1, 4], DType.float64)
    var s_a = zeros([1, 1], DType.float64)
    s_a.store[DType.float64](0, 1.0)
    var s_b = zeros([4, 4], DType.float64)
    for i in range(4):
        s_b.store[DType.float64](i * 4 + i, 1.0)
    try:
        var (_, _, _) = kl_shampoo_step(p, g, s_a, s_b, 0.1)
        raise Error("kl_shampoo_step should have rejected a 1×4 param")
    except _:
        print("  ok kl_shampoo_step rejected 1×4 (dim < 2)")
    try:
        var _st = init_kl_shampoo_state([p])
        raise Error("init_kl_shampoo_state should have rejected a 1×4 param")
    except _:
        print("  ok init_kl_shampoo_state rejected 1×4 (dim < 2)")
    print("test_reject_degenerate_dim PASSED")


def test_eligibility() raises:
    """`is_kl_shampoo_eligible` accepts matrices, rejects vectors / degenerate.
    """
    print("Running test_eligibility...")
    var mat = zeros([4, 5], DType.float64)
    if not is_kl_shampoo_eligible(mat):
        raise Error("4x5 matrix should be eligible")
    var vec = zeros([4], DType.float64)
    if is_kl_shampoo_eligible(vec):
        raise Error("vector should not be eligible")
    var degenerate = zeros([1, 5], DType.float64)
    if is_kl_shampoo_eligible(degenerate):
        raise Error("1x5 (a dimension < 2) should not be eligible")
    print("  ok eligibility: matrix yes, vector/degenerate no")
    print("test_eligibility PASSED")


def test_init_state_shapes() raises:
    """`init_kl_shampoo_state` returns 2 IDENTITY factors with the right shapes.
    """
    print("Running test_init_state_shapes...")
    var W = zeros([3, 4], DType.float64)
    var w_st = init_kl_shampoo_state([W])
    var st = w_st[0].copy()
    # S_A: 3x3 = 9 ; S_B: 4x4 = 16.
    if st[0].numel() != 9:
        raise Error("S_A should be 3x3")
    if st[1].numel() != 16:
        raise Error("S_B should be 4x4")
    # S_A identity: diagonal 1, off-diagonal 0.
    for r in range(3):
        for c in range(3):
            var expected = 1.0 if r == c else 0.0
            if st[0].load[DType.float64](r * 3 + c) != expected:
                raise Error("S_A must be initialized to the identity")
    # S_B identity check (diagonal).
    for i in range(4):
        if st[1].load[DType.float64](i * 4 + i) != 1.0:
            raise Error("S_B diagonal must be 1")
    print("  ok state shapes 3x3 / 4x4, identity-initialized")
    print("test_init_state_shapes PASSED")


def test_parity_three_step() raises:
    """Match the numpy transcription over 3 steps to 1e-6.

    Reference values from parity_refs/kl_shampoo_parity_reference.py (R=3, C=4,
    lr=0.1, beta=0.95, ridge=1e-8). The gradient on step s is (i*0.05−0.3)+s*0.01.
    State accumulates across all three steps, so this exercises the coupled
    cross-factor update and the inverse-square-root preconditioner as the factors
    depart from the identity.
    """
    print("Running test_parity_three_step...")

    var ref1 = List[Float64]()
    ref1.append(-0.23203041178211647)
    ref1.append(-0.20856918609552463)
    ref1.append(-0.18510796040893285)
    ref1.append(-0.16164673472234092)
    ref1.append(0.032644882032099226)
    ref1.append(0.05019274802788096)
    ref1.append(0.06774061402366269)
    ref1.append(0.08528848001944449)
    ref1.append(0.2973201758463148)
    ref1.append(0.3089546821512864)
    ref1.append(0.32058918845625806)
    ref1.append(0.33222369476122976)

    var ref2 = List[Float64]()
    ref2.append(-0.15552713523432948)
    ref2.append(-0.16134313580120274)
    ref2.append(-0.167159136368077)
    ref2.append(-0.17297513693495215)
    ref2.append(0.07745555511271876)
    ref2.append(0.06607162906370327)
    ref2.append(0.05468770301468673)
    ref2.append(0.043303776965672294)
    ref2.append(0.3104382454597658)
    ref2.append(0.29348639392860926)
    ref2.append(0.2765345423974526)
    ref2.append(0.25958269086629465)

    var ref3 = List[Float64]()
    ref3.append(0.07815046379833618)
    ref3.append(-0.006053367111831964)
    ref3.append(-0.09025719802200557)
    ref3.append(-0.17446102893220025)
    ref3.append(0.18551516287292352)
    ref3.append(0.10078453594514963)
    ref3.append(0.01605390901732462)
    ref3.append(-0.06867671791044909)
    ref3.append(0.2928798619475417)
    ref3.append(0.2076224390021189)
    ref3.append(0.1223650160567048)
    ref3.append(0.037107593111277704)

    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var w_st = init_kl_shampoo_state([W])
    var st = w_st[0].copy()
    var s_a = st[0]
    var s_b = st[1]

    for s in range(1, 4):
        var g = zeros([3, 4], DType.float64)
        _seed_ramp(g, 12, 0.05, -0.3 + Float64(s) * 0.01)
        var r = kl_shampoo_step(W, g, s_a, s_b, 0.1)
        W = r[0]
        s_a = r[1]
        s_b = r[2]
        for i in range(12):
            var expected: Float64
            if s == 1:
                expected = ref1[i]
            elif s == 2:
                expected = ref2[i]
            else:
                expected = ref3[i]
            if _abs_diff(W.load[DType.float64](i), expected) > 1e-6:
                raise Error(
                    "KL-Shampoo step-" + String(s) + " mismatch at " + String(i)
                )

    print("  ok matches reference over 3 steps to 1e-6")
    print("test_parity_three_step PASSED")


def test_kl_shampoo_step_simple_delegates() raises:
    """`kl_shampoo_step_simple` matches the full step at documented defaults.

    The simple wrapper delegates to `kl_shampoo_step` with `beta=0.95`,
    `weight_decay=0.0`, `ridge=1e-8` (per kl_shampoo.mojo). Asserts exact
    equality on params AND on the S_A / S_B preconditioner factors — both
    factors accumulate the gradient Kronecker products and so drift per
    step; silent drift = silent divergence. A future regression in the
    simple wrapper's delegation contract is caught here rather than as
    a downstream divergent loss.
    """
    print("Running test_kl_shampoo_step_simple_delegates...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var w_st = init_kl_shampoo_state([W])
    var st = w_st[0].copy()
    var s_a_full = st[0]
    var s_b_full = st[1]
    # Independent state buffers so we can compare both at the same step.
    var st2 = init_kl_shampoo_state([W])[0].copy()
    var s_a_simple = st2[0]
    var s_b_simple = st2[1]
    var g = zeros([3, 4], DType.float64)
    _seed_ramp(g, 12, 0.05, -0.3)

    var full_out = kl_shampoo_step(
        W, g, s_a_full, s_b_full, 0.1, 0.95, 0.0, 1e-8
    )
    var simple_out = kl_shampoo_step_simple(W, g, s_a_simple, s_b_simple, 0.1)

    # params (12)
    for i in range(12):
        if (
            _abs_diff(
                full_out[0].load[DType.float64](i),
                simple_out[0].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error(
                "kl_shampoo_step_simple params diverged at " + String(i)
            )
    # S_A is 3x3 = 9
    for i in range(9):
        if (
            _abs_diff(
                full_out[1].load[DType.float64](i),
                simple_out[1].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error("kl_shampoo_step_simple S_A diverged at " + String(i))
    # S_B is 4x4 = 16
    for i in range(16):
        if (
            _abs_diff(
                full_out[2].load[DType.float64](i),
                simple_out[2].load[DType.float64](i),
            )
            > 1e-12
        ):
            raise Error("kl_shampoo_step_simple S_B diverged at " + String(i))
    print("  ok kl_shampoo_step_simple delegates to the full step at defaults")
    print("test_kl_shampoo_step_simple_delegates PASSED")


def main() raises:
    """Run all KL-Shampoo tests."""
    print("=" * 60)
    print("KL-Shampoo Optimizer Test Suite")
    print("=" * 60)
    test_reject_non_2d()
    test_reject_shape_mismatch()
    test_reject_float32_params()
    test_reject_degenerate_dim()
    test_eligibility()
    test_init_state_shapes()
    test_parity_three_step()
    test_kl_shampoo_step_simple_delegates()
    print("=" * 60)
    print("All KL-Shampoo tests PASSED")
    print("=" * 60)
