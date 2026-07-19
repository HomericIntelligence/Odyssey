"""Unit tests for the diagonal state-space (S4-style) block.

Tests cover:
- Construction + validation (positive dim/state)
- forward shape (batch, seq, dim) -> (batch, seq, dim)
- parameter collection (5 tensors: a_log/b/c/d/log_dt)
- Numerical parity with an explicit torch diagonal-SSM recurrence (1e-5), for
  BOTH a single step AND the full multi-step sequence (state carried across steps)
- forward(seq) equals repeated step() calls threading the state

Reference values are the committed fixture
parity_refs/ssm_parity_reference.json, produced by
parity_refs/ssm_parity_reference.py (dim=3, state=2, batch=2, seq=4). Re-run the
generator and diff the fixture before push; the inline values below MUST equal it.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.ssm import DiagonalSSM


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
    """`forward` maps (batch, seq, dim) -> (batch, seq, dim)."""
    print("Running test_shape...")
    var ssm = DiagonalSSM[DType.float32](3, 2)
    var u = zeros([2, 4, 3], DType.float32)
    var out = ssm.forward(u)
    if out.shape()[0] != 2 or out.shape()[1] != 4 or out.shape()[2] != 3:
        raise Error("SSM forward must return (batch, seq, dim)")
    print("  ok (2, 4, 3)")
    print("test_shape PASSED")


def test_reject_bad_sizes() raises:
    """Non-positive sizes must raise."""
    print("Running test_reject_bad_sizes...")
    try:
        var _ = DiagonalSSM[DType.float32](0, 2)
        raise Error("Should have rejected dim = 0")
    except _:
        print("  ok rejected dim = 0")
    try:
        var _ = DiagonalSSM[DType.float32](3, 0)
        raise Error("Should have rejected state = 0")
    except _:
        print("  ok rejected state = 0")
    print("test_reject_bad_sizes PASSED")


def test_parameter_count() raises:
    """`parameters()` returns 5 tensors (a_log, b, c, d, log_dt)."""
    print("Running test_parameter_count...")
    var ssm = DiagonalSSM[DType.float32](3, 2)
    if len(ssm.parameters()) != 5:
        raise Error("SSM should expose 5 parameter tensors")
    print("  ok 5 parameter tensors")
    print("test_parameter_count PASSED")


def _make_seeded() raises -> DiagonalSSM[DType.float64]:
    """Build the dim=3,state=2 SSM with the fixture's ramp parameters."""
    var ssm = DiagonalSSM[DType.float64](3, 2)
    # a_log [3,2] = i*0.05 - 0.30 ; b = i*0.02 + 0.10 ; c = i*0.03 - 0.15
    _seed_ramp(ssm.a_log, 6, 0.05, -0.30)
    _seed_ramp(ssm.b, 6, 0.02, 0.10)
    _seed_ramp(ssm.c, 6, 0.03, -0.15)
    # d [3] = i*0.04 - 0.05 ; log_dt [3] = i*0.10 - 0.20
    _seed_ramp(ssm.d, 3, 0.04, -0.05)
    _seed_ramp(ssm.log_dt, 3, 0.10, -0.20)
    return ssm^


def test_parity_single_step() raises:
    """`step` from zero state must match the torch reference (1e-5).

    Reference: parity_refs/ssm_parity_reference.json "y_single_step" [B*D=6].
    """
    print("Running test_parity_single_step...")
    var ref_vals = List[Float64]()
    ref_vals.append(0.016981284442824887)
    ref_vals.append(0.0043357760126180355)
    ref_vals.append(-0.002909141883478728)
    ref_vals.append(-0.04007583128506674)
    ref_vals.append(-0.015897845379599467)
    ref_vals.append(0.019306123408540654)

    var ssm = _make_seeded()
    # u_0 = U[:, 0, :]: batch 0 = U flat [0,1,2], batch 1 = U flat [12,13,14].
    # U[i] = i*0.07 - 0.25 (flat over [B, SEQ*DIM]).
    var u0 = zeros([2, 3], DType.float64)
    u0.store[DType.float64](0, 0.0 * 0.07 - 0.25)
    u0.store[DType.float64](1, 1.0 * 0.07 - 0.25)
    u0.store[DType.float64](2, 2.0 * 0.07 - 0.25)
    u0.store[DType.float64](3, 12.0 * 0.07 - 0.25)
    u0.store[DType.float64](4, 13.0 * 0.07 - 0.25)
    u0.store[DType.float64](5, 14.0 * 0.07 - 0.25)

    var x0 = zeros([2, 3, 2], DType.float64)
    var yx = ssm.step(u0, x0)
    var y = yx[0]
    for i in range(6):
        if _abs_diff(y.load[DType.float64](i), ref_vals[i]) > 1e-5:
            raise Error("SSM single-step parity mismatch at " + String(i))
    print("  ok single step matches torch reference to 1e-5")
    print("test_parity_single_step PASSED")


def _seed_full_u(mut u: AnyTensor) raises:
    """Seed U [2,4,3] flat with U[i] = i*0.07 - 0.25 (24 elems)."""
    for i in range(24):
        u.store[DType.float64](i, Float64(i) * 0.07 - 0.25)


def test_parity_sequence() raises:
    """`forward` over 4 steps (state carried) must match the torch reference.

    Reference: parity_refs/ssm_parity_reference.json "y_sequence" [B*SEQ*D=24].
    """
    print("Running test_parity_sequence...")
    var ref_vals = List[Float64]()
    ref_vals.append(0.016981284442824887)
    ref_vals.append(0.0043357760126180355)
    ref_vals.append(-0.002909141883478728)
    ref_vals.append(0.005123994888213095)
    ref_vals.append(0.0004669355008566803)
    ref_vals.append(0.002802818628094236)
    ref_vals.append(-0.009868999965967996)
    ref_vals.append(-0.005421059448029987)
    ref_vals.append(0.008118709668720445)
    ref_vals.append(-0.0265464852330115)
    ref_vals.append(-0.012256506768259132)
    ref_vals.append(0.013274347966850338)
    ref_vals.append(-0.04007583128506674)
    ref_vals.append(-0.015897845379599467)
    ref_vals.append(0.019306123408540654)
    ref_vals.append(-0.06002060514761187)
    ref_vals.append(-0.02531798846072825)
    ref_vals.append(0.023810437389229805)
    ref_vals.append(-0.07935860132183747)
    ref_vals.append(-0.033811084936659554)
    ref_vals.append(0.028637705390878315)
    ref_vals.append(-0.09837100123054518)
    ref_vals.append(-0.04186947449922052)
    ref_vals.append(0.03359564306377222)

    var ssm = _make_seeded()
    var u = zeros([2, 4, 3], DType.float64)
    _seed_full_u(u)
    var out = ssm.forward(u)
    for i in range(24):
        if _abs_diff(out.load[DType.float64](i), ref_vals[i]) > 1e-5:
            raise Error("SSM sequence parity mismatch at " + String(i))
    print("  ok multi-step sequence matches torch reference to 1e-5")
    print("test_parity_sequence PASSED")


def test_forward_equals_stepped() raises:
    """`forward(seq)` must equal repeated `step` calls threading the state.

    Regression guard: the recurrent scan and the step API must produce the same
    trajectory (single-step and sequence forms of the SAME LTI system).
    """
    print("Running test_forward_equals_stepped...")
    var ssm = _make_seeded()
    var u = zeros([2, 4, 3], DType.float64)
    _seed_full_u(u)
    var out = ssm.forward(u)

    # Thread step() manually across the 4 timesteps from zero state.
    var x = zeros([2, 3, 2], DType.float64)
    for t in range(4):
        var u_t = zeros([2, 3], DType.float64)
        for bi in range(2):
            for ch in range(3):
                # u[bi, t, ch] flat index in [2,4,3]
                var src = (bi * 4 + t) * 3 + ch
                u_t.store[DType.float64](
                    bi * 3 + ch, u.load[DType.float64](src)
                )
        var yx = ssm.step(u_t, x)
        var y_t = yx[0]
        x = yx[1]
        for bi in range(2):
            for ch in range(3):
                var got = out.load[DType.float64]((bi * 4 + t) * 3 + ch)
                var want = y_t.load[DType.float64](bi * 3 + ch)
                if _abs_diff(got, want) > 1e-12:
                    raise Error(
                        "forward != stepped at t="
                        + String(t)
                        + " bi="
                        + String(bi)
                        + " ch="
                        + String(ch)
                    )
    print("  ok forward(seq) == threaded step() calls")
    print("test_forward_equals_stepped PASSED")


def main() raises:
    """Run all diagonal-SSM tests."""
    print("=" * 60)
    print("Diagonal SSM (S4-style) Block Test Suite")
    print("=" * 60)
    test_shape()
    test_reject_bad_sizes()
    test_parameter_count()
    test_parity_single_step()
    test_parity_sequence()
    test_forward_equals_stepped()
    print("=" * 60)
    print("All SSM tests PASSED")
    print("=" * 60)
