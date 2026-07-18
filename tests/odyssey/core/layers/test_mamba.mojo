"""Unit tests for the Mamba selective state-space (S6) block.

Tests cover:
- Construction + validation (positive dim/state/conv_kernel)
- forward shape (batch, seq, dim) -> (batch, seq, dim)
- parameter collection (12 tensors)
- Numerical parity with an explicit torch selective-scan (S6) reference (1e-5),
  for BOTH a single step (timestep 0) AND the full multi-step sequence (SSM
  state carried across all 4 steps)

Reference values are the committed fixture
parity_refs/mamba_parity_reference.json, produced by
parity_refs/mamba_parity_reference.py (dim=4, state=3, conv_kernel=3, batch=2,
seq=4). The seeded parameters below use the SAME ramp (scale, off) constants as
the generator; the inline y_* values MUST equal the fixture. Re-run the generator
and diff the fixture before push.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.mamba import MambaBlock


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
    var mamba = MambaBlock[DType.float32](4, 3, 3)
    var u = zeros([2, 4, 4], DType.float32)
    var out = mamba.forward(u)
    if out.shape()[0] != 2 or out.shape()[1] != 4 or out.shape()[2] != 4:
        raise Error("Mamba forward must return (batch, seq, dim)")
    print("  ok (2, 4, 4)")
    print("test_shape PASSED")


def test_reject_bad_sizes() raises:
    """Non-positive sizes must raise."""
    print("Running test_reject_bad_sizes...")
    try:
        var _ = MambaBlock[DType.float32](0, 3, 3)
        raise Error("Should have rejected dim = 0")
    except _:
        print("  ok rejected dim = 0")
    try:
        var _ = MambaBlock[DType.float32](4, 0, 3)
        raise Error("Should have rejected state = 0")
    except _:
        print("  ok rejected state = 0")
    try:
        var _ = MambaBlock[DType.float32](4, 3, 0)
        raise Error("Should have rejected conv_kernel = 0")
    except _:
        print("  ok rejected conv_kernel = 0")
    print("test_reject_bad_sizes PASSED")


def test_parameter_count() raises:
    """`parameters()` returns 12 tensors."""
    print("Running test_parameter_count...")
    var mamba = MambaBlock[DType.float32](4, 3, 3)
    if len(mamba.parameters()) != 12:
        raise Error("Mamba should expose 12 parameter tensors")
    print("  ok 12 parameter tensors")
    print("test_parameter_count PASSED")


def _make_seeded() raises -> MambaBlock[DType.float64]:
    """Build the dim=4,state=3,K=3 block with the fixture's ramp parameters.

    The (scale, off) pairs below MUST match mamba_parity_reference.py exactly.
    """
    var m = MambaBlock[DType.float64](4, 3, 3)
    # Wz [4,4] = i*0.011 - 0.07 ; bz [4] = i*0.03 - 0.04
    _seed_ramp(m.wz, 16, 0.011, -0.07)
    _seed_ramp(m.bz, 4, 0.03, -0.04)
    # conv_w [4,3] = i*0.02 + 0.05 ; conv_b [4] = i*0.01 - 0.02
    _seed_ramp(m.conv_w, 12, 0.02, 0.05)
    _seed_ramp(m.conv_b, 4, 0.01, -0.02)
    # WB [4,3] = i*0.013 - 0.06 ; WC [4,3] = i*0.017 - 0.05
    _seed_ramp(m.w_b, 12, 0.013, -0.06)
    _seed_ramp(m.w_c, 12, 0.017, -0.05)
    # Wdt [4,4] = i*0.009 - 0.03 ; dt_bias [4] = i*0.02 - 0.10
    _seed_ramp(m.w_dt, 16, 0.009, -0.03)
    _seed_ramp(m.dt_bias, 4, 0.02, -0.10)
    # A_log [4,3] = i*0.05 - 0.30 ; D [4] = i*0.04 - 0.05
    _seed_ramp(m.a_log, 12, 0.05, -0.30)
    _seed_ramp(m.d, 4, 0.04, -0.05)
    # Wo [4,4] = i*0.015 - 0.08 ; bo [4] = i*0.025 - 0.03
    _seed_ramp(m.wo, 16, 0.015, -0.08)
    _seed_ramp(m.bo, 4, 0.025, -0.03)
    return m^


def _seed_full_u(mut u: AnyTensor) raises:
    """Seed U [2,4,4] flat with U[i] = i*0.07 - 0.25 (32 elems)."""
    for i in range(32):
        u.store[DType.float64](i, Float64(i) * 0.07 - 0.25)


def test_parity_single_step() raises:
    """Timestep-0 output must match the torch reference (1e-5).

    Reference: parity_refs/mamba_parity_reference.json "y_single_step" [B*D=8].
    """
    print("Running test_parity_single_step...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.02999932872312777)
    ref_vals.append(-0.004999574093920268)
    ref_vals.append(0.020000180535287237)
    ref_vals.append(0.04499993516449475)
    ref_vals.append(-0.029874635247803344)
    ref_vals.append(-0.0048534268972167515)
    ref_vals.append(0.02016778145336984)
    ref_vals.append(0.04518898980395644)

    var m = _make_seeded()
    var u = zeros([2, 4, 4], DType.float64)
    _seed_full_u(u)
    var out = m.forward(u)
    # single step = timestep 0: out[bi, 0, d]. flat index (bi*4 + 0)*4 + d.
    for bi in range(2):
        for d0 in range(4):
            var got = out.load[DType.float64]((bi * 4 + 0) * 4 + d0)
            var want = ref_vals[bi * 4 + d0]
            if _abs_diff(got, want) > 1e-5:
                raise Error(
                    "Mamba single-step parity mismatch at bi="
                    + String(bi)
                    + " d="
                    + String(d0)
                )
    print("  ok single step matches torch reference to 1e-5")
    print("test_parity_single_step PASSED")


def test_parity_sequence() raises:
    """`forward` over 4 steps (state carried) must match the torch reference.

    Reference: parity_refs/mamba_parity_reference.json "y_sequence" [B*SEQ*D=32].
    """
    print("Running test_parity_sequence...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.02999932872312777)
    ref_vals.append(-0.004999574093920268)
    ref_vals.append(0.020000180535287237)
    ref_vals.append(0.04499993516449475)
    ref_vals.append(-0.02998902048855918)
    ref_vals.append(-0.004987624014437814)
    ref_vals.append(0.020013772459683554)
    ref_vals.append(0.04501516893380493)
    ref_vals.append(-0.02995083742263033)
    ref_vals.append(-0.004942994422121985)
    ref_vals.append(0.020064848578386363)
    ref_vals.append(0.045072691578894715)
    ref_vals.append(-0.0298538572620251)
    ref_vals.append(-0.004829482736714146)
    ref_vals.append(0.02019489178859681)
    ref_vals.append(0.04521926631390777)
    ref_vals.append(-0.029874635247803344)
    ref_vals.append(-0.0048534268972167515)
    ref_vals.append(0.02016778145336984)
    ref_vals.append(0.04518898980395644)
    ref_vals.append(-0.02960369922327681)
    ref_vals.append(-0.004536410067356734)
    ref_vals.append(0.020530879088563343)
    ref_vals.append(0.04559816824448342)
    ref_vals.append(-0.02897800149846976)
    ref_vals.append(-0.003802841459557155)
    ref_vals.append(0.02137231857935545)
    ref_vals.append(0.04654747861826806)
    ref_vals.append(-0.028202607480850184)
    ref_vals.append(-0.0028903443117072633)
    ref_vals.append(0.02242191885743566)
    ref_vals.append(0.04773418202657859)

    var m = _make_seeded()
    var u = zeros([2, 4, 4], DType.float64)
    _seed_full_u(u)
    var out = m.forward(u)
    for i in range(32):
        if _abs_diff(out.load[DType.float64](i), ref_vals[i]) > 1e-5:
            raise Error("Mamba sequence parity mismatch at " + String(i))
    print("  ok multi-step sequence matches torch reference to 1e-5")
    print("test_parity_sequence PASSED")


def main() raises:
    """Run all Mamba selective-SSM (S6) block tests."""
    print("=" * 60)
    print("Mamba Selective-SSM (S6) Block Test Suite")
    print("=" * 60)
    test_shape()
    test_reject_bad_sizes()
    test_parameter_count()
    test_parity_single_step()
    test_parity_sequence()
    print("=" * 60)
    print("All Mamba tests PASSED")
    print("=" * 60)
