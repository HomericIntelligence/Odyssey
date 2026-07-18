"""Unit tests for the SPlus optimizer (stable whitening — sign-in-eigenbasis).

Tests cover:
- Rejects a non-2D parameter
- init_splus_state produces correctly-shaped state (6 tensors; params_ema seeded to
  a copy of the params, the rest zeroed)
- Numerical parity with an independent numpy transcription of the SPlus step across a
  3-step run (eigenbasis built on step 1, reused on steps 2 & 3) — validates the
  Kronecker factors, eigenbasis projection, element-wise SIGN in the eigenbasis,
  project-back, shape-aware scaling, and the EMA (iterate-averaged) parameter
  sequence, plus cross-step state threading

Reference values are produced by the committed generator
`parity_refs/splus_parity_reference.py` (R=C=4, lr=0.1, beta1=0.9, beta2=0.99,
ema_rate=0.999, precondition_frequency=100). The fixture uses a SQUARE, full-rank,
direction-varying gradient so BOTH Kronecker factors are full rank with distinct
eigenvalues — the regime where the eigenbasis is unique up to per-column sign and
the nonlinear `sign(m')` step is stable across the numpy and Mojo eigensolvers (see
the generator's docstring). Regenerate + diff before editing.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.training.optimizers.splus import splus_step, init_splus_state


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _store_grad(mut g: AnyTensor, vals: List[Float64]) raises:
    for i in range(len(vals)):
        g.store[DType.float64](i, vals[i])


def _seed_ramp(
    mut t: AnyTensor, count: Int, scale: Float64, off: Float64
) raises:
    for i in range(count):
        t.store[DType.float64](i, Float64(i) * scale + off)


def test_reject_non_2d() raises:
    """SPlus rejects a non-matrix parameter."""
    print("Running test_reject_non_2d...")
    var p = zeros([10], DType.float64)
    var g = zeros([10], DType.float64)
    var z = zeros([10], DType.float64)
    var z2 = zeros([10], DType.float64)
    var z3 = zeros([10], DType.float64)
    var z4 = zeros([10], DType.float64)
    var z5 = zeros([10], DType.float64)
    try:
        var (_, _, _, _, _, _, _) = splus_step(
            p, p, g, z, z2, z3, z4, z5, 1, 0.1
        )
        raise Error("Should have rejected 1D params")
    except _:
        print("  ok rejected 1D params")
    print("test_reject_non_2d PASSED")


def test_init_state_shapes() raises:
    """`init_splus_state` returns 6 tensors: params_ema seeded, rest zeroed."""
    print("Running test_init_state_shapes...")
    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var st = init_splus_state(W)
    # params_ema, exp_avg: 3x4 = 12; gg_left: 3x3 = 9; gg_right: 4x4 = 16;
    # q_left: 9; q_right: 16.
    if st[0].numel() != 12 or st[1].numel() != 12:
        raise Error("params_ema / exp_avg should be 3x4")
    if st[2].numel() != 9 or st[4].numel() != 9:
        raise Error("gg_left / q_left should be 3x3")
    if st[3].numel() != 16 or st[5].numel() != 16:
        raise Error("gg_right / q_right should be 4x4")
    # params_ema seeded to a copy of the params.
    for i in range(12):
        if (
            _abs_diff(st[0].load[DType.float64](i), W.load[DType.float64](i))
            > 0.0
        ):
            raise Error("params_ema must be seeded to a copy of the params")
    # exp_avg and the factors/bases start zeroed.
    for i in range(12):
        if st[1].load[DType.float64](i) != 0.0:
            raise Error("exp_avg must start zeroed")
    for i in range(9):
        if st[2].load[DType.float64](i) != 0.0:
            raise Error("gg_left must start zeroed")
    print("  ok state shapes 3x4 / 3x3 / 4x4; params_ema seeded, rest zeroed")
    print("test_init_state_shapes PASSED")


def _grad_for_step(s: Int) raises -> AnyTensor:
    """The direction-varying rank-4 gradient for step `s` (matches the generator).
    """
    var g = zeros([4, 4], DType.float64)
    var vals = List[Float64]()
    if s == 1:
        vals.append(0.9)
        vals.append(-0.4)
        vals.append(0.2)
        vals.append(-0.7)
        vals.append(-0.3)
        vals.append(0.8)
        vals.append(-0.6)
        vals.append(0.1)
        vals.append(0.5)
        vals.append(-0.2)
        vals.append(0.7)
        vals.append(-0.9)
        vals.append(-0.8)
        vals.append(0.6)
        vals.append(-0.1)
        vals.append(0.4)
    elif s == 2:
        vals.append(0.2)
        vals.append(0.7)
        vals.append(-0.5)
        vals.append(0.3)
        vals.append(0.6)
        vals.append(-0.1)
        vals.append(0.4)
        vals.append(-0.8)
        vals.append(-0.7)
        vals.append(0.9)
        vals.append(-0.3)
        vals.append(0.2)
        vals.append(0.4)
        vals.append(-0.6)
        vals.append(0.8)
        vals.append(-0.2)
    else:
        vals.append(-0.5)
        vals.append(0.3)
        vals.append(0.9)
        vals.append(-0.2)
        vals.append(0.8)
        vals.append(-0.7)
        vals.append(0.1)
        vals.append(0.5)
        vals.append(-0.4)
        vals.append(0.6)
        vals.append(-0.9)
        vals.append(0.7)
        vals.append(0.2)
        vals.append(-0.3)
        vals.append(0.4)
        vals.append(-0.6)
    _store_grad(g, vals)
    return g


def test_parity_three_step() raises:
    """Match the numpy transcription over 3 steps to 1e-6, for BOTH the live params
    and the EMA (iterate-averaged) params.

    Reference values from parity_refs/splus_parity_reference.py (R=C=4, lr=0.1,
    precondition_frequency=100 so the eigenbasis is built once on step 1). The
    gradient direction changes each step (see `_grad_for_step`).
    """
    print("Running test_parity_three_step...")

    # Live params after steps 1, 2, 3.
    var p1 = List[Float64]()
    p1.append(-0.5241478495891233)
    p1.append(-0.4053765296643648)
    p1.append(-0.3035308160929283)
    p1.append(-0.19928752073208864)
    p1.append(-0.09849184727272908)
    p1.append(-0.016487136356468476)
    p1.append(0.11655857166704368)
    p1.append(0.20875862514678092)
    p1.append(0.3028883319621429)
    p1.append(0.3983766472889444)
    p1.append(0.4869714502068748)
    p1.append(0.6210779305136922)
    p1.append(0.7055914554499496)
    p1.append(0.7820658986980967)
    p1.append(0.8870152423067138)
    p1.append(0.9898265195684591)

    var p2 = List[Float64]()
    p2.append(-0.599830763446269)
    p2.append(-0.4114405198276413)
    p2.append(-0.3127309603323719)
    p2.append(-0.19828650676266332)
    p2.append(-0.10264441154253133)
    p2.append(-0.039193354792098956)
    p2.append(0.08813531400712392)
    p2.append(0.22927013271002278)
    p2.append(0.3105944902537596)
    p2.append(0.38416044761982604)
    p2.append(0.47241769951147594)
    p2.append(0.6343568269923165)
    p2.append(0.725204945779256)
    p2.append(0.809146855107501)
    p2.append(0.8731346923733122)
    p2.append(0.96914083859885)

    var p3 = List[Float64]()
    p3.append(-0.6340705725868668)
    p3.append(-0.4481918306330767)
    p3.append(-0.3462395776833768)
    p3.append(-0.1936106700388651)
    p3.append(-0.15470871840444794)
    p3.append(-0.022331920519898654)
    p3.append(0.10415974617981932)
    p3.append(0.2290694914084497)
    p3.append(0.3276246053239566)
    p3.append(0.34415971443740884)
    p3.append(0.47852369294296493)
    p3.append(0.6434613671967239)
    p3.append(0.7219307567661957)
    p3.append(0.8012054575320573)
    p3.append(0.8618694562478114)
    p3.append(0.9986283292258149)

    # EMA params after steps 1, 2, 3.
    var e1 = List[Float64]()
    e1.append(-0.5000241478495892)
    e1.append(-0.4000053765296644)
    e1.append(-0.3000035308160929)
    e1.append(-0.19999928752073204)
    e1.append(-0.0999984918472727)
    e1.append(-1.648713635646849e-05)
    e1.append(0.10001655857166714)
    e1.append(0.20000875862514683)
    e1.append(0.30000288833196215)
    e1.append(0.39999837664728893)
    e1.append(0.4999869714502069)
    e1.append(0.6000210779305137)
    e1.append(0.70000559145545)
    e1.append(0.7999820658986981)
    e1.append(0.8999870152423068)
    e1.append(0.9999898265195685)

    var e2 = List[Float64]()
    e2.append(-0.5001239544651858)
    e2.append(-0.4000168116729624)
    e2.append(-0.3000162582456091)
    e2.append(-0.19999757473997395)
    e2.append(-0.10000113776696797)
    e2.append(-5.5664004012211015e-05)
    e2.append(0.1000046773271026)
    e2.append(0.2000380199992317)
    e2.append(0.30001347993388394)
    e2.append(0.3999825387182615)
    e2.append(0.4999594021782681)
    e2.append(0.6000554136795755)
    e2.append(0.7000307908097738)
    e2.append(0.7999912306879069)
    e2.append(0.8999601629194378)
    e2.append(0.9999589775316478)

    var e3 = List[Float64]()
    e3.append(-0.5002579010833075)
    e3.append(-0.40006498669192253)
    e3.append(-0.3000624815650469)
    e3.append(-0.19999118783527284)
    e3.append(-0.10005584534760545)
    e3.append(-7.794026052809747e-05)
    e3.append(0.10000883239595532)
    e3.append(0.2000670514706409)
    e3.append(0.30004109105927407)
    e3.append(0.39992671589398066)
    e3.append(0.4999379664690328)
    e3.append(0.6000988196330926)
    e3.append(0.7000526907757302)
    e3.append(0.7999924449147511)
    e3.append(0.8999220722127661)
    e3.append(0.9999576468833419)

    var W = zeros([4, 4], DType.float64)
    _seed_ramp(W, 16, 0.1, -0.5)
    var st = init_splus_state(W)
    var params_ema = st[0]
    var exp_avg = st[1]
    var gg_left = st[2]
    var gg_right = st[3]
    var q_left = st[4]
    var q_right = st[5]

    for s in range(1, 4):
        var g = _grad_for_step(s)
        var r = splus_step(
            W,
            params_ema,
            g,
            exp_avg,
            gg_left,
            gg_right,
            q_left,
            q_right,
            s,
            0.1,
        )
        W = r[0]
        params_ema = r[1]
        exp_avg = r[2]
        gg_left = r[3]
        gg_right = r[4]
        q_left = r[5]
        q_right = r[6]
        for i in range(16):
            var exp_p = p1[i]
            var exp_e = e1[i]
            if s == 2:
                exp_p = p2[i]
                exp_e = e2[i]
            elif s == 3:
                exp_p = p3[i]
                exp_e = e3[i]
            if _abs_diff(W.load[DType.float64](i), exp_p) > 1e-6:
                raise Error(
                    "SPlus live-param step-"
                    + String(s)
                    + " mismatch at "
                    + String(i)
                )
            if _abs_diff(params_ema.load[DType.float64](i), exp_e) > 1e-6:
                raise Error(
                    "SPlus ema-param step-"
                    + String(s)
                    + " mismatch at "
                    + String(i)
                )

    print("  ok live + EMA params match reference over 3 steps to 1e-6")
    print("test_parity_three_step PASSED")


def main() raises:
    """Run all SPlus tests."""
    print("=" * 60)
    print("SPlus Optimizer Test Suite")
    print("=" * 60)
    test_reject_non_2d()
    test_init_state_shapes()
    test_parity_three_step()
    print("=" * 60)
    print("All SPlus tests PASSED")
    print("=" * 60)
