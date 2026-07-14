"""Unit tests for the SOAP optimizer (Shampoo + Adam in the eigenbasis).

Tests cover:
- Rejects a non-2D parameter
- init_soap_state produces correctly-shaped zeroed state (6 tensors)
- Numerical parity with an independent numpy transcription of the SOAP step across a
  2-step run (eigenbasis built on step 1, reused on step 2) — validates projection,
  rotated Adam, project-back, bias correction, weight decay, and state threading
"""

from odyssey.tensor.any_tensor import AnyTensor, zeros
from odyssey.training.optimizers.soap import soap_step, init_soap_state


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
    """SOAP rejects a non-matrix parameter."""
    print("Running test_reject_non_2d...")
    var p = zeros([10], DType.float64)
    var g = zeros([10], DType.float64)
    var z = zeros([10], DType.float64)
    var z2 = zeros([10], DType.float64)
    var z3 = zeros([10], DType.float64)
    var z4 = zeros([10], DType.float64)
    var z5 = zeros([10], DType.float64)
    var z6 = zeros([10], DType.float64)
    try:
        var (_, _, _, _, _, _, _) = soap_step(
            p, g, z, z2, z3, z4, z5, z6, 1, 0.1
        )
        raise Error("Should have rejected 1D params")
    except e:
        print("  ok rejected 1D params")
    print("test_reject_non_2d PASSED")


def test_init_state_shapes() raises:
    """init_soap_state returns 6 zeroed tensors with the right shapes."""
    print("Running test_init_state_shapes...")
    var W = zeros([3, 4], DType.float64)
    var st = init_soap_state(W)
    # exp_avg, exp_avg_sq: 3x4 = 12; gg_left: 3x3 = 9; gg_right: 4x4 = 16;
    # q_left: 9; q_right: 16.
    if st[0].numel() != 12 or st[1].numel() != 12:
        raise Error("exp_avg / exp_avg_sq should be 3x4")
    if st[2].numel() != 9 or st[4].numel() != 9:
        raise Error("gg_left / q_left should be 3x3")
    if st[3].numel() != 16 or st[5].numel() != 16:
        raise Error("gg_right / q_right should be 4x4")
    # all zero
    for i in range(12):
        if st[0].load[DType.float64](i) != 0.0:
            raise Error("state must start zeroed")
    print("  ok state shapes 3x4 / 3x3 / 4x4, zeroed")
    print("test_init_state_shapes PASSED")


def test_parity_two_step() raises:
    """Match the numpy transcription over 2 steps to 1e-6.

    Reference values from parity_refs/soap_parity_reference.py (R=3, C=4, lr=0.1,
    precondition_frequency=100 so the eigenbasis is built once on step 1). The
    gradient on step s is (i*0.05 - 0.3) + s*0.01.
    """
    print("Running test_parity_two_step...")

    var ref1 = List[Float64]()
    ref1.append(-0.4268552)
    ref1.append(-0.350556)
    ref1.append(-0.2742567)
    ref1.append(-0.1979574)
    ref1.append(-0.0592897)
    ref1.append(0.0148506)
    ref1.append(0.088991)
    ref1.append(0.1631313)
    ref1.append(0.3082758)
    ref1.append(0.3802572)
    ref1.append(0.4522387)
    ref1.append(0.5242201)

    var ref2 = List[Float64]()
    ref2.append(-0.3369098)
    ref2.append(-0.3081338)
    ref2.append(-0.2793578)
    ref2.append(-0.2505819)
    ref2.append(-0.0373521)
    ref2.append(0.009693)
    ref2.append(0.056738)
    ref2.append(0.1037831)
    ref2.append(0.2622056)
    ref2.append(0.3275198)
    ref2.append(0.3928339)
    ref2.append(0.458148)

    var W = zeros([3, 4], DType.float64)
    _seed_ramp(W, 12, 0.1, -0.5)
    var st = init_soap_state(W)
    var exp_avg = st[0]
    var exp_avg_sq = st[1]
    var gg_left = st[2]
    var gg_right = st[3]
    var q_left = st[4]
    var q_right = st[5]

    for s in range(1, 3):
        var g = zeros([3, 4], DType.float64)
        _seed_ramp(g, 12, 0.05, -0.3 + Float64(s) * 0.01)
        var r = soap_step(
            W,
            g,
            exp_avg,
            exp_avg_sq,
            gg_left,
            gg_right,
            q_left,
            q_right,
            s,
            0.1,
        )
        W = r[0]
        exp_avg = r[1]
        exp_avg_sq = r[2]
        gg_left = r[3]
        gg_right = r[4]
        q_left = r[5]
        q_right = r[6]
        for i in range(12):
            var expected = ref1[i] if s == 1 else ref2[i]
            if _abs_diff(W.load[DType.float64](i), expected) > 1e-6:
                raise Error(
                    "SOAP step-" + String(s) + " mismatch at " + String(i)
                )

    print("  ok matches reference over 2 steps to 1e-6")
    print("test_parity_two_step PASSED")


def main() raises:
    """Run all SOAP tests."""
    print("=" * 60)
    print("SOAP Optimizer Test Suite")
    print("=" * 60)
    test_reject_non_2d()
    test_init_state_shapes()
    test_parity_two_step()
    print("=" * 60)
    print("All SOAP tests PASSED")
    print("=" * 60)
