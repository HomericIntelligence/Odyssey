"""Unit tests for the Prodigy optimizer (arXiv:2306.06101, Algorithm 4).

Tests cover:
- Shape/dtype guards on prodigy_step (params/grad/m/v/s/x0 shape, params/grad
  dtype) — the raises-tests that pin the documented dtype/shape contract
- Numerical parity with the NumPy Algorithm-4 reference on FIVE steps, asserting
  BOTH the params AND the scalar distance estimate d at each step, with a
  gradient sequence chosen so d grows past d0 (1e-6 -> ~3.45e-5)
- The d-estimate is monotone non-decreasing and actually grows past d0
- prodigy_step_simple matches prodigy_step with the default hyperparameters
- growth_rate cap limits per-step growth of d
- descent direction (a positive gradient decreases a positive param)

Reference values are transcribed from parity_refs/prodigy_parity_reference.py
(re-run + diffed before committing). The official `prodigyopt` package is not
installed in the project interpreter, so the reference is a NumPy transcription
of the paper's Algorithm 4 (Adam variant), per the tracking issue's update rule.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full
from odyssey.training.optimizers.prodigy import (
    prodigy_step,
    prodigy_step_simple,
)


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_shape_mismatch() raises:
    """`prodigy_step` rejects a params/gradients shape mismatch."""
    print("Running test_reject_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([5], DType.float32)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    var s = zeros([4], DType.float32)
    var x0 = zeros([4], DType.float32)
    try:
        var _ = prodigy_step(p, g, m, v, s, x0, 0.0, 1e-6)
        raise Error("Should have rejected shape mismatch")
    except _:
        print("  ok rejected shape mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_reject_state_shape_mismatch() raises:
    """`prodigy_step` rejects a params/x0 (state) shape mismatch."""
    print("Running test_reject_state_shape_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float32)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    var s = zeros([4], DType.float32)
    var x0 = zeros([5], DType.float32)  # wrong shape
    try:
        var _ = prodigy_step(p, g, m, v, s, x0, 0.0, 1e-6)
        raise Error("Should have rejected x0 shape mismatch")
    except _:
        print("  ok rejected x0 shape mismatch")
    print("test_reject_state_shape_mismatch PASSED")


def test_reject_dtype_mismatch() raises:
    """`prodigy_step` rejects a params/gradients dtype mismatch.

    Pins the documented contract: params and gradients MUST share a dtype;
    mixed dtypes raise rather than silently mis-computing.
    """
    print("Running test_reject_dtype_mismatch...")
    var p = zeros([4], DType.float32)
    var g = zeros([4], DType.float16)
    var m = zeros([4], DType.float32)
    var v = zeros([4], DType.float32)
    var s = zeros([4], DType.float32)
    var x0 = zeros([4], DType.float32)
    try:
        var _ = prodigy_step(p, g, m, v, s, x0, 0.0, 1e-6)
        raise Error("Should have rejected dtype mismatch")
    except _:
        print("  ok rejected dtype mismatch")
    print("test_reject_dtype_mismatch PASSED")


def _make_state(n: Int) raises -> Tuple[AnyTensor, AnyTensor, AnyTensor]:
    return (
        zeros([n], DType.float64),
        zeros([n], DType.float64),
        zeros([n], DType.float64),
    )


def test_parity_with_reference() raises:
    """Five Prodigy steps must match the Algorithm-4 reference to 1e-9.

    Fixed inputs + reference outputs transcribed from
    parity_refs/prodigy_parity_reference.py (gamma=1.0, betas=(0.9,0.999),
    eps=1e-8, d0=1e-6). Asserts BOTH the params AND the scalar distance estimate
    d at each of the 5 steps. The constant positive gradient on a positive start
    makes <g, x0 - x> grow, driving d up from 1e-6 to ~3.45e-5.

    Step 1: x == x0 so the inner product is 0, r stays 0, and d stays at d0.
    Steps 2..5: d increases monotonically as the params move away from x0,
    exercising the full D-Adaptation numerator/denominator recurrence.
    """
    print("Running test_parity_with_reference...")

    var n = 4
    var x0 = zeros([n], DType.float64)
    x0.store[DType.float64](0, 1.0)
    x0.store[DType.float64](1, 2.0)
    x0.store[DType.float64](2, 3.0)
    x0.store[DType.float64](3, 4.0)

    var g = full([n], 0.5, DType.float64)

    # Reference params after each step (from the NumPy reference).
    var ref_p = List[List[Float64]]()
    var p1 = List[Float64]()
    p1.append(0.9999968377243398)
    p1.append(1.9999968377243398)
    p1.append(2.99999683772434)
    p1.append(3.99999683772434)
    ref_p.append(p1^)
    var p2 = List[Float64]()
    p2.append(0.9999925881345527)
    p2.append(1.9999925881345528)
    p2.append(2.999992588134553)
    p2.append(3.999992588134553)
    ref_p.append(p2^)
    var p3 = List[Float64]()
    p3.append(0.9999848264736261)
    p3.append(1.9999848264736262)
    p3.append(2.9999848264736264)
    p3.append(3.9999848264736264)
    ref_p.append(p3^)
    var p4 = List[Float64]()
    p4.append(0.9999622901162453)
    p4.append(1.9999622901162453)
    p4.append(2.9999622901162457)
    p4.append(3.9999622901162457)
    ref_p.append(p4^)
    var p5 = List[Float64]()
    p5.append(0.999901889528256)
    p5.append(1.999901889528256)
    p5.append(2.9999018895282563)
    p5.append(3.9999018895282563)
    ref_p.append(p5^)

    # Reference distance estimate d after each step.
    var ref_d = List[Float64]()
    ref_d.append(1e-06)
    ref_d.append(1.58153331227678e-06)
    ref_d.append(4.822405024989585e-06)
    ref_d.append(1.3496086407383853e-05)
    ref_d.append(3.4509673326090946e-05)

    # Initial state.
    var state = _make_state(n)
    var m = state[0]
    var v = state[1]
    var s = state[2]
    var r = 0.0
    var d = 1e-6

    var p = zeros([n], DType.float64)
    p.store[DType.float64](0, 1.0)
    p.store[DType.float64](1, 2.0)
    p.store[DType.float64](2, 3.0)
    p.store[DType.float64](3, 4.0)

    for step in range(5):
        var res = prodigy_step(p, g, m, v, s, x0, r, d, 1.0, 0.9, 0.999, 1e-8)
        p = res[0]
        m = res[1]
        v = res[2]
        s = res[3]
        r = res[4]
        d = res[5]
        for i in range(n):
            if _abs_diff(p.load[DType.float64](i), ref_p[step][i]) > 1e-9:
                raise Error(
                    "prodigy param parity mismatch at step "
                    + String(step + 1)
                    + " idx "
                    + String(i)
                )
        if _abs_diff(d, ref_d[step]) > 1e-12:
            raise Error("prodigy d parity mismatch at step " + String(step + 1))
    print("  ok 5 steps match the Algorithm-4 reference (params + d) to 1e-9")
    print("test_parity_with_reference PASSED")


def test_d_grows_and_is_monotone() raises:
    """The distance estimate d must be monotone non-decreasing and grow past d0.
    """
    print("Running test_d_grows_and_is_monotone...")
    var n = 4
    var x0 = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var state = _make_state(n)
    var m = state[0]
    var v = state[1]
    var s = state[2]
    var r = 0.0
    var d = 1e-6
    var p = full([n], 1.0, DType.float64)

    var prev_d = d
    for _ in range(6):
        var res = prodigy_step_simple(p, g, m, v, s, x0, r, d)
        p = res[0]
        m = res[1]
        v = res[2]
        s = res[3]
        r = res[4]
        d = res[5]
        if d < prev_d:
            raise Error("d must be monotone non-decreasing")
        prev_d = d
    if not (d > 1e-6):
        raise Error("d must grow past d0")
    print("  ok d monotone non-decreasing and grew past d0")
    print("test_d_grows_and_is_monotone PASSED")


def test_float32_d_grows() raises:
    """Prodigy works on FLOAT32: the d-estimate must grow past d0.

    Pins the documented float32 support and guards against the scalar-reduction
    dtype bug (reading a float32-backed reduced scalar as float64 froze d at d0).
    """
    print("Running test_float32_d_grows...")
    var n = 4
    var x0 = full([n], 1.0, DType.float32)
    var g = full([n], 0.5, DType.float32)
    var m = zeros([n], DType.float32)
    var v = zeros([n], DType.float32)
    var s = zeros([n], DType.float32)
    var r = 0.0
    var d = 1e-6
    var p = full([n], 1.0, DType.float32)
    var prev_d = d
    for _ in range(6):
        var res = prodigy_step_simple(p, g, m, v, s, x0, r, d)
        p = res[0]
        m = res[1]
        v = res[2]
        s = res[3]
        r = res[4]
        d = res[5]
        if d < prev_d:
            raise Error("float32: d must be monotone non-decreasing")
        prev_d = d
    if not (d > 1e-6):
        raise Error("float32: d must grow past d0 (dtype-read bug guard)")
    print("  ok float32 d grew past d0")
    print("test_float32_d_grows PASSED")


def test_step_simple_matches_defaults() raises:
    """`prodigy_step_simple` must equal `prodigy_step` with default hyperparams.
    """
    print("Running test_step_simple_matches_defaults...")
    var n = 4
    var x0 = full([n], 1.0, DType.float64)
    var g = full([n], 0.3, DType.float64)
    var state = _make_state(n)
    var p = full([n], 1.0, DType.float64)

    var simple = prodigy_step_simple(
        p, g, state[0], state[1], state[2], x0, 0.0, 1e-6
    )
    var full_call = prodigy_step(
        p, g, state[0], state[1], state[2], x0, 0.0, 1e-6, 1.0, 0.9, 0.999, 1e-8
    )
    var sp = simple[0]
    var fp = full_call[0]
    for i in range(n):
        if (
            _abs_diff(sp.load[DType.float64](i), fp.load[DType.float64](i))
            > 1e-12
        ):
            raise Error(
                "prodigy_step_simple != prodigy_step defaults at " + String(i)
            )
    if _abs_diff(simple[5], full_call[5]) > 1e-15:
        raise Error("prodigy_step_simple d != prodigy_step d")
    print("  ok prodigy_step_simple matches prodigy_step defaults")
    print("test_step_simple_matches_defaults PASSED")


def test_growth_rate_cap() raises:
    """`growth_rate` caps per-step growth of d (d' <= growth_rate * d)."""
    print("Running test_growth_rate_cap...")
    var n = 4
    var x0 = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var state = _make_state(n)
    var m = state[0]
    var v = state[1]
    var s = state[2]
    var r = 0.0
    var d = 1e-6
    var p = full([n], 1.0, DType.float64)
    var growth = 1.5

    # Drive several steps so the uncapped d would jump by more than 1.5x;
    # with the cap, each step's d must satisfy d' <= 1.5 * d_prev.
    for _ in range(6):
        var prev_d = d
        var res = prodigy_step(
            p, g, m, v, s, x0, r, d, 1.0, 0.9, 0.999, 1e-8, growth
        )
        p = res[0]
        m = res[1]
        v = res[2]
        s = res[3]
        r = res[4]
        d = res[5]
        if d > growth * prev_d + 1e-18:
            raise Error("growth_rate cap violated: d' > growth_rate * d")
    print("  ok d growth is capped by growth_rate")
    print("test_growth_rate_cap PASSED")


def test_descent_direction() raises:
    """A positive gradient must decrease the parameter."""
    print("Running test_descent_direction...")
    var n = 3
    var x0 = full([n], 1.0, DType.float64)
    var g = full([n], 0.5, DType.float64)
    var state = _make_state(n)
    var res = prodigy_step_simple(
        full([n], 1.0, DType.float64),
        g,
        state[0],
        state[1],
        state[2],
        x0,
        0.0,
        1e-3,
    )
    var np = res[0]
    for i in range(n):
        if not (np.load[DType.float64](i) < 1.0):
            raise Error("positive gradient should decrease param")
    print("  ok positive gradient decreased params")
    print("test_descent_direction PASSED")


def main() raises:
    """Run all Prodigy tests."""
    print("=" * 60)
    print("Prodigy Optimizer Test Suite")
    print("=" * 60)
    test_reject_shape_mismatch()
    test_reject_state_shape_mismatch()
    test_reject_dtype_mismatch()
    test_parity_with_reference()
    test_d_grows_and_is_monotone()
    test_float32_d_grows()
    test_step_simple_matches_defaults()
    test_growth_rate_cap()
    test_descent_direction()
    print("=" * 60)
    print("All Prodigy tests PASSED")
    print("=" * 60)
