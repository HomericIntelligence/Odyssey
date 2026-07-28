"""Tests for ``odyssey.autograd.jvp`` \u2014 Phase-1 forward-mode primitive.

The math invariants under test (each is a closed-form identity derivable
from the Dual-number chain rule, per Baydin et al. 2018 \u00a74):

- ``jvp(identity,   x, v)  == (x,   v)``  \u2014 trivial chain-rule passthrough.
- ``jvp(square,     x, v)  == (x*x, 2*x*v)``  \u2014 power rule.
- ``jvp(double,     x, v)  == (2*x, 2*v)``  \u2014 linear-scaling chain rule.
- ``jvp(add_const,  x, v)  == (x+1, v)``  \u2014 constant-cancellation.
- ``jvp(compose,    x, v)  == ((x+1)^2, 2*(x+1)*v)``  \u2014 composition.
- Dual arithmetic: ``(+), (-), (*), (/)`` propagate the chain rule via
  the operator overloads.
- ``sqrt_d`` chain rule: d(sqrt(x))/dx = 1/(2*sqrt(x)).
- Division-by-zero raises (no silent NaN propagation).
- `jvp_only` matches the imag component of `jvp(...)`.

The tests are organized as standalone ``def test_*()`` functions dispatched
from ``main()``, matching the project's Mojo 1.0 convention (``mojo run
<file>.mojo``, no ``mojo test`` subcommand).
"""

from odyssey.autograd.jvp import jvp, jvp_only, Dual, sqrt_d


def _abs_diff_f64(a: Float64, b: Float64) -> Float64:
    """``|a - b|``. Mirrors the canonical helper used by sibling tests
    (test_eigen.mojo, test_muon.mojo, test_shampoo.mojo etc.)."""
    var d = a - b
    if d < 0.0:
        d = -d
    return d


def test_jvp_identity() raises:
    """``jvp(identity, x, v)`` is the trivial passthrough trace."""
    print("Running test_jvp_identity...")

    def identity(x: Dual) raises -> Dual:
        return x

    var (p, t) = jvp[identity](3.0, 5.0)
    if _abs_diff_f64(p, 3.0) > 1e-12:
        raise Error(
            "jvp identity real mismatch: got " + String(p) + ", want 3.0"
        )
    if _abs_diff_f64(t, 5.0) > 1e-12:
        raise Error(
            "jvp identity imag mismatch: got " + String(t) + ", want 5.0"
        )
    print("  ok jvp identity = (3.0, 5.0)")


def test_jvp_square() raises:
    """``jvp(x^2, x, v)`` derives the power rule ``2x * v``."""
    print("Running test_jvp_square...")

    def square(x: Dual) raises -> Dual:
        return x * x

    # at x = 3.0, v = 1.0: real = 9.0, imag = 2*3*1 = 6.0
    var (p, t) = jvp[square](3.0, 1.0)
    if _abs_diff_f64(p, 9.0) > 1e-12:
        raise Error("square real: got " + String(p) + ", want 9.0")
    if _abs_diff_f64(t, 6.0) > 1e-12:
        raise Error("square imag (power rule): got " + String(t) + ", want 6.0")
    # at x = 2.5, v = 0.4: real = 6.25, imag = 2*2.5*0.4 = 2.0
    var (p2, t2) = jvp[square](2.5, 0.4)
    if _abs_diff_f64(p2, 6.25) > 1e-12:
        raise Error("square real@(2.5): got " + String(p2) + ", want 6.25")
    if _abs_diff_f64(t2, 2.0) > 1e-12:
        raise Error("square imag@(2.5,0.4): got " + String(t2) + ", want 2.0")
    print("  ok jvp square = (9.0, 6.0) and (6.25, 2.0)")


def test_jvp_linear() raises:
    """``jvp(2*x, x, v)`` produces ``(2x, 2v)`` \u2014 linear scaling chain rule.
    """
    print("Running test_jvp_linear...")

    def doubl(x: Dual) raises -> Dual:
        return Dual(2.0, 0.0) * x

    var (p, t) = jvp[doubl](5.0, 1.5)
    if _abs_diff_f64(p, 10.0) > 1e-12:
        raise Error("double real: got " + String(p) + ", want 10.0")
    if _abs_diff_f64(t, 3.0) > 1e-12:
        raise Error("double imag: got " + String(t) + ", want 3.0")
    print("  ok jvp double = (10.0, 3.0)")


def test_jvp_constant_offset() raises:
    """``jvp(x + 1, x, v)`` produces ``(x+1, v)``: constants have no imag."""
    print("Running test_jvp_constant_offset...")

    def add_one(x: Dual) raises -> Dual:
        return x + Dual(1.0, 0.0)

    var (p, t) = jvp[add_one](4.0, 0.7)
    if _abs_diff_f64(p, 5.0) > 1e-12:
        raise Error("constant add real: got " + String(p) + ", want 5.0")
    if _abs_diff_f64(t, 0.7) > 1e-12:
        raise Error("constant add imag: got " + String(t) + ", want 0.7")
    print("  ok jvp (x+1) = (5.0, 0.7)")


def test_jvp_composition() raises:
    """``jvp((x+1)^2, x, v)`` derives chain of ``x -> y=x+1, z=y^2``:

    dz/dx = dz/dy * dy/dx = 2*(x+1) * 1 = 2*(x+1). So at x=3.0, v=1.0:
    real = 16, imag = 8.0.
    """
    print("Running test_jvp_composition...")

    def composed(x: Dual) raises -> Dual:
        var y = x + Dual(1.0, 0.0)
        return y * y

    var (p, t) = jvp[composed](3.0, 1.0)
    if _abs_diff_f64(p, 16.0) > 1e-12:
        raise Error("compose real: got " + String(p) + ", want 16.0")
    if _abs_diff_f64(t, 8.0) > 1e-12:
        raise Error("compose imag: got " + String(t) + ", want 8.0")
    print("  ok jvp ((x+1)^2) = (16.0, 8.0)")


def test_dual_arithmetic() raises:
    """Direct Dual arithmetic primitives (no function-value indirection)."""
    print("Running test_dual_arithmetic...")
    var a = Dual(2.0, 1.0)
    var b = Dual(3.0, 4.0)

    # add: (2+3, 1+4) = (5, 5)
    var s = a + b
    if _abs_diff_f64(s.real, 5.0) > 1e-12 or _abs_diff_f64(s.imag, 5.0) > 1e-12:
        raise Error(
            "add mismatch: got (" + String(s.real) + ", " + String(s.imag) + ")"
        )

    # sub: (2-3, 1-4) = (-1, -3)
    var d = a - b
    if (
        _abs_diff_f64(d.real, -1.0) > 1e-12
        or _abs_diff_f64(d.imag, -3.0) > 1e-12
    ):
        raise Error("sub mismatch")

    # mul: real=6, imag=2*4 + 1*3 = 11
    var m = a * b
    if (
        _abs_diff_f64(m.real, 6.0) > 1e-12
        or _abs_diff_f64(m.imag, 11.0) > 1e-12
    ):
        raise Error(
            "mul mismatch: got (" + String(m.real) + ", " + String(m.imag) + ")"
        )

    # div: real=2/3, imag=(1/3 - 2*4/9) = (3/9 - 8/9) = -5/9
    var q = a / b
    var expected_real = 2.0 / 3.0
    var expected_imag = (1.0 / 3.0) - (2.0 * 4.0) / (9.0)
    if _abs_diff_f64(q.real, expected_real) > 1e-12:
        raise Error(
            "div real: got "
            + String(q.real)
            + ", want "
            + String(expected_real)
        )
    if _abs_diff_f64(q.imag, expected_imag) > 1e-12:
        raise Error(
            "div imag: got "
            + String(q.imag)
            + ", want "
            + String(expected_imag)
        )

    # neg
    var n = -a
    if (
        _abs_diff_f64(n.real, -2.0) > 1e-12
        or _abs_diff_f64(n.imag, -1.0) > 1e-12
    ):
        raise Error("neg mismatch")

    print("  ok dual +,-,*,/,- working correctly")


def test_dual_division_by_zero_raises() raises:
    """``Dual / Dual`` with zero primal raises (no silent NaN)."""
    print("Running test_dual_division_by_zero_raises...")
    var a = Dual(1.0, 50.0)
    var z = Dual(0.0, 7.0)
    var raised = False
    try:
        var _ = a / z
    except e:
        raised = True
        if "zero denom" not in String(e):
            raise Error("unexpected error message: " + String(e))
    if not raised:
        raise Error("Dual/0 should have raised")
    print("  ok dual/0 raises with 'zero denom'")


def test_sqrt_d_chain_rule() raises:
    """``sqrt_d`` chain rule: d(sqrt(x))/dx = 1/(2*sqrt(x))."""
    print("Running test_sqrt_d_chain_rule...")
    var x = Dual(4.0, 1.0)
    var y = sqrt_d(x)
    # real = 2.0, imag = 1.0 * 0.5 / 2.0 = 0.25
    if _abs_diff_f64(y.real, 2.0) > 1e-12:
        raise Error("sqrt(4).real: got " + String(y.real) + ", want 2.0")
    if _abs_diff_f64(y.imag, 0.25) > 1e-12:
        raise Error("sqrt_d imag: got " + String(y.imag) + ", want 0.25")
    print("  ok sqrt_d(x=4, v=1) = (2.0, 0.25)")


def test_sqrt_d_negative_raises() raises:
    """``sqrt_d(<0)`` raises \u2014 domain violation surfaced, not silent."""
    print("Running test_sqrt_d_negative_raises...")
    var x = Dual(-1.0, 1.0)
    var raised = False
    try:
        var _ = sqrt_d(x)
    except e:
        raised = True
    if not raised:
        raise Error("sqrt_d of negative should raise")
    print("  ok sqrt_d(-1) raises")


def test_jvp_only_matches_imag() raises:
    """``jvp_only(f, p, v)`` equals the imag component of ``jvp(f, p, v)``."""
    print("Running test_jvp_only_matches_imag...")

    def linear_combo(x: Dual) raises -> Dual:
        # y = 3*x^2 + 2*x
        return (x * x) * Dual(3.0, 0.0) + Dual(2.0, 0.0) * x

    var (p_full, t_full) = jvp[linear_combo](4.0, 1.0)
    var t_only = jvp_only[linear_combo](4.0, 1.0)
    # dy/dx = 6x + 2; at x=4: dy/dx = 26
    var expected_dydx = 6.0 * 4.0 + 2.0
    if _abs_diff_f64(t_only, expected_dydx) > 1e-12:
        raise Error(
            "jvp_only mismatch: got "
            + String(t_only)
            + ", want "
            + String(expected_dydx)
        )
    if _abs_diff_f64(t_only, t_full) > 1e-12:
        raise Error("jvp_only != jvp(...).imag")
    print("  ok jvp_only = jvp(...).imag = 26.0")


def main() raises:
    """Run all jvp test entry points."""
    print("=" * 60)
    print("odyssey.autograd.jvp \u2014 Phase-1 forward-mode test suite")
    print("=" * 60)
    test_jvp_identity()
    test_jvp_square()
    test_jvp_linear()
    test_jvp_constant_offset()
    test_jvp_composition()
    test_dual_arithmetic()
    test_dual_division_by_zero_raises()
    test_sqrt_d_chain_rule()
    test_sqrt_d_negative_raises()
    test_jvp_only_matches_imag()
    print("")
    print("=" * 60)
    print("ALL jvp TESTS PASSED")
    print("=" * 60)
