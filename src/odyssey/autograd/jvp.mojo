"""Forward-mode automatic differentiation: Jacobian-vector products (JVP).

This module provides the canonical forward-mode primitive used by the Hessian
estimators in ``odyssey.autograd.sophia_g`` (HutchinsonDiagonalHessian,
GaussNewtonDiagonalHessian). A real and imaginary (tangent) component are
tracked together via the ``Dual`` number convention, defined here as a
two-lane SIMD vector rather than a user-defined struct.

Mathematical identity: if we define ``Dual(x, v) = x + v*eps`` where
``eps^2 == 0``, then any arithmetic expression using the helper functions
below produces ``(real, imag)`` such that ``real = f(x)`` and
``imag = (df/dx)(x) * v`` -- the canonical forward-mode derivative of
``f`` at ``x`` along the direction ``v``.

Why ``alias Dual = SIMD[DType.float64, 2]`` (and not a struct):

    The user-defined-struct approach (``struct Dual(Copyable, Movable): ...``)
    hit a persistent Mojo 1.0b2 trait-default dispatch error
    (``missing required keyword-only argument: 'move'`` at
    ``std/builtin/value.mojo:50:1``) across six patch iterations of the JVP
    primitive. The failure happens when the trait-default unified copy/move
    initializer is synthesized for a user struct that flows through a
    parametric function-value signature.

    SIMD is a compiler intrinsic: Copyable/Movable are baked into the ABI,
    there is no user-defined trait implementation to synthesize, and the
    parametric function-value dispatch table agrees with the standard
    SIMD ABI. The structural problem disappears and the override path
    stays clean.

    The cost is that SIMD's overloaded ``*`` and ``/`` are component-wise,
    which is WRONG for dual multiply (``a*b -> (ac, ad+bc)``, not
    ``(ac, bd)``). We therefore expose explicit ``dual_mul`` /
    ``dual_div`` free functions. Addition, subtraction, and negation are
    component-wise SIMD-native and need no helper.

    Note: ``jvp`` returns a ``Dual`` (SIMD) rather than a
    ``Tuple[Float64, Float64]`` because the latter is itself a struct
    and would re-trigger the same ``move=`` trait-default synthesis path
    on return. ``Dual`` lanes ``[0]`` and ``[1]`` give the same primal /
    tangent split without the trait-default detour.

References:
- Baydin, Pearlmutter, Radul, Siskind (2018) "Automatic differentiation in
  machine learning: a survey", arXiv:1502.05767 §4 (forward mode).
- API shape mirrors ``torch.autograd.functional.jvp`` / ``jvp_only``.
"""

from std.math import sqrt as scalar_sqrt


# ---------------------------------------------------------------------------
# Dual number primitive: SIMD alias (lane 0 = real, lane 1 = tangent/imag).
# ---------------------------------------------------------------------------


alias Dual = SIMD[DType.float64, 2]


# ---------------------------------------------------------------------------
# Non-component-wise Dual arithmetic. Addition, subtraction, and negation
# delegate to native SIMD operators (component-wise, which is correct for
# dual numbers). Multiplication and division must do the chain-rule contract;
# SIMD's overloaded `*` and `/` are component-wise and would lose the bd
# cross term, so we expose them as free functions.
# ---------------------------------------------------------------------------


@always_inline
def dual_mul(a: Dual, b: Dual) -> Dual:
    """Dual multiplication. Discard eps^2 term (infinitesimal of order 2).

    Derivation: (a + b*e)(c + d*e) = ac + (ad + bc)*e + bd*e^2.
    Since eps^2 == 0, the bd term vanishes; the resulting real part is
    ac and the resulting tangent is ad + bc. SIMD's component-wise ``*``
    would give (ac, bd) -- the WRONG tangent -- so this helper exists.
    """
    return Dual(a[0] * b[0], a[0] * b[1] + a[1] * b[0])


@always_inline
def dual_div(a: Dual, b: Dual) raises -> Dual:
    """Dual division. Multiply by reciprocal and apply dual multiplication.

    For nonzero denominator ``c``:
        (a + b*e) / (c + d*e)
            = (a + b*e) * (1/c + (-d/c^2)*e)
            = a/c + (b/c - (a*d)/c^2)*e.

    Raises on a zero-denominator primal so callers do not silently
    propagate NaN tangents downstream.
    """
    var c = b[0]
    if c == 0.0:
        raise Error("dual_div: zero denom primal")
    var c_inv = 1.0 / c
    var c_inv_sq = c_inv * c_inv
    return Dual(a[0] * c_inv, a[1] * c_inv - a[0] * b[1] * c_inv_sq)


# ---------------------------------------------------------------------------
# Helper: sqrt with dual chain rule (used by Sophia-G benchmarks).
# ---------------------------------------------------------------------------


def sqrt_d(x: Dual) raises -> Dual:
    """Square root of a Dual. Chain rule: d(sqrt(x))/dx = 1/(2*sqrt(x)).

    Raises on a non-positive primal so callers must validate input;
    the error is surfaced at the JVP layer rather than silently
    producing a NaN tangent.
    """
    if x[0] < 0.0:
        raise Error("sqrt_d: negative primal")
    var root_real = scalar_sqrt(x[0])
    var coeff = 1.0 / (2.0 * root_real) if root_real != 0.0 else 0.0
    return Dual(root_real, x[1] * coeff)


# ---------------------------------------------------------------------------
# jvp / jvp_only -- canonical forward-mode derivative APIs.
# ---------------------------------------------------------------------------


def jvp[
    f: def(Dual) raises -> Dual
](primal: Float64, tangent: Float64) raises -> Dual:
    """Compute ``f(p)`` and ``d f(p)/dp * v`` simultaneously.

    The result is a ``Dual`` with lane ``[0]`` = ``f(p)`` and lane
    ``[1]`` = ``d f(p)/dp * v``. We deliberately avoid the
    ``Tuple[Float64, Float64]`` return shape because that tuple's
    struct traits re-trigger the Mojo 1.0b2 ``move=`` trait-default
    synthesis path on return, which would cascade into the same
    dispatch error that motivated the SIMD-alias pivot.

    The function-value parameter ``f`` is compile-time-bound via the
    parametric ``[f: ...]`` syntax (a Mojo 1.0 convention for
    function-value parameters). Docstring-side, ``f`` is intentionally
    omitted from any ``Args:`` block because Mojo 1.0b2's docstring
    parser does not recognize parametric function-value parameters
    in that section (it counts ``f`` as an unnamed index-0 argument
    and complains about subsequent ``primal:`` / ``tangent:`` index
    mismatches).

    Compute direction ``v`` is the value of ``tangent``. The function
    is evaluated at the point ``primal``. Both are Float64 scalars.
    """
    var out = f(Dual(primal, tangent))
    return out


def jvp_only[
    f: def(Dual) raises -> Dual
](primal: Float64, tangent: Float64) raises -> Float64:
    """Forward-mode derivative only: ``d f(p)/dp * v``.

    Sugar over the imaginary lane of ``jvp``. Useful when the caller
    has already computed ``f(p)`` and only needs the directional
    derivative (e.g. HutchinsonDiagonalHessian needs ``v^T H v``).

    Same docstring-side convention as ``jvp``: the parametric
    function-value parameter ``f`` is omitted from any ``Args:``
    block because Mojo 1.0b2's parser does not recognize it there.

    Compute point ``p`` is ``primal``; compute direction ``v`` is
    ``tangent``. Both are Float64 scalars.
    """
    var out = f(Dual(primal, tangent))
    return out[1]
