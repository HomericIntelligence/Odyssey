"""Forward-mode automatic differentiation: Jacobian-vector products (JVP).

This module provides the canonical forward-mode primitive used by Hessian
estimators (HutchinsonDiagonalHessian, GaussNewtonDiagonalHessian) in
``odyssey.autograd.sophia_g``. A real and imaginary component are tracked
together via the ``Dual`` number convention.

The mathematical identity: if we define ``Dual(x, v) = x + v*eps`` where
``eps^2 == 0``, then any arithmetic expression using the operator overloads
below produces ``(real, imag)`` such that ``real = f(x)`` and
``imag = (df/dx)(x) * v`` \u2014 the canonical forward-mode derivative of
``f`` at ``x`` along the direction ``v``.

This is intentionally a closed-form ``Dual`` surface for Phase 1 (scalar
primitives). It gives a self-contained JVP that lets the higher-order
Hessian estimators prime and verify themselves without requiring
``odyssey.autograd.jvp_only`` to wait for the full tensor-JVP surface
landed under #5721. A later Phase 2 implementation can replace the scalar
arithmetic with tensor-flavored rules (DualTensor) without changing the
``jvp`` / ``jvp_only`` API below.

References:
- Baydin, Pearlmutter, Radul, Siskind (2018) "Automatic differentiation in
  machine learning: a survey", arXiv:1502.05767 \u00a74 (forward mode).
- The API shape mirrors torch.autograd.functional.jvp /
  jvp_only but is constrained to Mojo 1.0.0b2's trait-style function-value
  parameters (with the `thin` keyword per AGENTS.md).
"""

from std.math import sqrt as scalar_sqrt


# ---------------------------------------------------------------------------
# Dual number primitive.
# ---------------------------------------------------------------------------


struct Dual(Copyable, Movable):
    """A scalar Dual number ``real + imag * eps`` with ``eps^2 == 0``.

    Used to propagate forward-mode directional derivatives through any
    arithmetic expression. Real and imaginary parts share dtype (Float64)
    since the Hessian estimators flap between fp32 intermediates only at
    the AnyTensor boundary, not inside the JVP rule.

    Attributes:
        real: The primal value ``x``.
        imag: The tangent coefficient ``v`` such that the Derivative
              along the direction ``v`` is ``imag = (df/dx) * v``.
    """

    var real: Float64
    var imag: Float64

    fn __init__(out self, real: Float64, imag: Float64):
        self.real = real
        self.imag = imag

    fn __init__(out self, real: Float64):
        # Treat a bare scalar as a Dual with zero tangent.
        self.real = real
        self.imag = 0.0

    fn __add__(self, other: Dual) -> Dual:
        # (a + b*e) + (c + d*e) = (a+c) + (b+d)*e.
        return Dual(self.real + other.real, self.imag + other.imag)

    fn __sub__(self, other: Dual) -> Dual:
        # (a + b*e) - (c + d*e) = (a-c) + (b-d)*e.
        return Dual(self.real - other.real, self.imag - other.imag)

    fn __neg__(self) -> Dual:
        return Dual(-self.real, -self.imag)

    fn __mul__(self, other: Dual) -> Dual:
        """Dual multiplication. Discard eps^2 term (infinitesimal of order 2).

        Derivation: (a + b*e)(c + d*e) = ac + (ad + bc)*e + bd*e^2.
        Since eps^2 == 0, the bd term vanishes and the resulting real part
        is ac; the resulting imag part is ad + bc.
        """
        return Dual(
            self.real * other.real,
            self.real * other.imag + self.imag * other.real,
        )

    fn __truediv__(self, other: Dual) -> Dual:
        """Dual division. Multiply by reciprocal and apply dual multiplication.

        For nonzero denominator ``c``:
            (a + b*e) / (c + d*e)
                = (a + b*e) * (1/c + (-d/c^2)*e)
                = a/c + (b/c - (a*d)/c^2)*e.

        Raises on a zero-denominator primal (which would silently produce
        NaN if propagated into a chain\u2014the user should fix the input
        rather than read garbage).
        """
        var c = other.real
        if c == 0.0:
            raise Error("Dual.__truediv__: zero denom primal")
        var c_inv = 1.0 / c
        var c_inv_sq = c_inv * c_inv
        var d = other.imag
        return Dual(
            self.real * c_inv,
            self.imag * c_inv - self.real * d * c_inv_sq,
        )


# ---------------------------------------------------------------------------
# Helper: sqrt with dual chain rule (used by Sophia-G benchmarks).
# ---------------------------------------------------------------------------


fn sqrt_d(x: Dual) -> Dual:
    """Square root of a Dual. Chain rule: d(sqrt(x))/dx = 1/(2*sqrt(x)).

    Same eps^2 contraction as the multiplication rule. Raises on a
    non-positive primal so callers must validate input; this surfaces the
    error at the JVP layer rather than silently producing NaN imag.
    """
    if x.real < 0.0:
        raise Error("sqrt_d: negative primal")
    var root_real = scalar_sqrt(x.real)
    var coeff = 1.0 / (2.0 * root_real) if root_real != 0.0 else 0.0
    return Dual(root_real, x.imag * coeff)


# ---------------------------------------------------------------------------
# jvp / jvp_only \u2014 canonical forward-mode derivative APIs.
# ---------------------------------------------------------------------------


def jvp(
    f: def(Dual) raises -> Dual thin,
    primal: Float64,
    tangent: Float64,
) raises -> Tuple[Float64, Float64]:
    """Compute ``f(p)`` and ``d f(p)/dp * v`` simultaneously.

    Semantically identical to ``torch.autograd.functional.jvp``: feed the
    Dual ``(primal, tangent)`` into ``f``, then split the output Dual into
    its real and imaginary parts.

    Args:
        f: A scalar-aware function ``f: Dual -> Dual`` whose arithmetic
           closure uses the ``Dual`` operator overloads above. The ``thin``
           keyword is required in Mojo 1.0.0b2 for function-value parameter
           types (per AGENTS.md).
        primal: The point ``p`` at which to evaluate ``f``.
        tangent: The direction ``v`` along which the directional derivative
                 is computed.

    Returns:
        ``(f(p), d f(p)/dp * v)`` as a 2-tuple of Float64.

    Raises:
        Error: Propagated from ``f`` (e.g. domain violations or tape
        errors). The Dual-arithmetic overloads in this file also raise on
        bad inputs.
    """
    var out = f(Dual(primal, tangent))
    return (out.real, out.imag)


def jvp_only(
    f: def(Dual) raises -> Dual thin,
    primal: Float64,
    tangent: Float64,
) raises -> Float64:
    """Forward-mode derivative only: ``d f(p)/dp * v``.

    Sugar over ``jvp(...)[1]``. Useful when the caller has already computed
    ``f(p)`` and needs only the derivative (e.g. HutchinsonDiagonalHessian
    needs ``v^T H v`` requires ``jvp_only(loss_layer, p, v)`` only).

    Args:
        f: A scalar-aware function `f: Dual -> Dual`.
        primal: The point ``p`` at which to evaluate.
        tangent: The direction ``v``.

    Returns:
        ``d f(p)/dp * v`` as a Float64.
    """
    var out = f(Dual(primal, tangent))
    return out.imag
