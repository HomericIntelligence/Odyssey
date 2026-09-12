"""Probe C: real AnyTensor GradientPair return shape — explicit ^ vs implicit.

Uses the actual odyssey AnyTensor + GradientPair types to test the exact
shape in arithmetic.mojo (`return GradientPair(grad_a^, grad_b^)`) vs the
implicit variant (`return GradientPair(grad_a, grad_b)`), on 1.0.0 stable.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full
from odyssey.core.gradient_types import GradientPair


def make_explicit() raises -> GradientPair:
    var ga = full([4], 2.0, DType.float32)
    var gb = full([4], 3.0, DType.float32)
    return GradientPair(ga^, gb^)


def make_implicit() raises -> GradientPair:
    var ga = full([4], 2.0, DType.float32)
    var gb = full([4], 3.0, DType.float32)
    return GradientPair(ga, gb)


def check(tag: String, p: GradientPair) -> Bool:
    var ok = True
    for i in range(4):
        var va = p.grad_a.load[DType.float32](i)
        var vb = p.grad_b.load[DType.float32](i)
        if va != 2.0 or vb != 3.0:
            ok = False
    print(tag, "=>", "OK" if ok else "CORRUPT")
    return ok


def main() raises:
    var exp_fail = False
    var imp_fail = False
    for i in range(10):
        var pe = make_explicit()
        if not check("explicit ^", pe):
            exp_fail = True
        var pi = make_implicit()
        if not check("implicit  ", pi):
            imp_fail = True
    print("explicit ^ failures:", "YES" if exp_fail else "NO")
    print("implicit   failures:", "YES" if imp_fail else "NO")
