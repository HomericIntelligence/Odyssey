"""Probe D: whole-owned-local AnyTensor return — the batchnorm `return output^` shape.

Tests `return output^` where `output` is an owned local AnyTensor (custom
`deinit move` + shared refcount), the exact shape in
src/odyssey/core/layers/batchnorm.mojo:152 and siblings, on 1.0.0 stable.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full


@always_inline
def _opaque(x: Float32) -> Float32:
    return x * Float32(2.0) + Float32(1.0)


def make_whole() raises -> AnyTensor:
    var output = full([64], 1.5, DType.float32)
    return output^


def make_whole_2() raises -> AnyTensor:
    var output = full([4096], 0.25, DType.float32)
    return output^


def check(tag: String, t: AnyTensor, expected: Float32) -> Bool:
    var ok = True
    for i in range(t.numel()):
        var v = t.load[DType.float32](i)
        if abs(v - expected) > Float32(0.01):
            ok = False
    print(tag, "=>", "OK" if ok else "CORRUPT")
    return ok


def main() raises:
    var fail1 = False
    var fail2 = False
    for i in range(10):
        var t1 = make_whole()
        if not check("small ^", t1, 1.5):
            fail1 = True
        var t2 = make_whole_2()
        if not check("large ^", t2, 0.25):
            fail2 = True
    print("small whole-local ^ failures:", "YES" if fail1 else "NO")
    print("large whole-local ^ failures:", "YES" if fail2 else "NO")
