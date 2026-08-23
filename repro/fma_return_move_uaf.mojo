"""Minimal reproducer: FM-A Return-move UAF / Memory corruption.
Tensor operations return garbage values on Mojo 1.0.0 stable.
Passes on Mojo 1.0.0b2, fails on 1.0.0 stable.
"""
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import full, ones, zeros
from odyssey.core.arithmetic import add


def main() raises:
    print("=== FM-A: Return-move UAF / Memory Corruption ===")

    var t1 = zeros([4], dtype=DType.float32)
    t1[0] = 5.0
    t1[1] = 3.0
    t1[2] = -2.0
    t1[3] = 7.0
    print("  t1[0] =", t1[0], "  (expected 5.0)")
    assert t1[0] == 5.0, "FAIL: t1[0] != 5.0"

    var t2 = make_tensor()
    print("  t2[0] =", t2[0], "  (expected 1.0)")
    assert t2[0] == 1.0, "FAIL: t2[0] != 1.0"

    var a = zeros([4], dtype=DType.float32)
    var b = zeros([4], dtype=DType.float32)
    a[0] = 3.0
    b[0] = 2.0
    var c = add(a, b)
    print("  c[0] =", c[0], "  (expected 5.0)")
    assert c[0] == 5.0, "FAIL: c[0] != 5.0"

    print("All assertions passed!")


def make_tensor() raises -> AnyTensor:
    var t = zeros([4], dtype=DType.float32)
    t[0] = 1.0
    t[1] = 2.0
    return t^
