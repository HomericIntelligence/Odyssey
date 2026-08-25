"""Control: same probe but read via t.load[i] (no escaped pointer)."""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros


def main() raises:
    var n = 1024
    var shape = List[Int]()
    shape.append(n)
    var t = zeros(shape, DType.float32)
    for i in range(n):
        t._set_float64(i, Float64(i % 7))

    var acc = Float32(0.0)
    for i in range(n):
        var v = t.load[DType.float32](i)
        acc += v * Float32(3.0) + Float32(1.0)

    var expected = Float32(0.0)
    for i in range(n):
        var v = Float32(i % 7)
        expected += v * Float32(3.0) + Float32(1.0)

    print("acc     =", acc)
    print("expected=", expected)
    if acc != expected:
        raise Error("MISMATCH (control)")
    print("control: OK")
