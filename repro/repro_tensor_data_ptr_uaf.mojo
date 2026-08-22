"""Probe: owned-local AnyTensor + data_ptr escape on Mojo 1.0.0 stable.

Mimics the shape in src/odyssey/training/evaluation.mojo (and the model
e2e tests): an owned local tensor, data_ptr captured into a local, then a
loop reads through the pointer with no further syntactic use of the tensor.

If the compiler ends the tensor's lifetime right after `data_ptr()` (the
premature __deinit__ UAF from modular/modular#6959/#6707), the refcount
decrement fires early, freeing the buffer, and the loop reads garbage.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros


@always_inline
def _opaque(x: Float32) -> Float32:
    """Opaque op the compiler cannot fold away."""
    return x * Float32(3.0) + Float32(1.0)


def main() raises:
    var n = 1024
    var shape = List[Int]()
    shape.append(n)
    var t = zeros(shape, DType.float32)
    for i in range(n):
        t._set_float64(i, Float64(i % 7))

    # The escape: last syntactic use of `t` is this call.
    var ptr = t.data_ptr[DType.float32]()

    # Loop reads exclusively through the escaped pointer.
    var acc = Float32(0.0)
    for i in range(n):
        acc += _opaque(ptr[unsafe_offset=i])

    # Expected: sum over i%7 of (3*(i%7)+1) for i in [0,1024).
    var expected = Float32(0.0)
    for i in range(n):
        var v = Float32(i % 7)
        expected += v * Float32(3.0) + Float32(1.0)

    print("acc     =", acc)
    print("expected=", expected)
    if acc != expected:
        raise Error("MISMATCH: escaped data_ptr read freed memory")
    print("tensor data_ptr escape: OK")
