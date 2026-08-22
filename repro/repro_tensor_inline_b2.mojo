"""Reproducer: escaped raw pointer / premature __deinit__ UAF — b2 mirror.

Byte-for-byte mirror of the logic in the 1.0.0 repro
(repro_tensor_inline.mojo), written with the b2-era stdlib APIs
(UnsafePointer/alloc/free/__del__) so it compiles on 1.0.0b2.

Run with:
    /tmp/mojob2/.venv/bin/mojo run repro_tensor_inline_b2.mojo

Observed (3/3 runs): PASS — the loop reads correct values on 1.0.0b2
(2cf4d08a); the identical shape fails on 1.0.0 stable.
"""

from std.memory import UnsafePointer, alloc


struct MiniTensor:
    """Minimal refcounted tensor with an escaping raw data_ptr()."""

    var _data: UnsafePointer[Float32, MutAnyOrigin]
    var _refcount: UnsafePointer[Int, MutAnyOrigin]
    var _n: Int

    def __init__(out self, n: Int):
        self._n = n
        self._data = alloc[Float32](n)
        self._refcount = alloc[Int](1)
        self._refcount[] = 1

    def data_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Escape: returns an untracked raw pointer into our storage."""
        return self._data

    def set(self, i: Int, v: Float32):
        self._data[i] = v

    def load(self, i: Int) -> Float32:
        return self._data[i]

    def __del__(deinit self):
        self._refcount[] -= 1
        if self._refcount[] == 0:
            self._data.free()
            self._refcount.free()


@always_inline
def _opaque(x: Float32) -> Float32:
    """Opaque op the compiler cannot fold away."""
    return x * Float32(3.0) + Float32(1.0)


def main() raises:
    var n = 1024
    var t = MiniTensor(n)
    for i in range(n):
        t.set(i, Float32(i % 7))

    # The escape: last syntactic use of `t` is this call.
    var ptr = t.data_ptr()

    # Loop reads exclusively through the escaped pointer.
    var acc = Float32(0.0)
    for i in range(n):
        acc += _opaque(ptr[i])

    var expected = Float32(0.0)
    for i in range(n):
        var v = Float32(i % 7)
        expected += v * Float32(3.0) + Float32(1.0)

    print("acc     =", acc)
    print("expected=", expected)
    if acc != expected:
        raise Error("MISMATCH: escaped data_ptr read freed memory")
    print("OK")
