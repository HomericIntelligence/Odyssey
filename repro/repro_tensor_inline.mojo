"""Self-contained repro: escaped raw pointer triggers premature __deinit__ UAF.

Mirrors the shape of AnyTensor.data_ptr() usage (owned local tensor,
raw pointer captured into a local, reads through it in a loop). Uses
only the stdlib. On Mojo 1.0.0 stable the compiler ends the struct's
lifetime right after data_ptr() -- the last syntactic use -- so the
refcount drops to 0 and the buffer is freed while the loop still reads
it. Same root cause as modular/modular#6959 / #6707.
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc


struct MiniTensor:
    """Minimal refcounted tensor with an escaping raw data_ptr()."""

    var _data: Pointer[Float32, MutUntrackedOrigin]
    var _refcount: Pointer[Int, MutUntrackedOrigin]
    var _n: Int

    def __init__(out self, n: Int):
        self._n = n
        self._data = unsafe_alloc[Float32](n)
        self._refcount = unsafe_alloc[Int](1)
        self._refcount[] = 1

    def __init__(out self, *, copy: Self):
        self._data = copy._data
        self._refcount = copy._refcount
        self._refcount[] += 1
        self._n = copy._n

    def __init__(out self, *, deinit move: Self):
        self._data = move._data
        self._refcount = move._refcount
        self._n = move._n

    def data_ptr(self) -> Pointer[Float32, MutUntrackedOrigin]:
        """Escape: returns an untracked raw pointer into our storage."""
        return self._data

    def set(self, i: Int, v: Float32):
        self._data[unsafe_offset=i] = v

    def load(self, i: Int) -> Float32:
        return self._data[unsafe_offset=i]

    def __deinit__(deinit self):
        self._refcount[] -= 1
        if self._refcount[] == 0:
            self._data.unsafe_free()
            self._refcount.unsafe_free()


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
        acc += _opaque(ptr[unsafe_offset=i])

    var expected = Float32(0.0)
    for i in range(n):
        var v = Float32(i % 7)
        expected += v * Float32(3.0) + Float32(1.0)

    print("acc     =", acc)
    print("expected=", expected)
    if acc != expected:
        raise Error("MISMATCH: escaped data_ptr read freed memory")
    print("OK")
