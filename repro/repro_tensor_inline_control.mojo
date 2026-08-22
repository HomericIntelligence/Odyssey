"""Control for repro_tensor_inline.mojo: same struct, but the loop reads
via load() (no escaped pointer). Should always pass on 1.0.0 stable,
proving the write path and math are correct.
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc


struct MiniTensor:
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

    def set(self, i: Int, v: Float32):
        self._data[unsafe_offset=i] = v

    def load(self, i: Int) -> Float32:
        return self._data[unsafe_offset=i]

    def __deinit__(deinit self):
        self._refcount[] -= 1
        if self._refcount[] == 0:
            self._data.unsafe_free()
            self._refcount.unsafe_free()


def main() raises:
    var n = 1024
    var t = MiniTensor(n)
    for i in range(n):
        t.set(i, Float32(i % 7))

    var acc = Float32(0.0)
    for i in range(n):
        var v = t.load(i)
        acc += v * Float32(3.0) + Float32(1.0)

    var expected = Float32(0.0)
    for i in range(n):
        var v = Float32(i % 7)
        expected += v * Float32(3.0) + Float32(1.0)

    print("acc     =", acc)
    print("expected=", expected)
    if acc != expected:
        raise Error("MISMATCH (control)")
    print("OK")
