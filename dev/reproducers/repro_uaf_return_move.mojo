"""BUG REPRODUCER (Mojo 1.0.0 regression): return-move runs moved-from deinit.

Environment: passes on 1.0.0b2 (2cf4d08a), fails on 1.0.0 stable (ed45d567).

Two-field version mirroring AnyTensor/GradientPair ownership: a `Pair` of two
buffer-owning structs returned by value. The moved-from source's field deinits
run after the return but before the caller reads the destination, freeing the
buffers the destination owns -> use-after-free (both fields).

ASAN (stable): heap-use-after-free. ASAN (b2): no UAF.

Run:
    mojo repro_uaf_return_move.mojo
    mojo build --sanitize address repro_uaf_return_move.mojo -o /tmp/r && /tmp/r
"""
from std.memory.alloc import *
from std.memory import UnsafePointer


struct Box(ImplicitlyCopyable, Movable):
    var ptr: UnsafePointer[Int, MutUntrackedOrigin]

    def __init__(out self, size: Int):
        self.ptr = alloc[Int](size)
        for i in range(size):
            self.ptr[i] = i

    def __copyinit__(mut self, other: Self):
        self.ptr = other.ptr

    def __init__(out self, *, deinit move: Self):
        self.ptr = move.ptr

    def __deinit__(deinit self):
        print("    deinit freeing ptr")
        self.ptr.free()


struct Pair(ImplicitlyCopyable, Movable):
    var a: Box
    var b: Box

    def __init__(out self):
        self.a = Box(2)
        self.b = Box(2)


def make() raises -> Pair:
    var r = Pair()
    return r^


def main() raises:
    var p = make()
    var v0 = p.a.ptr[0]
    var v1 = p.a.ptr[1]
    var w0 = p.b.ptr[0]
    var w1 = p.b.ptr[1]
    print("values:", v0, v1, w0, w1)
