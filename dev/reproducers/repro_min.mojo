"""MINIMAL BUG REPRODUCER (Mojo 1.0.0 regression): return-move runs moved-from deinit.

Environment: passes on 1.0.0b2 (2cf4d08a), fails on 1.0.0 stable (ed45d567).

A struct owning a buffer, with a `deinit move` constructor, returned by value
(`return r^`): the moved-from source's `__deinit__` runs after the return but
before the caller reads the destination, freeing the buffer the destination
owns -> use-after-free. Values print correctly only by allocator luck.

ASAN (stable): heap-use-after-free. ASAN (b2): no UAF.

Run:
    mojo repro_min.mojo
    mojo build --sanitize address repro_min.mojo -o /tmp/r && /tmp/r
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


def make(size: Int) -> Box:
    var r = Box(size)
    return r^


def main():
    var b = make(2)
    var v0 = b.ptr[0]
    var v1 = b.ptr[1]
    print("values:", v0, v1)
