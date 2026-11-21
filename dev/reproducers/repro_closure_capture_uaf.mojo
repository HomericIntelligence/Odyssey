"""Mojo 1.0.0 closure-capture UAF reproducer (FM-G).

Passing a `@parameter` capturing closure as a function-value parameter
(`def(Int) capturing -> None`) fires the captured local's `__deinit__` at
closure-construction time — BEFORE any call through the closure. When the
captured value owns a heap buffer, the buffer is freed before the closure
body reads it → use-after-free (first elements read 0.0 / garbage; the
freed block's head is reused by the closure context).

Passes on 1.0.0b2 (capture stays alive until end of scope).

Run:
    mojo run repro_closure_capture_uaf.mojo

Expected on 1.0.0 stable:
    creating box
        >>> Box.__deinit__ firing (freeing buffer)     <- premature
    calling closure via function-value param:
      param[ 0 ]: 0.0        <- UAF read (should be 0.25)
      param[ 1 ]: 0.0        <- UAF read (should be 0.75)
      param[ 2 ]: 1.25
      ...
"""

from std.memory.alloc import unsafe_alloc


struct Box:
    var data: Pointer[Float32, origin=MutUntrackedOrigin]
    var n: Int

    def __init__(out self, n: Int):
        self.n = n
        self.data = unsafe_alloc[Float32](n)

    def __deinit__(deinit self):
        print("    >>> Box.__deinit__ firing (freeing buffer)")
        if self.n > 0:
            self.data.unsafe_free()

    def fill(self):
        for i in range(self.n):
            self.data[unsafe_offset=i] = Float32(i) * 0.5 + 0.25

    def get(self, i: Int) -> Float32:
        return self.data[unsafe_offset=i]


def call_inline[func: def(Int) capturing -> None](n: Int):
    for i in range(n):
        func(i)


def main():
    print("creating box")
    var b = Box(8)
    b.fill()

    @parameter
    def worker_param(i: Int) capturing:
        print("  param[", i, "]:", b.get(i))

    print("calling closure via function-value param:")
    call_inline[worker_param](8)
    print("done main body")
