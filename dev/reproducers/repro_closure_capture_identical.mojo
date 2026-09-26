"""Closure-capture premature-deinit — IDENTICAL file for cross-version proof.

The exact same file compiles and runs unchanged on BOTH mojo 1.0.0b2 and
mojo 1.0.0 stable. The only difference between the versions is WHEN the
captured `Box` is destroyed:

    1.0.0b2  (2cf4d08a): no `__deinit__` print before reads; reads correct
                          (capture kept alive through the closure calls)
    1.0.0    (ed45d567): `>>> Box.__deinit__ firing` prints immediately
                          after the closure is defined/passed — BEFORE any
                          call through the function-value parameter.

With a heap-owning payload (see repro_closure_capture_uaf.mojo) the early
destruction frees the buffer the closure body still reads → use-after-free.

Run:
    mojo run repro_closure_capture_identical.mojo
"""


struct Box:
    var payload: Int

    def __init__(out self, payload: Int):
        self.payload = payload

    def __deinit__(deinit self):
        print("    >>> Box.__deinit__ firing (payload=", self.payload, ")")

    def get(self, i: Int) -> Int:
        return self.payload * 100 + i


def call_inline[func: def(Int) capturing -> None](n: Int):
    for i in range(n):
        func(i)


def main():
    print("creating box")
    var b = Box(7)

    @parameter
    def worker_param(i: Int) capturing:
        print("  param[", i, "]:", b.get(i))

    print("calling closure via function-value param:")
    call_inline[worker_param](4)
    print("done main body")
