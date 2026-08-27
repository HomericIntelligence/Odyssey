---
title: "[BUG] Returning a struct by value runs the moved-from source's `__deinit__`, freeing the destination's buffer (use-after-free)"
labels: [bug, mojo]
---

### Bug description

#### Summary

Returning a struct by value (`return r^`) runs `__deinit__` on the **moved-from source's
fields**, freeing the buffers that the destination now owns, **before the caller reads the
destination** → use-after-free. Simple shapes are unaffected: parameter moves
(`consume(x)`) and `var x = src` correctly skip the source's deinit. Only **return moves**
trigger it.

This **passes on `1.0.0b2` (2cf4d08a)** and **fails on `1.0.0` stable (ed45d567)**.

#### Actual behavior

The moved-from source's field deinit runs after the function returns but before the caller
reads the destination:

```text
$ mojo repro_min.mojo        # Mojo 1.0.0 stable
    deinit freeing ptr       # <- moved-from r's deinit runs (should be skipped)
values: 0 1                  #    reads freed memory; correct only by allocator luck
```

AddressSanitizer makes it unambiguous:

```text
$ mojo build --sanitize address repro_min.mojo -o /tmp/r && /tmp/r
==ERROR: AddressSanitizer: heap-use-after-free on address 0x502000000138
READ of size 8 ... in main
freed by thread T0 here:     # freed by the moved-from source's deinit
```

#### Expected behavior

The moved-from source's `__deinit__` must not run (or must be a no-op) after its
ownership was transferred via `return r^`. The destination should read its own buffer:

```text
$ mojo repro_min.mojo        # Mojo 1.0.0b2
values: 0 1                  # no spurious deinit; ASAN reports no UAF
```

#### Steps to reproduce

Save as `repro_min.mojo` (also at `docs/dev/reproducers/repro_min.mojo` in Odyssey):

```mojo
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
```

Run:

```bash
mojo repro_min.mojo                              # stable: spurious deinit print
mojo build --sanitize address repro_min.mojo -o /tmp/r && /tmp/r   # stable: heap-use-after-free
```

A two-field variant (`docs/dev/reproducers/repro_uaf_return_move.mojo`) mirrors
`GradientPair`-style returns and shows both buffers freed.

#### Impact

Any library that returns buffer-owning structs by value (e.g. a refcounted tensor type
with a custom `deinit move` constructor) gets silent memory corruption: the shared
refcount cell is decremented once by the moved-from source and once by the destination,
so the buffer frees while still referenced. In the Odyssey ML framework this manifests
as wrong tensor values, `List._realloc` crashes, and flaky failures across the test suite
— all passing on 1.0.0b2.

#### Environment

- Mojo version: 1.0.0 (ed45d567) — also reproduced with 1.0.0b2 (2cf4d08a) for comparison
- OS: Linux x86_64 (Ubuntu 24.04 container), glibc 2.39
- Installed via `pip install mojo==1.0.0` / `mojo==1.0.0b2` (PyPI)

#### Related issues

- #6187 (heap corruption, closed COMPLETED) — different signature: crash at alloc during
  heavy churn, not move semantics
- #6707 (stale read through `UnsafePointer[MutUntrackedOrigin]`, closed NOT_PLANNED) —
  different mechanism: origin-tracking limitation, no move involved
- #6475 (bitcast read in struct method, closed COMPLETED) — different mechanism, no moves involved
