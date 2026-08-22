"""Reproducer for modular/modular#6959 — Mojo 1.0.0b2-flavored variant.

Byte-for-byte mirror of the logic in the 1.0.0 repro (repro_6959_inline.mojo),
written with the b2-era stdlib APIs (UnsafePointer/alloc/bitcast/free) so it
compiles on 1.0.0b2.  The 1.0.0-style repro cannot compile on b2 because the
Pointer/Atomic APIs changed between b2 and stable.

Run with:
    mojo run repro_6959_inline_b2.mojo

Observed (5/5 runs): PASS — counter reaches 0 on Mojo 1.0.0b2 (2cf4d08a).
"""

from std.atomic import Atomic
from std.memory import UnsafePointer, alloc


struct SpinLock(Copyable, Movable):
    """Test-and-set spinlock over a heap-allocated Atomic[DType.int64].

    Identical logic to Odyssey's SpinLock at the pre-migration commit
    (src/odyssey/base/memory_pool.mojo @ febfcd23).
    """

    var _state: UnsafePointer[UInt8, MutAnyOrigin]
    """Heap-allocated 8-byte region reinterpreted as Atomic[DType.int64]."""

    def __init__(out self):
        self._state = alloc[UInt8](8)
        for i in range(8):
            self._state[i] = 0

    def _as_atomic(
        self,
    ) -> UnsafePointer[Atomic[DType.int64], MutAnyOrigin]:
        return self._state.bitcast[Atomic[DType.int64]]()

    def _lock_word(self) -> UnsafePointer[Int64, MutAnyOrigin]:
        return self._state.bitcast[Int64]()

    def lock(self):
        var word = self._lock_word()
        while True:
            while Atomic[DType.int64].load(word) != 0:
                pass
            if Atomic[DType.int64].fetch_add(word, Int64(1)) == 0:
                return
            _ = Atomic[DType.int64].fetch_add(word, Int64(-1))

    def unlock(self):
        _ = Atomic[DType.int64].fetch_add(self._lock_word(), Int64(-1))

    def __del__(deinit self):
        self._state.free()


def main() raises:
    print("=== Repro #6959: inlined SpinLock (b2-flavored) ===")

    var lk = SpinLock()
    lk.lock()

    var ptr = lk._as_atomic()

    # Simulate contender: fetch_add(1) increments counter to 2
    _ = ptr[].fetch_add(Int64(1))

    # Lock holder unlocks via fetch_add(-1): counter goes 2->1
    _ = ptr[].fetch_add(Int64(-1))
    var after_unlock = Int(ptr[].load())
    print("  after unlock: counter =", after_unlock, "(expected 1)")

    # Contender backs off: fetch_add(-1) -> counter should return to 0
    _ = ptr[].fetch_add(Int64(-1))
    var final = Int(ptr[].load())
    print("  after backoff: counter =", final, "(expected 0)")

    if after_unlock == 1 and final == 0:
        print("All assertions passed!")
    else:
        print("FAILED: counters wrong")
