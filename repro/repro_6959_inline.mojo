"""Standalone reproducer for modular/modular#6959 (Atomic regression).

Inlines the SpinLock struct from the Odyssey ML framework
(src/odyssey/base/memory_pool.mojo) so this file has zero external
dependencies — no imports from the Odyssey codebase.

Run with:
    mojo run repro_6959_inline.mojo

Observed (5/5 runs each):
    Mojo 1.0.0 stable (ed45d567): FAIL  — counter reaches -1
    Mojo 1.0.0b2    (2cf4d08a): PASS  — counter reaches 0
        (b2 needs the b2-flavored variant: repro_6959_inline_b2.mojo,
         because the Pointer/Atomic stdlib APIs changed between b2 and stable)

Root cause (see LLVM IR): the compiler hoists the struct's __deinit__
(unsafe_free of the heap storage) to immediately after lock(), while the
raw pointer returned by _as_atomic() is still live — a use-after-free.
The atomic values observed are just whatever the freed chunk happens to
hold.  Same bug class as modular/modular#6939 (premature lifetime end).
"""

from std.atomic import Atomic
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal


struct SpinLock(Copyable, Movable):
    """Test-and-set spinlock over a heap-allocated Atomic[DType.int64].

    Byte-for-byte identical to Odyssey's SpinLock
    (src/odyssey/base/memory_pool.mojo).  Value 0 = unlocked, 1 = locked.
    """

    var _state: Pointer[UInt8, MutUntrackedOrigin]
    """Heap-allocated 8-byte region reinterpreted as Atomic[DType.int64]."""

    def __init__(out self):
        """Initialize an unlocked spinlock."""
        self._state = unsafe_alloc[UInt8](8)
        for i in range(8):
            self._state[unsafe_offset=i] = 0

    def _as_atomic(
        self,
    ) -> Pointer[Atomic[DType.int64], MutUntrackedOrigin]:
        """Reinterpret backing store as an atomic int64."""
        return self._state.unsafe_bitcast[Atomic[DType.int64]]()

    def _lock_word(self) -> Pointer[Int64, MutUntrackedOrigin]:
        """Return the lock word as a plain Int64 pointer for static Atomic ops."""
        return self._state.unsafe_bitcast[Int64]()

    def lock(self):
        """Acquire the lock via the static Atomic API (fetch_add on Int64 word)."""
        var word = self._lock_word()
        while True:
            # Wait until lock looks free before attempting (reduces bus traffic)
            while Atomic[DType.int64].load(word) != 0:
                pass
            # Attempt to acquire: fetch_add returns the old value.
            if Atomic[DType.int64].fetch_add(word, Int64(1)) == 0:
                return
            # Another thread won the race; undo our increment atomically.
            _ = Atomic[DType.int64].fetch_add(word, Int64(-1))

    def unlock(self):
        """Release the lock via the static Atomic API."""
        _ = Atomic[DType.int64].fetch_add(self._lock_word(), Int64(-1))

    def __deinit__(deinit self):
        """Free the backing store."""
        self._state.unsafe_free()


def main() raises:
    print("=== Repro #6959: inlined SpinLock counter regression ===")

    var lk = SpinLock()
    lk.lock()

    var ptr = lk._as_atomic()

    # Simulate contender: instance-method fetch_add(1) increments counter to 2
    _ = ptr[].fetch_add(1)

    # Lock holder unlocks via instance-method fetch_add(-1): counter goes 2->1
    _ = ptr[].fetch_add(-1)
    var after_unlock = Int(ptr[].load())
    print("  after unlock: counter =", after_unlock, "(expected 1)")
    assert after_unlock >= 0, "counter must not go negative"

    # Contender backs off: fetch_add(-1) -> counter should return to 0
    _ = ptr[].fetch_add(-1)
    var final = Int(ptr[].load())
    print("  after backoff: counter =", final, "(expected 0)")
    assert_equal(final, 0, "counter must reach 0 after contender backs off")

    print("All assertions passed!")
