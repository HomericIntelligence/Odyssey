"""Reproducer test for SpinLock double-free bug (RC-1).

Before the fix, appending 5 SpinLocks to a List forces List reallocation
(capacity 0 → 1 → 2 → 4 → 8). Each reallocation shallow-copies SpinLocks
(pointer value only) and then destroys the old copies, calling __del__ →
_state.free(). The new copies still hold those freed pointers, causing a
double-free crash when the new copies are later destroyed.

This test:
1. Creates List[SpinLock]() with default capacity
2. Appends 5 SpinLocks (forces multiple reallocations)
3. Calls lock()/unlock() on locks[0]
4. If the bug is present this crashes; if fixed it passes.

"""

from std.testing import assert_true
from shared.base.memory_pool import SpinLock


def test_spinlock_list_no_double_free() raises:
    """SpinLock List reallocation must not produce double-free.

    Appending 5 items to a List forces capacity growth:
    0 -> 1 -> 2 -> 4 -> 8 elements requiring 3 reallocations.
    Each reallocation copies existing SpinLock structs.  Without explicit
    __moveinit__ the synthesised shallow copy duplicates the _state pointer
    value. When the old storage is destroyed, the same pointer is freed by
    every stale copy – a double-free that corrupts the heap and typically
    causes a crash or assertion failure at the next allocation.
    """
    var locks = List[SpinLock]()
    # Force reallocations: 5 appends from empty list triggers 0->1->2->4->8
    for _ in range(5):
        locks.append(SpinLock())

    # If we reach here without crashing, list-growth copies are safe.
    # Now exercise the locks to verify they are functional.
    locks[0].lock()
    locks[0].unlock()
    locks[4].lock()
    locks[4].unlock()
    print("PASS: test_spinlock_list_no_double_free")


def test_spinlock_basic_lock_unlock() raises:
    """SpinLock acquired once must be releasable."""
    var lk = SpinLock()
    lk.lock()
    lk.unlock()
    # A second lock/unlock cycle verifies the lock is in a clean state.
    lk.lock()
    lk.unlock()
    print("PASS: test_spinlock_basic_lock_unlock")


def main() raises:
    test_spinlock_list_no_double_free()
    test_spinlock_basic_lock_unlock()
    print("All test_spinlock_double_free tests passed!")
