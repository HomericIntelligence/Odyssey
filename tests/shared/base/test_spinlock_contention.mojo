"""Tests for SpinLock correctness under simulated contention (RC-2).

The original broken lock() algorithm:

    while ptr[].fetch_add(1) != 0:
        _ = ptr[].fetch_sub(1)

can drive the counter negative (or to zero when a lock holder is present),
creating a window where two logical threads can both believe they hold the
lock.  The race:

  1. Thread A holds lock; counter = 1.
  2. Thread B: fetch_add(1) → counter = 2, returns 1 (≠ 0, so B spins).
  3. Thread A: unlock, fetch_sub(1) → counter = 1.
  4. Thread C: fetch_add(1) → counter = 2, returns 1 (≠ 0, so C spins).
  5. Thread B: fetch_sub(1) → counter = 1.
  6. Thread C: fetch_sub(1) → counter = 0.
  7. Thread D: fetch_add(1) → counter = 1, returns 0 → D holds lock.
     Meanwhile no one else is "supposed" to be in, but the counter is
     now exactly 1 from D only — that part is fine.

The REAL hazard is that the subtraction in the spin path can make the
counter reach 0 while a holder is still in the critical section.  A new
contender then sees 0 and enters simultaneously with the real holder.

This file provides single-threaded behavioural tests that validate:
 a) fetch_add-based try-acquire is a strict binary transition: if we
    call fetch_add and it returns 0, the counter is exactly 1 afterwards.
 b) unlock (fetch_sub) restores the counter to 0 from exactly 1.
 c) The corrected lock() / unlock() cycle works end-to-end.

True multi-threaded contention is not testable in Mojo 0.26.1 without
`parallelize`; these tests verify the single-threaded invariants that the
fix preserves.

"""

from std.testing import assert_true, assert_equal
from shared.base.memory_pool import SpinLock
from std.os.atomic import Atomic
from std.memory import UnsafePointer, alloc


def test_lock_sets_counter_to_one() raises:
    """After lock(), the backing counter must be exactly 1 (locked state)."""
    var lk = SpinLock()
    lk.lock()
    # Peek at the raw counter value through the internal atomic pointer.
    var val = Int(lk._as_atomic()[].load())
    assert_equal(val, 1, "counter must be 1 after lock()")
    lk.unlock()
    print("PASS: test_lock_sets_counter_to_one")


def test_unlock_resets_counter_to_zero() raises:
    """After unlock(), the backing counter must return to 0."""
    var lk = SpinLock()
    lk.lock()
    lk.unlock()
    var val = Int(lk._as_atomic()[].load())
    assert_equal(val, 0, "counter must be 0 after unlock()")
    print("PASS: test_unlock_resets_counter_to_zero")


def test_double_lock_unlock_cycle() raises:
    """Two consecutive lock/unlock cycles must leave the counter at 0."""
    var lk = SpinLock()
    lk.lock()
    lk.unlock()
    lk.lock()
    lk.unlock()
    var val = Int(lk._as_atomic()[].load())
    assert_equal(val, 0, "counter must be 0 after two lock/unlock cycles")
    print("PASS: test_double_lock_unlock_cycle")


def test_failed_fetch_add_does_not_corrupt_counter() raises:
    """Simulates a contender backing off: counter must not go negative.

    This reproduces the core of the broken algorithm.  With the old code:
      - lock() holder has counter = 1.
      - Contender A: fetch_add(1) → 2 (non-zero → tries to back off).
      - Contender A: fetch_sub(1) → 1.
    Counter is still 1.  So the backing-off path is numerically safe as long
    as we add and then subtract exactly once per failed attempt.  The bug
    manifests when unlock() interleaves with the contender's subtract, making
    the counter reach 0 while the original holder has not yet finished.  We
    cannot reproduce the interleaved timing in a single thread, but we CAN
    verify that after a sequence of (add, sub) pairs representing N
    contenders all backing off, the counter stays at its pre-attempt value.
    """
    var lk = SpinLock()
    lk.lock()
    # Simulate 3 contenders each doing add-then-sub (backing off)
    var ptr = lk._as_atomic()
    for _ in range(3):
        _ = ptr[].fetch_add(1)  # contender increments
        _ = ptr[].fetch_sub(1)  # contender backs off
    # Counter must still be 1 (lock still held by original owner)
    var val = Int(ptr[].load())
    assert_equal(
        val, 1, "counter must stay 1 while lock is held despite contenders"
    )
    lk.unlock()
    var final = Int(ptr[].load())
    assert_equal(final, 0, "counter must be 0 after unlock")
    print("PASS: test_failed_fetch_add_does_not_corrupt_counter")


def main() raises:
    test_lock_sets_counter_to_one()
    test_unlock_resets_counter_to_zero()
    test_double_lock_unlock_cycle()
    test_failed_fetch_add_does_not_corrupt_counter()
    print("All test_spinlock_contention tests passed!")
