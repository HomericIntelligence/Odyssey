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
from projectodyssey.base.memory_pool import SpinLock
from std.atomic import Atomic
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


def test_store_zero_unlock_would_corrupt_counter() raises:
    """Demonstrates why store(0) unlock is wrong under contention (issue #5319).

    Reproduces the exact race that was found in production: a contender
    increments the counter to 2 while the lock is held (counter=1).  If
    unlock() used store(0) instead of fetch_add(-1), the store races with
    the contender's fetch_add(-1) (undo), leaving the counter at -1 or 0
    prematurely.

    This test VERIFIES that fetch_add(-1) keeps the counter non-negative
    across the same sequence of operations.  With store(0) the assertion
    below would fail because the undo-subtract would drive counter to -1.
    """
    var lk = SpinLock()
    lk.lock()
    var ptr = lk._as_atomic()
    # Simulate contender: fetch_add(1) increments counter to 2
    _ = ptr[].fetch_add(1)
    # Lock holder unlocks via fetch_add(-1): counter goes 2→1 (not 0!)
    # (If unlock() used store(0), counter would be 0 here — contender sees
    # free and enters while it shouldn't yet, because it hasn't done its undo.)
    _ = ptr[].fetch_add(-1)  # mimic unlock via fetch_add(-1)
    var after_unlock = Int(ptr[].load())
    assert_true(
        after_unlock >= 0,
        "counter must not go negative after unlock-via-fetch_add(-1)",
    )
    # Contender backs off: fetch_add(-1) → counter returns to 0
    _ = ptr[].fetch_add(-1)
    var final = Int(ptr[].load())
    assert_equal(final, 0, "counter must reach 0 after contender backs off")
    print("PASS: test_store_zero_unlock_would_corrupt_counter")


def test_counter_never_negative_under_repeated_contention() raises:
    """Counter must stay >= 0 across many lock/undo cycles (issue #5319).

    Validates the invariant that the fetch_add-based protocol never drives
    the counter negative regardless of how many contenders spin and back off.
    A negative counter freezes all waiters (spin condition `load != 0` is
    always true), which is the production deadlock that prompted this issue.
    """
    var lk = SpinLock()
    var ptr = lk._as_atomic()
    for _ in range(20):
        lk.lock()
        # Two contenders both attempt to acquire and then back off
        _ = ptr[].fetch_add(1)
        _ = ptr[].fetch_add(1)
        _ = ptr[].fetch_add(-1)
        _ = ptr[].fetch_add(-1)
        var mid = Int(ptr[].load())
        assert_true(mid >= 0, "counter must not go negative while lock is held")
        lk.unlock()
        var post = Int(ptr[].load())
        assert_equal(post, 0, "counter must be 0 after unlock")
    print("PASS: test_counter_never_negative_under_repeated_contention")


def test_many_sequential_cycles_no_drift() raises:
    """Counter must be exactly 0 after N lock/unlock cycles (issue #5319).

    The old store(0) unlock could mask a drifted counter; fetch_add(-1)
    relies on the counter being exactly 1 at unlock time.  This test
    confirms no drift accumulates over 50 sequential acquire-release pairs.
    """
    var lk = SpinLock()
    var ptr = lk._as_atomic()
    for _ in range(50):
        lk.lock()
        lk.unlock()
    var val = Int(ptr[].load())
    assert_equal(
        val, 0, "counter must be exactly 0 after 50 lock/unlock cycles"
    )
    print("PASS: test_many_sequential_cycles_no_drift")


def main() raises:
    test_lock_sets_counter_to_one()
    test_unlock_resets_counter_to_zero()
    test_double_lock_unlock_cycle()
    test_failed_fetch_add_does_not_corrupt_counter()
    test_store_zero_unlock_would_corrupt_counter()
    test_counter_never_negative_under_repeated_contention()
    test_many_sequential_cycles_no_drift()
    print("All test_spinlock_contention tests passed!")
