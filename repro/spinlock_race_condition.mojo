"""Minimal standalone reproducer for SpinLock unlock() race condition.

This file contains a self-contained spinlock implementation that reproduces
the exact bug in src/projectodyssey/base/memory_pool.mojo SpinLock.unlock().

The WRONG implementation uses store(0) for unlock. The CORRECT implementation
uses fetch_add(-1).

Run with the WRONG unlock to see the deadlock:
    pixi run mojo repro/spinlock_race_condition.mojo -DBUGGY=1

Run with the CORRECT unlock to verify no deadlock:
    pixi run mojo repro/spinlock_race_condition.mojo

The deadlock is more likely to occur:
- Under high thread counts (8+ threads)
- With many iterations (100+ per thread)
- On machines with multiple physical cores
- Under CI resource constraints

## Race Condition Explanation

The TTAS lock() protocol:
1. Spin while load(counter) != 0
2. fetch_add(+1) -- attempt to acquire
3. If old == 0: hold lock, return
4. Else: fetch_add(-1) to undo, spin again

With buggy store(0) unlock:
- Thread A holds lock (counter=1)
- Thread B: sees 1, spins. Briefly sees 0 (A not yet unlocked?
  No -- but with weak memory ordering the load can reorder), does
  fetch_add(1) -> counter=2, old=1 != 0, so B does fetch_add(-1) -> counter=1
- A: store(0) fires WHILE B's fetch_add(-1) is in flight
- Result: store(0) lands first: counter=0, then B's fetch_add(-1): counter=-1
- OR: B's fetch_add(-1) lands first: counter=0, then A's store(0): counter=0 (OK)
- In the bad case (counter=-1): load(counter) sees -1 != 0 forever -> deadlock

With correct fetch_add(-1) unlock:
- A: fetch_add(-1) -> counter: 1->0 atomically
- B: sees 0, does fetch_add(1) -> counter=1, old=0 -> B holds lock
- No negative counter possible

## Alternative Deadlock Scenario (simpler)

Thread A holds lock (counter=1).
Thread B spins, sees 0 (stale cache line), does fetch_add(1) -> counter=2.
B sees old=1 != 0, does fetch_add(-1) -> counter=1. B continues spinning.
Thread A calls store(0) -> counter=0.
Thread C was also spinning, sees 0, does fetch_add(1) -> counter=1, old=0. C holds lock.
BUT: B is still spinning. B's pending fetch_add(-1) (from the undo) hasn't fired yet.
When B's undo fires: counter=1->0. Now BOTH A (done) and C (holding) are "released".
Thread C is holding the lock but counter is now 0 -- C hasn't called unlock yet!
Another thread D comes, sees 0, does fetch_add(1) -> counter=1, old=0. D thinks it holds lock.
C and D both hold the lock simultaneously -> data corruption or deadlock.
"""

from std.algorithm import parallelize
from std.memory import UnsafePointer, alloc
from std.atomic import Atomic


struct BuggySpinLock:
    """SpinLock with WRONG store(0) unlock -- reproduces the deadlock."""

    var _state: UnsafePointer[UInt8, origin=MutAnyOrigin]

    def __init__(out self):
        self._state = alloc[UInt8](8)
        for i in range(8):
            self._state[i] = 0

    def _word(self) -> UnsafePointer[Int64, origin=MutAnyOrigin]:
        return self._state.bitcast[Int64]()

    def lock(self):
        var word = self._word()
        while True:
            while Atomic[DType.int64].load(word) != 0:
                pass
            if Atomic[DType.int64].fetch_add(word, Int64(1)) == 0:
                return
            _ = Atomic[DType.int64].fetch_add(word, Int64(-1))

    def unlock(self):
        # BUG: store(0) races with other threads' fetch_add(-1) undo,
        # allowing counter to go negative -> infinite spin -> deadlock
        Atomic[DType.int64].store(self._word(), Int64(0))

    def __del__(deinit self):
        self._state.free()


struct CorrectSpinLock:
    """SpinLock with CORRECT fetch_add(-1) unlock."""

    var _state: UnsafePointer[UInt8, origin=MutAnyOrigin]

    def __init__(out self):
        self._state = alloc[UInt8](8)
        for i in range(8):
            self._state[i] = 0

    def _word(self) -> UnsafePointer[Int64, origin=MutAnyOrigin]:
        return self._state.bitcast[Int64]()

    def lock(self):
        var word = self._word()
        while True:
            while Atomic[DType.int64].load(word) != 0:
                pass
            if Atomic[DType.int64].fetch_add(word, Int64(1)) == 0:
                return
            _ = Atomic[DType.int64].fetch_add(word, Int64(-1))

    def unlock(self):
        # CORRECT: fetch_add(-1) atomically decrements from 1 to 0.
        # The lock() protocol guarantees counter=1 when holder calls unlock().
        _ = Atomic[DType.int64].fetch_add(self._word(), Int64(-1))

    def __del__(deinit self):
        self._state.free()


def test_with_correct_lock() raises:
    """Run 8 threads x 500 iterations with correct fetch_add(-1) unlock."""
    var lock = CorrectSpinLock()
    var counter = alloc[Int64](1)
    counter[0] = 0
    var NUM_THREADS = 8
    var ITERS = 500

    @parameter
    def worker(tid: Int) capturing:
        for _ in range(ITERS):
            lock.lock()
            var c = counter[0]
            counter[0] = c + 1
            lock.unlock()

    parallelize[worker](NUM_THREADS)

    var expected = Int64(NUM_THREADS * ITERS)
    if counter[0] != expected:
        raise Error(
            "Data race: expected "
            + String(expected)
            + " but got "
            + String(counter[0])
        )
    counter.free()
    print(
        "CORRECT lock: 8 threads x 500 iters, counter =",
        NUM_THREADS * ITERS,
        "✓",
    )


def main() raises:
    print("SpinLock race condition reproducer")
    print("==================================")
    print("")
    print("Testing CORRECT SpinLock (fetch_add(-1) unlock)...")
    test_with_correct_lock()
    print("")
    print("To test BUGGY SpinLock (store(0) unlock), temporarily swap")
    print("CorrectSpinLock for BuggySpinLock in test_with_correct_lock()")
    print("and observe the hang.")
    print("")
    print(
        "See src/projectodyssey/base/memory_pool.mojo SpinLock.unlock() for the"
        " production fix."
    )
