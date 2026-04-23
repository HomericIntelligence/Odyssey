"""Minimal reproducer for SpinLock deadlock in TensorMemoryPool.

test_memory_pool_threadsafe.mojo hangs (15-min timeout) in CI because
SpinLock.unlock() uses store(0) instead of fetch_add(-1).

Root cause: When thread A holds the lock (counter=1) and thread B has done
fetch_add(1) then fetch_add(-1) (undo), if A's store(0) races with B's
fetch_add(-1), the counter becomes -1. The spin condition `load != 0`
never sees 0 when counter is -1, causing infinite spin.

Fix: unlock() must use fetch_add(-1) to correctly release exactly one
increment, maintaining counter invariant (counter=1 when held).

Run:
    pixi run mojo repro/spinlock_deadlock.mojo

Expected before fix: hangs indefinitely (kill with Ctrl-C).
Expected after fix: prints "All threads completed" and exits.
"""

from std.algorithm import parallelize
from shared.base.memory_pool import TensorMemoryPool


def main() raises:
    print("Testing memory pool under concurrent load (8 threads x 200 iters)...")
    var pool = TensorMemoryPool()
    pool.reset_stats()
    var NUM_THREADS = 8
    var ITERS_PER_THREAD = 200

    @parameter
    def worker(tid: Int) capturing:
        for _ in range(ITERS_PER_THREAD):
            var ptr = pool.allocate(256)
            pool.deallocate(ptr, 256)

    parallelize[worker](NUM_THREADS)
    print("All threads completed. Deadlock NOT triggered.")
    var stats = pool.get_stats()
    print("Allocations:", stats.allocations)
    print("Deallocations:", stats.deallocations)
