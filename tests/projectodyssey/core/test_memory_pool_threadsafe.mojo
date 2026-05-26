"""Concurrent stress tests for thread-safe memory pool.

Tests verify that TensorMemoryPool operates correctly under concurrent
access via `parallelize`, with no data corruption, crashes, or
inconsistent statistics.
"""

from std.algorithm import parallelize
from std.atomic import Atomic

from projectodyssey.base.memory_pool import (
    TensorMemoryPool,
    PoolConfig,
    PoolStats,
)


def test_concurrent_alloc_dealloc_small() raises:
    """Stress test: concurrent alloc/dealloc on small buckets.

    Multiple threads each perform alloc+dealloc cycles on 256B blocks.
    After all threads finish, stats must be consistent:
    - allocations == deallocations
    - bytes_allocated == 0 (all freed)
    """
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

    var stats = pool.get_stats()
    var expected = NUM_THREADS * ITERS_PER_THREAD
    if stats.allocations != expected:
        raise Error(
            "Expected "
            + String(expected)
            + " allocations, got "
            + String(stats.allocations)
        )
    if stats.deallocations != expected:
        raise Error(
            "Expected "
            + String(expected)
            + " deallocations, got "
            + String(stats.deallocations)
        )
    if stats.bytes_allocated != 0:
        raise Error(
            "Expected 0 bytes_allocated after all frees, got "
            + String(stats.bytes_allocated)
        )

    print("✓ test_concurrent_alloc_dealloc_small passed")


def test_concurrent_alloc_dealloc_medium() raises:
    """Stress test: concurrent alloc/dealloc on medium buckets (2KB)."""
    var pool = TensorMemoryPool()
    pool.reset_stats()
    var NUM_THREADS = 4
    var ITERS_PER_THREAD = 100

    @parameter
    def worker(tid: Int) capturing:
        for _ in range(ITERS_PER_THREAD):
            var ptr = pool.allocate(2048)
            pool.deallocate(ptr, 2048)

    parallelize[worker](NUM_THREADS)

    var stats = pool.get_stats()
    var expected = NUM_THREADS * ITERS_PER_THREAD
    if stats.allocations != expected:
        raise Error(
            "Expected "
            + String(expected)
            + " allocations, got "
            + String(stats.allocations)
        )
    if stats.deallocations != expected:
        raise Error(
            "Expected "
            + String(expected)
            + " deallocations, got "
            + String(stats.deallocations)
        )

    print("✓ test_concurrent_alloc_dealloc_medium passed")


def test_concurrent_same_bucket_contention() raises:
    """All threads hit the same 64B bucket to maximize contention.

    Verifies free list integrity under maximum contention on a single lock.
    """
    var pool = TensorMemoryPool()
    pool.reset_stats()
    var NUM_THREADS = 8
    var ITERS_PER_THREAD = 300

    @parameter
    def worker(tid: Int) capturing:
        for _ in range(ITERS_PER_THREAD):
            var ptr = pool.allocate(64)
            pool.deallocate(ptr, 64)

    parallelize[worker](NUM_THREADS)

    var stats = pool.get_stats()
    var expected = NUM_THREADS * ITERS_PER_THREAD
    if stats.allocations != expected:
        raise Error(
            "Contention test: expected "
            + String(expected)
            + " allocations, got "
            + String(stats.allocations)
        )
    if stats.bytes_allocated != 0:
        raise Error(
            "Contention test: bytes_allocated should be 0, got "
            + String(stats.bytes_allocated)
        )

    print("✓ test_concurrent_same_bucket_contention passed")


def test_concurrent_mixed_sizes() raises:
    """Threads allocate different sizes to test cross-bucket concurrency.

    Each thread uses a different size class to verify independent locking.
    """
    var pool = TensorMemoryPool()
    pool.reset_stats()
    var NUM_THREADS = 8
    var ITERS_PER_THREAD = 100

    @parameter
    def worker(tid: Int) capturing:
        # Each thread uses a different bucket based on tid
        var sizes = List[Int]()
        sizes.append(64)
        sizes.append(128)
        sizes.append(256)
        sizes.append(512)
        sizes.append(1024)
        sizes.append(2048)
        sizes.append(4096)
        sizes.append(8192)
        var size = sizes[tid % len(sizes)]
        for _ in range(ITERS_PER_THREAD):
            var ptr = pool.allocate(size)
            pool.deallocate(ptr, size)

    parallelize[worker](NUM_THREADS)

    var stats = pool.get_stats()
    var expected = NUM_THREADS * ITERS_PER_THREAD
    if stats.allocations != expected:
        raise Error(
            "Mixed sizes: expected "
            + String(expected)
            + " allocations, got "
            + String(stats.allocations)
        )
    if stats.deallocations != expected:
        raise Error(
            "Mixed sizes: expected "
            + String(expected)
            + " deallocations, got "
            + String(stats.deallocations)
        )
    if stats.bytes_allocated != 0:
        raise Error(
            "Mixed sizes: bytes_allocated should be 0, got "
            + String(stats.bytes_allocated)
        )

    print("✓ test_concurrent_mixed_sizes passed")


def test_concurrent_large_bypass() raises:
    """Threads allocate large sizes that bypass the pool entirely.

    Large allocations (>16KB) go directly to system malloc, which is
    thread-safe. This verifies stats remain consistent.
    """
    var pool = TensorMemoryPool()
    pool.reset_stats()
    var NUM_THREADS = 4
    var ITERS_PER_THREAD = 50

    @parameter
    def worker(tid: Int) capturing:
        for _ in range(ITERS_PER_THREAD):
            var ptr = pool.allocate(32768)
            pool.deallocate(ptr, 32768)

    parallelize[worker](NUM_THREADS)

    var stats = pool.get_stats()
    var expected = NUM_THREADS * ITERS_PER_THREAD
    if stats.allocations != expected:
        raise Error(
            "Large bypass: expected "
            + String(expected)
            + " allocations, got "
            + String(stats.allocations)
        )
    if stats.pool_hits != 0:
        raise Error(
            "Large bypass: should have 0 pool hits, got "
            + String(stats.pool_hits)
        )

    print("✓ test_concurrent_large_bypass passed")


def test_stats_consistency_after_concurrent_work() raises:
    """Verify pool_hits + pool_misses == allocations after concurrent use."""
    var pool = TensorMemoryPool()
    pool.reset_stats()
    var NUM_THREADS = 8
    var ITERS_PER_THREAD = 150

    @parameter
    def worker(tid: Int) capturing:
        for _ in range(ITERS_PER_THREAD):
            var ptr = pool.allocate(256)
            pool.deallocate(ptr, 256)

    parallelize[worker](NUM_THREADS)

    var stats = pool.get_stats()
    if stats.pool_hits + stats.pool_misses != stats.allocations:
        raise Error(
            "Stats inconsistency: hits("
            + String(stats.pool_hits)
            + ") + misses("
            + String(stats.pool_misses)
            + ") != allocations("
            + String(stats.allocations)
            + ")"
        )

    print("✓ test_stats_consistency_after_concurrent_work passed")


def main() raises:
    """Run thread-safety stress tests for memory pool."""
    print("Running memory pool thread-safety tests...")
    print("")

    test_concurrent_alloc_dealloc_small()
    test_concurrent_alloc_dealloc_medium()
    test_concurrent_same_bucket_contention()
    test_concurrent_mixed_sizes()
    test_concurrent_large_bypass()
    test_stats_consistency_after_concurrent_work()

    print("")
    print("All memory pool thread-safety tests passed!")
