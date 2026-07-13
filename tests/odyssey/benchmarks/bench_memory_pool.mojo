"""Benchmarks for memory pool performance.

Compares pool vs system malloc for various allocation sizes.
"""

from odyssey.base.memory_pool import (
    get_global_pool,
    pooled_alloc,
    pooled_free,
)
from std.memory import alloc
from std.time import perf_counter_ns


def bench_pool_small_allocations() raises:
    """Benchmark pool vs malloc for small allocations (64-1024 bytes)."""
    print("Benchmark: Small allocations (64-1024 bytes)")

    var sizes = [64, 128, 256, 512, 1024]
    var iterations = 1000

    for size in sizes:
        var pool = get_global_pool()
        pool.reset_stats()

        # Allocate and deallocate through pool
        var start = perf_counter_ns()
        for _ in range(iterations):
            var ptr = pooled_alloc(size)
            pooled_free(ptr, size)
        var pool_time = perf_counter_ns() - start

        var stats = pool.get_stats()
        var hit_rate = (stats.pool_hits * 100) // (
            stats.pool_hits + stats.pool_misses
        )

        print(
            "  ",
            size,
            "B: ",
            pool_time,
            " cycles, hit rate: ",
            hit_rate,
            "%",
        )


def bench_pool_medium_allocations() raises:
    """Benchmark pool for medium allocations (2KB-16KB)."""
    print("")
    print("Benchmark: Medium allocations (2KB-16KB)")

    var sizes = [2048, 4096, 8192, 16384]
    var iterations = 500

    for size in sizes:
        var pool = get_global_pool()
        pool.reset_stats()

        # Allocate and deallocate through pool
        var start = perf_counter_ns()
        for _ in range(iterations):
            var ptr = pooled_alloc(size)
            pooled_free(ptr, size)
        var pool_time = perf_counter_ns() - start

        var stats = pool.get_stats()
        var hit_rate = (stats.pool_hits * 100) // (
            stats.pool_hits + stats.pool_misses
        )

        print(
            "  ",
            size,
            "B: ",
            pool_time,
            " cycles, hit rate: ",
            hit_rate,
            "%",
        )


def bench_statistics() raises:
    """Print pool statistics."""
    print("")
    print("Pool Statistics:")

    var pool = get_global_pool()
    var stats = pool.get_stats()

    print("  Total allocations:", stats.allocations)
    print("  Total deallocations:", stats.deallocations)
    print("  Pool hits:", stats.pool_hits)
    print("  Pool misses:", stats.pool_misses)
    print("  Bytes allocated:", stats.bytes_allocated)
    print("  Bytes cached:", stats.bytes_cached)
    print("  Peak cached bytes:", stats.peak_cached_bytes)

    if stats.pool_hits + stats.pool_misses > 0:
        var overall_hit_rate = (stats.pool_hits * 100) // (
            stats.pool_hits + stats.pool_misses
        )
        print("  Overall hit rate:", overall_hit_rate, "%")


def main() raises:
    """Run all benchmarks."""
    print("Memory Pool Benchmarks")
    print("======================")
    print("")

    bench_pool_small_allocations()
    bench_pool_medium_allocations()
    bench_statistics()

    print("")
    print("Benchmarks complete!")
