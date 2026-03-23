# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_memory_pool.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for memory pool implementation (part 1 of 2).

Tests cover:
- Pool initialization and configuration
- Allocation and deallocation patterns
- Size bucket selection
- Statistics tracking
"""

from shared.base.memory_pool import (
    TensorMemoryPool,
    PoolConfig,
    PoolStats,
    pooled_alloc,
    pooled_free,
)
from shared.tensor.any_tensor import AnyTensor, zeros


fn test_pool_default_init() raises:
    """Test default pool initialization."""
    var pool = TensorMemoryPool()
    var stats = pool.get_stats()

    if stats.allocations != 0:
        raise Error("Initial allocations should be 0")
    if stats.deallocations != 0:
        raise Error("Initial deallocations should be 0")
    if stats.pool_hits != 0:
        raise Error("Initial hits should be 0")
    if stats.pool_misses != 0:
        raise Error("Initial misses should be 0")

    print("✓ test_pool_default_init passed")


fn test_pool_custom_config() raises:
    """Test pool with custom configuration."""
    var config = PoolConfig()
    config.small_block_count = 32
    config.medium_block_count = 16

    var _pool = TensorMemoryPool(config)
    # Just verify it initializes without crashing
    print("✓ test_pool_custom_config passed")


fn test_small_allocation_miss() raises:
    """Test allocation from pool hits pre-allocated block."""
    var pool = TensorMemoryPool()
    var ptr = pool.allocate(64)

    if not ptr:
        raise Error("Allocation should not be null")

    var stats = pool.get_stats()
    if stats.allocations != 1:
        raise Error("Should have 1 allocation")
    # Pool has pre-allocated blocks, so first allocation should be a hit
    if stats.pool_hits != 1:
        raise Error("Should have 1 hit from pre-allocated blocks")

    pool.deallocate(ptr, 64)
    print("✓ test_small_allocation_miss passed")


fn test_small_allocation_hit() raises:
    """Test allocation from populated pool (hit)."""
    var pool = TensorMemoryPool()

    # First allocation - hit from pre-allocated block
    var ptr1 = pool.allocate(64)
    pool.deallocate(ptr1, 64)

    # Second allocation - hit (reuses deallocated block)
    var ptr2 = pool.allocate(64)

    var stats = pool.get_stats()
    if stats.allocations != 2:
        raise Error("Should have 2 allocations")
    # Both should be hits since we have pre-allocated blocks
    if stats.pool_hits != 2:
        raise Error("Should have 2 hits (pre-allocated + reused)")
    if stats.pool_misses != 0:
        raise Error("Should have 0 misses with pre-allocated blocks")

    pool.deallocate(ptr2, 64)
    print("✓ test_small_allocation_hit passed")


fn test_medium_allocation() raises:
    """Test allocation from medium buckets."""
    var pool = TensorMemoryPool()

    # Allocate medium size
    var ptr = pool.allocate(2048)
    if not ptr:
        raise Error("Allocation should not be null")

    var stats = pool.get_stats()
    if stats.allocations != 1:
        raise Error("Should have 1 allocation")

    pool.deallocate(ptr, 2048)
    print("✓ test_medium_allocation passed")


fn test_large_allocation_bypass() raises:
    """Test large allocations work correctly."""
    var pool = TensorMemoryPool()
    pool.reset_stats()

    # Allocate large size (> 16KB) - should bypass pool
    var ptr = pool.allocate(32768)
    if not ptr:
        raise Error("Large allocation should succeed")

    var stats = pool.get_stats()
    if stats.allocations != 1:
        raise Error("Should have 1 allocation")
    if stats.pool_misses != 1:
        raise Error("Large alloc should be a miss (bypassed)")
    if stats.pool_hits != 0:
        raise Error("Large alloc should not count as hit")

    pool.deallocate(ptr, 32768)
    stats = pool.get_stats()
    if stats.deallocations != 1:
        raise Error("Should have 1 deallocation")

    print("✓ test_large_allocation_bypass passed")


fn test_bucket_selection_small() raises:
    """Test correct bucket selection for small sizes."""
    var pool = TensorMemoryPool()

    # Test boundary cases
    var ptr1 = pool.allocate(1)
    var ptr2 = pool.allocate(64)
    var ptr3 = pool.allocate(65)
    var ptr4 = pool.allocate(128)

    # All should succeed
    if not ptr1:
        raise Error("1 byte allocation should succeed")
    if not ptr2:
        raise Error("64 byte allocation should succeed")
    if not ptr3:
        raise Error("65 byte allocation should succeed")
    if not ptr4:
        raise Error("128 byte allocation should succeed")

    pool.deallocate(ptr1, 1)
    pool.deallocate(ptr2, 64)
    pool.deallocate(ptr3, 65)
    pool.deallocate(ptr4, 128)

    print("✓ test_bucket_selection_small passed")


fn test_statistics_tracking() raises:
    """Test statistics accuracy."""
    var pool = TensorMemoryPool()

    # Allocate and deallocate various sizes
    for i in range(5):
        var ptr = pool.allocate(256)
        pool.deallocate(ptr, 256)

    var stats = pool.get_stats()
    if stats.allocations != 5:
        raise Error("Should track all allocations")
    if stats.deallocations != 5:
        raise Error("Should track all deallocations")

    print("✓ test_statistics_tracking passed")


fn main() raises:
    """Run memory pool tests (part 1)."""
    print("Running memory pool tests (part 1)...")
    print("")

    test_pool_default_init()
    test_pool_custom_config()
    test_small_allocation_miss()
    test_small_allocation_hit()
    test_medium_allocation()
    test_large_allocation_bypass()
    test_bucket_selection_small()
    test_statistics_tracking()

    print("")
    print("All memory pool tests (part 1) passed!")
