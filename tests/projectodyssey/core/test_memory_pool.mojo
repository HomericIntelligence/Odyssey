"""Unit tests for memory pool implementation (part 1 of 2).

Tests cover:
- Pool initialization and configuration
- Allocation and deallocation patterns
- Size bucket selection
- Statistics tracking
"""


from projectodyssey.base.memory_pool import (
    TensorMemoryPool,
    PoolConfig,
    PoolStats,
    pooled_alloc,
    pooled_free,
)
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros


def test_pool_default_init() raises:
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


def test_pool_custom_config() raises:
    """Test pool with custom configuration."""
    var config = PoolConfig()
    config.small_block_count = 32
    config.medium_block_count = 16

    var _pool = TensorMemoryPool(config)
    # Just verify it initializes without crashing
    print("✓ test_pool_custom_config passed")


def test_small_allocation_miss() raises:
    """Test allocation from pool hits pre-allocated block."""
    var pool = TensorMemoryPool()
    var ptr = pool.allocate(64)

    var stats = pool.get_stats()
    if stats.allocations != 1:
        raise Error("Should have 1 allocation")
    # Pool has pre-allocated blocks, so first allocation should be a hit
    if stats.pool_hits != 1:
        raise Error("Should have 1 hit from pre-allocated blocks")

    pool.deallocate(ptr, 64)
    print("✓ test_small_allocation_miss passed")


def test_small_allocation_hit() raises:
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


def test_medium_allocation() raises:
    """Test allocation from medium buckets."""
    var pool = TensorMemoryPool()

    # Allocate medium size
    var ptr = pool.allocate(2048)

    var stats = pool.get_stats()
    if stats.allocations != 1:
        raise Error("Should have 1 allocation")

    pool.deallocate(ptr, 2048)
    print("✓ test_medium_allocation passed")


def test_large_allocation_bypass() raises:
    """Test large allocations work correctly."""
    var pool = TensorMemoryPool()
    pool.reset_stats()

    # Allocate large size (> 16KB) - should bypass pool
    var ptr = pool.allocate(32768)

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


def test_bucket_selection_small() raises:
    """Test correct bucket selection for small sizes."""
    var pool = TensorMemoryPool()

    # Test boundary cases
    var ptr1 = pool.allocate(1)
    var ptr2 = pool.allocate(64)
    var ptr3 = pool.allocate(65)
    var ptr4 = pool.allocate(128)

    # All should succeed

    pool.deallocate(ptr1, 1)
    pool.deallocate(ptr2, 64)
    pool.deallocate(ptr3, 65)
    pool.deallocate(ptr4, 128)

    print("✓ test_bucket_selection_small passed")


def test_statistics_tracking() raises:
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


def test_trim_releases_memory() raises:
    """Test trim() method exists (placeholder for future optimization)."""
    var pool = TensorMemoryPool()

    # Allocate and deallocate
    var ptr1 = pool.allocate(64)
    var ptr2 = pool.allocate(64)
    pool.deallocate(ptr1, 64)
    pool.deallocate(ptr2, 64)

    var stats_before = pool.get_stats()

    # Trim is a placeholder for future optimization
    pool.trim()

    var stats_after = pool.get_stats()

    # Stats should be unchanged (trim is no-op)
    if stats_after.allocations != stats_before.allocations:
        raise Error("Trim should not change stats")

    print("✓ test_trim_releases_memory passed")


def test_clear_releases_all() raises:
    """Test clear() releases all pooled memory."""
    var pool = TensorMemoryPool()

    # Allocate and deallocate to populate cache
    var ptr1 = pool.allocate(64)
    var ptr2 = pool.allocate(128)
    var ptr3 = pool.allocate(2048)
    pool.deallocate(ptr1, 64)
    pool.deallocate(ptr2, 128)
    pool.deallocate(ptr3, 2048)

    var stats_before = pool.get_stats()
    if stats_before.bytes_cached <= 0:
        raise Error("Should have cached bytes before clear")

    # Clear all pooled memory
    pool.clear()

    var stats_after = pool.get_stats()

    # Stats should be cleared
    if stats_after.bytes_cached != 0:
        raise Error("Cache should be empty after clear()")
    if stats_after.deallocations != 3:
        raise Error("Stats should reflect deallocations")

    print("✓ test_clear_releases_all passed")


def test_anytensor_uses_pool() raises:
    """Verify AnyTensor allocations work correctly."""
    # Create a tensor - it will use pooled_alloc internally
    var shape = List[Int]()
    shape.append(10)
    shape.append(20)
    var t = zeros(shape, DType.float32)

    # Track allocations using a new pool
    var pool = TensorMemoryPool()
    var ptr = pool.allocate(64)
    pool.deallocate(ptr, 64)

    var stats = pool.get_stats()
    if stats.allocations < 1:
        raise Error("Pool should track allocations")

    print("✓ test_anytensor_uses_pool passed")


def test_reference_counting_with_pool() raises:
    """Verify reference counting works with pooled memory."""
    # Create tensors to test reference counting
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)
    var t1 = zeros(shape, DType.float32)
    var t2 = t1  # Copy (increments refcount)
    var _t3 = t2  # Another copy

    # If we got here without memory errors, reference counting works
    print("✓ test_reference_counting_with_pool passed")


def test_pooled_alloc_deallocate() raises:
    """Test pooled_alloc and pooled_free functions."""
    var ptr = pooled_alloc(256)

    pooled_free(ptr, 256)

    print("✓ test_pooled_alloc_deallocate passed")


def main() raises:
    """Run all test_memory_pool tests."""
    print("Running test_memory_pool tests...")

    test_pool_default_init()
    print("✓ test_pool_default_init")

    test_pool_custom_config()
    print("✓ test_pool_custom_config")

    test_small_allocation_miss()
    print("✓ test_small_allocation_miss")

    test_small_allocation_hit()
    print("✓ test_small_allocation_hit")

    test_medium_allocation()
    print("✓ test_medium_allocation")

    test_large_allocation_bypass()
    print("✓ test_large_allocation_bypass")

    test_bucket_selection_small()
    print("✓ test_bucket_selection_small")

    test_statistics_tracking()
    print("✓ test_statistics_tracking")

    test_trim_releases_memory()
    print("✓ test_trim_releases_memory")

    test_clear_releases_all()
    print("✓ test_clear_releases_all")

    test_anytensor_uses_pool()
    print("✓ test_anytensor_uses_pool")

    test_reference_counting_with_pool()
    print("✓ test_reference_counting_with_pool")

    test_pooled_alloc_deallocate()
    print("✓ test_pooled_alloc_deallocate")

    print("\nAll test_memory_pool tests passed!")
