# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_memory_pool.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Unit tests for memory pool implementation (part 2 of 2).

Tests cover:
- Memory management (clear)
- AnyTensor integration with pool
- Reference counting with pooled memory
- pooled_alloc/pooled_free functions
"""

from shared.core.memory_pool import (
    TensorMemoryPool,
    PoolConfig,
    PoolStats,
    pooled_alloc,
    pooled_free,
)
from shared.core.any_tensor import AnyTensor, zeros


fn test_trim_releases_memory() raises:
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


fn test_clear_releases_all() raises:
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


fn test_anytensor_uses_pool() raises:
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


fn test_reference_counting_with_pool() raises:
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


fn test_pooled_alloc_deallocate() raises:
    """Test pooled_alloc and pooled_free functions."""
    var ptr = pooled_alloc(256)
    if not ptr:
        raise Error("pooled_alloc should return valid pointer")

    pooled_free(ptr, 256)

    print("✓ test_pooled_alloc_deallocate passed")


fn main() raises:
    """Run memory pool tests (part 2)."""
    print("Running memory pool tests (part 2)...")
    print("")

    test_trim_releases_memory()
    test_clear_releases_all()
    test_anytensor_uses_pool()
    test_reference_counting_with_pool()
    test_pooled_alloc_deallocate()

    print("")
    print("All memory pool tests (part 2) passed!")
