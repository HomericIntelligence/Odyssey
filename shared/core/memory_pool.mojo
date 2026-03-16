"""Memory Pool for Small Tensor Allocations.

Implements a high-performance memory pool to reduce allocation overhead for small
tensor allocations in ML workloads.

The pool uses a three-tier bucket strategy:
- Small buckets: 64B, 128B, 256B, 512B, 1KB (for very small tensors)
- Medium buckets: 2KB, 4KB, 8KB, 16KB (for typical gradient tensors)
- Large allocations: >16KB bypass the pool and go directly to system malloc

Each bucket maintains a free list of pre-allocated blocks that can be reused,
reducing allocation overhead and system malloc pressure.

Example:
    ```mojo
    from .memory_pool import get_global_pool, pooled_alloc, pooled_free

    # Allocate memory from pool
    var ptr = pooled_alloc(256)  # Will use 256B bucket

    # Use memory...

    # Return to pool for reuse
    pooled_free(ptr, 256)
    ```
"""

from collections import List
from memory import UnsafePointer, alloc

# Size bucket boundaries (in bytes)
comptime SMALL_SIZES_COUNT = 5
comptime MEDIUM_SIZES_COUNT = 4
comptime LARGE_THRESHOLD = 16384


fn _get_small_size(index: Int) -> Int:
    """Get small bucket size by index."""
    if index == 0:
        return 64
    elif index == 1:
        return 128
    elif index == 2:
        return 256
    elif index == 3:
        return 512
    else:
        return 1024


fn _get_medium_size(index: Int) -> Int:
    """Get medium bucket size by index."""
    if index == 0:
        return 2048
    elif index == 1:
        return 4096
    elif index == 2:
        return 8192
    else:
        return 16384


struct PoolStats(Copyable, ImplicitlyCopyable, Movable):
    """Statistics for memory pool performance monitoring.

    Tracks allocation patterns and pool efficiency metrics.

    Attributes:
        allocations: Total number of allocate() calls.
        deallocations: Total number of deallocate() calls.
        pool_hits: Allocations served from pool cache.
        pool_misses: Allocations requiring system malloc.
        bytes_allocated: Total bytes currently allocated.
        bytes_cached: Total bytes currently in pool.
        peak_cached_bytes: Peak bytes cached (high water mark).
    """

    var allocations: Int
    """Total allocate() calls."""
    var deallocations: Int
    """Total deallocate() calls."""
    var pool_hits: Int
    """Allocations served from pool."""
    var pool_misses: Int
    """Allocations requiring malloc."""
    var bytes_allocated: Int
    """Total bytes currently allocated."""
    var bytes_cached: Int
    """Total bytes in pool."""
    var peak_cached_bytes: Int
    """Peak bytes cached."""

    fn __init__(out self):
        """Initialize statistics to zero."""
        self.allocations = 0
        self.deallocations = 0
        self.pool_hits = 0
        self.pool_misses = 0
        self.bytes_allocated = 0
        self.bytes_cached = 0
        self.peak_cached_bytes = 0


struct PoolConfig(Copyable, Movable):
    """Configuration for TensorMemoryPool initialization.

    Attributes:
        small_block_count: Initial number of blocks pre-allocated per small bucket.
        medium_block_count: Initial number of blocks pre-allocated per medium bucket.
        max_cached_bytes: Maximum bytes to cache before trim() releases unused blocks.
    """

    var small_block_count: Int
    """Initial blocks per small size class."""
    var medium_block_count: Int
    """Initial blocks per medium size class."""
    var max_cached_bytes: Int
    """Maximum bytes to cache before trim."""

    fn __init__(out self):
        """Initialize with default configuration."""
        # Conservative defaults to avoid memory overhead
        self.small_block_count = 16
        self.medium_block_count = 8
        self.max_cached_bytes = 16 * 1024 * 1024  # 16 MB


struct _FreeListNode(Movable):
    """Node in an intrusive free list for pooled blocks.

    This allows us to maintain a linked list of free blocks without
    extra allocation overhead - the free list structure is stored
    at the beginning of each free block itself.

    Attributes:
        next: Pointer to the next free block, or null if this is the last.
    """

    var next: UnsafePointer[_FreeListNode, origin=MutAnyOrigin]
    """Next block in free list, or null if last."""


struct FreeList(Copyable, Movable):
    """Intrusive free list for pooled blocks.

    Maintains a LIFO (stack-like) list of free blocks of a fixed size.
    Each block contains space for the linked list node at the beginning.

    Attributes:
        head: Pointer to the first free block, or null if empty.
        block_size: Size of each block managed by this free list.
        count: Number of blocks currently in the free list.
    """

    var head: UnsafePointer[_FreeListNode, origin=MutAnyOrigin]
    """First node in free list, or null if empty."""
    var block_size: Int
    """Size of each block managed by this list."""
    var count: Int
    """Number of free blocks in this list."""

    fn __init__(out self, block_size: Int):
        """Initialize an empty free list for the given block size.

        Args:
            block_size: Size of each block this list will manage.
        """
        self.head = UnsafePointer[_FreeListNode, origin=MutAnyOrigin]()
        self.block_size = block_size
        self.count = 0

    fn is_empty(self) -> Bool:
        """Check if the free list is empty.

        Returns:
            True if there are no free blocks available.
        """
        return self.head == UnsafePointer[_FreeListNode, origin=MutAnyOrigin]()

    fn pop(mut self) -> UnsafePointer[UInt8, origin=MutAnyOrigin]:
        """Remove and return a block from the free list.

        Returns:
            UnsafePointer to the allocated block, or null if list is empty.
        """
        if self.is_empty():
            return UnsafePointer[UInt8, origin=MutAnyOrigin]()

        var node = self.head
        self.head = node[].next
        self.count -= 1

        # Cast the node back to UInt8 pointer
        return node.bitcast[UInt8]()

    fn push(mut self, ptr: UnsafePointer[UInt8, origin=MutAnyOrigin]):
        """Add a block back to the free list.

        Args:
            ptr: Pointer to the block to return to the pool.
        """
        var node = ptr.bitcast[_FreeListNode]()
        node[].next = self.head
        self.head = node
        self.count += 1


struct TensorMemoryPool(Copyable, Movable):
    """Memory pool for small tensor allocations.

    Implements a three-tier bucket strategy:
    - Small buckets (< 1KB): 64B, 128B, 256B, 512B, 1KB
    - Medium buckets (1-16KB): 2KB, 4KB, 8KB, 16KB
    - Large allocations (> 16KB): Direct system malloc

    Pre-allocates blocks to the free lists based on configuration.
    Allocations are served from the appropriate bucket's free list when available,
    otherwise a new block is allocated from the system.

    Attributes:
        small_lists: Free lists for small size classes.
        medium_lists: Free lists for medium size classes.
        stats: Performance statistics.
    """

    # Size bucket boundaries (in bytes)

    var small_lists: List[FreeList]
    """Free lists for < 1KB allocations."""
    var medium_lists: List[FreeList]
    """Free lists for 1KB-16KB allocations."""
    var stats: PoolStats
    """Pool statistics."""

    fn __init__(out self):
        """Initialize pool with default configuration."""
        self.small_lists = List[FreeList]()
        for i in range(SMALL_SIZES_COUNT):
            self.small_lists.append(FreeList(_get_small_size(i)))

        self.medium_lists = List[FreeList]()
        for i in range(MEDIUM_SIZES_COUNT):
            self.medium_lists.append(FreeList(_get_medium_size(i)))

        self.stats = PoolStats()

        # Pre-allocate default configuration
        self._preallocate_blocks(16, 8)

    fn __init__(out self, config: PoolConfig):
        """Initialize pool with custom configuration.

        Args:
            config: Pool configuration specifying initial block counts.
        """
        self.small_lists = List[FreeList]()
        for i in range(SMALL_SIZES_COUNT):
            self.small_lists.append(FreeList(_get_small_size(i)))

        self.medium_lists = List[FreeList]()
        for i in range(MEDIUM_SIZES_COUNT):
            self.medium_lists.append(FreeList(_get_medium_size(i)))

        self.stats = PoolStats()

        # Pre-allocate with custom configuration
        self._preallocate_blocks(
            config.small_block_count, config.medium_block_count
        )

    fn __del__(deinit self):
        """Destructor - release all pooled memory."""
        self.clear()

    fn _preallocate_blocks(mut self, small_count: Int, medium_count: Int):
        """Pre-allocate blocks to each bucket's free list.

        Args:
            small_count: Number of blocks to pre-allocate to each small bucket.
            medium_count: Number of blocks to pre-allocate to each medium bucket.
        """
        # Pre-allocate small blocks
        for i in range(len(self.small_lists)):
            var size = self.small_lists[i].block_size
            for _ in range(small_count):
                var ptr = alloc[UInt8](size)
                self.small_lists[i].push(ptr)
                self.stats.bytes_cached += size

        # Pre-allocate medium blocks
        for i in range(len(self.medium_lists)):
            var size = self.medium_lists[i].block_size
            for _ in range(medium_count):
                var ptr = alloc[UInt8](size)
                self.medium_lists[i].push(ptr)
                self.stats.bytes_cached += size

        # Update peak
        if self.stats.bytes_cached > self.stats.peak_cached_bytes:
            self.stats.peak_cached_bytes = self.stats.bytes_cached

    fn _find_bucket_index(self, size: Int) -> Int:
        """Find the smallest bucket that fits the requested size.

        Args:
            size: Number of bytes to allocate.

        Returns:
            Index into small_lists or medium_lists, or -1 if no bucket fits.
        """
        # Check small buckets
        for i in range(SMALL_SIZES_COUNT):
            if size <= _get_small_size(i):
                return i

        # Check medium buckets
        for i in range(MEDIUM_SIZES_COUNT):
            if size <= _get_medium_size(i):
                return i + SMALL_SIZES_COUNT

        # No bucket found
        return -1

    fn allocate(
        mut self, size: Int
    ) -> UnsafePointer[UInt8, origin=MutAnyOrigin]:
        """Allocate memory from pool or system allocator.

        Args:
            size: Number of bytes to allocate.

        Returns:
            UnsafePointer to allocated memory.
        """
        self.stats.allocations += 1

        # Large allocations bypass pool
        if size > LARGE_THRESHOLD:
            self.stats.pool_misses += 1
            self.stats.bytes_allocated += size
            return alloc[UInt8](size)

        var bucket_idx = self._find_bucket_index(size)

        # No suitable bucket found, allocate directly
        if bucket_idx < 0:
            self.stats.pool_misses += 1
            self.stats.bytes_allocated += size
            return alloc[UInt8](size)

        # Try to get from small bucket
        if bucket_idx < len(self.small_lists):
            if not self.small_lists[bucket_idx].is_empty():
                self.stats.pool_hits += 1
                var ptr = self.small_lists[bucket_idx].pop()
                self.stats.bytes_cached -= self.small_lists[
                    bucket_idx
                ].block_size
                self.stats.bytes_allocated += self.small_lists[
                    bucket_idx
                ].block_size
                return ptr
            else:
                # Pool miss - allocate new block
                self.stats.pool_misses += 1
                var actual_size = self.small_lists[bucket_idx].block_size
                self.stats.bytes_allocated += actual_size
                return alloc[UInt8](actual_size)

        # Try to get from medium bucket
        var medium_idx = bucket_idx - len(self.small_lists)
        if medium_idx < len(self.medium_lists):
            if not self.medium_lists[medium_idx].is_empty():
                self.stats.pool_hits += 1
                var ptr = self.medium_lists[medium_idx].pop()
                self.stats.bytes_cached -= self.medium_lists[
                    medium_idx
                ].block_size
                self.stats.bytes_allocated += self.medium_lists[
                    medium_idx
                ].block_size
                return ptr
            else:
                # Pool miss - allocate new block
                self.stats.pool_misses += 1
                var actual_size = self.medium_lists[medium_idx].block_size
                self.stats.bytes_allocated += actual_size
                return alloc[UInt8](actual_size)

        # Fallback (should not reach here)
        self.stats.pool_misses += 1
        self.stats.bytes_allocated += size
        return alloc[UInt8](size)

    fn deallocate(
        mut self, ptr: UnsafePointer[UInt8, origin=MutAnyOrigin], size: Int
    ):
        """Return allocation to pool or system allocator.

        Args:
            ptr: Pointer to memory to deallocate.
            size: Size of allocation.
        """
        self.stats.deallocations += 1

        # Large allocations bypass pool
        if size > LARGE_THRESHOLD:
            ptr.free()
            self.stats.bytes_allocated -= size
            return

        var bucket_idx = self._find_bucket_index(size)

        # No suitable bucket found, free directly
        if bucket_idx < 0:
            ptr.free()
            self.stats.bytes_allocated -= size
            return

        # Return to small bucket
        if bucket_idx < len(self.small_lists):
            var actual_size = self.small_lists[bucket_idx].block_size
            self.small_lists[bucket_idx].push(ptr)
            self.stats.bytes_cached += actual_size
            self.stats.bytes_allocated -= actual_size

            if self.stats.bytes_cached > self.stats.peak_cached_bytes:
                self.stats.peak_cached_bytes = self.stats.bytes_cached
            return

        # Return to medium bucket
        var medium_idx = bucket_idx - len(self.small_lists)
        if medium_idx < len(self.medium_lists):
            var actual_size = self.medium_lists[medium_idx].block_size
            self.medium_lists[medium_idx].push(ptr)
            self.stats.bytes_cached += actual_size
            self.stats.bytes_allocated -= actual_size

            if self.stats.bytes_cached > self.stats.peak_cached_bytes:
                self.stats.peak_cached_bytes = self.stats.bytes_cached
            return

        # Fallback (should not reach here)
        ptr.free()
        self.stats.bytes_allocated -= size

    fn get_stats(self) -> PoolStats:
        """Get current pool statistics.

        Returns:
            Copy of current statistics.
        """
        return self.stats

    fn reset_stats(mut self):
        """Reset all statistics to zero."""
        self.stats = PoolStats()

    fn trim(mut self):
        """Release unused blocks from pool (not implemented).

        This is a placeholder for future optimization to return
        cached blocks to the OS when they're not being used.
        """
        pass

    fn clear(mut self):
        """Release all pooled memory."""
        # Free all small bucket blocks
        for i in range(len(self.small_lists)):
            while not self.small_lists[i].is_empty():
                var ptr = self.small_lists[i].pop()
                ptr.free()

        # Free all medium bucket blocks
        for i in range(len(self.medium_lists)):
            while not self.medium_lists[i].is_empty():
                var ptr = self.medium_lists[i].pop()
                ptr.free()

        # Reset stats
        self.stats.bytes_cached = 0
        self.stats.bytes_allocated = 0


# Global memory pool singleton
# Since Mojo v0.26.1+ doesn't support global vars, we implement pooled_alloc
# and pooled_free to bypass the global pool and go directly to system allocator
# for now. This is a temporary workaround until Mojo adds proper global state support.


fn pooled_alloc(size: Int) -> UnsafePointer[UInt8, origin=MutAnyOrigin]:
    """Allocate memory - currently bypasses pool (direct malloc).

    Routes allocations directly to system malloc. The pool infrastructure
    is implemented but not used until Mojo v0.26+ supports global mutable state.

    Args:
        size: Number of bytes to allocate.

    Returns:
        UnsafePointer to allocated memory.

    Example:
        ```mojo
        var ptr = pooled_alloc(256)  # Allocated via malloc
        var large_ptr = pooled_alloc(1024*1024)  # Allocated via malloc
        ```
    """
    # Temporary: Direct malloc until we can use global pool
    return alloc[UInt8](size)


fn pooled_free(ptr: UnsafePointer[UInt8, origin=MutAnyOrigin], size: Int):
    """Return allocation to system allocator.

    Currently frees directly to system allocator. The pool infrastructure
    is implemented but not used until Mojo v0.26+ supports global mutable state.

    Args:
        ptr: Pointer to memory to deallocate.
        size: Size of allocation (unused in direct mode).

    Example:
        ```mojo
        pooled_free(ptr, 256)  # Freed to system allocator
        ```
    """
    # Temporary: Direct free until we can use global pool
    ptr.free()


fn get_global_pool() -> TensorMemoryPool:
    """Get a new memory pool instance.

    Note: This returns a new instance since Mojo v0.26.1+ doesn't support
    global mutable state. The pool infrastructure is fully implemented but
    not used until Mojo v0.26+ adds support for global vars.

    Returns:
        A new TensorMemoryPool instance with default configuration.

    Example:
        ```mojo
        var pool = get_global_pool()
        var stats = pool.get_stats()
        ```
    """
    return TensorMemoryPool()
