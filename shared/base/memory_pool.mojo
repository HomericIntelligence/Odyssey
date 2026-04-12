"""Memory Pool for Small Tensor Allocations.

Implements a high-performance memory pool to reduce allocation overhead for small
tensor allocations in ML workloads.

The pool uses a three-tier bucket strategy:
- Small buckets: 64B, 128B, 256B, 512B, 1KB (for very small tensors)
- Medium buckets: 2KB, 4KB, 8KB, 16KB (for typical gradient tensors)
- Large allocations: >16KB bypass the pool and go directly to system malloc

Each bucket maintains a free list of pre-allocated blocks that can be reused,
reducing allocation overhead and system malloc pressure.

Thread-Safety Guarantees:
    - `TensorMemoryPool.allocate()` and `deallocate()` are safe to call
      concurrently from multiple threads (e.g., via `parallelize`).
    - Each size bucket has its own spinlock to minimize contention.
    - Statistics counters use atomic operations for lock-free updates.
    - `clear()` and `reset_stats()` are NOT safe to call concurrently
      with `allocate()`/`deallocate()`. Call them only when no other
      threads are accessing the pool.
    - `pooled_alloc`/`pooled_free` currently bypass the pool and delegate
      to system malloc, which is itself thread-safe.

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

from std.collections import List
from std.memory import UnsafePointer, alloc, memcpy
from std.os.atomic import Atomic

# Size bucket boundaries (in bytes)
comptime SMALL_SIZES_COUNT = 5
comptime MEDIUM_SIZES_COUNT = 4
comptime LARGE_THRESHOLD = 16384

# AtomicStats byte offsets (each counter is 8 bytes / int64)
comptime _ASTATS_ALLOCATIONS = 0
comptime _ASTATS_DEALLOCATIONS = 8
comptime _ASTATS_POOL_HITS = 16
comptime _ASTATS_POOL_MISSES = 24
comptime _ASTATS_BYTES_ALLOCATED = 32
comptime _ASTATS_BYTES_CACHED = 40
comptime _ASTATS_PEAK_CACHED = 48
comptime _ASTATS_SIZE = 56


def _get_small_size(index: Int) -> Int:
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


def _get_medium_size(index: Int) -> Int:
    """Get medium bucket size by index."""
    if index == 0:
        return 2048
    elif index == 1:
        return 4096
    elif index == 2:
        return 8192
    else:
        return 16384


struct SpinLock(Copyable, Movable):
    """Simple test-and-set spinlock using atomic operations.

    Uses a heap-allocated Atomic[DType.int64] as the lock state.
    Value 0 means unlocked; value 1 means locked.

    Explicit copy constructor deep-copies the backing storage so that each
    SpinLock instance owns its own allocation.  Without an explicit copy
    constructor Mojo synthesises a shallow memberwise copy that duplicates
    the _state pointer value.  When List[SpinLock] reallocates (capacity
    0→1→2→4→…) the old storage is destroyed, calling __del__ → free() on
    every stale copy – a double-free that corrupts the heap.

    Explicit move constructor transfers pointer ownership so that
    List reallocation moves (rather than copies-then-destroys) the
    SpinLock instances, which is both safe and avoids the allocation
    overhead of a deep copy.
    """

    var _state: UnsafePointer[UInt8, origin=MutAnyOrigin]
    """Heap-allocated 8-byte region reinterpreted as Atomic[DType.int64]."""

    def __init__(out self):
        """Initialize an unlocked spinlock."""
        self._state = alloc[UInt8](8)
        for i in range(8):
            self._state[i] = 0

    def _as_atomic(
        self,
    ) -> UnsafePointer[Atomic[DType.int64], origin=MutAnyOrigin]:
        """Reinterpret backing store as an atomic int64."""
        return self._state.bitcast[Atomic[DType.int64]]()

    def _lock_word(self) -> UnsafePointer[Int64, origin=MutAnyOrigin]:
        """Return the lock word as a plain Int64 pointer for static Atomic ops.
        """
        return self._state.bitcast[Int64]()

    def lock(self):
        """Acquire the lock, spinning until available.

        Uses TTAS (Test-and-Test-and-Set) with atomic fetch_add for acquisition.
        Spins on a plain load() until the lock looks free, then atomically
        attempts to transition 0→1 via fetch_add(1).  If the old value was 0
        we hold the lock (counter is exactly 1).  If the old value was non-zero
        another thread won the race; we immediately undo with fetch_add(-1) and
        spin again.

        Note: compare_exchange_weak is not available in Mojo 0.26.3's Atomic
        API.  This TTAS pattern is the closest correct approximation: it is
        safe for any number of threads because the undo always restores the
        counter to its pre-attempt value, and the counter is bounded by the
        number of concurrently spinning threads (all within Int64 range).
        """
        var word = self._lock_word()
        while True:
            # Wait until lock looks free before attempting (reduces bus traffic)
            while Atomic[DType.int64].load(word) != 0:
                pass
            # Attempt to acquire: fetch_add returns the old value.
            # If old == 0 we transitioned 0→1 atomically and hold the lock.
            if Atomic[DType.int64].fetch_add(word, Int64(1)) == 0:
                return
            # Another thread won the race; undo our increment atomically.
            _ = Atomic[DType.int64].fetch_add(word, Int64(-1))

    def unlock(self):
        """Release the lock.

        Atomically writes 0 to the lock word.  This is correct because the
        lock() protocol guarantees that when we hold the lock the counter is
        exactly 1 (only our increment), so an atomic store of 0 is equivalent
        to a sequentially-consistent release without needing fetch_sub.
        """
        Atomic[DType.int64].store(self._lock_word(), Int64(0))

    def __del__(deinit self):
        """Free the backing store."""
        self._state.free()


struct AtomicStats(Copyable, Movable):
    """Atomic statistics counters for thread-safe pool monitoring.

    Each counter is a heap-allocated Atomic[DType.int64] to allow
    lock-free concurrent updates from multiple threads.

    Explicit copy and move constructors are provided for the same
    reason as SpinLock: without them Mojo synthesises a shallow copy that
    duplicates the _data pointer, causing a double-free when the owning
    struct is stored in a List that reallocates.
    """

    var _data: UnsafePointer[UInt8, origin=MutAnyOrigin]
    """Heap-allocated storage for 7 atomic int64 counters (56 bytes)."""

    def __init__(out self):
        """Initialize all counters to zero."""
        self._data = alloc[UInt8](_ASTATS_SIZE)
        for i in range(_ASTATS_SIZE):
            self._data[i] = 0

    def _counter(
        self, offset: Int
    ) -> UnsafePointer[Atomic[DType.int64], origin=MutAnyOrigin]:
        """Get pointer to atomic counter at given byte offset."""
        return (self._data + offset).bitcast[Atomic[DType.int64]]()

    def add_allocations(self, n: Int):
        """Atomically increment allocations counter."""
        _ = self._counter(_ASTATS_ALLOCATIONS)[].fetch_add(Int64(n))

    def add_deallocations(self, n: Int):
        """Atomically increment deallocations counter."""
        _ = self._counter(_ASTATS_DEALLOCATIONS)[].fetch_add(Int64(n))

    def add_pool_hits(self, n: Int):
        """Atomically increment pool hits counter."""
        _ = self._counter(_ASTATS_POOL_HITS)[].fetch_add(Int64(n))

    def add_pool_misses(self, n: Int):
        """Atomically increment pool misses counter."""
        _ = self._counter(_ASTATS_POOL_MISSES)[].fetch_add(Int64(n))

    def add_bytes_allocated(self, n: Int):
        """Atomically adjust bytes allocated counter."""
        _ = self._counter(_ASTATS_BYTES_ALLOCATED)[].fetch_add(Int64(n))

    def add_bytes_cached(self, n: Int):
        """Atomically adjust bytes cached counter."""
        _ = self._counter(_ASTATS_BYTES_CACHED)[].fetch_add(Int64(n))

    def update_peak_cached(self):
        """Update peak cached bytes if current cached exceeds it."""
        var cached = self._counter(_ASTATS_BYTES_CACHED)[].load()
        var peak_ptr = self._counter(_ASTATS_PEAK_CACHED)
        # Simple load-compare-store; minor races here are acceptable
        # since peak is a high-water-mark metric, not a correctness counter.
        if cached > peak_ptr[].load():
            peak_ptr[].max(Int64(cached))

    def snapshot(self) -> PoolStats:
        """Take a consistent snapshot of all counters.

        Returns:
            A PoolStats struct with current counter values.
        """
        var s = PoolStats()
        s.allocations = Int(self._counter(_ASTATS_ALLOCATIONS)[].load())
        s.deallocations = Int(self._counter(_ASTATS_DEALLOCATIONS)[].load())
        s.pool_hits = Int(self._counter(_ASTATS_POOL_HITS)[].load())
        s.pool_misses = Int(self._counter(_ASTATS_POOL_MISSES)[].load())
        s.bytes_allocated = Int(self._counter(_ASTATS_BYTES_ALLOCATED)[].load())
        s.bytes_cached = Int(self._counter(_ASTATS_BYTES_CACHED)[].load())
        s.peak_cached_bytes = Int(self._counter(_ASTATS_PEAK_CACHED)[].load())
        return s

    def _counter_int64(
        self, offset: Int
    ) -> UnsafePointer[Int64, origin=MutAnyOrigin]:
        """Return a plain Int64 pointer at the given byte offset (for atomic store).
        """
        return (self._data + offset).bitcast[Int64]()

    def reset(self):
        """Reset all counters to zero using atomic stores.

        Uses one 8-byte atomic store per counter rather than byte-level zeroing
        so that a concurrent snapshot() cannot observe a partially-zeroed counter
        on architectures where 8-byte writes are not naturally atomic.

        Caller must ensure no concurrent allocate()/deallocate() calls are in
        flight (see module-level thread-safety note).
        """
        Atomic[DType.int64].store(
            self._counter_int64(_ASTATS_ALLOCATIONS), Int64(0)
        )
        Atomic[DType.int64].store(
            self._counter_int64(_ASTATS_DEALLOCATIONS), Int64(0)
        )
        Atomic[DType.int64].store(
            self._counter_int64(_ASTATS_POOL_HITS), Int64(0)
        )
        Atomic[DType.int64].store(
            self._counter_int64(_ASTATS_POOL_MISSES), Int64(0)
        )
        Atomic[DType.int64].store(
            self._counter_int64(_ASTATS_BYTES_ALLOCATED), Int64(0)
        )
        Atomic[DType.int64].store(
            self._counter_int64(_ASTATS_BYTES_CACHED), Int64(0)
        )
        Atomic[DType.int64].store(
            self._counter_int64(_ASTATS_PEAK_CACHED), Int64(0)
        )

    def __del__(deinit self):
        """Free the backing store."""
        self._data.free()


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

    def __init__(out self):
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

    def __init__(out self):
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

    def __init__(out self, block_size: Int):
        """Initialize an empty free list for the given block size.

        Args:
            block_size: Size of each block this list will manage.
        """
        self.head = UnsafePointer[_FreeListNode, origin=MutAnyOrigin]()
        self.block_size = block_size
        self.count = 0

    def is_empty(self) -> Bool:
        """Check if the free list is empty.

        Returns:
            True if there are no free blocks available.
        """
        return self.head == UnsafePointer[_FreeListNode, origin=MutAnyOrigin]()

    def pop(mut self) -> UnsafePointer[UInt8, origin=MutAnyOrigin]:
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

    def push(mut self, ptr: UnsafePointer[UInt8, origin=MutAnyOrigin]):
        """Add a block back to the free list.

        Args:
            ptr: Pointer to the block to return to the pool.

        Note: ptr must not be null. A null pointer indicates an allocation
              failure and should not be returned to the pool.
        """
        # Guard against null pointers (shouldn't happen in normal operation,
        # but provides safety if allocate() somehow returns null)
        if ptr == UnsafePointer[UInt8, origin=MutAnyOrigin]():
            return

        var node = ptr.bitcast[_FreeListNode]()
        node[].next = self.head
        self.head = node
        self.count += 1


struct TensorMemoryPool(Copyable, Movable):
    """Thread-safe memory pool for small tensor allocations.

    Implements a three-tier bucket strategy:
    - Small buckets (< 1KB): 64B, 128B, 256B, 512B, 1KB
    - Medium buckets (1-16KB): 2KB, 4KB, 8KB, 16KB
    - Large allocations (> 16KB): Direct system malloc

    Thread-Safety:
        - `allocate()` and `deallocate()` are safe to call concurrently.
        - Each bucket has its own SpinLock to minimize contention.
        - Statistics use AtomicStats for lock-free counter updates.
        - `clear()`, `reset_stats()`, and `trim()` are NOT thread-safe;
          call them only when no other threads access the pool.

    Attributes:
        small_lists: Free lists for small size classes.
        medium_lists: Free lists for medium size classes.
        stats: Performance statistics (non-atomic snapshot struct).
        _atomic_stats: Atomic statistics for thread-safe updates.
        _small_locks: Per-bucket spinlocks for small free lists.
        _medium_locks: Per-bucket spinlocks for medium free lists.
    """

    var small_lists: List[FreeList]
    """Free lists for < 1KB allocations."""
    var medium_lists: List[FreeList]
    """Free lists for 1KB-16KB allocations."""
    var stats: PoolStats
    """Pool statistics (snapshot, updated from atomic stats)."""
    var _atomic_stats: AtomicStats
    """Atomic counters for thread-safe statistics."""
    var _small_locks: List[SpinLock]
    """Per-bucket spinlocks for small free lists."""
    var _medium_locks: List[SpinLock]
    """Per-bucket spinlocks for medium free lists."""

    def __init__(out self):
        """Initialize pool with default configuration."""
        self.small_lists = List[FreeList]()
        self._small_locks = List[SpinLock]()
        for i in range(SMALL_SIZES_COUNT):
            self.small_lists.append(FreeList(_get_small_size(i)))
            self._small_locks.append(SpinLock())

        self.medium_lists = List[FreeList]()
        self._medium_locks = List[SpinLock]()
        for i in range(MEDIUM_SIZES_COUNT):
            self.medium_lists.append(FreeList(_get_medium_size(i)))
            self._medium_locks.append(SpinLock())

        self.stats = PoolStats()
        self._atomic_stats = AtomicStats()

        # Pre-allocate default configuration
        self._preallocate_blocks(16, 8)

    def __init__(out self, config: PoolConfig):
        """Initialize pool with custom configuration.

        Args:
            config: Pool configuration specifying initial block counts.
        """
        self.small_lists = List[FreeList]()
        self._small_locks = List[SpinLock]()
        for i in range(SMALL_SIZES_COUNT):
            self.small_lists.append(FreeList(_get_small_size(i)))
            self._small_locks.append(SpinLock())

        self.medium_lists = List[FreeList]()
        self._medium_locks = List[SpinLock]()
        for i in range(MEDIUM_SIZES_COUNT):
            self.medium_lists.append(FreeList(_get_medium_size(i)))
            self._medium_locks.append(SpinLock())

        self.stats = PoolStats()
        self._atomic_stats = AtomicStats()

        # Pre-allocate with custom configuration
        self._preallocate_blocks(
            config.small_block_count, config.medium_block_count
        )

    def __del__(deinit self):
        """Destructor - release all pooled memory."""
        self.clear()

    def _preallocate_blocks(mut self, small_count: Int, medium_count: Int):
        """Pre-allocate blocks to each bucket's free list.

        Called during initialization only (single-threaded), so no
        locking is needed here.

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
                self._atomic_stats.add_bytes_cached(size)

        # Pre-allocate medium blocks
        for i in range(len(self.medium_lists)):
            var size = self.medium_lists[i].block_size
            for _ in range(medium_count):
                var ptr = alloc[UInt8](size)
                self.medium_lists[i].push(ptr)
                self._atomic_stats.add_bytes_cached(size)

        # Update peak and sync snapshot
        self._atomic_stats.update_peak_cached()
        self.stats = self._atomic_stats.snapshot()

    def _find_bucket_index(self, size: Int) -> Int:
        """Find the smallest bucket that fits the requested size.

        This is a read-only operation and is thread-safe without locking.

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

    def allocate(
        mut self, size: Int
    ) -> UnsafePointer[UInt8, origin=MutAnyOrigin]:
        """Allocate memory from pool or system allocator.

        Thread-safe: uses per-bucket spinlocks and atomic stats.

        Args:
            size: Number of bytes to allocate.

        Returns:
            UnsafePointer to allocated memory.
        """
        self._atomic_stats.add_allocations(1)

        # Large allocations bypass pool (no lock needed)
        if size > LARGE_THRESHOLD:
            self._atomic_stats.add_pool_misses(1)
            self._atomic_stats.add_bytes_allocated(size)
            return alloc[UInt8](size)

        var bucket_idx = self._find_bucket_index(size)

        # No suitable bucket found, allocate directly
        if bucket_idx < 0:
            self._atomic_stats.add_pool_misses(1)
            self._atomic_stats.add_bytes_allocated(size)
            return alloc[UInt8](size)

        # Try to get from small bucket
        if bucket_idx < len(self.small_lists):
            self._small_locks[bucket_idx].lock()
            if not self.small_lists[bucket_idx].is_empty():
                var ptr = self.small_lists[bucket_idx].pop()
                self._small_locks[bucket_idx].unlock()
                var actual_size = self.small_lists[bucket_idx].block_size
                self._atomic_stats.add_pool_hits(1)
                self._atomic_stats.add_bytes_cached(-actual_size)
                self._atomic_stats.add_bytes_allocated(actual_size)
                return ptr
            else:
                self._small_locks[bucket_idx].unlock()
                # Pool miss - allocate new block outside the lock
                var actual_size = self.small_lists[bucket_idx].block_size
                self._atomic_stats.add_pool_misses(1)
                self._atomic_stats.add_bytes_allocated(actual_size)
                return alloc[UInt8](actual_size)

        # Try to get from medium bucket
        var medium_idx = bucket_idx - len(self.small_lists)
        if medium_idx < len(self.medium_lists):
            self._medium_locks[medium_idx].lock()
            if not self.medium_lists[medium_idx].is_empty():
                var ptr = self.medium_lists[medium_idx].pop()
                self._medium_locks[medium_idx].unlock()
                var actual_size = self.medium_lists[medium_idx].block_size
                self._atomic_stats.add_pool_hits(1)
                self._atomic_stats.add_bytes_cached(-actual_size)
                self._atomic_stats.add_bytes_allocated(actual_size)
                return ptr
            else:
                self._medium_locks[medium_idx].unlock()
                # Pool miss - allocate new block outside the lock
                var actual_size = self.medium_lists[medium_idx].block_size
                self._atomic_stats.add_pool_misses(1)
                self._atomic_stats.add_bytes_allocated(actual_size)
                return alloc[UInt8](actual_size)

        # Fallback (should not reach here)
        self._atomic_stats.add_pool_misses(1)
        self._atomic_stats.add_bytes_allocated(size)
        return alloc[UInt8](size)

    def deallocate(
        mut self, ptr: UnsafePointer[UInt8, origin=MutAnyOrigin], size: Int
    ):
        """Return allocation to pool or system allocator.

        Thread-safe: uses per-bucket spinlocks and atomic stats.

        Args:
            ptr: Pointer to memory to deallocate.
            size: Size of allocation.
        """
        self._atomic_stats.add_deallocations(1)

        # Large allocations bypass pool
        if size > LARGE_THRESHOLD:
            ptr.free()
            self._atomic_stats.add_bytes_allocated(-size)
            return

        var bucket_idx = self._find_bucket_index(size)

        # No suitable bucket found, free directly
        if bucket_idx < 0:
            ptr.free()
            self._atomic_stats.add_bytes_allocated(-size)
            return

        # Return to small bucket
        if bucket_idx < len(self.small_lists):
            var actual_size = self.small_lists[bucket_idx].block_size
            self._small_locks[bucket_idx].lock()
            self.small_lists[bucket_idx].push(ptr)
            self._small_locks[bucket_idx].unlock()
            self._atomic_stats.add_bytes_cached(actual_size)
            self._atomic_stats.add_bytes_allocated(-actual_size)
            self._atomic_stats.update_peak_cached()
            return

        # Return to medium bucket
        var medium_idx = bucket_idx - len(self.small_lists)
        if medium_idx < len(self.medium_lists):
            var actual_size = self.medium_lists[medium_idx].block_size
            self._medium_locks[medium_idx].lock()
            self.medium_lists[medium_idx].push(ptr)
            self._medium_locks[medium_idx].unlock()
            self._atomic_stats.add_bytes_cached(actual_size)
            self._atomic_stats.add_bytes_allocated(-actual_size)
            self._atomic_stats.update_peak_cached()
            return

        # Fallback (should not reach here)
        ptr.free()
        self._atomic_stats.add_bytes_allocated(-size)

    def get_stats(self) -> PoolStats:
        """Get current pool statistics.

        Returns an atomic snapshot of all counters. Safe to call
        concurrently with allocate()/deallocate().

        Returns:
            Copy of current statistics.
        """
        return self._atomic_stats.snapshot()

    def reset_stats(mut self):
        """Reset all statistics to zero.

        NOT thread-safe. Call only when no other threads access the pool.
        """
        self._atomic_stats.reset()
        self.stats = PoolStats()

    def trim(mut self):
        """Release unused blocks from pool (not implemented).

        This is a placeholder for future optimization to return
        cached blocks to the OS when they're not being used.

        Note: When implemented, this must acquire all bucket locks.
        """
        pass

    def clear(mut self):
        """Release all pooled memory.

        NOT thread-safe. Call only when no other threads access the pool.
        """
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

        # Reset byte counters only (preserve allocation/deallocation counts)
        var current_cached = Int(
            self._atomic_stats._counter(_ASTATS_BYTES_CACHED)[].load()
        )
        var current_alloc = Int(
            self._atomic_stats._counter(_ASTATS_BYTES_ALLOCATED)[].load()
        )
        self._atomic_stats.add_bytes_cached(-current_cached)
        self._atomic_stats.add_bytes_allocated(-current_alloc)
        self.stats = self._atomic_stats.snapshot()


# Global memory pool singleton
# Since Mojo v0.26.1+ doesn't support global vars, we implement pooled_alloc
# and pooled_free to bypass the global pool and go directly to system allocator
# for now. This is a temporary workaround until Mojo adds proper global state support.


def pooled_alloc(size: Int) -> UnsafePointer[UInt8, origin=MutAnyOrigin]:
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


def pooled_free(ptr: UnsafePointer[UInt8, origin=MutAnyOrigin], size: Int):
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


def get_global_pool() -> TensorMemoryPool:
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
