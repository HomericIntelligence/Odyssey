# ADR-003: Memory Pool Architecture for Small Tensor Allocations

**Status**: Accepted (Deferred)

**Date**: 2025-12-28

**Decision Owner**: Chief Architect

## Executive Summary

This ADR documents the memory pool architecture for small tensor allocations in ML Odyssey. The
pool uses a three-tier bucket strategy to reduce allocation overhead for small tensors common in
gradient computation and intermediate layer outputs.

## Context

### Problem Statement

ML training workloads create and destroy many small tensors during forward and backward passes.
Standard system allocators (malloc/free) have overhead that becomes significant when handling
thousands of small allocations per training iteration:

- Gradient tensors for individual weights
- Intermediate activation tensors
- Small temporary buffers for operations

### Performance Impact

Without pooling:

- Each allocation requires system call overhead
- Memory fragmentation accumulates over time
- Cache locality suffers from scattered allocations
- GC pressure increases with allocation frequency

### Requirements

1. **Low Overhead**: Allocation should be O(1) for common sizes
2. **Reuse**: Previously freed blocks should be reused
3. **Configurable**: Support different workload patterns
4. **Statistics**: Track allocation patterns for debugging
5. **Safe Fallback**: Large allocations bypass pool

## Decision

### Three-Tier Bucket Strategy

Implement a memory pool with three tiers based on allocation size:

**Tier 1: Small Buckets (64B - 1KB)**

- Bucket sizes: 64B, 128B, 256B, 512B, 1KB
- Pre-allocated blocks per bucket: 16 (configurable)
- Target: Individual gradient scalars, small vectors

**Tier 2: Medium Buckets (2KB - 16KB)**

- Bucket sizes: 2KB, 4KB, 8KB, 16KB
- Pre-allocated blocks per bucket: 8 (configurable)
- Target: Typical gradient tensors, batch normalization stats

**Tier 3: Large Allocations (>16KB)**

- Bypass pool entirely
- Direct system malloc/free
- Target: Large weight matrices, activation maps

### Data Structures

```mojo
struct TensorMemoryPool:
    """Memory pool for small tensor allocations."""
    var small_lists: List[FreeList]   # 5 free lists for <1KB
    var medium_lists: List[FreeList]  # 4 free lists for 1-16KB
    var stats: PoolStats              # Performance tracking

struct FreeList:
    """Intrusive free list for pooled blocks."""
    var head: UnsafePointer[_FreeListNode]
    var block_size: Int
    var count: Int
```

### Allocation Strategy

1. **Size Classification**: Find smallest bucket that fits requested size
2. **Pool Hit**: If free list has blocks, pop from list (O(1))
3. **Pool Miss**: If free list empty, allocate new block from system
4. **Large Bypass**: Sizes >16KB go directly to system allocator

### Deallocation Strategy

1. **Size Classification**: Determine which bucket the block belongs to
2. **Return to Pool**: Push block to appropriate free list (O(1))
3. **Large Bypass**: Sizes >16KB freed directly to system

### Current Limitation

Mojo v0.26.1+ does not support global mutable state. The pool infrastructure is fully
implemented, but `pooled_alloc()` and `pooled_free()` currently bypass the pool and use
direct malloc/free. This will be enabled when Mojo v0.26+ adds global variable support.

```mojo
fn pooled_alloc(size: Int) -> UnsafePointer[UInt8]:
    # Temporary: Direct malloc until global pool available
    return alloc[UInt8](size)

fn pooled_free(ptr: UnsafePointer[UInt8], size: Int):
    # Temporary: Direct free until global pool available
    ptr.free()
```

## Rationale

### Why Bucket Sizes?

The bucket sizes (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384) were chosen based on:

1. **Power-of-two alignment**: Cache-friendly allocation boundaries
2. **Common tensor sizes**: Matches typical gradient and activation sizes
3. **Internal fragmentation**: Maximum 50% waste (allocating 65B uses 128B bucket)

### Why Free Lists?

Free lists provide O(1) allocation and deallocation:

1. **Intrusive design**: Node stored in freed block itself (no extra memory)
2. **LIFO ordering**: Recently freed blocks are warmer in cache
3. **Simple implementation**: Minimal complexity, easy to verify

### Why 16KB Threshold?

Large allocations (>16KB) bypass the pool because:

1. **Amortization**: System allocator overhead is proportionally smaller
2. **Fragmentation**: Large blocks are harder to reuse efficiently
3. **Memory pressure**: Caching large blocks consumes excessive memory

## Consequences

### Positive

- **Reduced allocation overhead**: O(1) for pooled sizes
- **Memory reuse**: Blocks returned to pool for future use
- **Performance tracking**: Statistics enable workload analysis
- **Configurable**: Pre-allocation counts are adjustable
- **Safe design**: Large allocations handled correctly

### Negative

- **Internal fragmentation**: Up to 50% waste within buckets
- **Memory overhead**: Pre-allocated blocks consume memory upfront
- **Complexity**: More code than direct malloc/free
- **Global state blocked**: Full pooling awaits Mojo v0.26+

### Neutral

- **Local pools work**: Per-module or per-function pools work today
- **Statistics available**: PoolStats tracks all allocation patterns

## Alternatives Considered

### Alternative 1: No Pooling (Direct Malloc)

**Description**: Use system allocator for all allocations.

**Pros**:

- Simple implementation
- No memory overhead
- Works with all allocation sizes

**Cons**:

- System call overhead per allocation
- Memory fragmentation over time
- No allocation tracking

**Why Rejected**: Overhead too high for ML workload patterns.

### Alternative 2: Arena Allocator

**Description**: Allocate from contiguous memory regions, free all at once.

**Pros**:

- Very fast allocation (bump pointer)
- No fragmentation
- Simple deallocation (reset pointer)

**Cons**:

- Cannot free individual allocations
- Requires clear lifecycle boundaries
- May hold memory longer than needed

**Why Rejected**: ML training requires fine-grained deallocation.

### Alternative 3: Slab Allocator (Fixed Size Only)

**Description**: Single fixed-size allocator per type.

**Pros**:

- Zero internal fragmentation
- Optimal for uniform sizes

**Cons**:

- Requires separate allocator per size
- Complex to manage multiple allocators

**Why Rejected**: Multiple buckets achieve similar benefit with less complexity.

## Implementation Details

### File Location

`shared/core/memory_pool.mojo`

### Key Functions

```mojo
# Public API
fn pooled_alloc(size: Int) -> UnsafePointer[UInt8]
fn pooled_free(ptr: UnsafePointer[UInt8], size: Int)
fn get_global_pool() -> TensorMemoryPool

# Pool methods
fn allocate(mut self, size: Int) -> UnsafePointer[UInt8]
fn deallocate(mut self, ptr: UnsafePointer[UInt8], size: Int)
fn get_stats(self) -> PoolStats
fn trim(mut self)  # Release unused blocks
fn clear(mut self) # Release all pooled memory
```

### Statistics Tracking

```mojo
struct PoolStats:
    var allocations: Int       # Total allocate() calls
    var deallocations: Int     # Total deallocate() calls
    var pool_hits: Int         # Served from pool cache
    var pool_misses: Int       # Required system malloc
    var bytes_allocated: Int   # Currently allocated bytes
    var bytes_cached: Int      # Currently in pool
    var peak_cached_bytes: Int # High water mark
```

### Configuration

```mojo
struct PoolConfig:
    var small_block_count: Int   # Initial blocks per small bucket (default: 16)
    var medium_block_count: Int  # Initial blocks per medium bucket (default: 8)
    var max_cached_bytes: Int    # Maximum cache before trim (default: 16MB)
```

## Future Considerations

### When Mojo Supports Global State

1. Enable global pool singleton
2. Update `pooled_alloc()` and `pooled_free()` to use global pool
3. Add thread-safety if needed for parallel training

### Potential Optimizations

- Thread-local pools for parallel training
- Adaptive bucket sizing based on workload
- Memory pressure callbacks for system integration

## References

### Related Files

- `shared/core/memory_pool.mojo`: Implementation
- `tests/shared/core/test_memory_pool.mojo`: Unit tests
- `tests/shared/benchmarks/bench_memory_pool.mojo`: Performance benchmarks

### Related Issues

- Issue #2548: Memory management infrastructure

### External Documentation

- [jemalloc Design](https://jemalloc.net/jemalloc.3.html): Inspiration for bucket sizing
- [TCMalloc](https://google.github.io/tcmalloc/): Thread-caching allocator patterns

## Revision History

| Version | Date       | Author          | Changes     |
| ------- | ---------- | --------------- | ----------- |
| 1.0     | 2025-12-28 | Chief Architect | Initial ADR |

---

## Document Metadata

- **Location**: `/docs/adr/ADR-003-memory-pool-architecture.md`
- **Status**: Accepted (Deferred)
- **Review Frequency**: As-needed
- **Next Review**: When Mojo v0.26+ releases
- **Supersedes**: None
- **Superseded By**: None
