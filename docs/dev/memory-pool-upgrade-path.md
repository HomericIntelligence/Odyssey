# Memory Pool Upgrade Path

## Current State (Mojo 1.0)

The functions `pooled_alloc()` and `pooled_free()` in `src/odyssey/base/memory_pool.mojo`
bypass the `TensorMemoryPool` entirely and delegate directly to system `malloc`/`free`. This is a
temporary workaround because **Mojo 1.0 does not support global mutable state**.

### Why This Matters

The `TensorMemoryPool` class is fully implemented with:

- Three-tier bucket strategy for O(1) cache hits on common allocation sizes
- Spinlock-protected free lists for thread-safety
- Atomic statistics counters for allocation/deallocation tracking
- Large allocation bypass (>16KB goes directly to system allocator)

However, without a global singleton, there's no way to use these optimizations in the module-level
`pooled_alloc`/`pooled_free` functions.

### Current Workaround

Applications that want pooling must:

1. Create their own `TensorMemoryPool` instance
2. Pass it explicitly to code that needs allocation (not feasible for internal allocation)
3. Or accept the performance penalty of direct `malloc`/`free`

## Future State (When Mojo Adds Global Vars)

When Mojo adds support for `global var`, upgrading is a **one-line change** in three places:

### Upgrade Steps

#### 1. Add Global Pool Singleton

After line 760 in `src/odyssey/base/memory_pool.mojo`, replace the comment block with:

```mojo
# Global memory pool singleton - one per process
global_pool = TensorMemoryPool()
```

#### 2. Update pooled_alloc()

Replace the implementation at line ~785:

```mojo
# OLD (before)
return alloc[UInt8](size)

# NEW (after)
return global_pool.allocate(size)
```

#### 3. Update pooled_free()

Replace the implementation at line ~816:

```mojo
# OLD (before)
ptr.free()

# NEW (after)
global_pool.deallocate(ptr, size)
```

#### 4. Update get_global_pool()

Simplify the return statement to:

```mojo
# OLD (before)
return TensorMemoryPool()

# NEW (after)
return global_pool
```

### Verification After Upgrade

Run the test suite to verify the upgrade:

```bash
# Unit tests for memory pool behavior
just test-group "tests/odyssey/core" "test_memory_pool.mojo"

# Thread-safety tests
just test-group "tests/odyssey/core" "test_memory_pool_threadsafe.mojo"

# Full validation
just validate
```

Expected outcomes:

- All existing tests pass without modification
- `pooled_alloc()` will show `pool_hits > 0` in stats after repeated allocations
- `pooled_free()` will properly return blocks to buckets for reuse
- No performance regression (likely improvement from caching)

## Why This Limitation Exists

Mojo's memory model prevents unsafe global mutable state to preserve memory safety guarantees.
Once Mojo designs a safe mechanism (e.g., global immutable singletons, thread-local storage, or
explicit pool-passing APIs), this limitation will be lifted.

See issue #5132 and ADR-009 for related discussions on global state and memory safety.

## Alternative Patterns (Today)

Until global vars are supported, applications can:

1. **Scope-Limited Pooling**: Create a `TensorMemoryPool` for a specific operation or module scope
2. **Dependency Injection**: Pass a pool as a parameter through the call stack
3. **Thread-Local Storage**: Use platform-specific TLS if needed for per-thread pools

These patterns can be implemented immediately without waiting for Mojo language updates.
