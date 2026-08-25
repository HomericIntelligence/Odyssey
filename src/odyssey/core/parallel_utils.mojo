"""Parallel processing utilities for batch operations.

This module provides adaptive batch execution based on batch size,
avoiding overhead for small batches while enabling parallelism
for large batches.

Note (Mojo 1.0.0): `std.algorithm.parallelize` was removed in the stable
release; the replacement lives in the `max` package, which is not bundled
with the `mojo` PyPI wheel. This module re-implements the same contract on
top of the stdlib's async runtime (`std.runtime.asyncrt.TaskGroup`): work
items are distributed across the runtime's worker threads and executed in
parallel, matching the semantics of the removed `std.algorithm.parallelize`.

FM-G caveat (modular/modular#6965): on Mojo 1.0.0, passing a capturing
closure whose captures own heap memory as a function-value parameter
destroys those captures at closure-construction time (premature
`__deinit__` -> use-after-free). `parallelize` is therefore only safe when
the closure captures scalars, or when every owned capture is used again in
the enclosing frame after the call (keep-alive). Heap-owning captures that
are not re-used afterwards (e.g. tensors) must dispatch inline via
`TaskGroup` instead — see the batched paths in `pooling.mojo`, `conv.mojo`,
and `normalization.mojo` for the canonical inline pattern.
"""

from std.runtime.asyncrt import TaskGroup, parallelism_level

# Minimum batch size to warrant parallelization
comptime PARALLEL_BATCH_THRESHOLD: Int = 4

# Default worker count (0 = runtime decides via parallelism_level)
comptime DEFAULT_NUM_WORKERS: Int = 0


def should_parallelize(
    batch_size: Int, threshold: Int = PARALLEL_BATCH_THRESHOLD
) -> Bool:
    """Determine if batch size warrants parallel execution.

    Args:
        batch_size: Number of batch elements.
        threshold: Minimum batch size for parallelization.

    Returns:
        True if parallelization is beneficial.
    """
    return batch_size >= threshold


def parallelize[
    func: def(Int) capturing -> None
](num_work_items: Int, var num_workers: Int = DEFAULT_NUM_WORKERS):
    """Execute function across work-item indices in parallel.

    Re-implements the `std.algorithm.parallelize` removed in Mojo 1.0.0
    (its replacement ships in the `max` package) using the stdlib async
    runtime's `TaskGroup`. Work items are coalesced into contiguous chunks
    (one per worker) and executed as coroutine tasks on the runtime's
    worker threads; `wait()` blocks until all chunks complete.

    Parameters:
        func: Function to execute for each work-item index.
    Args:
        num_work_items: Number of work items.
        num_workers: Number of worker threads (0 = runtime decides).

    Note:
        See the module docstring for the FM-G caveat (modular/modular#6965):
        do not pass closures that capture heap-owning locals unless they are
        kept alive afterwards.
    """
    if num_work_items <= 0:
        return

    if num_workers <= 0:
        num_workers = parallelism_level()
    if num_workers > num_work_items:
        num_workers = num_work_items

    # Single worker (or single work item): run inline, avoid task overhead.
    if num_workers == 1:
        for i in range(num_work_items):
            func(i)
        return

    var chunk_size, extra_items = divmod(num_work_items, num_workers)

    @parameter
    async def worker(thread_idx: Int):
        var start_idx = thread_idx * chunk_size + min(thread_idx, extra_items)
        for i in range(chunk_size + Int(thread_idx < extra_items)):
            func(start_idx + i)

    var tg = TaskGroup()
    for t in range(num_workers):
        tg.create_task(worker(t))
    tg.wait()


def parallel_for_batch[
    func: def(Int) capturing -> None
](batch_size: Int, num_workers: Int = DEFAULT_NUM_WORKERS):
    """Execute function across batch indices.

    Parameters:
        func: Function to execute for each batch index.
    Args:
        batch_size: Number of batch elements.
        num_workers: Number of worker threads (0 = auto).
    """
    parallelize[func](batch_size, num_workers)
