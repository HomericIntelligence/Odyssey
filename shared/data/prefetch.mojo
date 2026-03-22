"""Prefetch buffer infrastructure for batching.

This module provides utilities for pre-computing batches ahead of consumption
to improve data pipeline efficiency. Since Mojo lacks async primitives and
thread-safe queues in the stdlib, this implementation uses synchronous
pre-computation at the start of each epoch.

Note on Async: True async prefetching would require Mojo stdlib additions
(TaskGroup, Queue, Mutex). The current synchronous approach provides benefits
by separating data preparation from training iterations. Future versions
can be upgraded to async when Mojo adds these primitives.

Example:
    from shared.data import AnyTensorDataset, BatchLoader, RandomSampler
    from shared.data.prefetch import PrefetchDataLoader

    var dataset = AnyTensorDataset(images, labels)
    var sampler = RandomSampler(dataset.__len__())
    var loader = BatchLoader(dataset, sampler, batch_size=32)
    var prefetch_loader = PrefetchDataLoader(loader, prefetch_factor=2)
"""

from shared.core import AnyTensor
from shared.data.loaders import Batch, BatchLoader
from shared.data.datasets import Dataset
from shared.data.samplers import Sampler


struct PrefetchBuffer(Copyable, Movable):
    """Ring buffer for holding pre-computed batches.

    Manages a fixed-size buffer of batches. Currently uses synchronous
    pre-computation. Can be upgraded to async when Mojo adds Task primitives.

    Note: Batch storage requires careful memory management since AnyTensor
    uses heap allocation. This buffer owns the batches until consumption.
    """

    var batches: List[Batch]
    """Pre-computed batches."""
    var capacity: Int
    """Maximum number of batches to buffer."""

    fn __init__(out self, capacity: Int = 2) raises:
        """Create a prefetch buffer.

        Args:
            capacity: Maximum number of batches to hold (default 2).

        Raises:
            Error: If capacity is not positive.
        """
        if capacity <= 0:
            raise Error(
                "Prefetch capacity must be positive, got: " + String(capacity)
            )

        self.batches = List[Batch]()
        self.capacity = capacity

    fn is_full(self) -> Bool:
        """Check if buffer has reached capacity.

        Returns:
            True if buffer has capacity batches, False otherwise.
        """
        return self.batches.__len__() >= self.capacity

    fn is_empty(self) -> Bool:
        """Check if buffer is empty.

        Returns:
            True if no batches in buffer, False otherwise.
        """
        return self.batches.__len__() == 0

    fn append(mut self, var batch: Batch) raises:
        """Add a batch to the buffer.

        Args:
            batch: Batch to add.

        Raises:
            Error: If buffer is full.
        """
        if self.is_full():
            raise Error(
                "Prefetch buffer is full (capacity="
                + String(self.capacity)
                + ")"
            )

        self.batches.append(batch^)

    fn pop(mut self) raises -> Batch:
        """Remove and return the first batch in buffer.

        Returns:
            First batch in buffer.

        Raises:
            Error: If buffer is empty.
        """
        if self.is_empty():
            raise Error("Prefetch buffer is empty")

        return self.batches.pop(0)

    fn clear(mut self):
        """Clear all batches from buffer."""
        self.batches = List[Batch]()


struct PrefetchDataLoader[
    D: Dataset & Copyable & Movable, S: Sampler & Copyable & Movable
](Copyable, Movable):
    """Data loader with batch prefetching.

    Wraps a BatchLoader and pre-computes batches ahead of consumption.
    The prefetch factor controls how many batches are pre-computed.

    Since Mojo lacks async/threading, batches are pre-computed synchronously
    at the start of iteration. This still provides benefits by:
    - Separating data loading logic from training logic
    - Allowing easy future async upgrades
    - Improving code structure and readability

    Parameters:
        D: Dataset type conforming to Dataset trait.
        S: Sampler type conforming to Sampler trait.

    Example:
        ```mojo
        var dataset = AnyTensorDataset(images, labels)
        var sampler = RandomSampler(dataset.__len__())
        var loader = BatchLoader(dataset, sampler, batch_size=32)

        # Pre-compute 2 batches ahead (default)
        var prefetch = PrefetchDataLoader(loader, prefetch_factor=2)
        var batches = prefetch.__iter__()
        ```
    """

    var base_loader: BatchLoader[Self.D, Self.S]
    """Underlying batch loader."""
    var prefetch_factor: Int
    """Number of batches to pre-compute."""

    fn __init__(
        out self,
        var loader: BatchLoader[Self.D, Self.S],
        prefetch_factor: Int = 2,
    ) raises:
        """Create a prefetch data loader.

        Args:
            loader: Base BatchLoader to wrap.
            prefetch_factor: Number of batches to pre-compute (default 2).

        Raises:
            Error: If prefetch_factor is invalid.
        """
        if prefetch_factor <= 0:
            raise Error(
                "Prefetch factor must be positive, got: "
                + String(prefetch_factor)
            )

        self.base_loader = loader^
        self.prefetch_factor = prefetch_factor

    fn __iter__(mut self) raises -> List[Batch]:
        """Pre-compute and return all batches for this epoch.

        Returns:
            List of all batches with pre-computation applied.

        Note: Currently performs synchronous pre-computation. Future versions
        can upgrade to async when Mojo adds Task primitives.
        """
        # Get all batches from base loader
        # Since Mojo lacks lazy iterators, we must get all batches upfront
        return self.base_loader.__iter__()

    fn _prefetch_batches(
        mut self, var batches: List[Batch], prefetch_factor: Int
    ) raises -> List[Batch]:
        """Process batches with prefetch awareness.

        This internal method can be used to implement custom prefetching logic.
        Currently returns batches as-is since true async isn't available.

        Args:
            batches: List of batches to process (ownership transferred).
            prefetch_factor: Prefetch buffer size.

        Returns:
            Processed batches (same as input in current implementation).
        """
        # In a true async implementation, this would:
        # 1. Create a PrefetchBuffer with the given prefetch_factor
        # 2. Start a background task to populate the buffer
        # 3. Return an iterator that consumes from the buffer
        #
        # For now, we just return the batches as-is since Mojo lacks
        # async primitives. The structural improvements still apply:
        # - Separation of concerns between data loading and training
        # - Clear interface for future async upgrades
        # - Benchmark point for measuring any async benefits

        return batches^
