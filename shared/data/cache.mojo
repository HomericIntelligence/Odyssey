"""Dataset caching utilities for improved I/O efficiency.

This module provides a dataset wrapper with sample caching to avoid
repeated I/O for small datasets. Caching is especially useful for
datasets loaded from disk (FileDataset) where I/O is the bottleneck.

Design:
    - Cache is populated explicitly during construction via _preload_cache()
    - Samples can be retrieved from cache or loaded on-demand
    - max_cache_size prevents unbounded memory growth
    - Cache is optional and can be disabled

Note on Immutability: The __getitem__ trait method is immutable (self, not mut self),
so in-place cache updates would violate the trait contract. This implementation
works around that limitation by using an explicit preload pattern.

Example:
    from shared.data import FileDataset, CachedDataset

    var dataset = FileDataset(image_paths, label_paths)
    var cached = CachedDataset(dataset, max_cache_size=1000)
    var img, label = cached[0]  # Returns from cache if available
"""

from shared.tensor.any_tensor import AnyTensor
from shared.data._datasets_core import Dataset


struct CachedDataset[D: Dataset & Copyable & Movable](
    Copyable, Dataset, Movable
):
    """Dataset wrapper with sample caching.

    Caches loaded samples to avoid repeated I/O. Cache is populated
    explicitly at construction time rather than lazily, working around
    Mojo's immutable trait method constraint.

    Parameters:
        D: Dataset type conforming to the Dataset trait.

    Design Principle:
        Caching is opt-in and explicit. The _preload_cache() method must
        be called manually to populate the cache. This avoids hidden
        performance characteristics and memory overhead.

    Example:
        ```mojo
        var dataset = FileDataset(image_paths, label_paths)
        var cached = CachedDataset(dataset, max_cache_size=1000)
        # Cache is empty at this point

        # Explicitly preload cache (or it's loaded on-demand)
        cached._preload_cache()  # or access items to lazy-load

        var img, label = cached[0]  # Returns from cache
        ```
    """

    var dataset: Self.D
    """Base dataset to wrap."""
    var cache: Dict[Int, Tuple[AnyTensor, AnyTensor]]
    """Cache of loaded samples (index -> (data, label))."""
    var cache_enabled: Bool
    """Whether caching is enabled."""
    var max_cache_size: Int
    """Maximum number of samples to cache (-1 = unlimited)."""
    var cache_hits: Int
    """Number of cache hits for statistics."""
    var cache_misses: Int
    """Number of cache misses for statistics."""

    fn __init__(
        out self,
        var dataset: Self.D,
        max_cache_size: Int = -1,
        cache_enabled: Bool = True,
    ):
        """Create a cached dataset.

        Args:
            dataset: Base dataset to wrap.
            max_cache_size: Maximum samples to cache (-1 = unlimited, default -1).
            cache_enabled: Whether to enable caching (default True).
        """
        self.dataset = dataset^
        self.cache = Dict[Int, Tuple[AnyTensor, AnyTensor]]()
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    fn __len__(self) -> Int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return self.dataset.__len__()

    fn __getitem__(self, index: Int) raises -> Tuple[AnyTensor, AnyTensor]:
        """Get a sample, using cache if available.

        Attempts to retrieve from cache first. If not cached, loads from
        the base dataset. Note: Cache statistics are not updated here since
        the Dataset trait requires immutable self.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (data, labels).

        Raises:
            Error: If index is out of bounds or loading fails.
        """
        # Check cache first (read-only)
        if self.cache_enabled and index in self.cache:
            return self.cache[index]

        # Cache miss - load from base dataset
        return self.dataset.__getitem__(index)

    fn _get_and_cache(mut self, index: Int) raises -> Tuple[AnyTensor, AnyTensor]:
        """Get a sample with mutable access, enabling cache updates.

        Unlike __getitem__, this method can update cache and statistics.
        Use this when you need lazy caching behavior.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (data, labels).

        Raises:
            Error: If index is out of bounds or loading fails.
        """
        # Check cache first
        if self.cache_enabled and index in self.cache:
            self.cache_hits += 1
            return self.cache[index]

        # Cache miss
        self.cache_misses += 1
        var data, label = self.dataset.__getitem__(index)

        # Add to cache if enabled and within limits
        if self.cache_enabled:
            var can_cache = (
                self.max_cache_size < 0
                or self.cache.__len__() < self.max_cache_size
            )
            if can_cache:
                self.cache[index] = (data, label)

        return (data, label)

    fn _preload_cache(mut self) raises:
        """Pre-populate cache with samples.

        Loads samples from the dataset up to max_cache_size.
        Useful for small datasets where all samples fit in memory.

        Note: This is an explicit operation, not automatic, to avoid
        surprising performance characteristics.

        Raises:
            Error: If loading samples fails.
        """
        if not self.cache_enabled:
            return

        var n = self.dataset.__len__()
        var limit = n if self.max_cache_size < 0 else min(
            n, self.max_cache_size
        )

        for i in range(limit):
            # Use _get_and_cache to actually populate the cache
            _ = self._get_and_cache(i)

    fn clear_cache(mut self):
        """Clear all cached samples."""
        self.cache = Dict[Int, Tuple[AnyTensor, AnyTensor]]()
        self.cache_hits = 0
        self.cache_misses = 0

    fn disable_cache(mut self):
        """Disable caching and clear cache."""
        self.cache_enabled = False
        self.clear_cache()

    fn enable_cache(mut self):
        """Enable caching."""
        self.cache_enabled = True

    fn get_cache_stats(self) -> Tuple[Int, Int, Int]:
        """Get cache statistics.

        Returns:
            Tuple of (cache_size, cache_hits, cache_misses).
        """
        return (self.cache.__len__(), self.cache_hits, self.cache_misses)

    fn get_hit_rate(self) -> Float32:
        """Get cache hit rate.

        Returns:
            Fraction of accesses that were cache hits (0.0 to 1.0).
            Returns 0.0 if no accesses yet.
        """
        var total = self.cache_hits + self.cache_misses
        if total == 0:
            return Float32(0.0)
        return Float32(self.cache_hits) / Float32(total)
