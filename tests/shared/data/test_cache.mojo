"""Tests for CachedDataset wrapper (part 1 of 2).

Tests that CachedDataset correctly caches samples and respects limits.
"""


from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)
from shared.data.datasets import AnyTensorDataset
from shared.data.cache import CachedDataset
from shared.tensor.any_tensor import AnyTensor, ones, zeros
from collections import List


fn test_cached_dataset_creation() raises:
    """Test creating CachedDataset.

    CachedDataset should wrap a base dataset.
    """
    var data_shape: List[Int] = [10, 3, 32, 32]
    var label_shape: List[Int] = [10, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    assert_equal(cached.__len__(), 10)
    assert_true(cached.cache_enabled)


fn test_cached_dataset_length() raises:
    """Test that CachedDataset.__len__ matches base dataset.

    The length should reflect the number of samples.
    """
    var data_shape: List[Int] = [42, 3, 32, 32]
    var label_shape: List[Int] = [42, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    assert_equal(cached.__len__(), 42)


fn test_cached_dataset_stores_samples() raises:
    """Test that CachedDataset stores samples in cache.

    Accessing a sample via _get_and_cache should add it to the cache.
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    # Access a sample via mutable method
    var _, _ = cached._get_and_cache(0)

    # Check that it's in cache
    assert_equal(cached.cache.__len__(), 1)


fn test_cached_dataset_cache_hit() raises:
    """Test that cache hits are tracked.

    Multiple accesses to same sample should count as cache hits.
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    # First access - cache miss
    var _d1, _l1 = cached._get_and_cache(0)
    assert_equal(cached.cache_hits, 0)
    assert_equal(cached.cache_misses, 1)

    # Second access - cache hit
    var _d2, _l2 = cached._get_and_cache(0)
    assert_equal(cached.cache_hits, 1)
    assert_equal(cached.cache_misses, 1)


fn test_cached_dataset_max_cache_size() raises:
    """Test that max_cache_size limits number of cached samples.

    When cache size reaches limit, new samples shouldn't be added.
    """
    var data_shape: List[Int] = [10, 1, 8, 8]
    var label_shape: List[Int] = [10, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=3)

    # Access 5 samples via mutable method
    for i in range(5):
        var _d, _l = cached._get_and_cache(i)

    # Cache size should be limited to 3
    assert_equal(cached.cache.__len__(), 3)


fn test_cached_dataset_disabled_cache() raises:
    """Test that caching can be disabled.

    When cache_enabled=False, nothing should be cached.
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(
        base_dataset^, max_cache_size=-1, cache_enabled=False
    )

    var _d, _l = cached._get_and_cache(0)

    # Cache should be empty
    assert_true(cached.cache.__len__() == 0)


fn test_cached_dataset_preload_cache() raises:
    """Test preloading entire cache.

    _preload_cache should populate cache with all samples.
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    cached._preload_cache()

    # Cache should have all 5 samples
    assert_equal(cached.cache.__len__(), 5)


fn test_cached_dataset_clear_cache() raises:
    """Test clearing the cache.

    clear_cache should remove all cached samples.
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    # Add to cache
    var _d, _l = cached._get_and_cache(0)
    assert_equal(cached.cache.__len__(), 1)

    # Clear
    cached.clear_cache()
    assert_equal(cached.cache.__len__(), 0)


fn test_cached_dataset_enable_disable() raises:
    """Test enabling and disabling cache.

    Should be able to toggle caching on/off.
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    # Cache is enabled by default
    assert_true(cached.cache_enabled)

    # Disable
    cached.disable_cache()
    assert_true(not cached.cache_enabled)

    # Enable again
    cached.enable_cache()
    assert_true(cached.cache_enabled)


fn test_cached_dataset_hit_rate() raises:
    """Test cache hit rate calculation.

    Hit rate should be hits / (hits + misses).
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    # No accesses yet - should return 0.0
    assert_equal(cached.get_hit_rate(), Float32(0.0))

    # 2 accesses to same sample - 1 hit, 1 miss
    var _d1, _l1 = cached._get_and_cache(0)
    var _d2, _l2 = cached._get_and_cache(0)

    var hit_rate = cached.get_hit_rate()
    assert_almost_equal(hit_rate, Float32(0.5), Float32(0.01))


fn test_cached_dataset_get_stats() raises:
    """Test cache statistics.

    get_cache_stats should return (cache_size, hits, misses).
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = AnyTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    var _d1, _l1 = cached._get_and_cache(0)
    var _d2, _l2 = cached._get_and_cache(0)

    var cache_size, hits, misses = cached.get_cache_stats()

    assert_equal(cache_size, 1)
    assert_equal(hits, 1)
    assert_equal(misses, 1)


fn main() raises:
    """Run all test_cache tests."""
    print("Running test_cache tests...")

    test_cached_dataset_creation()
    print("✓ test_cached_dataset_creation")

    test_cached_dataset_length()
    print("✓ test_cached_dataset_length")

    test_cached_dataset_stores_samples()
    print("✓ test_cached_dataset_stores_samples")

    test_cached_dataset_cache_hit()
    print("✓ test_cached_dataset_cache_hit")

    test_cached_dataset_max_cache_size()
    print("✓ test_cached_dataset_max_cache_size")

    test_cached_dataset_disabled_cache()
    print("✓ test_cached_dataset_disabled_cache")

    test_cached_dataset_preload_cache()
    print("✓ test_cached_dataset_preload_cache")

    test_cached_dataset_clear_cache()
    print("✓ test_cached_dataset_clear_cache")

    test_cached_dataset_enable_disable()
    print("✓ test_cached_dataset_enable_disable")

    test_cached_dataset_hit_rate()
    print("✓ test_cached_dataset_hit_rate")

    test_cached_dataset_get_stats()
    print("✓ test_cached_dataset_get_stats")

    print("\nAll test_cache tests passed!")
