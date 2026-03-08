# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_cache.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for CachedDataset wrapper (part 2 of 2).

Tests cache enable/disable, hit rate, and statistics.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
)
from shared.data import ExTensorDataset, CachedDataset
from shared.core.extensor import ExTensor, ones, zeros
from collections import List


# ============================================================================
# CachedDataset Enable/Disable and Statistics Tests
# ============================================================================


fn test_cached_dataset_enable_disable() raises:
    """Test enabling and disabling cache.

    Should be able to toggle caching on/off.
    """
    var data_shape: List[Int] = [5, 1, 8, 8]
    var label_shape: List[Int] = [5, 10]

    var data = ones(data_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    var base_dataset = ExTensorDataset(data^, labels^)
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

    var base_dataset = ExTensorDataset(data^, labels^)
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

    var base_dataset = ExTensorDataset(data^, labels^)
    var cached = CachedDataset(base_dataset^, max_cache_size=-1)

    var _d1, _l1 = cached._get_and_cache(0)
    var _d2, _l2 = cached._get_and_cache(0)

    var cache_size, hits, misses = cached.get_cache_stats()

    assert_equal(cache_size, 1)
    assert_equal(hits, 1)
    assert_equal(misses, 1)


fn main() raises:
    """Run all tests."""
    print("Testing CachedDataset (part 2)...")
    print("  test_cached_dataset_enable_disable...")
    test_cached_dataset_enable_disable()
    print("  test_cached_dataset_hit_rate...")
    test_cached_dataset_hit_rate()
    print("  test_cached_dataset_get_stats...")
    test_cached_dataset_get_stats()
    print("All CachedDataset part 2 tests passed!")
