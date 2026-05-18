"""Shared utilities for sampler implementations.

Provides common patterns for index generation, random operations, and
weight-based sampling across different sampler types.
"""

from std.random import random_si64, seed


# ============================================================================
# Common Index Generation Utilities
# ============================================================================


def validate_range(
    start_index: Int, end_index: Int, data_source_len: Int
) -> Tuple[Int, Int]:
    """Validate and normalize range parameters for index generation.

    Args:
            start_index: Starting index (inclusive).
            end_index: Ending index (exclusive), -1 for end of dataset.
            data_source_len: Total length of dataset.

    Returns:
            Tuple of (normalized_start, normalized_end).
    """
    var normalized_start = max(0, start_index)
    var normalized_end: Int

    if end_index == -1:
        normalized_end = data_source_len
    else:
        normalized_end = min(data_source_len, end_index)

    return Tuple[Int, Int](normalized_start, normalized_end)


def create_sequential_indices(capacity: Int, start_index: Int = 0) -> List[Int]:
    """Create sequential indices for range iteration.

    Args:
            capacity: Number of indices to generate.
            start_index: Starting value for indices.

    Returns:
            List of sequential indices [start_index, start_index+1, ..., start_index+capacity-1].
    """
    var indices = List[Int](capacity=capacity)
    for i in range(capacity):
        indices.append(start_index + i)
    return indices^


def create_range_indices(start_index: Int, end_index: Int) -> List[Int]:
    """Create sequential indices for a range.

    Args:
            start_index: Starting index (inclusive).
            end_index: Ending index (exclusive).

    Returns:
            List of sequential indices [start_index, start_index+1, ..., end_index-1].
    """
    var capacity = end_index - start_index
    var indices = List[Int](capacity=capacity)
    for i in range(start_index, end_index):
        indices.append(i)
    return indices^


# ============================================================================
# Random Seed Management
# ============================================================================


def set_random_seed(seed_value: Int):
    """Set random seed if provided.

    Args:
            seed_value: Seed value (-1 = no seed, randomness is not seeded).
    """
    if seed_value >= 0:
        seed(seed_value)


# ============================================================================
# Fisher-Yates Shuffle Implementation
# ============================================================================


def shuffle_indices(var indices: List[Int]) -> List[Int]:
    """Shuffle indices using Fisher-Yates algorithm.

    Args:
            indices: List of indices to shuffle.

    Returns:
            Shuffled list of indices.
    """
    var size = len(indices)
    if size <= 1:
        return indices^

    # Fisher-Yates shuffle: iterate from end to beginning
    for i in range(size - 1, 0, -1):
        var j = Int(random_si64(Int64(0), Int64(i)))
        var temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp

    return indices^


# ============================================================================
# Random Sampling with/without Replacement
# ============================================================================


def sample_with_replacement(
    data_source_len: Int, num_samples: Int
) -> List[Int]:
    """Generate indices by sampling with replacement.

    Args:
            data_source_len: Size of dataset to sample from.
            num_samples: Number of samples to draw.

    Returns:
            List of sampled indices (may contain duplicates).
    """
    var indices = List[Int](capacity=num_samples)
    for _ in range(num_samples):
        indices.append(Int(random_si64(Int64(0), Int64(data_source_len - 1))))
    return indices^


def sample_without_replacement(
    data_source_len: Int, num_samples: Int
) -> List[Int]:
    """Generate indices by sampling without replacement.

        Creates a shuffled permutation and returns first num_samples indices.

    Args:
            data_source_len: Size of dataset to sample from.
            num_samples: Number of samples to draw (must be <= data_source_len).

    Returns:
            List of sampled indices (no duplicates).
    """
    # Create full list of indices
    var all_indices = create_sequential_indices(data_source_len)

    # Shuffle all indices
    all_indices = shuffle_indices(all_indices^)

    # Take first num_samples
    var indices = List[Int](capacity=min(num_samples, data_source_len))
    for i in range(min(num_samples, data_source_len)):
        indices.append(all_indices[i])

    return indices^


# ============================================================================
# Weighted Sampling with Cumulative Distribution
# ============================================================================


def build_cumulative_weights(weights: List[Float64]) -> List[Float64]:
    """Build cumulative distribution from weights.

    Args:
            weights: Normalized weight values.

    Returns:
            Cumulative sum of weights for binary search sampling.
    """
    var cumsum = List[Float64](capacity=len(weights))
    var total = Float64(0)
    for i in range(len(weights)):
        total += weights[i]
        cumsum.append(total)
    return cumsum^


def sample_index_from_distribution(
    cumsum: List[Float64], random_value: Float64
) -> Int:
    """Find index corresponding to random value in cumulative distribution.

        Performs linear search (can be optimized to binary search for large weights).

    Args:
            cumsum: Cumulative weight distribution.
            random_value: Random value in [0, 1).

    Returns:
            Index in the weight distribution.
    """
    var idx = 0
    for i in range(len(cumsum)):
        if random_value < cumsum[i]:
            idx = i
            break
    return idx


def renormalize_weights(var weights: List[Float64]) -> List[Float64]:
    """Renormalize weights to sum to 1.0.

        Used after removing sampled elements in weighted sampling without replacement.

    Args:
            weights: Weight values (may not sum to 1.0).

    Returns:
            Normalized weights.
    """
    var total = Float64(0)
    for i in range(len(weights)):
        total += weights[i]

    if total == 0:
        return weights^

    var normalized = List[Float64](capacity=len(weights))
    for i in range(len(weights)):
        normalized.append(weights[i] / total)

    return normalized^


# ============================================================================
# Random Sampling in [0, 1) Range
# ============================================================================


def generate_random_float(max_value: Int = 1000000) -> Float64:
    """Generate a random float in [0, 1) range.

    Args:
            max_value: Resolution for random float (default 1M for ~6 decimal places).

    Returns:
            Random float value in [0, 1).
    """
    return Float64(random_si64(Int64(0), Int64(max_value - 1))) / Float64(
        max_value
    )
