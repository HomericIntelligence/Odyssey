"""Sampling strategies for data loading.

This module provides various sampling strategies for iterating through datasets.
"""

from projectodyssey.data.sampler_utils import (
    validate_range,
    create_range_indices,
    set_random_seed,
    sample_with_replacement,
    sample_without_replacement,
    build_cumulative_weights,
    sample_index_from_distribution,
    renormalize_weights,
    generate_random_float,
)


# ============================================================================
# Sampler Trait
# ============================================================================


trait Sampler(ImplicitlyDestructible):
    """Base interface for all samplers.

    Samplers determine the order in which samples are accessed from a dataset.
    """

    def __len__(self) -> Int:
        """Return the number of samples."""
        ...

    def __iter__(mut self) -> List[Int]:
        """Return an iterator over sample indices.

        Returns:
            List of indices in the order they should be accessed.
        """
        ...


# ============================================================================
# SequentialSampler Implementation
# ============================================================================


struct SequentialSampler(Copyable, Movable, Sampler):
    """Samples elements sequentially in order.

    Always returns indices in the same order: 0, 1, 2, ..., n-1.
    """

    var data_source_len: Int
    """Total length of the data source."""
    var start_index: Int
    """Starting index (inclusive)."""
    var end_index: Int
    """Ending index (exclusive)."""

    def __init__(
        out self,
        data_source_len: Int,
        start_index: Int = 0,
        end_index: Int = -1,
    ):
        """Create sequential sampler.

        Args:
            data_source_len: Length of the dataset.
            start_index: Starting index (inclusive).
            end_index: Ending index (exclusive), -1 for end of dataset.
        """
        self.data_source_len = data_source_len
        var normalized = validate_range(start_index, end_index, data_source_len)
        self.start_index = normalized[0]
        self.end_index = normalized[1]

    def __len__(self) -> Int:
        """Return number of samples."""
        return self.end_index - self.start_index

    def __iter__(mut self) -> List[Int]:
        """Return sequential indices.

        Returns:
            List of indices from start to end.
        """
        return create_range_indices(self.start_index, self.end_index)


# ============================================================================
# RandomSampler Implementation
# ============================================================================


struct RandomSampler(Copyable, Movable, Sampler):
    """Samples elements randomly without replacement.

    Generates a random permutation of indices for each iteration.
    """

    var data_source_len: Int
    """Total length of the data source."""
    var replacement: Bool
    """Whether to sample with replacement."""
    var num_samples: Int
    """Number of samples to draw."""
    var seed_value: Int
    """Random seed for reproducibility (-1 = no seed)."""

    def __init__(
        out self,
        data_source_len: Int,
        replacement: Bool = False,
        num_samples: Int = -1,
        seed_value: Int = -1,
    ):
        """Create random sampler.

        Args:
            data_source_len: Length of the dataset.
            replacement: Whether to sample with replacement.
            num_samples: Number of samples to draw (-1 = all).
            seed_value: Random seed for reproducibility (-1 = no seed).
        """
        self.data_source_len = data_source_len
        self.replacement = replacement
        self.seed_value = seed_value

        if num_samples > 0:
            self.num_samples = num_samples
        else:
            self.num_samples = data_source_len

    def __len__(self) -> Int:
        """Return number of samples."""
        return self.num_samples

    def __iter__(mut self) -> List[Int]:
        """Return random indices.

        Returns:
            List of randomly shuffled or sampled indices.
        """
        set_random_seed(self.seed_value)

        if self.replacement:
            return sample_with_replacement(
                self.data_source_len, self.num_samples
            )
        else:
            return sample_without_replacement(
                self.data_source_len, self.num_samples
            )


# ============================================================================
# WeightedSampler Implementation
# ============================================================================


struct WeightedSampler(Copyable, Movable, Sampler):
    """Samples elements according to given weights.

    Each sample is drawn with probability proportional to its weight.
    """

    var weights: List[Float64]
    """Normalized weights for each sample."""
    var num_samples: Int
    """Number of samples to draw."""
    var replacement: Bool
    """Whether to sample with replacement."""
    var seed_value: Int
    """Random seed for reproducibility (-1 = no seed)."""

    def __init__(
        out self,
        var weights: List[Float64],
        num_samples: Int,
        replacement: Bool = True,
        seed_value: Int = -1,
    ) raises:
        """Create weighted sampler.

        Args:
            weights: Weight for each sample.
            num_samples: Number of samples to draw.
            replacement: Whether to sample with replacement.
            seed_value: Random seed for reproducibility (-1 = no seed).

        Raises:
            Error: If weights are invalid.
        """
        # Validate weights
        var total_weight = Float64(0)
        for i in range(len(weights)):
            if weights[i] < 0:
                raise Error("Weights must be non-negative")
            total_weight += weights[i]

        if total_weight == 0:
            raise Error("At least one weight must be positive")

        # Normalize weights
        self.weights = List[Float64](capacity=len(weights))
        for i in range(len(weights)):
            self.weights.append(weights[i] / total_weight)

        self.num_samples = num_samples
        self.replacement = replacement
        self.seed_value = seed_value

    def __len__(self) -> Int:
        """Return number of samples."""
        return self.num_samples

    def __iter__(mut self) -> List[Int]:
        """Return weighted random indices.

        Returns:
            List of indices sampled according to weights.
        """
        set_random_seed(self.seed_value)

        var indices = List[Int](capacity=self.num_samples)

        # Sample indices
        for _ in range(self.num_samples):
            var r = generate_random_float()

            # Get index from cumulative distribution
            var cumsum = build_cumulative_weights(self.weights)
            var idx = sample_index_from_distribution(cumsum, r)
            indices.append(idx)

            # If without replacement, set weight to 0 and renormalize
            if not self.replacement:
                self.weights[idx] = 0
                self.weights = renormalize_weights(self.weights^)

        return indices^
