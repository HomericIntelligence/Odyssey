"""Batch utilities for efficient mini-batch processing.

This module provides utilities for extracting mini-batches from datasets
for training and evaluation.
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros


def extract_batch(
    data: AnyTensor, start_idx: Int, batch_size: Int
) raises -> AnyTensor:
    """Extract a mini-batch from a dataset tensor using zero-copy slicing.

    Extracts a contiguous slice of samples from the dataset starting at
    start_idx and containing up to batch_size samples. Uses AnyTensor's
    slice() method for efficient memory views instead of copying data.

    Args:
        data: Full dataset tensor of shape (N, ...) where N is number of samples.
        start_idx: Starting index for batch extraction (0-indexed).
        batch_size: Number of samples to extract.

    Returns:
        Batch tensor of shape (actual_batch_size, ...) where actual_batch_size
        is min(batch_size, N - start_idx). Returns a view that shares memory
        with the original tensor (no copy).

    Raises:
        Error: If start_idx is out of bounds.

    Example:
        ```mojo
        from projectodyssey.data import extract_batch

        # Extract batch of 128 images from dataset (zero-copy view)
        var images = load_dataset()  # Shape: (50000, 3, 32, 32)
        var batch = extract_batch(images, start_idx=0, batch_size=128)
        # batch shape: (128, 3, 32, 32) - memory shared with images
        ```

    Performance:
        - O(1) time complexity (no data copying)
        - Zero memory allocation for batch extraction
        - Ideal for training loops processing hundreds/thousands of batches
        - Memory is shared until a batch is modified or freed

    Note:
        - Handles edge cases where remaining samples < batch_size
        - Works with any tensor dimensionality (2D, 3D, 4D, etc.)
        - Uses memory-efficient slicing with reference counting
        - Original tensor must remain valid while batch is in use.
    """
    var data_shape = data.shape()
    var num_samples = data_shape[0]

    # Handle edge case: start_idx beyond dataset
    if start_idx >= num_samples:
        raise Error(
            "start_idx ("
            + String(start_idx)
            + ") >= num_samples ("
            + String(num_samples)
            + ")"
        )

    # Compute actual batch size (handle partial batches at end)
    var end_idx = min(start_idx + batch_size, num_samples)

    # Use zero-copy slice() instead of allocating and copying
    # This creates a view that shares memory with the original tensor
    return data.slice(start_idx, end_idx, axis=0)


def extract_batch_pair(
    data: AnyTensor, labels: AnyTensor, start_idx: Int, batch_size: Int
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Extract matching mini-batches of data and labels using zero-copy slicing.

    Convenience function that extracts matching slices from both data and
    label tensors with a single call. Uses AnyTensor's slice() method for
    efficient memory views.

    Args:
        data: Full dataset tensor of shape (N, ...).
        labels: Full labels tensor of shape (N,) or (N, ...).
        start_idx: Starting index for batch extraction (0-indexed).
        batch_size: Number of samples to extract.

    Returns:
        Tuple of (batch_data, batch_labels) with matching first dimension.
        Both are views that share memory with the original tensors.

    Raises:
        Error: If data and label sizes don't match.

    Example:
        ```mojo
        from projectodyssey.data import extract_batch_pair

        var images = load_images()      # Shape: (50000, 3, 32, 32)
        var labels = load_labels()      # Shape: (50000,)

        var (batch_images, batch_labels) = extract_batch_pair(
            images, labels, start_idx=0, batch_size=128
        )
        # batch_images shape: (128, 3, 32, 32) - zero-copy view
        # batch_labels shape: (128,) - zero-copy view
        ```

    Performance:
        - O(1) time complexity (no data copying)
        - Two slice operations instead of two allocations
        - Ideal for training loops with paired data/label batches

    Note:
        - Ensures data and labels have matching number of samples
        - Both tensors extracted with same start_idx and batch_size
        - Uses efficient memory views with reference counting
        - Original tensors must remain valid while batches are in use.
    """
    # Verify matching sizes
    var data_samples = data.shape()[0]
    var label_samples = labels.shape()[0]

    if data_samples != label_samples:
        raise Error(
            "Data samples ("
            + String(data_samples)
            + ") != label samples ("
            + String(label_samples)
            + ")"
        )

    # Extract both batches using zero-copy slicing
    # Both call extract_batch which now uses slice() instead of copying
    var batch_data = extract_batch(data, start_idx, batch_size)
    var batch_labels = extract_batch(labels, start_idx, batch_size)

    return (batch_data, batch_labels)


def compute_num_batches(num_samples: Int, batch_size: Int) -> Int:
    """Compute the number of batches needed to process all samples.

    Args:
            num_samples: Total number of samples in dataset.
            batch_size: Number of samples per batch.

    Returns:
            Number of batches needed (rounded up).

        Example:
            ```mojo
            from projectodyssey.data import compute_num_batches

            var num_batches = compute_num_batches(50000, 128)
            # Returns: 391 batches (50000 / 128 = 390.625 → 391)
            ```

    Note:
            - Uses ceiling division: (num_samples + batch_size - 1) // batch_size.
            - Accounts for partial batch at end.
    """
    return (num_samples + batch_size - 1) // batch_size


def get_batch_indices(
    batch_idx: Int, batch_size: Int, num_samples: Int
) -> Tuple[Int, Int, Int]:
    """Compute start index, end index, and actual size for a batch.

        Helper function to compute batch boundaries with proper handling
        of the final partial batch

    Args:
            batch_idx: Batch index (0-indexed).
            batch_size: Desired batch size.
            num_samples: Total number of samples in dataset.

    Returns:
            Tuple of (start_idx, end_idx, actual_batch_size) where:
            - start_idx: Starting sample index (inclusive)
            - end_idx: Ending sample index (exclusive)
            - actual_batch_size: Number of samples in this batch

    Example:
        ```mojo
        from projectodyssey.data import get_batch_indices

        # Get indices for batch 390 of size 128 from 50000 samples
        var (start, end, size) = get_batch_indices(390, 128, 50000)
        # Returns: (49920, 50000, 80) - partial batch with 80 samples
        ```

    Note:
            - Handles partial batches automatically.
            - start_idx = batch_idx * batch_size.
            - end_idx = min(start_idx + batch_size, num_samples).
            - actual_batch_size = end_idx - start_idx.
    """
    var start_idx = batch_idx * batch_size
    var end_idx = min(start_idx + batch_size, num_samples)
    var actual_batch_size = end_idx - start_idx

    return (start_idx, end_idx, actual_batch_size)
