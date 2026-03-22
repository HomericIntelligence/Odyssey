"""Data loading utilities with batching and shuffling.

This module provides the main DataLoader class and related utilities
for efficient batch loading during training.
"""

from shared.core import AnyTensor, zeros
from shared.data._datasets_core import Dataset
from shared.data.samplers import Sampler, SequentialSampler, RandomSampler


# ============================================================================
# Batch Container
# ============================================================================


struct Batch(Copyable, Movable):
    """Container for a batch of samples.

    Holds data and labels for a batch, along with batch metadata.
    """

    var data: AnyTensor
    """Batch data tensor."""
    var labels: AnyTensor
    """Batch labels tensor."""
    var batch_size: Int
    """Number of samples in the batch."""
    var indices: List[Int]
    """Original indices of samples in the batch."""

    fn __init__(
        out self,
        var data: AnyTensor,
        var labels: AnyTensor,
        var indices: List[Int],
    ) raises:
        """Create a batch.

        Args:
            data: Batch data tensor.
            labels: Batch labels tensor.
            indices: Original indices of samples in the batch.

        Raises:
            Error: If batch initialization fails.
        """
        self.data = data^
        self.labels = labels^
        self.indices = indices^
        self.batch_size = self.data.shape()[0]


# ============================================================================
# BaseLoader Implementation
# ============================================================================


struct BaseLoader[D: Dataset & Copyable & Movable](Copyable, Movable):
    """Base data loader with core functionality.

    Provides the foundation for all data loading operations.

    Parameters:
        D: Dataset type that conforms to the Dataset trait and is Copyable & Movable.
    """

    var dataset: Self.D
    """Dataset to load from."""
    var batch_size: Int
    """Number of samples per batch."""
    var drop_last: Bool
    """Whether to drop the last incomplete batch."""
    var _len: Int
    """Cached number of batches."""

    fn __init__(
        out self,
        var dataset: Self.D,
        batch_size: Int = 1,
        drop_last: Bool = False,
    ) raises:
        """Create base loader.

        Args:
            dataset: Dataset to load from.
            batch_size: Number of samples per batch.
            drop_last: Whether to drop the last incomplete batch.

        Raises:
            Error: If batch_size is invalid.
        """
        if batch_size <= 0:
            raise Error(
                "Batch size must be positive, got: " + String(batch_size)
            )

        self.dataset = dataset^
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Calculate number of batches
        var dataset_len = self.dataset.__len__()
        if self.drop_last:
            self._len = dataset_len // batch_size
        else:
            self._len = (dataset_len + batch_size - 1) // batch_size

    fn __len__(self) -> Int:
        """Return number of batches."""
        return self._len


# ============================================================================
# BatchLoader Implementation
# ============================================================================


struct BatchLoader[
    D: Dataset & Copyable & Movable, S: Sampler & Copyable & Movable
](Copyable, Movable):
    """Data loader with batching and optional shuffling.

    Loads data in batches, optionally shuffling the order of samples.

    Parameters:
        D: Dataset type that conforms to the Dataset trait and is Copyable & Movable.
        S: Sampler type that conforms to the Sampler trait and is Copyable & Movable.
    """

    var dataset: Self.D
    """Dataset to load from."""
    var batch_size: Int
    """Number of samples per batch."""
    var drop_last: Bool
    """Whether to drop the last incomplete batch."""
    var _len: Int
    """Cached number of batches."""
    var sampler: Self.S
    """Sampler for generating indices."""
    var shuffle: Bool
    """Whether to shuffle the data."""

    fn __init__(
        out self,
        var dataset: Self.D,
        var sampler: Self.S,
        batch_size: Int = 32,
        shuffle: Bool = False,
        drop_last: Bool = False,
    ) raises:
        """Create batch loader.

        Args:
            dataset: Dataset to load from.
            sampler: Sampler to use for generating indices.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data (informational, sampler controls actual behavior).
            drop_last: Whether to drop the last incomplete batch.

        Raises:
            Error: If batch_size is invalid.
        """
        # Validate batch size
        if batch_size <= 0:
            raise Error(
                "Batch size must be positive, got: " + String(batch_size)
            )

        # Initialize base fields (composition pattern instead of inheritance)
        self.dataset = dataset^
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Calculate number of batches
        var dataset_len = self.dataset.__len__()
        if self.drop_last:
            self._len = dataset_len // batch_size
        else:
            self._len = (dataset_len + batch_size - 1) // batch_size

        self.shuffle = shuffle

        # Set up sampler
        self.sampler = sampler^

    fn __len__(self) -> Int:
        """Return number of batches."""
        return self._len

    fn __iter__(mut self) raises -> List[Batch]:
        """Iterate over batches.

        Returns:
            List of batches for the epoch.

        Raises:
            Error: If operation fails.
        """
        var batches = List[Batch](capacity=self.__len__())
        var indices = self.sampler.__iter__()

        var batch_start = 0
        while batch_start < len(indices):
            var batch_end = min(batch_start + self.batch_size, len(indices))
            var batch_indices = List[Int](capacity=batch_end - batch_start)

            # Collect batch indices
            for i in range(batch_start, batch_end):
                batch_indices.append(indices[i])

            # Skip incomplete batch if drop_last
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            # Load batch data
            var batch_data = List[AnyTensor](capacity=len(batch_indices))
            var batch_labels = List[AnyTensor](capacity=len(batch_indices))

            for idx in batch_indices:
                var sample = self.dataset.__getitem__(idx)
                batch_data.append(sample[0])
                batch_labels.append(sample[1])

            # Stack into batch tensors
            var data_tensor = self._stack_tensors(batch_data)
            var labels_tensor = self._stack_tensors(batch_labels)

            batches.append(Batch(data_tensor^, labels_tensor^, batch_indices^))
            batch_start = batch_end

        return batches^

    fn _stack_tensors(self, tensors: List[AnyTensor]) raises -> AnyTensor:
        """Stack list of tensors into a batch tensor.

        Creates a new tensor with shape [batch_size, *tensor_shape] by
        stacking the input tensors along a new first dimension.
        Supports arbitrary N-dimensional tensors. All tensors must have
        identical shapes for proper stacking.

        Args:
            tensors: List of tensors to stack.

        Returns:
            AnyTensor with batch dimension prepended (shape: [batch_size, *tensor_shape]).

        Raises:
            Error: If tensors list is empty or tensors have incompatible shapes.
        """
        if len(tensors) == 0:
            raise Error("Cannot stack empty list of tensors")

        var batch_size = len(tensors)
        var first_shape = tensors[0].shape()
        var first_dtype = tensors[0].dtype()

        # Verify all tensors have compatible shapes and dtype
        for i in range(len(tensors)):
            if len(tensors[i].shape()) != len(first_shape):
                raise Error(
                    "All tensors must have same number of dimensions. "
                    + "Got tensors with "
                    + String(len(first_shape))
                    + " and "
                    + String(len(tensors[i].shape()))
                    + " dimensions"
                )

            for j in range(len(first_shape)):
                if tensors[i].shape()[j] != first_shape[j]:
                    raise Error(
                        "All tensors must have same shape. "
                        + "Dimension "
                        + String(j)
                        + " mismatch: "
                        + String(first_shape[j])
                        + " vs "
                        + String(tensors[i].shape()[j])
                    )

            if tensors[i].dtype() != first_dtype:
                raise Error("All tensors must have same dtype")

        # Build the new shape: [batch_size, *tensor_shape]
        var new_shape = List[Int]()
        new_shape.append(batch_size)
        for i in range(len(first_shape)):
            new_shape.append(first_shape[i])

        # Calculate total elements in output tensor
        var total_elements = batch_size
        for i in range(len(first_shape)):
            total_elements *= first_shape[i]

        # Calculate number of elements per tensor
        var elements_per_tensor = tensors[0].num_elements()

        # Build stacked data as a List and create tensor
        var stacked_data = List[Float32](capacity=total_elements)

        # Copy each tensor's data into the stacked list
        for tensor_idx in range(len(tensors)):
            # Copy all elements from this tensor
            for elem_idx in range(elements_per_tensor):
                stacked_data.append(Float32(tensors[tensor_idx][elem_idx]))

        # Create output tensor from the list
        var stacked = AnyTensor(stacked_data^)

        return stacked^
