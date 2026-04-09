"""Dataset wrapper that applies transforms during data loading.

This module provides a dataset wrapper that chains a base dataset with
a transform pipeline. Transforms are applied to data but not labels,
enabling data augmentation during training.

Example:
    from shared.data import AnyTensorDataset, TransformedDataset
    from shared.data.transforms import Normalize

    var dataset = AnyTensorDataset(images, labels)
    var normalize = Normalize(mean=0.5, std=0.5)
    var transformed = TransformedDataset(dataset, normalize)
    var img, label = transformed[0]  # Returns normalized image
"""

from shared.tensor.any_tensor import AnyTensor
from shared.data._datasets_core import Dataset
from shared.data.transforms import Transform


struct TransformedDataset[
    D: Dataset & Copyable & Movable, T: Transform & Copyable & Movable
](Copyable, Dataset, Movable):
    """Dataset wrapper that applies transforms to data.

    Applies a transform to the data component of samples while leaving
    labels unchanged. Useful for data augmentation during training.

    Parameters:
        D: Dataset type that conforms to the Dataset trait.
        T: Transform type with __call__(AnyTensor) -> AnyTensor method.

    Examples:
        ```mojo
        var dataset = AnyTensorDataset(images, labels)
        var normalize = Normalize(mean=0.5, std=0.5)
        var transformed = TransformedDataset(dataset, normalize)

        # Get a sample with transform applied
        var img, label = transformed[0]
        ```
    """

    var dataset: Self.D
    """Base dataset to wrap."""
    var transform: Self.T
    """Transform to apply to data."""

    def __init__(out self, var dataset: Self.D, var transform: Self.T):
        """Create a transformed dataset.

        Args:
            dataset: Base dataset to wrap.
            transform: Transform to apply to data samples.
        """
        self.dataset = dataset^
        self.transform = transform^

    def __len__(self) -> Int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return self.dataset.__len__()

    def __getitem__(self, index: Int) raises -> Tuple[AnyTensor, AnyTensor]:
        """Get a sample with transform applied.

        The transform is applied to the data but not the labels.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (transformed_data, labels).

        Raises:
            Error: If index is out of bounds or transform fails.
        """
        var data, labels = self.dataset.__getitem__(index)

        # Apply transform to data only, not labels
        # The transform must have __call__(AnyTensor) -> AnyTensor method
        var transformed_data = self.transform(data)

        return (transformed_data, labels)
