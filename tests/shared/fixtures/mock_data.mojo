"""Mock dataset and data loader implementations for testing.

This module provides simple mock implementations of datasets and data loaders
for testing data pipelines, training loops, and data transformations.

Key components:
- MockDataset: Simple dataset with configurable samples
- MockClassificationDataset: Classification data generator
- MockRegressionDataset: Regression data generator
- MockDataLoader: Basic data loader with batching

All mocks use deterministic seeds for reproducible tests.
"""

from std.random import seed, randn, randint
from tests.shared.fixtures.mock_tensors import (
    create_random_tensor,
    create_zeros_tensor,
    create_ones_tensor,
)


# ============================================================================
# Mock Dataset Implementations
# ============================================================================


struct MockDataset:
    """Simple mock dataset for testing.

    Provides a minimal dataset implementation with configurable number of
    samples and dimensions. Useful for testing data loading pipelines.

    Attributes:
        num_samples: Number of samples in dataset.
        input_dim: Dimension of input features.
        output_dim: Dimension of output/labels.
        random_seed: Seed for reproducible data generation.
    """

    var num_samples: Int
    var input_dim: Int
    var output_dim: Int
    var random_seed: Int

    def __init__(
        mut self,
        num_samples: Int = 100,
        input_dim: Int = 10,
        output_dim: Int = 1,
        random_seed: Int = 42,
    ):
        """Initialize mock dataset.

        Args:
            num_samples: Number of samples (default: 100).
            input_dim: Input feature dimension (default: 10).
            output_dim: Output dimension (default: 1).
            random_seed: Random seed (default: 42).

        Example:
            ```mojo
            var dataset = MockDataset(
                num_samples=50,
                input_dim=20,
                output_dim=5
            )
            ```
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_seed = random_seed

    def __len__(self) -> Int:
        """Get number of samples in dataset.

        Returns:
            Number of samples.
        """
        return self.num_samples

    def get_item(self, index: Int) -> Tuple[List[Float32], List[Float32]]:
        """Get a single sample by index.

        Args:
            index: Sample index (0 to num_samples-1).

        Returns:
            Tuple of (input, output) tensors as flat lists.

        Example:
            ```mojo
            var dataset = MockDataset()
            var item = dataset.get_item(0)
            var input = item[0]
            var output = item[1]
            # input has shape [input_dim]
            # output has shape [output_dim]
            ```

        Note:
            Uses deterministic seeding based on index for reproducibility.
        """
        # Create deterministic but varied data per index
        var item_seed = self.random_seed + index

        # Generate input
        var input = create_random_tensor(
            [self.input_dim], random_seed=item_seed
        )

        # Generate output
        var output = create_random_tensor(
            [self.output_dim], random_seed=item_seed + 1000
        )

        return (input, output)


struct MockClassificationDataset:
    """Mock classification dataset.

    Generates synthetic classification data with configurable classes.
    Each sample is labeled with a class index (0 to num_classes-1).

    Attributes:
        num_samples: Number of samples.
        input_dim: Input feature dimension.
        num_classes: Number of classification classes.
        random_seed: Seed for reproducibility.
    """

    var num_samples: Int
    var input_dim: Int
    var num_classes: Int
    var random_seed: Int

    def __init__(
        mut self,
        num_samples: Int = 100,
        input_dim: Int = 10,
        num_classes: Int = 5,
        random_seed: Int = 42,
    ):
        """Initialize classification dataset.

        Args:
            num_samples: Number of samples (default: 100).
            input_dim: Input dimension (default: 10).
            num_classes: Number of classes (default: 5).
            random_seed: Random seed (default: 42).

        Example:
            ```mojo
            var dataset = MockClassificationDataset(
                num_samples=1000,
                input_dim=784,  # Like MNIST
                num_classes=10
            )
            ```
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.random_seed = random_seed

    def __len__(self) -> Int:
        """Get dataset size."""
        return self.num_samples

    def get_item(self, index: Int) -> Tuple[List[Float32], Int]:
        """Get classification sample.

        Args:
            index: Sample index.

        Returns:
            Tuple of (input tensor, class label).
            - input: Flat list of Float32 features
            - label: Integer class index (0 to num_classes-1)

        Example:
            ```mojo
            var dataset = MockClassificationDataset()
            var item2 = dataset.get_item(0)
            var input2 = item2[0]
            var label = item2[1]
            # label is in range [0, num_classes)
            ```
        """
        var item_seed = self.random_seed + index

        # Generate input features
        var input = create_random_tensor(
            [self.input_dim], random_seed=item_seed
        )

        # Generate class label (deterministic but varied)
        seed(item_seed)
        var label = randint(0, self.num_classes - 1)

        return (input, label)


struct MockRegressionDataset:
    """Mock regression dataset.

    Generates synthetic regression data where output is a simple
    function of the input (for testing purposes).

    Attributes:
        num_samples: Number of samples.
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        random_seed: Seed for reproducibility.
        noise_scale: Scale of noise to add to outputs.
    """

    var num_samples: Int
    var input_dim: Int
    var output_dim: Int
    var random_seed: Int
    var noise_scale: Float32

    def __init__(
        mut self,
        num_samples: Int = 100,
        input_dim: Int = 10,
        output_dim: Int = 1,
        random_seed: Int = 42,
        noise_scale: Float32 = 0.1,
    ):
        """Initialize regression dataset.

        Args:
            num_samples: Number of samples (default: 100).
            input_dim: Input dimension (default: 10).
            output_dim: Output dimension (default: 1).
            random_seed: Random seed (default: 42).
            noise_scale: Noise magnitude (default: 0.1).

        Example:
            ```mojo
            var dataset = MockRegressionDataset(
                num_samples=500,
                input_dim=5,
                output_dim=3,
                noise_scale=0.05
            )
            ```
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_seed = random_seed
        self.noise_scale = noise_scale

    def __len__(self) -> Int:
        """Get dataset size."""
        return self.num_samples

    def get_item(self, index: Int) -> Tuple[List[Float32], List[Float32]]:
        """Get regression sample.

        Args:
            index: Sample index.

        Returns:
            Tuple of (input, target) tensors as flat lists.
            Output is mean(input) + noise for simple regression task.

        Example:
            ```mojo
            var dataset = MockRegressionDataset()
            var item3 = dataset.get_item(0)
            var input3 = item3[0]
            var target = item3[1]
            # target is correlated with input (mean + noise)
            ```
        """
        var item_seed = self.random_seed + index

        # Generate input
        var input = create_random_tensor(
            [self.input_dim], random_seed=item_seed
        )

        # Generate output as simple function of input
        # Output = mean(input) + small noise
        var input_mean = Float32(0.0)
        for i in range(len(input)):
            input_mean += input[i]
        input_mean /= Float32(len(input))

        # Add noise
        seed(item_seed + 1000)
        var output = List[Float32]()
        for _ in range(self.output_dim):
            var noise = Float32(randn()) * self.noise_scale
            output.append(input_mean + noise)

        return (input, output)


# ============================================================================
# Mock Data Loader Implementation
# ============================================================================


struct MockDataLoader:
    """Simple mock data loader with batching.

    Provides basic batching functionality for testing training loops
    and data pipeline integration.

    Attributes:
        dataset: The dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data (not implemented - always False).
        num_batches: Total number of batches.
    """

    var num_samples: Int
    var batch_size: Int
    var shuffle: Bool
    var num_batches: Int

    def __init__(
        mut self, num_samples: Int, batch_size: Int, shuffle: Bool = False
    ):
        """Initialize data loader.

        Args:
            num_samples: Total number of samples in dataset.
            batch_size: Samples per batch.
            shuffle: Whether to shuffle (not implemented, always False).

        Example:
            ```mojo
            var dataset = MockDataset(num_samples=100)
            var loader = MockDataLoader(
                num_samples=dataset.__len__(),
                batch_size=32
            )
            ```

        Note:
            Shuffle is not yet implemented - data is always in order.
            The last batch may be smaller than batch_size.
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate number of batches (ceiling division)
        self.num_batches = (num_samples + batch_size - 1) // batch_size

    def __len__(self) -> Int:
        """Get number of batches.

        Returns:
            Total number of batches.
        """
        return self.num_batches

    def get_batch_size(self, batch_index: Int) -> Int:
        """Get size of specific batch.

        Args:
            batch_index: Index of batch (0 to num_batches-1).

        Returns:
            Number of samples in this batch.
            Last batch may be smaller than batch_size.

        Example:
            ```mojo
            var loader = MockDataLoader(num_samples=100, batch_size=32)
            var last_batch_size = loader.get_batch_size(3)  # Returns 4
            # Batches: [32, 32, 32, 4]
            ```
        """
        var start_idx = batch_index * self.batch_size
        var end_idx = min(start_idx + self.batch_size, self.num_samples)
        return end_idx - start_idx

    def get_batch_indices(self, batch_index: Int) -> List[Int]:
        """Get sample indices for a batch.

        Args:
            batch_index: Index of batch.

        Returns:
            List of sample indices in this batch.

        Example:
            ```mojo
            var loader = MockDataLoader(num_samples=10, batch_size=3)
            var indices = loader.get_batch_indices(0)  # [0, 1, 2]
            var indices2 = loader.get_batch_indices(3)  # [9]
            ```
        """
        var start_idx = batch_index * self.batch_size
        var end_idx = min(start_idx + self.batch_size, self.num_samples)

        var indices = List[Int]()
        for i in range(start_idx, end_idx):
            indices.append(i)

        return indices


# ============================================================================
# Helper Functions
# ============================================================================


def create_mock_batch(
    batch_size: Int, input_dim: Int, output_dim: Int = 1, random_seed: Int = 42
) -> Tuple[List[List[Float32]], List[List[Float32]]]:
    """Create a mock batch of data.

    Convenience function for creating a batch without full dataset/loader setup.

    Args:
        batch_size: Number of samples in batch.
        input_dim: Input feature dimension.
        output_dim: Output dimension (default: 1).
        random_seed: Random seed (default: 42).

    Returns:
        Tuple of (inputs, outputs) where each is a list of samples.

    Example:
        ```mojo
        var batch_result = create_mock_batch(
            batch_size=32,
            input_dim=10,
            output_dim=5
        )
        var inputs = batch_result[0]
        var outputs = batch_result[1]
        # inputs is List[List[Float32]] with 32 samples
        # outputs is List[List[Float32]] with 32 samples
        ```
    """
    var inputs = List[List[Float32]]()
    var outputs = List[List[Float32]]()

    for i in range(batch_size):
        var input = create_random_tensor(
            [input_dim], random_seed=random_seed + i
        )
        var output = create_random_tensor(
            [output_dim], random_seed=random_seed + i + 1000
        )

        inputs.append(input)
        outputs.append(output)

    return (inputs, outputs)


def create_mock_classification_batch(
    batch_size: Int, input_dim: Int, num_classes: Int = 5, random_seed: Int = 42
) -> Tuple[List[List[Float32]], List[Int]]:
    """Create a mock classification batch.

    Args:
        batch_size: Number of samples.
        input_dim: Input dimension.
        num_classes: Number of classes (default: 5).
        random_seed: Random seed (default: 42).

    Returns:
        Tuple of (inputs, labels) where:
        - inputs: List[List[Float32]] of features
        - labels: List[Int] of class indices

    Example:
        ```mojo
        var classification_batch = create_mock_classification_batch(
            batch_size=16,
            input_dim=784,
            num_classes=10
        )
        var inputs2 = classification_batch[0]
        var labels2 = classification_batch[1]
        ```
    """
    var inputs = List[List[Float32]]()
    var labels = List[Int]()

    for i in range(batch_size):
        var input = create_random_tensor(
            [input_dim], random_seed=random_seed + i
        )
        inputs.append(input)

        # Generate class label
        seed(random_seed + i + 1000)
        var label = randint(0, num_classes - 1)
        labels.append(label)

    return (inputs, labels)
