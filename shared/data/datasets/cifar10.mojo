"""CIFAR-10 Dataset Wrapper

High-level dataset interface for CIFAR-10 with convenient methods for train/test splits

CIFAR-10 Properties:
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - Images: 32x32 RGB (3 channels), values in [0, 1] after normalization
    - Training: 50,000 images
    - Test: 10,000 images

Architecture:
    - CIFAR10Dataset: Main struct providing dataset access
    - Lazy loading: Data loaded on-demand for memory efficiency
    - Train/test splits: Separate methods for accessing different portions
    - Iterator support: Can be used with data loaders

Example:
    var dataset = CIFAR10Dataset("/path/to/cifar10")
    var n_samples = dataset.__len__()
    var image, label = dataset.__getitem__(0)

    var train_data, train_labels = dataset.get_train_data()
    var test_data, test_labels = dataset.get_test_data()

References:
    - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
    - Array API: https://data-apis.org/array-api/latest/
"""

from shared.tensor.any_tensor import AnyTensor, zeros
from shared.data.formats import load_cifar10_batch
from std.collections import List


# ============================================================================
# CIFAR-10 Class Labels
# ============================================================================


def _get_cifar10_classes() -> List[String]:
    """Get CIFAR-10 class names."""
    var classes: List[String] = []
    classes.append("airplane")
    classes.append("automobile")
    classes.append("bird")
    classes.append("cat")
    classes.append("deer")
    classes.append("dog")
    classes.append("frog")
    classes.append("horse")
    classes.append("ship")
    classes.append("truck")
    return classes^


# CIFAR10 class names - created at runtime when needed
def get_cifar10_classes() -> List[String]:
    """Get CIFAR-10 class names."""
    return _get_cifar10_classes()


# ============================================================================
# CIFAR10Dataset Implementation
# ============================================================================


struct CIFAR10Dataset(Copyable, Movable):
    """CIFAR-10 Dataset wrapper for convenient access to training and test data.

    Provides methods to:
    - Load individual samples by index
    - Get all training data
    - Get all test data
    - Query dataset properties

    The dataset expects data to be organized as CIFAR-10 IDX format files:
        data_dir/
            train_batch_1_images.idx
            train_batch_1_labels.idx
            train_batch_2_images.idx
            train_batch_2_labels.idx
            ... (5 batches total)
            test_batch_images.idx
            test_batch_labels.idx

    Attributes:
        data_dir: Path to directory containing CIFAR-10 data files
        _train_data: Cached training images (lazy loaded)
        _train_labels: Cached training labels (lazy loaded)
        _test_data: Cached test images (lazy loaded)
        _test_labels: Cached test labels (lazy loaded)
        _train_loaded: Whether training data has been loaded
        _test_loaded: Whether test data has been loaded
    """

    var data_dir: String
    var _train_data: AnyTensor
    var _train_labels: AnyTensor
    var _test_data: AnyTensor
    var _test_labels: AnyTensor
    var _train_loaded: Bool
    var _test_loaded: Bool

    def __init__(out self, data_dir: String) raises:
        """Initialize CIFAR10Dataset.

        Args:
            data_dir: Path to directory containing CIFAR-10 IDX format files.

        Raises:
            Error: If data_dir is empty or invalid.
        """
        if len(data_dir) == 0:
            raise Error("data_dir cannot be empty")

        self.data_dir = data_dir

        # Initialize placeholder tensors
        self._train_data = zeros([1], DType.float32)
        self._train_labels = zeros([1], DType.uint8)
        self._test_data = zeros([1], DType.float32)
        self._test_labels = zeros([1], DType.uint8)

        self._train_loaded = False
        self._test_loaded = False

    def __len__(self) -> Int:
        """Return total number of training samples.

        Returns:
            Number of training samples (50,000 for standard CIFAR-10).
        """
        return 50000

    def __getitem__(mut self, index: Int) raises -> Tuple[AnyTensor, AnyTensor]:
        """Get a sample from the training set.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Tuple of (image, label) where:
                - image: AnyTensor of shape (3, 32, 32) with float32 values.
                - label: AnyTensor of shape (1,) with uint8 class index.

        Raises:
            Error: If index is out of bounds or data cannot be loaded.
        """
        # Load training data if not already loaded
        if not self._train_loaded:
            self._load_train_data()

        # Validate index
        if index < 0 or index >= 50000:
            raise Error(
                "Index "
                + String(index)
                + " out of bounds for training set of size 50000"
            )

        # Return owned copies (clones) for safe lifetime management.
        # See AnyTensorDataset.__getitem__ comment for details.
        var image_slice = self._train_data.slice(index, index + 1, axis=0).clone()
        var label_slice = self._train_labels.slice(index, index + 1, axis=0).clone()

        return (image_slice, label_slice)

    def _load_train_data(mut self) raises:
        """Load all training data from CIFAR-10 batch files.

        Loads 5 batches (10,000 images each) from files:
            train_batch_1_images.idx, train_batch_1_labels.idx
            train_batch_2_images.idx, train_batch_2_labels.idx
            ... (up to train_batch_5)

        Data is normalized and stored as float32 with shape (50000, 3, 32, 32)

        Raises:
            Error: If batch files cannot be read or loaded
        """
        var all_images: List[AnyTensor] = []
        var all_labels: List[AnyTensor] = []

        # Load 5 training batches
        for batch_num in range(1, 6):
            var batch_name = "train_batch_" + String(batch_num)
            var images, labels = load_cifar10_batch(self.data_dir, batch_name)
            all_images.append(images)
            all_labels.append(labels)

        # Concatenate all batches
        self._train_data = self._concatenate_tensors(all_images)
        self._train_labels = self._concatenate_tensors(all_labels)
        self._train_loaded = True

    def get_train_data(mut self) raises -> Tuple[AnyTensor, AnyTensor]:
        """Get all training data.

        Returns:
            Tuple of (images, labels) where:
                - images: AnyTensor of shape (50000, 3, 32, 32) with float32 values.
                - labels: AnyTensor of shape (50000,) with uint8 class indices.

        Raises:
            Error: If data files cannot be loaded.

        Note:
            Data is loaded once and cached for efficiency.
            Subsequent calls return the cached data.
        """
        if not self._train_loaded:
            self._load_train_data()

        return (self._train_data, self._train_labels)

    def _load_test_data(mut self) raises:
        """Load test data from CIFAR-10 test batch file.

        Loads test set (10,000 images) from files:
            test_batch_images.idx
            test_batch_labels.idx

        Data is normalized and stored as float32 with shape (10000, 3, 32, 32)

        Raises:
            Error: If test batch files cannot be read
        """
        var images, labels = load_cifar10_batch(self.data_dir, "test_batch")
        self._test_data = images
        self._test_labels = labels
        self._test_loaded = True

    def get_test_data(mut self) raises -> Tuple[AnyTensor, AnyTensor]:
        """Get all test data.

        Returns:
            Tuple of (images, labels) where:
                - images: AnyTensor of shape (10000, 3, 32, 32) with float32 values.
                - labels: AnyTensor of shape (10000,) with uint8 class indices.

        Raises:
            Error: If test data files cannot be loaded.

        Note:
            Data is loaded once and cached for efficiency.
            Subsequent calls return the cached data.
        """
        if not self._test_loaded:
            self._load_test_data()

        return (self._test_data, self._test_labels)

    def _concatenate_tensors(self, tensors: List[AnyTensor]) raises -> AnyTensor:
        """Concatenate a list of tensors along the first (batch) dimension.

        Args:
            tensors: List of AnyTensor objects with same shape except first dimension

        Returns:
            Concatenated tensor with first dimension equal to sum of input dimensions

        Raises:
            Error: If tensors list is empty or tensors have incompatible shapes
        """
        if len(tensors) == 0:
            raise Error("Cannot concatenate empty list of tensors")

        if len(tensors) == 1:
            return tensors[0]

        from shared.core.shape import concatenate
        return concatenate(tensors, axis=0)

    def get_class_name(self, class_idx: Int) raises -> String:
        """Get human-readable class name from class index.

        Args:
            class_idx: Integer class index (0-9).

        Returns:
            Class name string (e.g., "airplane", "cat").

        Raises:
            Error: If class_idx is not in range [0, 9].
        """
        if class_idx < 0 or class_idx >= 10:
            raise Error(
                "Class index " + String(class_idx) + " out of range [0, 9]"
            )

        var classes = get_cifar10_classes()
        return classes[class_idx]

    def num_classes(self) -> Int:
        """Get number of classes in CIFAR-10.

        Returns:
            Number of classes (10).
        """
        return 10

    def num_train_samples(self) -> Int:
        """Get number of training samples.

        Returns:
            Number of training samples (50,000).
        """
        return 50000

    def num_test_samples(self) -> Int:
        """Get number of test samples.

        Returns:
            Number of test samples (10,000).
        """
        return 10000

    def image_shape(self) -> List[Int]:
        """Get shape of individual images.

        Returns:
            List containing image dimensions [3, 32, 32] (channels, height, width).
        """
        var shape: List[Int] = []
        shape.append(3)  # RGB channels
        shape.append(32)  # Height
        shape.append(32)  # Width
        return shape^
