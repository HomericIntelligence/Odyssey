"""Dataset Constants.

Centralized constants for dataset-specific information like class names, sizes, and metadata

This module provides:
    - Dataset class names (CIFAR-10, EMNIST variants)
    - DatasetInfo struct for dataset metadata
    - Helper functions to access dataset properties

Example:
    from projectodyssey.data import CIFAR10_CLASS_NAMES, EMNIST_BALANCED_CLASSES, DatasetInfo

    # Access class names directly
    var cifar10_classes = CIFAR10_CLASS_NAMES()
    var emnist_letters = EMNIST_BALANCED_CLASSES()

    # Or use DatasetInfo for comprehensive metadata
    var info = DatasetInfo("cifar10")
    var num_classes = info.num_classes()
    var class_name = info.class_name(0)
    ```
"""

from std.collections import List


# ============================================================================
# CIFAR-10 Format Constants
# ============================================================================

comptime CIFAR10_IMAGE_SIZE: Int = 32
comptime CIFAR10_CHANNELS: Int = 3
comptime CIFAR10_BYTES_PER_IMAGE: Int = 3073
comptime CIFAR10_NUM_CLASSES: Int = 10


# ============================================================================
# CIFAR-100 Format Constants
# ============================================================================

comptime CIFAR100_IMAGE_SIZE: Int = 32
comptime CIFAR100_CHANNELS: Int = 3
comptime CIFAR100_BYTES_PER_IMAGE: Int = 3074
comptime CIFAR100_NUM_CLASSES_FINE: Int = 100
comptime CIFAR100_NUM_CLASSES_COARSE: Int = 20


# ============================================================================
# CIFAR-10 Class Names
# ============================================================================


def CIFAR10_CLASS_NAMES() -> List[String]:
    """Get CIFAR-10 class names.

        CIFAR-10 contains 10 object classes commonly used for image classification

    Returns:
            List of 10 class name strings in order:
            ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    Note:
            Creates a new List each time it's called. Cache the result if used repeatedly.
    """
    var classes = List[String]()
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


# ============================================================================
# EMNIST Class Names (Balanced Split)
# ============================================================================


def EMNIST_BALANCED_CLASSES() -> List[String]:
    """Get EMNIST Balanced class names.

        EMNIST Balanced contains 47 classes: 10 digits (0-9) and 37 letters
        The balanced split has roughly equal numbers of samples per class
        Letters are: A-Z (26 classes) + a-k (11 classes) = 37 letter classes

    Returns:
            List of 47 class name strings representing digits and letters

    Note:
            Classes are ordered as: 0-9 (digits), then A-Z (uppercase), then a-k (lowercase)
            Creates a new List each time it's called. Cache the result if used repeatedly.
    """
    var classes = List[String]()

    # Digits 0-9
    for i in range(10):
        classes.append(String(i))

    # Uppercase letters A-Z
    var uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(26):
        classes.append(String(uppercase[byte = i : i + 1]))

    # Lowercase letters a-k (11 classes)
    var lowercase = "abcdefghijk"
    for i in range(11):
        classes.append(String(lowercase[byte = i : i + 1]))

    return classes^


# ============================================================================
# EMNIST Class Names (By Class Split)
# ============================================================================


def EMNIST_BYCLASS_CLASSES() -> List[String]:
    """Get EMNIST By Class class names.

        EMNIST By Class contains 62 classes: 10 digits (0-9) and 52 letters (A-Z, a-z)
        The by-class split maintains separate classes for uppercase and lowercase letters

    Returns:
            List of 62 class name strings representing digits and letters

    Note:
            Classes are ordered as: 0-9 (digits), then A-Z (uppercase), then a-z (lowercase)
            Creates a new List each time it's called. Cache the result if used repeatedly.
    """
    var classes = List[String]()

    # Digits 0-9
    for i in range(10):
        classes.append(String(i))

    # Uppercase letters A-Z
    var uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(26):
        classes.append(String(uppercase[byte = i : i + 1]))

    # Lowercase letters a-z
    var lowercase = "abcdefghijklmnopqrstuvwxyz"
    for i in range(26):
        classes.append(String(lowercase[byte = i : i + 1]))

    return classes^


# ============================================================================
# EMNIST Class Names (By Merge Split)
# ============================================================================


def EMNIST_BYMERGE_CLASSES() -> List[String]:
    """Get EMNIST By Merge class names.

        EMNIST By Merge contains 36 classes where uppercase and lowercase letters
        are merged into single classes (e.g., 'A' and 'a' -> 'A')

    Returns:
            List of 36 class name strings: 10 digits and 26 merged letter classes

    Note:
            Classes are ordered as: 0-9 (digits), then A-Z (merged uppercase/lowercase)
            Creates a new List each time it's called. Cache the result if used repeatedly.
    """
    var classes = List[String]()

    # Digits 0-9
    for i in range(10):
        classes.append(String(i))

    # Uppercase letters A-Z (merged with lowercase)
    var uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(26):
        classes.append(String(uppercase[byte = i : i + 1]))

    return classes^


# ============================================================================
# EMNIST Class Names (Digits Only)
# ============================================================================


def EMNIST_DIGITS_CLASSES() -> List[String]:
    """Get EMNIST Digits class names.

        EMNIST Digits contains only the 10 digit classes (0-9)
        This is equivalent to MNIST

    Returns:
            List of 10 class name strings: "0", "1", ..., "9"

    Note:
            Creates a new List each time it's called. Cache the result if used repeatedly.
    """
    var classes = List[String]()
    for i in range(10):
        classes.append(String(i))
    return classes^


# ============================================================================
# EMNIST Class Names (Letters Only)
# ============================================================================


def EMNIST_LETTERS_CLASSES() -> List[String]:
    """Get EMNIST Letters class names.

        EMNIST Letters contains only letter characters with uppercase and lowercase
        treated as separate classes. Total of 52 classes (A-Z, a-z)

    Returns:
            List of 52 class name strings: A-Z followed by a-z

    Note:
            Creates a new List each time it's called. Cache the result if used repeatedly.
    """
    var classes = List[String]()

    # Uppercase letters A–Z
    var uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(26):
        classes.append(String(uppercase[byte = i : i + 1]))

    # Lowercase letters a–z
    var lowercase = "abcdefghijklmnopqrstuvwxyz"
    for i in range(26):
        classes.append(String(lowercase[byte = i : i + 1]))

    return classes^


# ============================================================================
# DatasetInfo Struct
# ============================================================================


struct DatasetInfo(Copyable, Movable):
    """Dataset metadata container.

    Provides convenient access to dataset properties like number of classes,
    image shape, and class names for supported datasets

    Supported datasets:
        - "cifar10": CIFAR-10 (10 classes, 32x32 RGB)
        - "emnist_balanced": EMNIST Balanced split (47 classes, 28x28 grayscale)
        - "emnist_byclass": EMNIST By Class split (62 classes, 28x28 grayscale)
        - "emnist_bymerge": EMNIST By Merge split (47 classes, 28x28 grayscale)
        - "emnist_digits": EMNIST Digits (10 classes, 28x28 grayscale)
        - "emnist_letters": EMNIST Letters (52 classes, 28x28 grayscale)

    Attributes:
        dataset_name: Name of the dataset (e.g., "cifar10", "emnist_balanced")

    Example:
        ```mojo
        var info = DatasetInfo("cifar10")
        var num_classes = info.num_classes()           # Returns 10
        var shape = info.image_shape()                 # Returns [3, 32, 32]
        var class_name = info.class_name(0)           # Returns "airplane"
        var classes = info.class_names()              # Returns List of all 10 classes
        ```
    """

    var dataset_name: String

    def __init__(out self, dataset_name: String) raises:
        """Initialize DatasetInfo.

        Args:
            dataset_name: Name of the dataset. Must be one of the supported datasets.

        Raises:
            Error: If dataset_name is not supported.

        Supported datasets:
        ```
            - "cifar10"
            - "emnist_balanced"
            - "emnist_byclass"
            - "emnist_bymerge"
            - "emnist_digits"
            - "emnist_letters"
        ```
        """
        # Initialize field first
        self.dataset_name = dataset_name
        # Validate dataset name
        if not self._is_valid_dataset(dataset_name):
            raise Error(
                "Unknown dataset: "
                + dataset_name
                + ". Supported: cifar10, emnist_balanced, emnist_byclass,"
                " emnist_bymerge, emnist_digits, emnist_letters"
            )

    def _is_valid_dataset(self, name: String) -> Bool:
        """Check if dataset name is valid.

        Args:
            name: Dataset name to validate

        Returns:
            True if dataset is supported, False otherwise
        """
        var valid_datasets = List[String]()
        valid_datasets.append("cifar10")
        valid_datasets.append("emnist_balanced")
        valid_datasets.append("emnist_byclass")
        valid_datasets.append("emnist_bymerge")
        valid_datasets.append("emnist_digits")
        valid_datasets.append("emnist_letters")

        for valid_name in valid_datasets:
            if name == valid_name:
                return True
        return False

    def num_classes(self) -> Int:
        """Get number of classes in the dataset.

        Returns:
            Number of classes:
            - cifar10: 10
            - emnist_balanced: 47
            - emnist_byclass: 62
            - emnist_bymerge: 36
            - emnist_digits: 10
            - emnist_letters: 52
            - -1 if dataset is not recognized (shouldn't happen after validation).
        """
        if self.dataset_name == "cifar10":
            return 10
        elif self.dataset_name == "emnist_balanced":
            return 47
        elif self.dataset_name == "emnist_byclass":
            return 62
        elif self.dataset_name == "emnist_bymerge":
            return 36
        elif self.dataset_name == "emnist_digits":
            return 10
        elif self.dataset_name == "emnist_letters":
            return 52
        else:
            # This shouldn't happen due to validation in __init__
            return -1

    def image_shape(self) -> List[Int]:
        """Get shape of individual images in the dataset.

        Returns:
            Shape as List[Int]:
            - cifar10: [3, 32, 32] (RGB, height, width)
            - emnist_*: [1, 28, 28] (grayscale, height, width)

        Note:
            The shape represents [channels, height, width] in channel-first format.
        """
        if self.dataset_name == "cifar10":
            var shape = List[Int]()
            shape.append(3)  # RGB channels
            shape.append(32)  # Height
            shape.append(32)  # Width
            return shape^
        else:
            # All EMNIST variants use 28x28 grayscale
            var shape = List[Int]()
            shape.append(1)  # Grayscale
            shape.append(28)  # Height
            shape.append(28)  # Width
            return shape^

    def class_names(self) -> List[String]:
        """Get list of all class names for the dataset.

        Returns:
            List of class name strings in order.
            -1 error if dataset is not recognized (shouldn't happen after validation).
        """
        if self.dataset_name == "cifar10":
            return CIFAR10_CLASS_NAMES()
        elif self.dataset_name == "emnist_balanced":
            return EMNIST_BALANCED_CLASSES()
        elif self.dataset_name == "emnist_byclass":
            return EMNIST_BYCLASS_CLASSES()
        elif self.dataset_name == "emnist_bymerge":
            return EMNIST_BYMERGE_CLASSES()
        elif self.dataset_name == "emnist_digits":
            return EMNIST_DIGITS_CLASSES()
        elif self.dataset_name == "emnist_letters":
            return EMNIST_LETTERS_CLASSES()
        else:
            # This shouldn't happen, but return empty list as fallback
            return List[String]()

    def class_name(self, class_idx: Int) raises -> String:
        """Get name of a specific class by index.

        Args:
            class_idx: Index of the class (0-based).

        Returns:
            Class name string at the given index.

        Raises:
            Error: If class_idx is out of range for the dataset.
        """
        if class_idx < 0 or class_idx >= self.num_classes():
            raise Error(
                "Class index "
                + String(class_idx)
                + " out of range [0, "
                + String(self.num_classes() - 1)
                + "]"
            )

        var classes = self.class_names()
        return classes[class_idx]

    def num_train_samples(self) -> Int:
        """Get number of training samples in the dataset.

        Returns:
            Number of training samples:
            - cifar10: 50000
            - emnist_balanced: ~112589
            - emnist_byclass: ~814255
            - emnist_bymerge: ~814255
            - emnist_digits: ~60000
            - emnist_letters: ~103600

        Note:
            These are approximate sizes; actual sizes may vary slightly.
        """
        if self.dataset_name == "cifar10":
            return 50000
        elif self.dataset_name == "emnist_balanced":
            return 112589
        elif self.dataset_name == "emnist_byclass":
            return 814255
        elif self.dataset_name == "emnist_bymerge":
            return 814255
        elif self.dataset_name == "emnist_digits":
            return 60000
        elif self.dataset_name == "emnist_letters":
            return 103600
        else:
            return -1

    def num_test_samples(self) -> Int:
        """Get number of test samples in the dataset.

        Returns:
            Number of test samples:
            - cifar10: 10000
            - emnist_balanced: ~18822
            - emnist_byclass: ~135800
            - emnist_bymerge: ~135800
            - emnist_digits: ~10000
            - emnist_letters: ~17383

        Note:
            These are approximate sizes; actual sizes may vary slightly.
        """
        if self.dataset_name == "cifar10":
            return 10000
        elif self.dataset_name == "emnist_balanced":
            return 18822
        elif self.dataset_name == "emnist_byclass":
            return 135800
        elif self.dataset_name == "emnist_bymerge":
            return 135800
        elif self.dataset_name == "emnist_digits":
            return 10000
        elif self.dataset_name == "emnist_letters":
            return 17383
        else:
            return -1

    def description(self) -> String:
        """Get human-readable description of the dataset.

        Returns:
            Description string including dataset name, number of classes, and image shape

        Example:
            ```mojo
            CIFAR-10: 10 classes, image shape [3, 32, 32]"
        ```
        """
        var shape = self.image_shape()
        var description = (
            self.dataset_name
            + ": "
            + String(self.num_classes())
            + " classes, image shape ["
        )
        description += (
            String(shape[0])
            + ", "
            + String(shape[1])
            + ", "
            + String(shape[2])
            + "]"
        )
        return description
