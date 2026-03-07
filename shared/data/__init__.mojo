"""Data Loading Package

Provides data loaders for common ML datasets and utilities for handling various data formats

Modules:
    formats: Low-level data format loaders (IDX, CIFAR, etc.)
    datasets: High-level dataset interfaces (CIFAR-10, EMNIST, etc.)

Architecture:
    - `formats` provides low-level file I/O and format parsing
    - `datasets` provides high-level, user-friendly interfaces
    - All data is returned as ExTensor for consistency with core library

Example:
    from shared.data import load_emnist_train, load_emnist_test

    # Load EMNIST
    images, labels = load_emnist_train("/path/to/emnist", split="balanced")

    # Or use the EMNISTDataset class directly
    from shared.data import EMNISTDataset
    dataset = EMNISTDataset("/path/to/emnist", split="balanced", train=True)
    sample_img, sample_label = dataset[0]
    ```
"""

# Package version
from shared.version import VERSION

# ============================================================================
# Format Loaders (Low-Level File I/O)
# ============================================================================

# IDX format utilities
from shared.data.formats import (
    read_uint32_be,  # Read big-endian uint32
    load_idx_labels,  # Load IDX label file
    load_idx_images,  # Load IDX grayscale images
    load_idx_images_rgb,  # Load IDX RGB images
    normalize_images,  # Normalize uint8 images to [0, 1] float32
    normalize_images_rgb,  # Normalize RGB images with ImageNet parameters
    one_hot_encode,  # One-hot encode integer labels
    load_cifar10_batch,  # Load single CIFAR-10 batch
)

# ============================================================================
# Dataset Constants and Metadata
# ============================================================================

# Dataset-specific constants and metadata
from shared.data.constants import (
    CIFAR10_IMAGE_SIZE,  # CIFAR-10 image size (32x32)
    CIFAR10_CHANNELS,  # CIFAR-10 color channels (3)
    CIFAR10_BYTES_PER_IMAGE,  # CIFAR-10 bytes per image (3073)
    CIFAR10_NUM_CLASSES,  # CIFAR-10 number of classes (10)
    CIFAR100_IMAGE_SIZE,  # CIFAR-100 image size (32x32)
    CIFAR100_CHANNELS,  # CIFAR-100 color channels (3)
    CIFAR100_BYTES_PER_IMAGE,  # CIFAR-100 bytes per image (3074)
    CIFAR100_NUM_CLASSES_FINE,  # CIFAR-100 fine-grained classes (100)
    CIFAR100_NUM_CLASSES_COARSE,  # CIFAR-100 coarse classes (20)
    CIFAR10_CLASS_NAMES,  # CIFAR-10 class names (10 classes)
    EMNIST_BALANCED_CLASSES,  # EMNIST Balanced class names (47 classes)
    EMNIST_BYCLASS_CLASSES,  # EMNIST By Class class names (62 classes)
    EMNIST_BYMERGE_CLASSES,  # EMNIST By Merge class names (36 classes)
    EMNIST_DIGITS_CLASSES,  # EMNIST Digits class names (10 classes)
    EMNIST_LETTERS_CLASSES,  # EMNIST Letters class names (52 classes)
    DatasetInfo,  # Dataset metadata container
)

# ============================================================================
# Dataset Loaders (High-Level Interfaces)
# ============================================================================

# Dataset classes and loaders
from shared.data.datasets import (
    Dataset,  # Base dataset interface
    ExTensorDataset,  # In-memory tensor dataset wrapper
    FileDataset,  # File-based lazy-loading dataset
    CIFAR10Dataset,  # CIFAR-10 dataset with train/test splits
    get_cifar10_classes,  # CIFAR-10 class names
)

# Dataset wrappers and utilities
from shared.data.dataset_with_transform import (
    TransformedDataset,  # Wrapper that applies transforms to data
)

# EMNIST dataset is defined in _datasets_core.mojo
from shared.data._datasets_core import (
    EMNISTDataset,  # EMNIST dataset with multiple splits
    load_emnist_train,  # Load EMNIST training set
    load_emnist_test,  # Load EMNIST test set
)

# ============================================================================
# Public API
# ============================================================================

# NOTE: Mojo does not support __all__ for controlling exports.
# All imported symbols are automatically available to package consumers.
#
# High-level usage:
#   from shared.data import load_emnist_train, EMNISTDataset
#   images, labels = load_emnist_train("/path/to/emnist", split="balanced")
#
# Low-level usage:
#   from shared.data import load_idx_images, load_idx_labels, read_uint32_be
#   images = load_idx_images("/path/to/custom-images-idx3-ubyte")

# ============================================================================
# Transform Base Classes and Utilities
# ============================================================================

from shared.data.random_transform_base import (
    RandomTransformBase,  # Base for probabilistic transforms
    random_float,  # Random float generation utility
)

# Core transforms (most commonly used)
from shared.data.transforms import (
    Normalize,  # Normalize tensor values: (x - mean) / std
    Compose,  # Chain multiple transforms into a single transform
)

# ============================================================================
# Data Loaders and Samplers
# ============================================================================

# Core data loading infrastructure
from shared.data.loaders import (
    Batch,  # Batch container with data, labels, and indices
    BatchLoader,  # Main data loader with shuffling and batching
)

# Sampling strategies for data iteration
from shared.data.samplers import (
    Sampler,  # Base sampler interface
    SequentialSampler,  # Sequential ordering without shuffling
    RandomSampler,  # Random permutation with shuffling
    WeightedSampler,  # Weighted sampling (with replacement)
)

# Prefetching utilities
from shared.data.prefetch import (
    PrefetchBuffer,  # Ring buffer for batches
    PrefetchDataLoader,  # Loader with prefetching
)

# Caching utilities
from shared.data.cache import (
    CachedDataset,  # Dataset wrapper with caching
)

# ============================================================================
# Batch Processing Utilities
# ============================================================================

from shared.data.batch_utils import (
    extract_batch,
    extract_batch_pair,
    compute_num_batches,
    get_batch_indices,
)
