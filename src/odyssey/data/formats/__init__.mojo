"""Data Format Loaders.

Provides functions and classes for loading various data formats used in ML Odyssey

Modules:
    idx_loader: IDX file format (MNIST, EMNIST, CIFAR-10 as IDX)
    cifar_loader: CIFAR-10 and CIFAR-100 binary format loader
"""

# IDX Format Loaders
from odyssey.data.formats.idx_loader import (
    read_uint32_be,
    load_idx_labels,
    load_idx_images,
    load_idx_images_rgb,
    normalize_images,
    normalize_images_rgb,
    one_hot_encode,
    load_cifar10_batch,
)

# CIFAR Binary Format Loaders
from odyssey.data.formats.cifar_loader import CIFARLoader
from odyssey.data.constants import (
    CIFAR10_IMAGE_SIZE,
    CIFAR10_CHANNELS,
    CIFAR10_BYTES_PER_IMAGE,
    CIFAR10_NUM_CLASSES,
    CIFAR100_IMAGE_SIZE,
    CIFAR100_CHANNELS,
    CIFAR100_BYTES_PER_IMAGE,
    CIFAR100_NUM_CLASSES_FINE,
    CIFAR100_NUM_CLASSES_COARSE,
)
