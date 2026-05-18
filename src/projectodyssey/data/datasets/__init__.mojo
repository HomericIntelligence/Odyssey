"""Dataset implementations and utilities.

Provides high-level dataset interfaces for common ML datasets including CIFAR-10.

Modules:
    cifar10: CIFAR-10 dataset wrapper for image classification

Classes:
    Dataset: Base trait for all datasets
    AnyTensorDataset: In-memory tensor dataset
    TensorDataset: Alias for AnyTensorDataset
    FileDataset: Lazy-loading file-based dataset
    CIFAR10Dataset: High-level interface for CIFAR-10 data access

Example:
    from projectodyssey.data.datasets import CIFAR10Dataset, TensorDataset

    # Create CIFAR-10 dataset
    var cifar = CIFAR10Dataset("/path/to/cifar10/data")

    # Create in-memory tensor dataset
    var dataset = TensorDataset(data_tensor, label_tensor)
"""

# Core dataset types from _datasets_core.mojo
from projectodyssey.data._datasets_core import (
    Dataset,
    AnyTensorDataset,
    TensorDataset,
    FileDataset,
)

# CIFAR-10 dataset
from projectodyssey.data.datasets.cifar10 import (
    CIFAR10Dataset,
    get_cifar10_classes,
)
