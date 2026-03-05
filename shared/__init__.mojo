"""
Shared Library for ML Odyssey Paper Implementations

This package provides reusable ML/AI components including:
- Core neural network components (layers, activations, tensors)
- Training infrastructure (optimizers, schedulers, metrics, callbacks)
- Data processing utilities (datasets, loaders, transforms)
- Helper utilities (logging, visualization, configuration)

Usage:
    # Import commonly used components directly
    from shared import Linear, relu, sigmoid, tanh, softmax
    from shared import StepLR, CosineAnnealingLR
    from shared import EarlyStopping, ModelCheckpoint
    from shared import Logger, plot_training_curves

    # Import from specific modules for less common items
    from shared.core.layers import Conv2dLayer, ReLULayer, DropoutLayer
    from shared.data.transforms import Normalize, Compose
    from shared.training.metrics.loss_tracker import LossTracker

Example:
    ```mojo
    from shared import Linear, relu, SGD

    # Create a linear layer and apply activation
    layer = Linear(784, 256)
    optimizer = SGD(learning_rate=0.01)
    ```

Note:
    Mojo v0.26.1+ does not support ``__all__`` module-level assignments.
    In Mojo, all public symbols (those not prefixed with ``_``) are automatically
    exported when the module is imported. The convenience re-exports below are
    uncommented as implementations reach stable naming conventions.

    Some components are available under different names than originally planned:
    - Conv2D -> Conv2dLayer (shared.core.layers.conv2d)
    - ReLU -> ReLULayer (shared.core.layers.relu)
    - Dropout -> DropoutLayer (shared.core.layers.dropout)
    - Tensor -> ExTensor (shared.core.extensor)
    - Accuracy -> AccuracyMetric (shared.training.metrics.accuracy)

    Not yet implemented (see Issue #49 for tracking):
    - Sequential, MaxPool2D, Flatten as standalone structs
    - AdamW optimizer
    - train_epoch, validate_epoch functions
    - TensorDataset, ImageDataset, DataLoader (stable names)
    - ToTensor transform

Placeholder tests in tests/shared/integration/test_packaging.mojo require implementation.
See Issue #3033 for tracking: 12 placeholder tests for packaging integration.
Tests require corresponding modules to be implemented first.
"""

# Package version and metadata
from shared.version import VERSION, AUTHOR, LICENSE

# ============================================================================
# Core Exports - Most commonly used components
# ============================================================================

# Core layers - Linear is implemented; others use different names (see module docstring)
from shared.core.layers.linear import Linear

# Core activations (function form) - all implemented in shared.core.activation
# Note: module is 'activation' (not 'activations')
from shared.core.activation import relu, sigmoid, tanh, softmax

# Core module system - Module trait is implemented; Sequential not yet available
from shared.core.module import Module

# Core tensors - ExTensor is the tensor type; Tensor alias not available
# NOT YET IMPLEMENTED (see Issue #49): Tensor, zeros, ones, randn
# Use: from shared.core.extensor import ExTensor

# Training optimizers - SGD and Adam available in shared.training and shared.autograd
# NOT YET IMPLEMENTED (see Issue #49): AdamW
# Note: Use 'from shared.training import SGD' or 'from shared.autograd.optimizers import Adam'

# Training schedulers - both implemented
from shared.training.schedulers import StepLR, CosineAnnealingLR

# Training metrics - LossTracker implemented; Accuracy is AccuracyMetric
# NOT YET IMPLEMENTED under these names (see Issue #49): Accuracy
# Use: from shared.training.metrics.loss_tracker import LossTracker
# Use: from shared.training.metrics.accuracy import AccuracyMetric

# Training callbacks - both implemented
from shared.training.callbacks import EarlyStopping, ModelCheckpoint

# Training loops - not yet implemented (see Issue #49)
# NOT YET IMPLEMENTED: train_epoch, validate_epoch

# Data components - Normalize and Compose implemented; others unavailable or renamed
# NOT YET IMPLEMENTED under these names (see Issue #49):
#   TensorDataset -> use ExTensorDataset (shared.data._datasets_core)
#   ImageDataset -> not yet available
#   DataLoader -> partial stub only
#   ToTensor -> not yet implemented
# Use: from shared.data.transforms import Normalize, Compose

# Utils - both implemented
from shared.utils.logging import Logger
from shared.utils.visualization import plot_training_curves

# ============================================================================
# Convenience: Make subpackages accessible
# ============================================================================
# This allows users to do: from shared import core, training, data, utils
# Then access via: shared.core.layers.Linear, shared.training.optimizers.SGD
#
# Public API (modules and symbols exposed at package level):
# - VERSION, AUTHOR, LICENSE - Package metadata
# - core - Core neural network components
# - training - Training infrastructure and optimizers
# - data - Data loading and transformation utilities
# - utils - Helper utilities
# - autograd - Automatic differentiation (when available)
# - testing - Test utilities and fixtures
#
from shared import core
from shared import training
from shared import data
from shared import utils
from shared import autograd
from shared import testing
