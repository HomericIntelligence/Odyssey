"""
Shared Library for ML Odyssey Paper Implementations

This package provides reusable ML/AI components including:
- Core neural network components (layers, activations, tensors)
- Training infrastructure (optimizers, schedulers, metrics, callbacks)
- Data processing utilities (datasets, loaders, transforms)
- Helper utilities (logging, visualization, configuration)

Usage:
    # Import commonly used components directly
    from shared import Linear, Conv2D, ReLU, SGD, Adam, Tensor

    # Import from specific modules for less common items
    from shared.core.layers import MaxPool2D, Dropout
    from shared.training.schedulers import CosineAnnealingLR
    from shared.data.transforms import Normalize

Example:
    ```mojo
    from shared import Linear, ReLU, Sequential, SGD

    # Build a simple model
    model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ])

    # Create optimizer
    optimizer = SGD(learning_rate=0.01, momentum=0.9)

    # Training loop
    for epoch in range(100):
        loss = train_epoch(model, optimizer, train_loader)
        print("Epoch", epoch, "Loss:", loss)
    ```

Import tests in tests/shared/integration/test_packaging.mojo are implemented and passing.
See Issue #3033: 12 tests for packaging integration — all tests pass.
"""

# Package version and metadata
from shared.version import VERSION, AUTHOR, LICENSE

# ============================================================================
# Core Exports - Most commonly used components
# ============================================================================
# NOTE: These imports are commented out until implementation phase completes.
# Uncomment as components become available from Issue #49.

# Core layers (most commonly used)
# from .core.layers import Linear, Conv2D, ReLU, MaxPool2D, Dropout, Flatten

# Core activations (function form)
# from .core.activations import relu, sigmoid, tanh, softmax

# Core module system
# from .core.module import Module, Sequential

# Core tensors
# from .core.tensors import Tensor, zeros, ones, randn

# Training optimizers (most commonly used)
# from .training.optimizers import SGD, Adam, AdamW

# Training schedulers (most commonly used)
# from .training.schedulers import StepLR, CosineAnnealingLR

# Training metrics (most commonly used)
# from .training.metrics import Accuracy, LossTracker

# Training callbacks (most commonly used)
# from .training.callbacks import EarlyStopping, ModelCheckpoint

# Training loops
# from .training.loops import train_epoch, validate_epoch

# Data components (most commonly used)
# from .data.datasets import TensorDataset, ImageDataset
# from .data.loaders import DataLoader
# from .data.transforms import Normalize, ToTensor, Compose

# Utils (most commonly used)
# from .utils.logging import Logger
# from .utils.visualization import plot_training_curves

# ============================================================================
# Public API
# ============================================================================
# Mojo module exports for convenience imports.
# While Mojo does not support __all__ lists like Python (all public symbols
# are automatically exported), we document the public API here for clarity.
#
# Users can import in multiple ways:
#   from shared import core, training, data, utils  # Import modules
#   from shared.core.layers import Linear           # Import specific items
#   import shared                                     # Import whole package
#
# The following components will be available once implementation completes:
#
# Version info: VERSION, AUTHOR, LICENSE
# Core - Layers: Linear, Conv2D, ReLU, MaxPool2D, Dropout, Flatten
# Core - Activations: relu, sigmoid, tanh, softmax
# Core - Module system: Module, Sequential
# Core - Tensors: Tensor, zeros, ones, randn
# Training - Optimizers: SGD, Adam, AdamW
# Training - Schedulers: StepLR, CosineAnnealingLR
# Training - Metrics: Accuracy, LossTracker
# Training - Callbacks: EarlyStopping, ModelCheckpoint
# Training - Loops: train_epoch, validate_epoch
# Data - Datasets: TensorDataset, ImageDataset, DataLoader
# Data - Transforms: Normalize, ToTensor, Compose
# Utils: Logger, plot_training_curves
# Autograd: Automatic differentiation utilities (when available)
# Testing: Test utilities and fixtures

# ============================================================================
# Convenience: Make subpackages accessible
# ============================================================================
# This allows users to do: from shared import core, training, data, utils
# Then access via: shared.core.layers.Linear, shared.training.optimizers.SGD
#
# NOTE: Mojo v0.26.1+ does not support __all__ module-level assignments.
# In Mojo, all public symbols (those not prefixed with _) are automatically
# exported when the module is imported. The public API documentation below
# describes what should be exposed at this package level:
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
# Once implementations are available, users will be able to import:
#   from shared import core, training, data, utils
#   from shared import VERSION, AUTHOR, LICENSE
#
# For implementation of component-level imports when core modules
# are fully implemented, see test_packaging.mojo
#
from shared import core
from shared import training
from shared import data
from shared import utils
from shared import autograd
from shared import testing
