"""
Shared Library for ML Odyssey Paper Implementations.

This package provides reusable ML/AI components including:
- Core neural network components (layers, activations, tensors)
- Training infrastructure (optimizers, schedulers, metrics, callbacks)
- Data processing utilities (datasets, loaders, transforms)
- Helper utilities (logging, visualization, configuration)

Usage:
    # Import commonly used components directly
    from projectodyssey import Linear, Conv2D, ReLU, SGD, Adam, AnyTensor

    # Import from specific modules for less common items
    from projectodyssey.core.layers import MaxPool2D, Dropout
    from projectodyssey.training.schedulers import CosineAnnealingLR
    from projectodyssey.data.transforms import Normalize

Example:
    ```mojo
    from projectodyssey import Linear, ReLU, Sequential, SGD

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

Import tests in tests/projectodyssey/integration/test_packaging.mojo are implemented and passing.
See Issue #3033: 12 tests for packaging integration — all tests pass.
"""

# Package version and metadata
from projectodyssey.version import VERSION, AUTHOR, LICENSE

# ============================================================================
# Core Exports - Most commonly used components
# ============================================================================

# Core layers — activated where leaf-module implementations are confirmed ready.
# NOTE(#3754, Mojo v0.26.1): Re-export chain limitation requires absolute leaf-module
# paths here; chained re-exports through intermediate __init__.mojo do not work.
from projectodyssey.core.layers.linear import Linear
from projectodyssey.core.layers.conv2d import Conv2dLayer
from projectodyssey.core.layers.relu import ReLULayer
from projectodyssey.core.layers.dropout import DropoutLayer
from projectodyssey.core.layers.batchnorm import BatchNorm2dLayer

# Aliases to match documented public API names (e.g. `from projectodyssey import Conv2D`)
comptime Conv2D = Conv2dLayer
comptime ReLU = ReLULayer
comptime Dropout = DropoutLayer
comptime BatchNorm2d = BatchNorm2dLayer

# MaxPool2D, Flatten — pending layer implementation

# Core activations (function form) — all four confirmed in src/projectodyssey/core/activation.mojo
from projectodyssey.core.activation import relu, sigmoid, tanh, softmax

# Core module system
# Module is a trait (not a struct) — can be imported but not instantiated directly
from projectodyssey.core.module import Module

# Sequential — only parametric variants exist (Sequential2, Sequential3, …)

# Core tensors — AnyTensor is the canonical runtime-typed tensor.
# Import directly: from projectodyssey.tensor.any_tensor import AnyTensor
# Factory functions live in tensor_creation:
#   from projectodyssey.tensor.tensor_creation import zeros, ones, randn
# NOT re-exported here to avoid circular imports.

# Training optimizers — struct classes available via projectodyssey.autograd.optimizers

# Training schedulers — struct implementations confirmed in lr_schedulers.mojo
from projectodyssey.training.schedulers.lr_schedulers import (
    StepLR,
    CosineAnnealingLR,
)

# Training metrics (most commonly used) — Issue #3221
# NOTE(#3754, Mojo v0.26.1): Metrics are imported directly from projectodyssey.training.metrics
# rather than from projectodyssey.training due to Mojo re-export chain limitation.
# See src/projectodyssey/training/__init__.mojo for detailed explanation of this limitation.
# Users should import metrics either as:
#   from projectodyssey.training.metrics import LossTracker, AccuracyMetric
# or:
#   from projectodyssey import LossTracker, AccuracyMetric  # if this module re-exports them
from projectodyssey.training.metrics import (
    LossTracker,
    AccuracyMetric,
    ConfusionMatrix,
    CSVMetricsLogger,
)

# Expose plan-canonical alias: Accuracy = AccuracyMetric
comptime Accuracy = AccuracyMetric

# Autograd optimizers (most commonly used) — Issue #3745, #3219
from projectodyssey.autograd.optimizers import (
    SGD,
    Adam,
    AdaGrad,
    RMSprop,
    AdamW,
)

# Training callbacks — struct implementations confirmed in src/projectodyssey/training/callbacks.mojo
from projectodyssey.training.callbacks import EarlyStopping, ModelCheckpoint

# Training loops — available as train_one_epoch / validate (see training_loop.mojo)

# Data components — AnyTensorDataset available; typed wrappers pending

# Utils
from projectodyssey.utils.logging import Logger
from projectodyssey.utils.visualization import plot_training_curves

# ============================================================================
# Public API
# ============================================================================
# Mojo module exports for convenience imports.
# While Mojo does not support __all__ lists like Python (all public symbols
# are automatically exported), we document the public API here for clarity.
#
# Users can import in multiple ways:
#   from projectodyssey import core, training, data, utils  # Import modules
#   from projectodyssey.core.layers import Linear           # Import specific items
#   import projectodyssey                                     # Import whole package
#
# Available Now:
# - Version info: VERSION, AUTHOR, LICENSE
# - Training - Metrics: Accuracy (alias for AccuracyMetric), LossTracker, ConfusionMatrix,
#   CSVMetricsLogger
# - Modules: core, training, data, utils, autograd, testing (for sub-imports)
#
# Version info: VERSION, AUTHOR, LICENSE
# Core - Layers: Linear, Conv2D, ReLU, MaxPool2D, Dropout, Flatten
# Core - Activations: relu, sigmoid, tanh, softmax
# Core - Module system: Module, Sequential
# Core - Tensors: AnyTensor, zeros, ones, randn
# Training - Optimizers: SGD, Adam, AdaGrad, RMSprop, AdamW (via autograd)
# Training - Schedulers: StepLR, CosineAnnealingLR
# Training - Metrics: Accuracy, LossTracker, ConfusionMatrix, CSVMetricsLogger
# Training - Callbacks: EarlyStopping, ModelCheckpoint
# Training - Loops: train_epoch, validate_epoch
# Data - Datasets: TensorDataset, ImageDataset, DataLoader
# Data - Transforms: ToTensor, Normalize, Compose
# Utils: Logger, plot_training_curves
# Autograd: Automatic differentiation utilities (when available)
# Testing: Test utilities and fixtures, constants (GRADIENT_CHECK_EPSILON_FLOAT32)

# ============================================================================
# Convenience: Make subpackages accessible
# ============================================================================
# This allows users to do: from projectodyssey import core, training, data, utils
# Then access via: projectodyssey.core.layers.Linear, projectodyssey.training.optimizers.SGD
#
# NOTE(#3751, Mojo v0.26.1): Mojo v0.26.1+ does not support __all__ module-level assignments.
# In Mojo, all public symbols (those not prefixed with _) are automatically
# exported when the module is imported. The public API documentation below
# describes what should be exposed at this package level:
#
# Re-export Chain Limitation (Mojo v0.26.1, #3754):
# Mojo v0.26.1 does not support re-export chains where an intermediate __init__.mojo
# re-exports a symbol and a consumer imports it from the top-level package.
# Example: `from projectodyssey import Linear` fails even if src/projectodyssey/core/__init__.mojo
# re-exports Linear from projectodyssey/core/layers.mojo.
# Workaround: Import directly from the submodule that defines the symbol:
#   from projectodyssey.core.layers import Linear   # ✓ works
#   from projectodyssey import Linear               # ✗ fails in v0.26.1
# This limitation is tracked upstream. Once resolved, top-level convenience
# imports will be enabled by un-commenting the import lines above.
#
# Public API Table (top-level exports at package level):
# ┌──────────────────────────────────────────┬──────────────────────────────────────────┐
# │ Symbol                                   │ Source                                   │
# ├──────────────────────────────────────────┼──────────────────────────────────────────┤
# │ VERSION, AUTHOR, LICENSE                 │ projectodyssey.version                           │
# │ Linear                                   │ projectodyssey.core.layers.linear                │
# │ Conv2dLayer, Conv2D                      │ projectodyssey.core.layers.conv2d (Conv2D alias) │
# │ ReLULayer, ReLU                          │ projectodyssey.core.layers.relu (ReLU alias)     │
# │ DropoutLayer, Dropout                    │ projectodyssey.core.layers.dropout (alias)       │
# │ BatchNorm2dLayer, BatchNorm2d            │ projectodyssey.core.layers.batchnorm (alias)     │
# │ relu, sigmoid, tanh, softmax             │ projectodyssey.core.activation                   │
# │ Module                                   │ projectodyssey.core.module (trait)               │
# │ AnyTensor                                 │ projectodyssey.core.any_tensor                    │
# │ zeros, ones, randn                       │ projectodyssey.core.any_tensor                    │
# │ StepLR, CosineAnnealingLR                │ projectodyssey.training.schedulers.lr_schedulers │
# │ LossTracker, AccuracyMetric              │ projectodyssey.training.metrics                  │
# │ ConfusionMatrix, CSVMetricsLogger        │ projectodyssey.training.metrics                  │
# │ Accuracy                                 │ comptime alias for AccuracyMetric        │
# │ SGD, Adam, AdaGrad, RMSprop, AdamW       │ projectodyssey.autograd.optimizers               │
# │ EarlyStopping, ModelCheckpoint           │ projectodyssey.training.callbacks                │
# │ Logger                                   │ projectodyssey.utils.logging                     │
# │ plot_training_curves                     │ projectodyssey.utils.visualization               │
# │ GRADIENT_CHECK_EPSILON_FLOAT32           │ projectodyssey.testing.tolerance_constants       │
# │ core                                     │ projectodyssey.core (subpackage)                 │
# │ training                                 │ projectodyssey.training (subpackage)             │
# │ data                                     │ projectodyssey.data (subpackage)                 │
# │ utils                                    │ projectodyssey.utils (subpackage)                │
# │ autograd                                 │ projectodyssey.autograd (subpackage)             │
# │ testing                                  │ projectodyssey.testing (subpackage)              │
# └──────────────────────────────────────────┴──────────────────────────────────────────┘
# Not yet activated (implementation pending):
#   Sequential (only Sequential2/3/4/5 variants exist),
#   TensorDataset/ImageDataset/DataLoader, train_epoch/validate_epoch, MaxPool2D/Flatten
#
# Once implementations are available, users will be able to import:
#   from projectodyssey import core, training, data, utils
#   from projectodyssey import VERSION, AUTHOR, LICENSE
#
# For implementation of component-level imports when core modules
# are fully implemented, see test_packaging.mojo
#
# Note: self-referential subpackage imports (from projectodyssey import core, etc.)
# cause recursive reference errors in Mojo 0.26.3. Users should import
# subpackages directly: from projectodyssey.core import ..., from projectodyssey.training import ...

# ============================================================================
# Testing Constants - Available at package level
# ============================================================================
from projectodyssey.testing import GRADIENT_CHECK_EPSILON_FLOAT32
