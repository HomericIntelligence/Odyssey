"""
Import Validation Tests

Tests that all public imports work correctly for the shared library.
These tests verify both import functionality and basic component behavior.

"""


from std.testing import assert_true


def test_core_imports() raises:
    """Test core package imports work correctly."""
    from shared.tensor.any_tensor import AnyTensor, zeros, ones, randn
    from shared.core import relu, sigmoid, tanh, softmax, gelu

    # Test that functions are actually callable and work correctly
    var test_tensor = zeros([3, 3], DType.float32)
    assert_true(
        test_tensor.dim() == 2, "zeros should create tensor with correct rank"
    )
    var test_shape = test_tensor.shape()
    assert_true(
        test_shape[0] == 3,
        "zeros should create tensor with correct first dimension",
    )
    assert_true(
        test_shape[1] == 3,
        "zeros should create tensor with correct second dimension",
    )

    print("✓ Core imports test passed")


def test_core_layers_imports() raises:
    """Test core layer operations imports."""
    from shared.core import linear, conv2d, flatten
    from shared.core import maxpool2d, avgpool2d

    print("✓ Core layer operations imports test passed")


def test_core_activations_imports() raises:
    """Test core activation function imports."""
    from shared.core import (
        relu,
        sigmoid,
        tanh,
        softmax,
        leaky_relu,
        elu,
        gelu,
        swish,
        mish,
        selu,
    )

    print("✓ Core activation functions imports test passed")


def test_core_types_imports() raises:
    """Test core types imports."""
    from shared.tensor.any_tensor import AnyTensor
    from shared.core import FP8, BF8

    print("✓ Core types imports test passed")


def test_core_activations_direct_imports() raises:
    """Test activations are importable directly from shared.core.activation sub-module.
    """
    from shared.core.activation import relu, sigmoid, tanh, gelu

    print("✓ Core activations direct imports test passed")


def test_core_layers_direct_imports() raises:
    """Test layers are importable directly from their shared.core sub-modules.

    Note: linear, conv2d, flatten are pure functions in shared.core (not in
    shared.core.layers which contains struct-based layer wrappers like Linear).
    """
    from shared.core.linear import linear
    from shared.core.conv import conv2d
    from shared.core.shape import flatten

    print("✓ Core layers direct imports test passed")


def test_core_types_direct_imports() raises:
    """Test types are importable directly from their shared.core sub-modules.

    Note: AnyTensor lives in shared.core.any_tensor (not shared.core.types which
    contains dtype aliases like FP8, BF8, BF16).
    """
    from shared.tensor.any_tensor import AnyTensor
    from shared.core.types import FP8, BF8

    print("✓ Core types direct imports test passed")


def test_training_imports() raises:
    """Test training package imports work correctly."""
    from shared.training import SGD, MSELoss
    from shared.training import StepLR, CosineAnnealingLR, ExponentialLR
    from shared.training import EarlyStopping, ModelCheckpoint

    print("✓ Training imports test passed")


def test_training_optimizers_imports() raises:
    """Test training optimizers imports."""
    from shared.training import SGD

    print("✓ Training optimizers imports test passed")


def test_shared_optimizer_imports() raises:
    """Test that SGD, Adam, AdamW, AdaGrad, RMSprop are importable from shared package.

    Covers Issue #3745: AdaGrad and RMSprop exposed as top-level shared imports.
    """
    from shared import SGD, Adam, AdamW, AdaGrad, RMSprop

    print("✓ Shared optimizer imports test passed")


def test_training_schedulers_imports() raises:
    """Test training schedulers imports."""
    from shared.training import (
        StepLR,
        CosineAnnealingLR,
        ExponentialLR,
        WarmupLR,
        MultiStepLR,
        ReduceLROnPlateau,
    )

    print("✓ Training schedulers imports test passed")


def test_training_metrics_imports() raises:
    """Test training metrics imports."""
    # Metrics are in shared.training for now
    from shared.training import base

    print("✓ Training metrics imports test passed")


def test_training_callbacks_imports() raises:
    """Test training callbacks imports."""
    from shared.training import (
        EarlyStopping,
        ModelCheckpoint,
        LoggingCallback,
    )

    print("✓ Training callbacks imports test passed")


def test_training_optimizers_direct_imports() raises:
    """Test optimizers are importable directly from shared.training.optimizers sub-module.

    Validates the canonical import path for optimizers sub-module.

    Note: The SGD struct is defined in shared.training (importable from there).
    The shared.training.optimizers sub-module provides functional step functions
    (sgd_step, adam_step, etc.) rather than struct-based optimizers.
    """
    from shared.training.optimizers import sgd_step, adam_step

    print("✓ Training optimizers direct imports test passed")


def test_training_schedulers_direct_imports() raises:
    """Test schedulers are importable directly from shared.training.schedulers sub-module.

    Validates the canonical import path for schedulers sub-module.
    """
    from shared.training.schedulers import (
        StepLR,
        CosineAnnealingLR,
        ExponentialLR,
    )

    print("✓ Training schedulers direct imports test passed")


def test_training_base_direct_imports() raises:
    """Test base classes are importable directly from shared.training.base sub-module.

    Validates the canonical import path for base sub-module.
    """
    from shared.training.base import Callback, TrainingState

    print("✓ Training base direct imports test passed")


def test_training_loops_direct_imports() raises:
    """Test loops and base components are importable from their direct sub-modules.

    Validates the canonical import paths for loops and base sub-modules.

    Note: TrainingState is defined in shared.training.base (not shared.training.loops).
    The loops sub-module provides TrainingLoop and ValidationLoop.
    """
    from shared.training.base import TrainingState
    from shared.training.loops import TrainingLoop

    print("✓ Training loops direct imports test passed")


def test_training_callbacks_direct_imports() raises:
    """Test callbacks are importable directly from shared.training.callbacks sub-module.

    This validates the canonical import path documented in Issue #3211:
        from shared.training.callbacks import EarlyStopping

    NOTE: A negative test for the wrong import path cannot be written because
    Mojo import failures are compile-time errors, not runtime exceptions.
    There is no equivalent of pytest.raises() for compile-time errors.
    """
    from shared.training.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        LoggingCallback,
    )

    # Instantiate each type to confirm the import is functional, not just parseable
    var early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        min_delta=0.001,
        mode="min",
        verbose=False,
    )
    assert_true(
        early_stop.patience == 3, "EarlyStopping should have patience=3"
    )
    assert_true(
        early_stop.stopped == False,
        "EarlyStopping should not be stopped initially",
    )

    var checkpoint = ModelCheckpoint(
        filepath="test_checkpoint.pt",
        save_best_only=False,
        save_frequency=1,
        mode="min",
    )
    assert_true(
        checkpoint.save_count == 0,
        "ModelCheckpoint should have save_count=0 initially",
    )

    var logger = LoggingCallback(log_interval=2)
    assert_true(
        logger.log_interval == 2, "LoggingCallback should have log_interval=2"
    )
    assert_true(
        logger.log_count == 0,
        "LoggingCallback should have log_count=0 initially",
    )

    print("✓ Training callbacks direct imports test passed")


def test_training_loops_imports() raises:
    """Test training loops imports."""
    from shared.training import TrainingState, Callback

    print("✓ Training loops imports test passed")


def test_data_imports() raises:
    """Test data package imports work correctly."""
from shared.data.datasets import AnyTensorDataset, CIFAR10Dataset, Dataset, EMNISTDataset

    print("✓ Data imports test passed")


def test_data_datasets_imports() raises:
    """Test data datasets imports."""
    from shared.data import Dataset, AnyTensorDataset, FileDataset

    print("✓ Data datasets imports test passed")


def test_data_loaders_imports() raises:
    """Test data loaders imports."""
    from shared.data import Batch

    print("✓ Data loaders imports test passed")


def test_data_transforms_imports() raises:
    """Test data transforms imports."""
    # Data transforms are provided as utility functions, not classes
    from shared.data import normalize_images, one_hot_encode

    print("✓ Data transforms imports test passed")


def test_data_datasets_direct_imports() raises:
    """Test datasets are importable directly from shared.data.datasets sub-module.
    """
    from shared.data.datasets import Dataset, AnyTensorDataset

    print("✓ Data datasets direct imports test passed")


def test_data_loaders_direct_imports() raises:
    """Test loaders are importable directly from shared.data.loaders sub-module.
    """
    from shared.data.loaders import Batch

    print("✓ Data loaders direct imports test passed")


def test_utils_imports() raises:
    """Test utils package imports work correctly."""
    from shared.utils import Logger, LogLevel, get_logger
    from shared.utils import load_config, save_config, Config

    print("✓ Utils imports test passed")


def test_utils_logging_imports() raises:
    """Test utils logging imports."""
    from shared.utils import (
        Logger,
        LogLevel,
        get_logger,
        StreamHandler,
        FileHandler,
    )

    print("✓ Utils logging imports test passed")


def test_utils_visualization_imports() raises:
    """Test utils visualization imports."""
    # Visualization functions require Python interop
    # For now, just verify utils imports work
    from shared.utils import Logger

    print("✓ Utils visualization imports test passed")


def test_utils_config_imports() raises:
    """Test utils config imports."""
    from shared.utils import Config, load_config, save_config, ConfigValidator

    print("✓ Utils config imports test passed")


def test_root_imports() raises:
    """Test root package convenience imports work."""
    # Root package doesn't re-export all components
    # Users should import from subpackages
    from shared.tensor.any_tensor import AnyTensor
    from shared.training import SGD
    from shared.utils import Logger

    print("✓ Root imports test passed")


def test_subpackage_imports() raises:
    """Test importing subpackages themselves."""
    from shared import core, training, data, utils

    print("✓ Subpackage imports test passed")


def test_nested_optimizer_imports() raises:
    """Test nested imports from optimizer subpackages."""
    from shared.training import SGD

    print("✓ Nested optimizer imports test passed")


def test_nested_scheduler_imports() raises:
    """Test nested imports from scheduler subpackages."""
    from shared.training import StepLR, CosineAnnealingLR

    print("✓ Nested scheduler imports test passed")


def test_nested_metric_imports() raises:
    """Test nested imports from metrics subpackages."""
    # Metrics are in shared.training
    from shared.training import Callback

    print("✓ Nested metric imports test passed")


def test_version_info() raises:
    """Test version info is accessible and has proper format."""
    from shared import VERSION, AUTHOR, LICENSE

    # Critical validation - ensure values are not empty/None
    assert_true(VERSION != "", "VERSION should not be empty")
    assert_true(AUTHOR != "", "AUTHOR should not be empty")
    assert_true(LICENSE != "", "LICENSE should not be empty")

    # Ensure these are actual string values, not None
    assert_true(VERSION.__len__() > 0, "VERSION string should have length > 0")
    assert_true(AUTHOR.__len__() > 0, "AUTHOR string should have length > 0")
    assert_true(LICENSE.__len__() > 0, "LICENSE string should have length > 0")

    # Test version format follows semantic versioning (major.minor.patch)
    var version_parts = VERSION.split(".")
    assert_true(
        version_parts.__len__() == 3,
        "Version should have 3 parts (major.minor.patch)",
    )

    # Test that version parts are numeric by checking they only contain digits
    for i in range(version_parts.__len__()):
        var part = version_parts[i]
        assert_true(part.__len__() > 0, "Version part should not be empty")

        # Check each character is a digit (0-9)
        var is_numeric = True
        var part_bytes = part.as_bytes()
        for j in range(part.__len__()):
            var ch = Int(part_bytes[j])
            if ch < 48 or ch > 57:  # ord("0") == 48, ord("9") == 57
                is_numeric = False
                break

        assert_true(is_numeric, "Version part should contain only digits")

    print("✓ Version info test passed")


def test_training_dataloader_imports() raises:
    """Test DataLoader and DataBatch are importable from trainer_interface.

    Verifies Issue #3851: DataLoader and DataBatch defined in
    shared/training/trainer_interface.mojo.
    """
    from shared.training.trainer_interface import DataLoader, DataBatch

    print("✓ Training DataLoader/DataBatch package imports test passed")


def main() raises:
    """Run all test_imports tests."""
    print("Running test_imports tests...")

    test_core_imports()
    print("✓ test_core_imports")

    test_core_layers_imports()
    print("✓ test_core_layers_imports")

    test_core_activations_imports()
    print("✓ test_core_activations_imports")

    test_core_types_imports()
    print("✓ test_core_types_imports")

    test_core_activations_direct_imports()
    print("✓ test_core_activations_direct_imports")

    test_core_layers_direct_imports()
    print("✓ test_core_layers_direct_imports")

    test_core_types_direct_imports()
    print("✓ test_core_types_direct_imports")

    test_training_imports()
    print("✓ test_training_imports")

    test_training_optimizers_imports()
    print("✓ test_training_optimizers_imports")

    test_shared_optimizer_imports()
    print("✓ test_shared_optimizer_imports")

    test_training_schedulers_imports()
    print("✓ test_training_schedulers_imports")

    test_training_metrics_imports()
    print("✓ test_training_metrics_imports")

    test_training_callbacks_imports()
    print("✓ test_training_callbacks_imports")

    test_training_optimizers_direct_imports()
    print("✓ test_training_optimizers_direct_imports")

    test_training_schedulers_direct_imports()
    print("✓ test_training_schedulers_direct_imports")

    test_training_base_direct_imports()
    print("✓ test_training_base_direct_imports")

    test_training_loops_direct_imports()
    print("✓ test_training_loops_direct_imports")

    test_training_callbacks_direct_imports()
    print("✓ test_training_callbacks_direct_imports")

    test_training_loops_imports()
    print("✓ test_training_loops_imports")

    test_data_imports()
    print("✓ test_data_imports")

    test_data_datasets_imports()
    print("✓ test_data_datasets_imports")

    test_data_loaders_imports()
    print("✓ test_data_loaders_imports")

    test_data_transforms_imports()
    print("✓ test_data_transforms_imports")

    test_data_datasets_direct_imports()
    print("✓ test_data_datasets_direct_imports")

    test_data_loaders_direct_imports()
    print("✓ test_data_loaders_direct_imports")

    test_utils_imports()
    print("✓ test_utils_imports")

    test_utils_logging_imports()
    print("✓ test_utils_logging_imports")

    test_utils_visualization_imports()
    print("✓ test_utils_visualization_imports")

    test_utils_config_imports()
    print("✓ test_utils_config_imports")

    test_root_imports()
    print("✓ test_root_imports")

    test_subpackage_imports()
    print("✓ test_subpackage_imports")

    test_nested_optimizer_imports()
    print("✓ test_nested_optimizer_imports")

    test_nested_scheduler_imports()
    print("✓ test_nested_scheduler_imports")

    test_nested_metric_imports()
    print("✓ test_nested_metric_imports")

    test_version_info()
    print("✓ test_version_info")

    test_training_dataloader_imports()
    print("✓ test_training_dataloader_imports")

    print("\nAll test_imports tests passed!")
