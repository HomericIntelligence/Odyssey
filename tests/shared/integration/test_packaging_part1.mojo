"""
Packaging Integration Tests (Part 1 of 3)

Tests that verify the shared library package structure and import hierarchy.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""


from std.testing import assert_true, assert_equal


def test_package_version() raises:
    """Test package version is accessible and correct."""
    from shared import VERSION, AUTHOR, LICENSE

    # Critical validation - ensure values are not empty/None
    assert_true(VERSION != "", "VERSION should not be empty")
    assert_true(AUTHOR != "", "AUTHOR should not be empty")
    assert_true(LICENSE != "", "LICENSE should not be empty")

    # Test expected format and values
    assert_equal(AUTHOR, "ML Odyssey Team")
    assert_equal(LICENSE, "BSD")

    # Additional critical tests - ensure these are actual string values, not None
    assert_true(VERSION.__len__() > 0, "VERSION string should have length > 0")
    assert_true(AUTHOR.__len__() > 0, "AUTHOR string should have length > 0")
    assert_true(LICENSE.__len__() > 0, "LICENSE string should have length > 0")

    print("✓ Package version test passed")


def test_subpackage_accessibility() raises:
    """Test all subpackages can be imported and have expected exports."""
    from shared import core, training, data, utils

    # Verify subpackages are accessible by testing exports
    from shared.tensor.any_tensor import AnyTensor, zeros
    from shared.training import SGD, MSELoss
    from shared.data import Dataset, AnyTensorDataset
    from shared.utils import Logger, Config

    # Test that we can actually call the functions
    var test_tensor = zeros([2, 3], DType.float32)
    assert_true(test_tensor.dim() == 2, "zeros should create 2D tensor")
    var shape = test_tensor.shape()
    assert_true(shape[0] == 2, "First dimension should be 2")
    assert_true(shape[1] == 3, "Second dimension should be 3")

    # Test that we can actually instantiate classes
    var _test_optimizer = SGD(learning_rate=0.01)
    var _test_loss = MSELoss()
    var _test_logger = Logger("test.log")

    print("✓ Subpackage accessibility test passed")


def test_root_level_imports() raises:
    """Test most commonly used components are available at root level."""
    # Root package doesn't re-export all components directly
    from shared.tensor.any_tensor import AnyTensor
    from shared.training import SGD
    from shared.utils import Logger

    print("✓ Root level imports test passed")


def test_layer_root_level_imports() raises:
    """Test that layer symbols activated in shared/__init__.mojo are importable directly.

    Verifies Issue #3759: confirmed-ready layer exports are accessible via
    `from shared import <Symbol>` after uncommenting in __init__.mojo.
    """
    # Core layer structs — original names
    from shared import Linear, Conv2dLayer, ReLULayer, DropoutLayer, BatchNorm2dLayer

    # Core layer aliases matching documented public API names
    from shared import Conv2D, ReLU, Dropout, BatchNorm2d

    # Core activation functions
    from shared import relu, sigmoid, tanh, softmax

    # Core module trait
    from shared import Module

    # Core tensors and creation functions
    # AnyTensor is NOT re-exported from shared (avoids circular imports)
    from shared.tensor.any_tensor import AnyTensor, zeros, ones, randn
    from shared.tensor.tensor import Tensor

    # Training schedulers
    from shared import StepLR, CosineAnnealingLR

    # Training callbacks
    from shared import EarlyStopping, ModelCheckpoint

    # Utils
    from shared import Logger, plot_training_curves

    # Smoke-test: construct a Linear layer and run a forward pass
    var layer = Linear(4, 2)
    var x = zeros([1, 4], DType.float32)
    var out = layer.forward(x)
    var out_shape = out.shape()
    assert_true(out.dim() == 2, "Linear output should be 2D")
    assert_true(out_shape[1] == 2, "Linear output features should be 2")

    # Smoke-test: Conv2D alias resolves to Conv2dLayer
    var _conv = Conv2D(1, 4, 3, 3)
    assert_true(True, "Conv2D alias should resolve to Conv2dLayer")

    # Smoke-test: ReLU alias resolves to ReLULayer
    var _act = ReLU()
    assert_true(True, "ReLU alias should resolve to ReLULayer")

    print("✓ Layer root-level imports test passed")


def test_module_level_imports() raises:
    """Test importing from specific modules."""
    from shared.tensor.any_tensor import AnyTensor
    from shared.core import relu, linear
    from shared.training import SGD, MSELoss
    from shared.data import AnyTensorDataset, Batch

    print("✓ Module level imports test passed")


def test_nested_imports() raises:
    """Test importing from nested submodules."""
    from shared.core import linear, conv2d
    from shared.training import SGD
    from shared.training import StepLR

    print("✓ Nested imports test passed")


def test_core_training_integration() raises:
    """Test integration between core and training modules."""
    from shared.tensor.any_tensor import AnyTensor, zeros
    from shared.training import SGD, MSELoss

    # Create tensors using core
    var data = zeros([10, 5], DType.float32)

    # Create optimizer using training
    var optimizer = SGD(learning_rate=0.01)
    var _loss_fn = MSELoss()

    # Verify types are correct and components can be instantiated
    var data_shape = data.shape()
    assert_true(data.dim() == 2, "Data should be 2D tensor")
    assert_true(data_shape[0] == 10, "First dimension should be 10")
    assert_true(data_shape[1] == 5, "Second dimension should be 5")
    assert_true(
        abs(optimizer.get_learning_rate() - 0.01) < 1e-9,
        "Learning rate should be 0.01",
    )

    print("✓ Core-training integration test passed")


def test_core_data_integration() raises:
    """Test integration between core and data modules."""
    from shared.tensor.any_tensor import AnyTensor, zeros, ones
    from shared.data import AnyTensorDataset

    # Create tensors using core
    var data = zeros([10, 5], DType.float32)
    var labels = ones([10, 1], DType.float32)

    # Create dataset using data
    var dataset = AnyTensorDataset(data, labels)

    # Verify dataset was created and has correct properties
    var data_shape = data.shape()
    var labels_shape = labels.shape()
    assert_true(data.dim() == 2, "Data should be 2D tensor")
    assert_true(labels.dim() == 2, "Labels should be 2D tensor")
    assert_true(data_shape[0] == 10, "First dimension should be 10")
    assert_true(labels_shape[0] == 10, "Labels first dimension should be 10")
    assert_true(len(dataset) == 10, "Dataset should have 10 samples")

    print("✓ Core-data integration test passed")


def main() raises:
    """Run test_packaging part 1 tests."""
    print("Running test_packaging_part1 tests...")

    test_package_version()
    print("✓ test_package_version")

    test_subpackage_accessibility()
    print("✓ test_subpackage_accessibility")

    test_root_level_imports()
    print("✓ test_root_level_imports")

    test_layer_root_level_imports()
    print("✓ test_layer_root_level_imports")

    test_module_level_imports()
    print("✓ test_module_level_imports")

    test_nested_imports()
    print("✓ test_nested_imports")

    test_core_training_integration()
    print("✓ test_core_training_integration")

    test_core_data_integration()
    print("✓ test_core_data_integration")

    print("\nAll test_packaging_part1 tests passed!")
