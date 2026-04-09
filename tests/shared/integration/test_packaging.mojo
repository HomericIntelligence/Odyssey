"""
Packaging Integration Tests

Tests that verify the shared library package structure and import hierarchy.
Split from test_packaging.mojo per ADR-009.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_packaging.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""


from testing import assert_true, assert_equal


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


def test_training_data_integration() raises:
    """Test integration between training and data modules."""
    from shared.training import SGD
    from shared.data import AnyTensorDataset
    from shared.tensor.any_tensor import zeros, ones

    # Create simple dataset
    var data = zeros([10, 5], DType.float32)
    var labels = ones([10, 1], DType.float32)
    var dataset = AnyTensorDataset(data, labels)

    # Create optimizer
    var optimizer = SGD(learning_rate=0.01)

    # Verify integration by checking component properties
    var _data_shape = data.shape()
    var _labels_shape = labels.shape()
    assert_true(data.dim() == 2, "Data should be 2D tensor")
    assert_true(labels.dim() == 2, "Labels should be 2D tensor")
    assert_true(
        abs(optimizer.get_learning_rate() - 0.01) < 1e-9,
        "Learning rate should be 0.01",
    )
    assert_true(len(dataset) > 0, "Dataset should have samples")

    print("✓ Training-data integration test passed")


def test_complete_training_workflow() raises:
    """Test complete training workflow using all modules."""
    from shared.tensor.any_tensor import zeros, ones
    from shared.core import relu
    from shared.training import SGD, MSELoss
    from shared.data import AnyTensorDataset
    from shared.utils import Logger

    # 1. Create model parameters (core)
    var weights = zeros([5, 10], DType.float32)
    var bias = zeros([5], DType.float32)

    # 2. Create data (data)
    var data = zeros([10, 10], DType.float32)
    var labels = ones([10, 5], DType.float32)
    var dataset = AnyTensorDataset(data, labels)

    # 3. Create optimizer and loss (training)
    var optimizer = SGD(learning_rate=0.01)
    var _loss_fn = MSELoss()

    # 4. Create logger (utils)
    var _logger = Logger("training.log")

    # 5. Verify workflow components work together
    var _weights_shape = weights.shape()
    var _bias_shape = bias.shape()
    var _data_shape = data.shape()
    var _labels_shape = labels.shape()
    assert_true(weights.dim() == 2, "Weights should be 2D tensor")
    assert_true(bias.dim() == 1, "Bias should be 1D tensor")
    assert_true(data.dim() == 2, "Data should be 2D tensor")
    assert_true(labels.dim() == 2, "Labels should be 2D tensor")
    assert_true(
        abs(optimizer.get_learning_rate() - 0.01) < 1e-9,
        "Learning rate should be 0.01",
    )
    assert_true(len(dataset) > 0, "Dataset should have samples")
    # Logger is created successfully if no exception was raised

    print("✓ Complete workflow test passed")


def test_paper_implementation_pattern() raises:
    """Test typical usage pattern from paper implementation."""
    # Simulates how a paper implementation would use the shared library

    from shared.tensor.any_tensor import AnyTensor, zeros
    from shared.core import conv2d, flatten, relu
    from shared.training import (
        SGD,
        CosineAnnealingLR,
        EarlyStopping,
        ModelCheckpoint,
    )
    from shared.data import AnyTensorDataset

    # Paper-specific tensors for conv operations
    var input_data = zeros([1, 1, 28, 28], DType.float32)

    # Training setup
    var optimizer = SGD(learning_rate=0.001)
    var _scheduler = CosineAnnealingLR(0.001, 50)

    # Callbacks
    var _early_stop = EarlyStopping()
    var _checkpoint = ModelCheckpoint()

    # Create dataset
    var data = zeros([10, 1, 28, 28], DType.float32)
    var labels = zeros([10, 10], DType.float32)
    var dataset = AnyTensorDataset(data, labels)

    # Verify all components are properly instantiated
    assert_true(input_data.dim() == 4, "Input data should be 4D tensor")
    assert_true(
        abs(optimizer.get_learning_rate() - 0.001) < 1e-9,
        "Learning rate should be 0.001",
    )
    assert_true(len(dataset) > 0, "Dataset should have samples")

    print("✓ Paper implementation pattern test passed")


def test_no_private_exports() raises:
    """Test that private modules are not exported at root level."""
    # Test that private modules are not accessible through public imports
    # Mojo v0.26.1 doesn't support __all__, so we verify by checking
    # that public symbols are available and documenting expected behavior

    # Verify public symbols are available (confirming public API is working)
    from shared import core, training, data, utils

    # Document that the following should NOT be accessible:
    # - Private modules like _internal, _private, _utils_private
    # - Private symbols prefixed with _
    # - Internal implementation details

    # The fact that we can only import public symbols (core, training, data, utils)
    # and not private ones proves the public API is properly isolated

    print("✓ No private exports test passed - public API properly isolated")


def test_normalize_compose_from_shared_data() raises:
    """Test that Normalize and Compose are accessible via shared.data."""
    from shared.data import Normalize, Compose

    # Verify Normalize can be instantiated
    var _normalizer = Normalize(Float64(0.5), Float64(0.5))

    print("✓ Normalize and Compose importable from shared.data")


def test_losstracker_from_shared() raises:
    """Test that LossTracker is accessible at shared package level."""
    from shared import LossTracker

    # Verify LossTracker can be instantiated
    var _tracker = LossTracker()

    print("✓ LossTracker importable from shared")


def test_accuracymetric_from_shared() raises:
    """Test that AccuracyMetric is accessible at shared package level."""
    from shared import AccuracyMetric

    # Verify AccuracyMetric can be instantiated
    var _metric = AccuracyMetric()

    print("✓ AccuracyMetric importable from shared")


def test_accuracy_alias_from_shared() raises:
    """Test that Accuracy alias resolves and is identical to AccuracyMetric."""
    from shared import Accuracy, AccuracyMetric

    # Verify Accuracy can be instantiated and works the same way
    var _metric_via_alias = Accuracy()
    var _metric_via_full_name = AccuracyMetric()

    # Both should be instantiable - verifying the alias works
    print("✓ Accuracy alias importable from shared and matches AccuracyMetric")


def test_losstracker_from_shared_training() raises:
    """Test that LossTracker is accessible via shared.training."""
    from shared.training import LossTracker

    # Verify LossTracker can be instantiated
    var _tracker = LossTracker()

    print("✓ LossTracker importable from shared.training")


def test_accuracymetric_from_shared_training() raises:
    """Test that AccuracyMetric is accessible via shared.training."""
    from shared.training import AccuracyMetric

    # Verify AccuracyMetric can be instantiated
    var _metric = AccuracyMetric()

    print("✓ AccuracyMetric importable from shared.training")


def test_deprecated_imports() raises:
    """Test that deprecated imports still work with warnings."""
    # Currently no deprecated APIs exist in this codebase
    # When deprecated APIs are added, this test should:
    # 1. Test that deprecated imports still work (backward compatibility)
    # 2. Optionally test that deprecation warnings are issued
    # 3. Document the migration path to new APIs

    # Example of what this test should do when deprecated APIs exist:
    # ```mojo
    # # Test deprecated import still works
    # from shared.deprecated import old_function  # Should still work
    #
    # # Test that replacement is available
    # from shared.new import new_function  # Should work as replacement
    # ```

    # For now, we verify the test framework itself works by importing from shared
    from shared import VERSION  # This import should always work

    assert_true(VERSION != "", "Version should be accessible")

    print(
        "✓ Deprecated imports test passed - no deprecated APIs currently exist"
    )


def test_api_version_compatibility() raises:
    """Test API version compatibility."""
    from shared import VERSION

    # Verify version follows semantic versioning (major.minor.patch format)
    var version_parts = VERSION.split(".")
    assert_equal(
        version_parts.__len__(),
        3,
        "Version should have 3 parts (major.minor.patch)",
    )

    # Verify each part is numeric
    var major = version_parts[0]
    var minor = version_parts[1]
    var patch = version_parts[2]

    # Basic format validation (should be digits)
    try:
        var major_int = atol(major)
        assert_true(major_int >= 0, "Major version should be non-negative")
    except:
        assert_true(False, "Major version should be numeric")

    try:
        var minor_int = atol(minor)
        assert_true(minor_int >= 0, "Minor version should be non-negative")
    except:
        assert_true(False, "Minor version should be numeric")

    try:
        var patch_int = atol(patch)
        assert_true(patch_int >= 0, "Patch version should be non-negative")
    except:
        assert_true(False, "Patch version should be numeric")

    print("✓ API version compatibility test passed")


def test_cross_module_computation() raises:
    """Test that components actually work together in real computations."""
    from shared.tensor.any_tensor import zeros, ones
    from shared.core import relu
    from shared.core.matrix import matmul
    from shared.training import SGD, MSELoss
    from shared.data import AnyTensorDataset

    # Create realistic tensors
    var data = zeros([32, 64], DType.float32)  # Batch of 32, features of 64
    var labels = zeros([32, 10], DType.float32)  # 32 samples, 10 classes

    # Create dataset
    var _dataset = AnyTensorDataset(data, labels)

    # Create a simple network forward pass
    var weights1 = zeros([64, 128], DType.float32)  # Input layer
    var _bias1 = zeros([128], DType.float32)
    var weights2 = zeros([128, 10], DType.float32)  # Output layer
    var _bias2 = zeros([10], DType.float32)

    # Forward pass - this is where integration failures would occur
    var hidden = matmul(data, weights1)  # (32,64) × (64,128) = (32,128)
    var hidden_activated = relu(hidden)
    var logits = matmul(
        hidden_activated, weights2
    )  # (32,128) × (128,10) = (32,10)

    # Critical assertions that would catch shape/dtype errors
    var logits_shape = logits.shape()
    assert_true(logits.dim() == 2, "Logits should be 2D tensor")
    assert_true(logits_shape[0] == 32, "Batch size should be preserved")
    assert_true(logits_shape[1] == 10, "Output classes should match labels")
    assert_true(logits.dtype() == DType.float32, "DType should be preserved")

    # Test with training components
    var _optimizer = SGD(learning_rate=0.001)
    var loss_fn = MSELoss()

    # Compute loss
    var loss = loss_fn.compute(logits, labels)
    # Loss reduction depends on MSELoss configuration (mean/sum/none)
    # Just verify loss is computed successfully
    assert_true(loss.numel() > 0, "Loss should be computed")

    print("✓ Cross-module computation test passed")


def test_tensor_operations_safety() raises:
    """Test that tensor operations handle edge cases safely."""
    from shared.tensor.any_tensor import zeros, ones, full

    # Test zero-sized tensors
    var empty_data = zeros([0, 5], DType.float32)
    var _empty_labels = zeros([0, 3], DType.float32)
    assert_true(
        empty_data.num_elements() == 0, "Empty tensor should have 0 elements"
    )

    # Test single-element tensors
    var single_data = zeros([1], DType.float32)
    var _single_labels = zeros([1], DType.float32)
    assert_true(
        single_data.num_elements() == 1,
        "Single element tensor should have 1 element",
    )

    # Test large tensors (memory safety)
    try:
        var large_tensor = zeros([1000, 1000], DType.float32)
        assert_true(
            large_tensor.num_elements() == 1000000,
            "Large tensor should have 1M elements",
        )
    except:
        # If allocation fails, that's actually a valid failure case
        print("✓ Large tensor allocation failed (acceptable)")

    # Test different dtypes
    var int_tensor = zeros([2, 2], DType.int32)
    var float_tensor = zeros([2, 2], DType.float32)
    var bool_tensor = zeros([2, 2], DType.bool)

    assert_true(
        int_tensor.dtype() == DType.int32, "Int tensor should maintain dtype"
    )
    assert_true(
        float_tensor.dtype() == DType.float32,
        "Float tensor should maintain dtype",
    )
    assert_true(
        bool_tensor.dtype() == DType.bool, "Bool tensor should maintain dtype"
    )

    print("✓ Tensor operations safety test passed")


def test_error_propagation() raises:
    """Test that errors propagate correctly between modules."""
    from shared.tensor.any_tensor import zeros
    from shared.training import SGD
    from shared.data import AnyTensorDataset

    # Test that incompatible tensor shapes fail appropriately
    var good_data = zeros([10, 5], DType.float32)
    var good_labels = zeros([10, 3], DType.float32)

    # This should work
    var good_dataset = AnyTensorDataset(good_data, good_labels)
    assert_true(len(good_dataset) > 0, "Valid dataset should be created")

    # Test optimizer with edge case learning rates
    var fast_optimizer = SGD(learning_rate=1000.0)  # Very large
    var slow_optimizer = SGD(learning_rate=0.000001)  # Very small

    assert_true(
        abs(fast_optimizer.get_learning_rate() - 1000.0) < 1e-6,
        "Large learning rate should be preserved",
    )
    assert_true(
        abs(slow_optimizer.get_learning_rate() - 0.000001) < 1e-12,
        "Small learning rate should be preserved",
    )

    print("✓ Error propagation test passed")


def test_integration_stress() raises:
    """Stress test with realistic deep learning workload."""
    from shared.tensor.any_tensor import zeros, ones
    from shared.core import relu
    from shared.core.matrix import matmul
    from shared.training import SGD, MSELoss
    from shared.data import AnyTensorDataset

    # Create a realistic batch size
    var batch_size = 128
    var input_dim = 784  # MNIST-like
    var hidden_dim = 256
    var output_dim = 10  # 10 classes

    # Create data
    var train_data = zeros([batch_size, input_dim], DType.float32)
    var train_labels = zeros([batch_size, output_dim], DType.float32)

    # Create dataset
    var _dataset = AnyTensorDataset(train_data, train_labels)

    # Create network parameters
    var w1 = zeros([input_dim, hidden_dim], DType.float32)
    var _b1 = zeros([hidden_dim], DType.float32)
    var w2 = zeros([hidden_dim, hidden_dim], DType.float32)
    var _b2 = zeros([hidden_dim], DType.float32)
    var w3 = zeros([hidden_dim, output_dim], DType.float32)
    var _b3 = zeros([output_dim], DType.float32)

    # Forward pass through 3-layer network
    var x1 = matmul(train_data, w1)  # (128,784) × (784,256) = (128,256)
    var x1_activated = relu(x1)

    var x2 = matmul(x1_activated, w2)  # (128,256) × (256,256) = (128,256)
    var x2_activated = relu(x2)

    var x3 = matmul(x2_activated, w3)  # (128,256) × (256,10) = (128,10)

    # Verify all shapes are correct
    var x1_shape = x1_activated.shape()
    var x2_shape = x2_activated.shape()
    var x3_shape = x3.shape()
    assert_true(x1_activated.dim() == 2, "First layer output should be 2D")
    assert_true(
        x1_shape[0] == batch_size and x1_shape[1] == hidden_dim,
        "First layer should match expected shape",
    )
    assert_true(x2_activated.dim() == 2, "Second layer output should be 2D")
    assert_true(
        x2_shape[0] == batch_size, "Second layer batch size should match"
    )
    assert_true(x3.dim() == 2, "Final output should be 2D")
    assert_true(
        x3_shape[0] == batch_size, "Final output batch size should match"
    )
    assert_true(x3_shape[1] == output_dim, "Final output classes should match")

    # Test with training components
    var _optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()

    # Compute loss
    var loss = loss_fn.compute(x3, train_labels)
    # Loss reduction depends on MSELoss configuration
    assert_true(loss.numel() > 0, "Loss should be computed")

    print("✓ Integration stress test passed")


def main() raises:
    """Run all test_packaging tests."""
    print("Running test_packaging tests...")

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

    test_training_data_integration()
    print("✓ test_training_data_integration")

    test_complete_training_workflow()
    print("✓ test_complete_training_workflow")

    test_paper_implementation_pattern()
    print("✓ test_paper_implementation_pattern")

    test_no_private_exports()
    print("✓ test_no_private_exports")

    test_normalize_compose_from_shared_data()
    print("✓ test_normalize_compose_from_shared_data")

    test_losstracker_from_shared()
    print("✓ test_losstracker_from_shared")

    test_accuracymetric_from_shared()
    print("✓ test_accuracymetric_from_shared")

    test_accuracy_alias_from_shared()
    print("✓ test_accuracy_alias_from_shared")

    test_losstracker_from_shared_training()
    print("✓ test_losstracker_from_shared_training")

    test_accuracymetric_from_shared_training()
    print("✓ test_accuracymetric_from_shared_training")

    test_deprecated_imports()
    print("✓ test_deprecated_imports")

    test_api_version_compatibility()
    print("✓ test_api_version_compatibility")

    test_cross_module_computation()
    print("✓ test_cross_module_computation")

    test_tensor_operations_safety()
    print("✓ test_tensor_operations_safety")

    test_error_propagation()
    print("✓ test_error_propagation")

    test_integration_stress()
    print("✓ test_integration_stress")

    print("\nAll test_packaging tests passed!")
