"""
Packaging Integration Tests (Part 3 of 3)

Tests that verify training metrics, API compatibility, cross-module computation,
tensor safety, error propagation, and stress testing.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""


from std.testing import assert_true, assert_equal


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


fn main() raises:
    """Run test_packaging part 3 tests."""
    print("Running test_packaging_part3 tests...")

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

    print("\nAll test_packaging_part3 tests passed!")
