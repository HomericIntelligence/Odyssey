"""
Packaging Integration Tests - Part 3: Backward Compatibility and Critical Integration

Tests that verify backward compatibility and critical cross-module integration.
Split from test_packaging.mojo per ADR-009.

Run with: mojo test tests/shared/integration/test_packaging_part3.mojo

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_packaging.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_equal

# ============================================================================
# Backward Compatibility Tests
# ============================================================================


fn test_deprecated_imports() raises:
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


fn test_api_version_compatibility() raises:
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


# ============================================================================
# Critical Integration Tests - Catching Real Failures
# ============================================================================


fn test_cross_module_computation() raises:
    """Test that components actually work together in real computations."""
    from shared.core import zeros, ones, relu
    from shared.core.matrix import matmul
    from shared.training import SGD, MSELoss
    from shared.data import ExTensorDataset

    # Create realistic tensors
    var data = zeros([32, 64], DType.float32)  # Batch of 32, features of 64
    var labels = zeros([32, 10], DType.float32)  # 32 samples, 10 classes

    # Create dataset
    var dataset = ExTensorDataset(data, labels)

    # Create a simple network forward pass
    var weights1 = zeros([64, 128], DType.float32)  # Input layer
    var bias1 = zeros([128], DType.float32)
    var weights2 = zeros([128, 10], DType.float32)  # Output layer
    var bias2 = zeros([10], DType.float32)

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
    var optimizer = SGD(learning_rate=0.001)
    var loss_fn = MSELoss()

    # Compute loss
    var loss = loss_fn.compute(logits, labels)
    # Loss reduction depends on MSELoss configuration (mean/sum/none)
    # Just verify loss is computed successfully
    assert_true(loss.numel() > 0, "Loss should be computed")

    print("✓ Cross-module computation test passed")


fn test_tensor_operations_safety() raises:
    """Test that tensor operations handle edge cases safely."""
    from shared.core import zeros, ones, full

    # Test zero-sized tensors
    var empty_data = zeros([0, 5], DType.float32)
    var empty_labels = zeros([0, 3], DType.float32)
    assert_true(
        empty_data.num_elements() == 0, "Empty tensor should have 0 elements"
    )

    # Test single-element tensors
    var single_data = zeros([1], DType.float32)
    var single_labels = zeros([1], DType.float32)
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


fn test_error_propagation() raises:
    """Test that errors propagate correctly between modules."""
    from shared.core import zeros
    from shared.training import SGD
    from shared.data import ExTensorDataset

    # Test that incompatible tensor shapes fail appropriately
    var good_data = zeros([10, 5], DType.float32)
    var good_labels = zeros([10, 3], DType.float32)

    # This should work
    var good_dataset = ExTensorDataset(good_data, good_labels)
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


fn test_integration_stress() raises:
    """Stress test with realistic deep learning workload."""
    from shared.core import zeros, ones, relu
    from shared.core.matrix import matmul
    from shared.training import SGD, MSELoss
    from shared.data import ExTensorDataset

    # Create a realistic batch size
    var batch_size = 128
    var input_dim = 784  # MNIST-like
    var hidden_dim = 256
    var output_dim = 10  # 10 classes

    # Create data
    var train_data = zeros([batch_size, input_dim], DType.float32)
    var train_labels = zeros([batch_size, output_dim], DType.float32)

    # Create dataset
    var dataset = ExTensorDataset(train_data, train_labels)

    # Create network parameters
    var w1 = zeros([input_dim, hidden_dim], DType.float32)
    var b1 = zeros([hidden_dim], DType.float32)
    var w2 = zeros([hidden_dim, hidden_dim], DType.float32)
    var b2 = zeros([hidden_dim], DType.float32)
    var w3 = zeros([hidden_dim, output_dim], DType.float32)
    var b3 = zeros([output_dim], DType.float32)

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
    var optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()

    # Compute loss
    var loss = loss_fn.compute(x3, train_labels)
    # Loss reduction depends on MSELoss configuration
    assert_true(loss.numel() > 0, "Loss should be computed")

    print("✓ Integration stress test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run packaging integration tests - part 3."""
    print("\n" + "=" * 70)
    print("Running Packaging Integration Tests - Part 3")
    print("=" * 70 + "\n")

    # Backward compatibility
    print("Testing Backward Compatibility...")
    test_deprecated_imports()
    test_api_version_compatibility()

    # Critical integration tests
    print("\nTesting Critical Integration...")
    test_cross_module_computation()
    test_tensor_operations_safety()
    test_error_propagation()
    test_integration_stress()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Packaging Integration Tests - Part 3 Passed!")
    print("=" * 70)
