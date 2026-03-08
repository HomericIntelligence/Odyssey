"""
Packaging Integration Tests - Part 1: Package Structure and Import Hierarchy

Tests that verify the shared library package structure and import hierarchy.
Split from test_packaging.mojo per ADR-009.

Run with: mojo test tests/shared/integration/test_packaging_part1.mojo

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_packaging.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

from testing import assert_true, assert_equal

# ============================================================================
# Package Structure Tests
# ============================================================================


fn test_package_version() raises:
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


fn test_subpackage_accessibility() raises:
    """Test all subpackages can be imported and have expected exports."""
    from shared import core, training, data, utils

    # Verify subpackages are accessible by testing exports
    from shared.core import ExTensor, zeros
    from shared.training import SGD, MSELoss
    from shared.data import Dataset, ExTensorDataset
    from shared.utils import Logger, Config

    # Test that we can actually call the functions
    var test_tensor = zeros([2, 3], DType.float32)
    assert_true(test_tensor.dim() == 2, "zeros should create 2D tensor")
    var shape = test_tensor.shape()
    assert_true(shape[0] == 2, "First dimension should be 2")
    assert_true(shape[1] == 3, "Second dimension should be 3")

    # Test that we can actually instantiate classes
    var test_optimizer = SGD(learning_rate=0.01)
    var test_loss = MSELoss()
    var test_logger = Logger("test.log")

    print("✓ Subpackage accessibility test passed")


# ============================================================================
# Import Hierarchy Tests
# ============================================================================


fn test_root_level_imports() raises:
    """Test most commonly used components are available at root level."""
    # Root package doesn't re-export all components directly
    from shared.core import ExTensor
    from shared.training import SGD
    from shared.utils import Logger

    print("✓ Root level imports test passed")


fn test_module_level_imports() raises:
    """Test importing from specific modules."""
    from shared.core import ExTensor, relu, linear
    from shared.training import SGD, MSELoss
    from shared.data import ExTensorDataset, Batch

    print("✓ Module level imports test passed")


fn test_nested_imports() raises:
    """Test importing from nested submodules."""
    from shared.core import linear, conv2d
    from shared.training import SGD
    from shared.training import StepLR

    print("✓ Nested imports test passed")


fn test_core_training_integration() raises:
    """Test integration between core and training modules."""
    from shared.core import ExTensor, zeros
    from shared.training import SGD, MSELoss

    # Create tensors using core
    var data = zeros([10, 5], DType.float32)

    # Create optimizer using training
    var optimizer = SGD(learning_rate=0.01)
    var loss_fn = MSELoss()

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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run packaging integration tests - part 1."""
    print("\n" + "=" * 70)
    print("Running Packaging Integration Tests - Part 1")
    print("=" * 70 + "\n")

    # Package structure
    print("Testing Package Structure...")
    test_package_version()
    test_subpackage_accessibility()

    # Import hierarchy
    print("\nTesting Import Hierarchy...")
    test_root_level_imports()
    test_module_level_imports()
    test_nested_imports()

    # Cross-module integration (first test)
    print("\nTesting Cross-Module Integration...")
    test_core_training_integration()

    # Summary
    print("\n" + "=" * 70)
    print("✅ All Packaging Integration Tests - Part 1 Passed!")
    print("=" * 70)
