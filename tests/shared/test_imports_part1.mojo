# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_imports.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

"""
Import Validation Tests - Part 1: Core and Training Package Imports

Tests that all public imports work correctly for the shared library.
These tests verify both import functionality and basic component behavior.

Run with: mojo test tests/shared/test_imports_part1.mojo
"""

from testing import assert_true

# ============================================================================
# Core Package Imports
# ============================================================================


fn test_core_imports() raises:
    """Test core package imports work correctly."""
    from shared.core import AnyTensor, zeros, ones, randn
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


fn test_core_layers_imports() raises:
    """Test core layer operations imports."""
    from shared.core import linear, conv2d, flatten
    from shared.core import maxpool2d, avgpool2d

    print("✓ Core layer operations imports test passed")


fn test_core_activations_imports() raises:
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


fn test_core_types_imports() raises:
    """Test core types imports."""
    from shared.core import AnyTensor, FP8, BF8

    print("✓ Core types imports test passed")


# ============================================================================
# Training Package Imports
# ============================================================================


fn test_training_imports() raises:
    """Test training package imports work correctly."""
    from shared.training import SGD, MSELoss
    from shared.training import StepLR, CosineAnnealingLR, ExponentialLR
    from shared.training import EarlyStopping, ModelCheckpoint

    print("✓ Training imports test passed")


fn test_training_optimizers_imports() raises:
    """Test training optimizers imports."""
    from shared.training import SGD

    print("✓ Training optimizers imports test passed")


fn test_training_schedulers_imports() raises:
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


fn test_training_metrics_imports() raises:
    """Test training metrics imports."""
    # Metrics are in shared.training for now
    from shared.training import base

    print("✓ Training metrics imports test passed")


fn test_training_dataloader_imports() raises:
    """Test DataLoader and DataBatch are importable from trainer_interface.

    Verifies Issue #3851: DataLoader and DataBatch defined in
    shared/training/trainer_interface.mojo.
    """
    from shared.training.trainer_interface import DataLoader, DataBatch

    print("✓ Training DataLoader/DataBatch package imports test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run part 1 import validation tests."""
    print("\n" + "=" * 70)
    print("Running Import Validation Tests - Part 1 (Core & Training)")
    print("=" * 70 + "\n")

    # Core package tests
    print("Testing Core Package...")
    test_core_imports()
    test_core_layers_imports()
    test_core_activations_imports()
    test_core_types_imports()

    # Training package tests (first half)
    print("\nTesting Training Package (Part 1)...")
    test_training_imports()
    test_training_optimizers_imports()
    test_training_schedulers_imports()
    test_training_metrics_imports()
    test_training_dataloader_imports()

    # Summary
    print("\n" + "=" * 70)
    print("✅ Import Validation Tests - Part 1 Passed!")
    print("=" * 70)
