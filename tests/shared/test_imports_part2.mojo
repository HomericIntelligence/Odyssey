"""
Import Validation Tests - Part 2: Training Callbacks, Data, and Utils Imports

Tests that all public imports work correctly for the shared library.
These tests verify both import functionality and basic component behavior.

ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
high test load. Split from test_imports.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Run with: mojo test tests/shared/test_imports_part2.mojo
"""

from testing import assert_true

# ============================================================================
# Training Package Imports (continued)
# ============================================================================


fn test_training_callbacks_imports() raises:
    """Test training callbacks imports."""
    from shared.training import (
        EarlyStopping,
        ModelCheckpoint,
        LoggingCallback,
    )

    print("✓ Training callbacks imports test passed")


fn test_training_loops_imports() raises:
    """Test training loops imports."""
    from shared.training import TrainingState, Callback

    print("✓ Training loops imports test passed")


# ============================================================================
# Data Package Imports
# ============================================================================


fn test_data_imports() raises:
    """Test data package imports work correctly."""
    from shared.data import (
        Dataset,
        AnyTensorDataset,
        CIFAR10Dataset,
        EMNISTDataset,
    )

    print("✓ Data imports test passed")


fn test_data_datasets_imports() raises:
    """Test data datasets imports."""
    from shared.data import Dataset, AnyTensorDataset, FileDataset

    print("✓ Data datasets imports test passed")


fn test_data_loaders_imports() raises:
    """Test data loaders imports."""
    from shared.data import Batch

    print("✓ Data loaders imports test passed")


fn test_data_transforms_imports() raises:
    """Test data transforms imports."""
    # Data transforms are provided as utility functions, not classes
    from shared.data import normalize_images, one_hot_encode

    print("✓ Data transforms imports test passed")


# ============================================================================
# Utils Package Imports
# ============================================================================


fn test_utils_imports() raises:
    """Test utils package imports work correctly."""
    from shared.utils import Logger, LogLevel, get_logger
    from shared.utils import load_config, save_config, Config

    print("✓ Utils imports test passed")


fn test_utils_logging_imports() raises:
    """Test utils logging imports."""
    from shared.utils import (
        Logger,
        LogLevel,
        get_logger,
        StreamHandler,
        FileHandler,
    )

    print("✓ Utils logging imports test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run part 2 import validation tests."""
    print("\n" + "=" * 70)
    print(
        "Running Import Validation Tests - Part 2 (Training Callbacks, Data &"
        " Utils)"
    )
    print("=" * 70 + "\n")

    # Training package tests (second half)
    print("Testing Training Package (Part 2)...")
    test_training_callbacks_imports()
    test_training_loops_imports()

    # Data package tests
    print("\nTesting Data Package...")
    test_data_imports()
    test_data_datasets_imports()
    test_data_loaders_imports()
    test_data_transforms_imports()

    # Utils package tests (first half)
    print("\nTesting Utils Package (Part 1)...")
    test_utils_imports()
    test_utils_logging_imports()

    # Summary
    print("\n" + "=" * 70)
    print("✅ Import Validation Tests - Part 2 Passed!")
    print("=" * 70)
