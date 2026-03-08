# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_results_printer.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for results printer module - Part 1.

Covers:
- Training progress printer (5 tests)
- Evaluation summary printer (3 tests)
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
)
from shared.core import ExTensor, zeros, ones, full
from shared.training.metrics import (
    print_evaluation_summary,
    print_per_class_accuracy,
    print_confusion_matrix,
    print_training_progress,
    print_training_summary,
)
from collections import List


# ============================================================================
# Training Progress Printer Tests (#2353)
# ============================================================================


fn test_print_training_progress_basic() raises:
    """Test basic training progress output."""
    print("Testing print_training_progress basic...")

    # Should not raise any errors
    print_training_progress(
        epoch=1,
        total_epochs=100,
        batch=10,
        total_batches=50,
        loss=0.523,
        learning_rate=0.01,
    )

    print("   Training progress basic test passed")


fn test_print_training_progress_early_epoch() raises:
    """Test training progress in early epoch."""
    print("Testing print_training_progress early epoch...")

    # First batch of first epoch
    print_training_progress(
        epoch=1,
        total_epochs=200,
        batch=1,
        total_batches=100,
        loss=2.453,
        learning_rate=0.1,
    )

    print("   Training progress early epoch test passed")


fn test_print_training_progress_late_epoch() raises:
    """Test training progress in late epoch."""
    print("Testing print_training_progress late epoch...")

    # Last batch of last epoch
    print_training_progress(
        epoch=200,
        total_epochs=200,
        batch=1000,
        total_batches=1000,
        loss=0.012,
        learning_rate=0.0001,
    )

    print("   Training progress late epoch test passed")


fn test_print_training_progress_very_small_loss() raises:
    """Test training progress with very small loss values."""
    print("Testing print_training_progress with very small loss...")

    print_training_progress(
        epoch=100,
        total_epochs=200,
        batch=50,
        total_batches=100,
        loss=Float32(1e-6),
        learning_rate=Float32(1e-8),
    )

    print("   Training progress very small loss test passed")


fn test_print_training_progress_large_loss() raises:
    """Test training progress with large loss values."""
    print("Testing print_training_progress with large loss...")

    print_training_progress(
        epoch=1,
        total_epochs=100,
        batch=1,
        total_batches=100,
        loss=100.234,
        learning_rate=0.1,
    )

    print("   Training progress large loss test passed")


# ============================================================================
# Evaluation Summary Printer Tests (#2353)
# ============================================================================


fn test_print_evaluation_summary_basic() raises:
    """Test basic evaluation summary output."""
    print("Testing print_evaluation_summary basic...")

    print_evaluation_summary(
        epoch=1,
        total_epochs=100,
        train_loss=0.523,
        train_accuracy=0.923,
        test_loss=0.645,
        test_accuracy=0.891,
    )

    print("   Evaluation summary basic test passed")


fn test_print_evaluation_summary_perfect_accuracy() raises:
    """Test evaluation summary with perfect accuracy."""
    print("Testing print_evaluation_summary perfect accuracy...")

    print_evaluation_summary(
        epoch=50,
        total_epochs=100,
        train_loss=0.001,
        train_accuracy=1.0,
        test_loss=0.005,
        test_accuracy=1.0,
    )

    print("   Evaluation summary perfect accuracy test passed")


fn test_print_evaluation_summary_zero_accuracy() raises:
    """Test evaluation summary with zero accuracy."""
    print("Testing print_evaluation_summary zero accuracy...")

    print_evaluation_summary(
        epoch=1,
        total_epochs=100,
        train_loss=4.605,
        train_accuracy=0.0,
        test_loss=4.605,
        test_accuracy=0.0,
    )

    print("   Evaluation summary zero accuracy test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run results printer tests - part 1."""
    print("\n" + "=" * 60)
    print("Running Results Printer Tests - Part 1")
    print("=" * 60 + "\n")

    # Training progress tests
    test_print_training_progress_basic()
    test_print_training_progress_early_epoch()
    test_print_training_progress_late_epoch()
    test_print_training_progress_very_small_loss()
    test_print_training_progress_large_loss()

    # Evaluation summary tests
    test_print_evaluation_summary_basic()
    test_print_evaluation_summary_perfect_accuracy()
    test_print_evaluation_summary_zero_accuracy()

    print("\n" + "=" * 60)
    print("All results printer tests - part 1 passed!")
    print("=" * 60 + "\n")
