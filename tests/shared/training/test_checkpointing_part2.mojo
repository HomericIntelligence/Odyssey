"""Unit tests for Checkpointing Callback - Part 2.

Tests cover:
- Mode-specific behavior (min/max)
- Edge cases (missing metrics, error tracking, save count API)

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_checkpointing.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

All tests use the real ModelCheckpoint implementation.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    assert_greater,
    TestFixtures,
)
from shared.training.callbacks import ModelCheckpoint
from shared.training.base import TrainingState


# ============================================================================
# Mode Tests
# ============================================================================


fn test_checkpointing_mode_min() raises:
    """Test ModelCheckpoint with mode='min' for loss minimization."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Decreasing loss should save
    state.metrics["val_loss"] = 1.0
    _ = checkpoint.on_epoch_end(state)
    var count1 = checkpoint.save_count

    state.epoch = 2
    state.metrics["val_loss"] = 0.5
    _ = checkpoint.on_epoch_end(state)
    var count2 = checkpoint.save_count

    assert_greater(count2, count1)  # Should have saved


fn test_checkpointing_mode_max() raises:
    """Test ModelCheckpoint with mode='max' for accuracy maximization."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/model.pt",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Increasing accuracy should save
    state.metrics["val_accuracy"] = 0.5
    _ = checkpoint.on_epoch_end(state)
    var count1 = checkpoint.save_count

    state.epoch = 2
    state.metrics["val_accuracy"] = 0.8
    _ = checkpoint.on_epoch_end(state)
    var count2 = checkpoint.save_count

    assert_greater(count2, count1)  # Should have saved


# ============================================================================
# Edge Cases
# ============================================================================


fn test_checkpointing_missing_monitored_metric() raises:
    """Test ModelCheckpoint handles missing monitored metric gracefully."""
    var checkpoint = ModelCheckpoint(
        filepath="/tmp/model.pt",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Metric not in state - should not save
    state.metrics["train_loss"] = 0.5  # Different metric
    _ = checkpoint.on_epoch_end(state)

    # Should not have saved (no val_loss)
    assert_equal(checkpoint.save_count, 0)


fn test_checkpointing_error_count_tracking() raises:
    """Test ModelCheckpoint tracks error count."""
    var checkpoint = ModelCheckpoint(filepath="/tmp/model.pt")

    # Error count should start at 0
    assert_equal(checkpoint.error_count, 0)

    # Note: Actual file I/O is stubbed, so error_count won't change
    # This test just verifies the attribute exists and is accessible


fn test_checkpointing_get_save_count() raises:
    """Test get_save_count returns correct count."""
    var checkpoint = ModelCheckpoint(filepath="/tmp/model.pt", save_frequency=1)
    var state = TrainingState(epoch=1, learning_rate=0.1)

    # Initially 0
    assert_equal(checkpoint.get_save_count(), 0)

    # After one save
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 1)

    # After two saves
    state.epoch = 2
    _ = checkpoint.on_epoch_end(state)
    assert_equal(checkpoint.get_save_count(), 2)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run checkpointing callback tests - Part 2."""
    print("Running mode tests...")
    test_checkpointing_mode_min()
    test_checkpointing_mode_max()

    print("Running edge cases...")
    test_checkpointing_missing_monitored_metric()
    test_checkpointing_error_count_tracking()
    test_checkpointing_get_save_count()

    print("\nAll checkpointing callback tests (Part 2) passed! ✓")
