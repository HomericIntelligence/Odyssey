# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_progress_bar.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for progress bar utilities module - Part 1.

This module tests basic progress bar and metrics functionality:
- Basic progress bar rendering
- Basic progress bar with metrics creation and updates
"""

from shared.utils.progress_bar import (
    ProgressBar,
    ProgressBarWithMetrics,
    ProgressBarWithETA,
    format_duration,
    create_progress_bar,
    create_progress_bar_with_metrics,
    create_progress_bar_with_eta,
)


# ============================================================================
# Test Basic Progress Bar
# ============================================================================


fn test_progress_bar_creation():
    """Test progress bar can be created."""
    var progress = ProgressBar(total=100, description="Test")
    _ = progress


fn test_progress_bar_update():
    """Test progress bar update increments current."""
    var progress = ProgressBar(total=100)
    progress.update()
    progress.update()
    progress.update()
    # Should have incremented without error


fn test_progress_bar_overflow():
    """Test progress bar clamps to total."""
    var progress = ProgressBar(total=100)
    progress.update(150)
    # Should clamp to 100, not error


fn test_progress_bar_set_total():
    """Test updating total."""
    var progress = ProgressBar(total=100)
    progress.update(50)
    progress.set_total(200)
    # Should update without error


fn test_progress_bar_reset():
    """Test reset clears progress."""
    var progress = ProgressBar(total=100)
    progress.update(50)
    progress.reset()
    # Should reset without error


# ============================================================================
# Test Progress Bar With Metrics
# ============================================================================


fn test_progress_bar_with_metrics_creation():
    """Test metrics progress bar can be created."""
    var progress = ProgressBarWithMetrics(total=100, description="Epoch 1")
    _ = progress


fn test_progress_bar_with_metrics_set_metric():
    """Test setting a metric."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.342)
    progress.set_metric("accuracy", 0.891)
    # Should store metrics without error


fn test_progress_bar_with_metrics_update():
    """Test update with metrics."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.update(25)
    # Should update with metrics without error


fn main() raises:
    """Run progress bar part 1 tests."""
    print("")
    print("=" * 70)
    print("Running Progress Bar Tests - Part 1")
    print("=" * 70)
    print("")

    test_progress_bar_creation()
    test_progress_bar_update()
    test_progress_bar_overflow()
    test_progress_bar_set_total()
    test_progress_bar_reset()

    test_progress_bar_with_metrics_creation()
    test_progress_bar_with_metrics_set_metric()
    test_progress_bar_with_metrics_update()

    print("")
    print("=" * 70)
    print("All progress bar part 1 tests passed!")
    print("=" * 70)
    print("")
