# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_progress_bar.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for progress bar utilities module - Part 2.

This module tests metrics clear/reset and ETA progress bar functionality:
- Progress bar with metrics clear and reset
- Progress bar with ETA creation and updates
- Factory functions for basic and metrics progress bars
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
# Test Progress Bar With Metrics (continued)
# ============================================================================


fn test_progress_bar_with_metrics_clear():
    """Test clearing metrics."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.clear_metrics()
    # Should clear without error


fn test_progress_bar_with_metrics_reset():
    """Test reset clears metrics and progress."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.update(50)
    progress.reset()
    # Should reset without error


# ============================================================================
# Test Progress Bar With ETA
# ============================================================================


fn test_progress_bar_with_eta_creation():
    """Test ETA progress bar can be created."""
    var progress = ProgressBarWithETA(total=100)
    _ = progress


fn test_progress_bar_with_eta_update():
    """Test ETA progress bar update."""
    var progress = ProgressBarWithETA(total=100)
    progress.update(25)
    progress.update(25)
    progress.update(25)
    progress.update(25)
    # Should update without error


fn test_progress_bar_with_eta_metrics():
    """Test ETA progress bar supports metrics."""
    var progress = ProgressBarWithETA(total=100)
    progress.set_metric("loss", 0.5)
    progress.set_metric("accuracy", 0.9)
    progress.update(50)
    # Should work with metrics


fn test_progress_bar_with_eta_reset():
    """Test ETA progress bar reset."""
    var progress = ProgressBarWithETA(total=100)
    progress.update(50)
    progress.set_metric("loss", 0.5)
    progress.reset()
    # Should reset without error


# ============================================================================
# Test Factory Functions
# ============================================================================


fn test_factory_create_progress_bar():
    """Test factory for simple progress bar."""
    var progress = create_progress_bar(total=100, description="Test")
    _ = progress


fn test_factory_create_progress_bar_with_metrics():
    """Test factory for metrics progress bar."""
    var progress = create_progress_bar_with_metrics(
        total=100, description="Test"
    )
    _ = progress


fn main() raises:
    """Run progress bar part 2 tests."""
    print("")
    print("=" * 70)
    print("Running Progress Bar Tests - Part 2")
    print("=" * 70)
    print("")

    test_progress_bar_with_metrics_clear()
    test_progress_bar_with_metrics_reset()

    test_progress_bar_with_eta_creation()
    test_progress_bar_with_eta_update()
    test_progress_bar_with_eta_metrics()
    test_progress_bar_with_eta_reset()

    test_factory_create_progress_bar()
    test_factory_create_progress_bar_with_metrics()

    print("")
    print("=" * 70)
    print("All progress bar part 2 tests passed!")
    print("=" * 70)
    print("")
