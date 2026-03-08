# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_progress_bar.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for progress bar utilities module - Part 3.

This module tests factory functions, helper functions, and integration scenarios:
- Factory function for ETA progress bar
- format_duration helper function
- Integration scenarios (training loop, ETA tracking, rapid updates)
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
# Test Factory Functions (continued)
# ============================================================================


fn test_factory_create_progress_bar_with_eta():
    """Test factory for ETA progress bar."""
    var progress = create_progress_bar_with_eta(total=100, description="Test")
    _ = progress


# ============================================================================
# Test Helper Functions
# ============================================================================


fn test_format_duration():
    """Test format_duration produces output."""
    var result = format_duration(45.0)
    # Should produce non-empty string


fn test_format_duration_long():
    """Test format_duration with longer duration."""
    var result = format_duration(3665.0)
    # Should produce non-empty string


# ============================================================================
# Test Integration Scenarios
# ============================================================================


fn test_training_loop_simulation():
    """Test progress bar in simulated training loop."""
    var progress = ProgressBarWithMetrics(total=100, description="Training")

    var i = 0
    while i < 100:
        progress.set_metric("loss", 1.0 - (Float32(i) / 100.0) * 0.5)
        progress.update(1)
        i = i + 1

    # Should complete training loop without error


fn test_eta_time_tracking():
    """Test ETA progress bar tracks time."""
    var progress = ProgressBarWithETA(total=10)

    var i = 0
    while i < 10:
        progress.set_metric("loss", 1.0 - (Float32(i) / 10.0) * 0.5)
        progress.update(1)
        i = i + 1

    # Should complete tracking without error


fn test_rapid_updates():
    """Test progress bar handles rapid updates."""
    var progress = ProgressBar(total=1000)

    var i = 0
    while i < 1000:
        progress.update(1)
        i = i + 1

    # Should complete rapid updates without error


fn main() raises:
    """Run progress bar part 3 tests."""
    print("")
    print("=" * 70)
    print("Running Progress Bar Tests - Part 3")
    print("=" * 70)
    print("")

    test_factory_create_progress_bar_with_eta()

    test_format_duration()
    test_format_duration_long()

    test_training_loop_simulation()
    test_eta_time_tracking()
    test_rapid_updates()

    print("")
    print("=" * 70)
    print("All progress bar part 3 tests passed!")
    print("=" * 70)
    print("")
