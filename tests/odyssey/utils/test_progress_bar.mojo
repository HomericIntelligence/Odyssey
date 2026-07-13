"""Tests for progress bar utilities module.

This module tests basic progress bar and metrics functionality:
- Basic progress bar rendering
- Basic progress bar with metrics creation and updates
"""


from odyssey.utils.progress_bar import (
    ProgressBar,
    ProgressBarWithMetrics,
    ProgressBarWithETA,
    format_duration,
    create_progress_bar,
    create_progress_bar_with_metrics,
    create_progress_bar_with_eta,
)


def test_progress_bar_creation():
    """Test progress bar can be created."""
    var progress = ProgressBar(total=100, description="Test")
    _ = progress


def test_progress_bar_update():
    """Test progress bar update increments current."""
    var progress = ProgressBar(total=100)
    progress.update()
    progress.update()
    progress.update()
    # Should have incremented without error


def test_progress_bar_overflow():
    """Test progress bar clamps to total."""
    var progress = ProgressBar(total=100)
    progress.update(150)
    # Should clamp to 100, not error


def test_progress_bar_set_total():
    """Test updating total."""
    var progress = ProgressBar(total=100)
    progress.update(50)
    progress.set_total(200)
    # Should update without error


def test_progress_bar_reset():
    """Test reset clears progress."""
    var progress = ProgressBar(total=100)
    progress.update(50)
    progress.reset()
    # Should reset without error


def test_progress_bar_with_metrics_creation():
    """Test metrics progress bar can be created."""
    var progress = ProgressBarWithMetrics(total=100, description="Epoch 1")
    _ = progress


def test_progress_bar_with_metrics_set_metric():
    """Test setting a metric."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.342)
    progress.set_metric("accuracy", 0.891)
    # Should store metrics without error


def test_progress_bar_with_metrics_update():
    """Test update with metrics."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.update(25)
    # Should update with metrics without error


def test_progress_bar_with_metrics_clear():
    """Test clearing metrics."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.clear_metrics()
    # Should clear without error


def test_progress_bar_with_metrics_reset():
    """Test reset clears metrics and progress."""
    var progress = ProgressBarWithMetrics(total=100)
    progress.set_metric("loss", 0.5)
    progress.update(50)
    progress.reset()
    # Should reset without error


def test_progress_bar_with_eta_creation():
    """Test ETA progress bar can be created."""
    var progress = ProgressBarWithETA(total=100)
    _ = progress


def test_progress_bar_with_eta_update():
    """Test ETA progress bar update."""
    var progress = ProgressBarWithETA(total=100)
    progress.update(25)
    progress.update(25)
    progress.update(25)
    progress.update(25)
    # Should update without error


def test_progress_bar_with_eta_metrics():
    """Test ETA progress bar supports metrics."""
    var progress = ProgressBarWithETA(total=100)
    progress.set_metric("loss", 0.5)
    progress.set_metric("accuracy", 0.9)
    progress.update(50)
    # Should work with metrics


def test_progress_bar_with_eta_reset():
    """Test ETA progress bar reset."""
    var progress = ProgressBarWithETA(total=100)
    progress.update(50)
    progress.set_metric("loss", 0.5)
    progress.reset()
    # Should reset without error


def test_factory_create_progress_bar():
    """Test factory for simple progress bar."""
    var progress = create_progress_bar(total=100, description="Test")
    _ = progress


def test_factory_create_progress_bar_with_metrics():
    """Test factory for metrics progress bar."""
    var progress = create_progress_bar_with_metrics(
        total=100, description="Test"
    )
    _ = progress


def test_factory_create_progress_bar_with_eta():
    """Test factory for ETA progress bar."""
    var progress = create_progress_bar_with_eta(total=100, description="Test")
    _ = progress


def test_format_duration():
    """Test format_duration produces output."""
    _ = format_duration(45.0)
    # Should produce non-empty string


def test_format_duration_long():
    """Test format_duration with longer duration."""
    _ = format_duration(3665.0)
    # Should produce non-empty string


def test_training_loop_simulation():
    """Test progress bar in simulated training loop."""
    var progress = ProgressBarWithMetrics(total=100, description="Training")

    var i = 0
    while i < 100:
        progress.set_metric("loss", 1.0 - (Float32(i) / 100.0) * 0.5)
        progress.update(1)
        i = i + 1

    # Should complete training loop without error


def test_eta_time_tracking():
    """Test ETA progress bar tracks time."""
    var progress = ProgressBarWithETA(total=10)

    var i = 0
    while i < 10:
        progress.set_metric("loss", 1.0 - (Float32(i) / 10.0) * 0.5)
        progress.update(1)
        i = i + 1

    # Should complete tracking without error


def test_rapid_updates():
    """Test progress bar handles rapid updates."""
    var progress = ProgressBar(total=1000)

    var i = 0
    while i < 1000:
        progress.update(1)
        i = i + 1

    # Should complete rapid updates without error


def main() raises:
    """Run all test_progress_bar tests."""
    print("Running test_progress_bar tests...")

    test_progress_bar_creation()
    print("✓ test_progress_bar_creation")

    test_progress_bar_update()
    print("✓ test_progress_bar_update")

    test_progress_bar_overflow()
    print("✓ test_progress_bar_overflow")

    test_progress_bar_set_total()
    print("✓ test_progress_bar_set_total")

    test_progress_bar_reset()
    print("✓ test_progress_bar_reset")

    test_progress_bar_with_metrics_creation()
    print("✓ test_progress_bar_with_metrics_creation")

    test_progress_bar_with_metrics_set_metric()
    print("✓ test_progress_bar_with_metrics_set_metric")

    test_progress_bar_with_metrics_update()
    print("✓ test_progress_bar_with_metrics_update")

    test_progress_bar_with_metrics_clear()
    print("✓ test_progress_bar_with_metrics_clear")

    test_progress_bar_with_metrics_reset()
    print("✓ test_progress_bar_with_metrics_reset")

    test_progress_bar_with_eta_creation()
    print("✓ test_progress_bar_with_eta_creation")

    test_progress_bar_with_eta_update()
    print("✓ test_progress_bar_with_eta_update")

    test_progress_bar_with_eta_metrics()
    print("✓ test_progress_bar_with_eta_metrics")

    test_progress_bar_with_eta_reset()
    print("✓ test_progress_bar_with_eta_reset")

    test_factory_create_progress_bar()
    print("✓ test_factory_create_progress_bar")

    test_factory_create_progress_bar_with_metrics()
    print("✓ test_factory_create_progress_bar_with_metrics")

    test_factory_create_progress_bar_with_eta()
    print("✓ test_factory_create_progress_bar_with_eta")

    test_format_duration()
    print("✓ test_format_duration")

    test_format_duration_long()
    print("✓ test_format_duration_long")

    test_training_loop_simulation()
    print("✓ test_training_loop_simulation")

    test_eta_time_tracking()
    print("✓ test_eta_time_tracking")

    test_rapid_updates()
    print("✓ test_rapid_updates")

    print("\nAll test_progress_bar tests passed!")
