"""Tests for visualization utilities module - Part 1.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_visualization.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

This module tests PlotData, PlotSeries, ConfusionMatrixData structs
and initial training curve plotting functionality.
"""

from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
)
from shared.utils.visualization import (
    PlotData,
    PlotSeries,
    ConfusionMatrixData,
    plot_training_curves,
    plot_loss_only,
    plot_accuracy_only,
)


# ============================================================================
# Test PlotData Struct
# ============================================================================


fn test_plot_data_default_init() raises:
    """Test PlotData default initialization."""
    var plot = PlotData()
    assert_equal(plot.title, "")
    assert_equal(plot.xlabel, "")
    assert_equal(plot.ylabel, "")
    assert_equal(len(plot.x_data), 0)
    assert_equal(len(plot.y_data), 0)
    assert_equal(plot.label, "")


fn test_plot_data_set_attributes() raises:
    """Test setting PlotData attributes."""
    var plot = PlotData()
    plot.title = "Training Loss"
    plot.xlabel = "Epoch"
    plot.ylabel = "Loss"
    plot.label = "train"
    plot.x_data.append(1.0)
    plot.x_data.append(2.0)
    plot.y_data.append(0.5)
    plot.y_data.append(0.3)

    assert_equal(plot.title, "Training Loss")
    assert_equal(plot.xlabel, "Epoch")
    assert_equal(plot.ylabel, "Loss")
    assert_equal(plot.label, "train")
    assert_equal(len(plot.x_data), 2)
    assert_equal(len(plot.y_data), 2)


# ============================================================================
# Test PlotSeries Struct
# ============================================================================


fn test_plot_series_default_init() raises:
    """Test PlotSeries default initialization."""
    var series = PlotSeries()
    assert_equal(series.title, "")
    assert_equal(series.xlabel, "")
    assert_equal(series.ylabel, "")
    assert_equal(len(series.series_data), 0)


fn test_plot_series_add_series() raises:
    """Test adding series to PlotSeries."""
    var plot_series = PlotSeries()
    plot_series.title = "Training Curves"

    var train_data = PlotData()
    train_data.label = "Training"
    train_data.y_data.append(0.5)
    train_data.y_data.append(0.3)

    var val_data = PlotData()
    val_data.label = "Validation"
    val_data.y_data.append(0.6)
    val_data.y_data.append(0.4)

    plot_series.add_series(train_data^)
    plot_series.add_series(val_data^)

    assert_equal(len(plot_series.series_data), 2)


# ============================================================================
# Test ConfusionMatrixData Struct
# ============================================================================


fn test_confusion_matrix_data_default_init() raises:
    """Test ConfusionMatrixData default initialization."""
    var cm = ConfusionMatrixData()
    assert_equal(len(cm.class_names), 0)
    assert_equal(len(cm.matrix), 0)
    assert_equal(cm.accuracy, 0.0)
    assert_equal(cm.precision, 0.0)
    assert_equal(cm.recall, 0.0)


# ============================================================================
# Test Training Curve Plotting
# ============================================================================


fn test_plot_training_loss() raises:
    """Test plotting training loss over epochs."""
    var train_losses = List[Float32]()
    train_losses.append(0.5)
    train_losses.append(0.4)
    train_losses.append(0.3)
    train_losses.append(0.25)
    train_losses.append(0.2)

    var val_losses = List[Float32]()
    val_losses.append(0.6)
    val_losses.append(0.5)
    val_losses.append(0.4)
    val_losses.append(0.35)
    val_losses.append(0.3)

    var result = plot_training_curves(train_losses, val_losses)
    assert_true(result)


fn test_plot_training_and_validation_loss() raises:
    """Test plotting both training and validation loss."""
    var train_losses = List[Float32]()
    train_losses.append(0.5)
    train_losses.append(0.4)
    train_losses.append(0.3)

    var val_losses = List[Float32]()
    val_losses.append(0.6)
    val_losses.append(0.5)
    val_losses.append(0.4)

    var result = plot_training_curves(train_losses, val_losses)
    assert_true(result)


fn test_plot_accuracy_curves() raises:
    """Test plotting accuracy curves over epochs."""
    var train_losses = List[Float32]()
    train_losses.append(0.5)
    train_losses.append(0.3)

    var val_losses = List[Float32]()
    val_losses.append(0.6)
    val_losses.append(0.4)

    var train_accs = List[Float32]()
    train_accs.append(0.6)
    train_accs.append(0.8)

    var val_accs = List[Float32]()
    val_accs.append(0.55)
    val_accs.append(0.75)

    var result = plot_training_curves(
        train_losses, val_losses, train_accs, val_accs
    )
    assert_true(result)


fn main() raises:
    """Run all tests."""
    print("Test Visualization Utilities - Part 1")
    print("=" * 50)

    print("  test_plot_data_default_init...", end="")
    test_plot_data_default_init()
    print(" OK")

    print("  test_plot_data_set_attributes...", end="")
    test_plot_data_set_attributes()
    print(" OK")

    print("  test_plot_series_default_init...", end="")
    test_plot_series_default_init()
    print(" OK")

    print("  test_plot_series_add_series...", end="")
    test_plot_series_add_series()
    print(" OK")

    print("  test_confusion_matrix_data_default_init...", end="")
    test_confusion_matrix_data_default_init()
    print(" OK")

    print("  test_plot_training_loss...", end="")
    test_plot_training_loss()
    print(" OK")

    print("  test_plot_training_and_validation_loss...", end="")
    test_plot_training_and_validation_loss()
    print(" OK")

    print("  test_plot_accuracy_curves...", end="")
    test_plot_accuracy_curves()
    print(" OK")

    print()
    print("All visualization part 1 tests passed (8/8)")
