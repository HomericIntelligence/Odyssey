"""Tests for visualization utilities module

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
from shared.utils.visualization import (
    plot_loss_only,
    plot_accuracy_only,
    compute_confusion_matrix,
    plot_confusion_matrix,
)
from shared.utils.visualization import (
    normalize_confusion_matrix,
    compute_matrix_metrics,
    compute_confusion_matrix,
    visualize_model_architecture,
    visualize_tensor_shapes,
)
from shared.utils.visualization import (
    visualize_gradient_flow,
    detect_gradient_issues,
    show_images,
)
from shared.utils.visualization import (
    show_images,
    show_augmented_images,
    visualize_feature_maps,
    save_figure,
    clear_figure,
    show_figure,
)


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


fn test_confusion_matrix_data_default_init() raises:
    """Test ConfusionMatrixData default initialization."""
    var cm = ConfusionMatrixData()
    assert_equal(len(cm.class_names), 0)
    assert_equal(len(cm.matrix), 0)
    assert_equal(cm.accuracy, 0.0)
    assert_equal(cm.precision, 0.0)
    assert_equal(cm.recall, 0.0)


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


fn test_plot_loss_only_single_series() raises:
    """Test plotting single loss series."""
    var losses = List[Float32]()
    losses.append(0.5)
    losses.append(0.4)
    losses.append(0.3)

    var result = plot_loss_only(losses, "Training Loss")
    assert_true(result)


fn test_plot_accuracy_only_single_series() raises:
    """Test plotting single accuracy series."""
    var accuracies = List[Float32]()
    accuracies.append(0.6)
    accuracies.append(0.75)
    accuracies.append(0.85)

    var result = plot_accuracy_only(accuracies, "Validation Accuracy")
    assert_true(result)


fn test_plot_with_save_path() raises:
    """Test plotting with save path specified."""
    var losses = List[Float32]()
    losses.append(0.5)
    losses.append(0.3)

    var result = plot_loss_only(losses, "Loss", "output.png")
    assert_true(result)


fn test_create_confusion_matrix() raises:
    """Test creating confusion matrix from predictions."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(1)
    y_true.append(2)
    y_true.append(0)
    y_true.append(1)
    y_true.append(2)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(2)
    y_pred.append(2)
    y_pred.append(0)
    y_pred.append(1)
    y_pred.append(1)

    var matrix = compute_confusion_matrix(y_true, y_pred)

    # Verify matrix shape: 3x3 (for 3 classes)
    assert_equal(len(matrix), 3)
    assert_equal(len(matrix[0]), 3)

    # Verify diagonal values (correct predictions)
    assert_equal(matrix[0][0], 2)  # Class 0: 2 correct
    assert_equal(matrix[1][1], 1)  # Class 1: 1 correct
    assert_equal(matrix[2][2], 1)  # Class 2: 1 correct


fn test_confusion_matrix_with_num_classes() raises:
    """Test confusion matrix with specified number of classes."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(1)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(1)

    # Request 4 classes even though only 2 are present
    var matrix = compute_confusion_matrix(y_true, y_pred, num_classes=4)

    assert_equal(len(matrix), 4)
    assert_equal(len(matrix[0]), 4)


fn test_plot_confusion_matrix() raises:
    """Test plotting confusion matrix as heatmap."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(1)
    y_true.append(0)
    y_true.append(1)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(0)
    y_pred.append(0)
    y_pred.append(1)

    var class_names = List[String]()
    class_names.append("cat")
    class_names.append("dog")

    var result = plot_confusion_matrix(y_true, y_pred, class_names)
    assert_true(result)


fn test_confusion_matrix_with_class_names() raises:
    """Test confusion matrix with custom class names."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(1)
    y_true.append(2)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(1)
    y_pred.append(2)

    var class_names = List[String]()
    class_names.append("cat")
    class_names.append("dog")
    class_names.append("bird")

    var result = plot_confusion_matrix(y_true, y_pred, class_names)
    assert_true(result)


fn test_empty_confusion_matrix() raises:
    """Test confusion matrix with empty inputs."""
    var y_true = List[Int]()
    var y_pred = List[Int]()

    var matrix = compute_confusion_matrix(y_true, y_pred)
    # Empty inputs should produce empty matrix
    assert_equal(len(matrix), 0)


fn test_confusion_matrix_normalization() raises:
    """Test normalizing confusion matrix by row (true labels)."""
    var matrix = List[List[Int]]()
    var row0 = List[Int]()
    row0.append(8)
    row0.append(2)
    matrix.append(row0^)

    var row1 = List[Int]()
    row1.append(1)
    row1.append(9)
    matrix.append(row1^)

    var normalized = normalize_confusion_matrix(matrix)

    # Row 0: [8, 2] -> [0.8, 0.2]
    assert_true(normalized[0][0] > 0.79 and normalized[0][0] < 0.81)
    assert_true(normalized[0][1] > 0.19 and normalized[0][1] < 0.21)

    # Row 1: [1, 9] -> [0.1, 0.9]
    assert_true(normalized[1][0] > 0.09 and normalized[1][0] < 0.11)
    assert_true(normalized[1][1] > 0.89 and normalized[1][1] < 0.91)


fn test_confusion_matrix_accuracy() raises:
    """Test computing accuracy from confusion matrix."""
    var matrix = List[List[Int]]()

    var row0 = List[Int]()
    row0.append(8)
    row0.append(1)
    row0.append(0)
    matrix.append(row0^)

    var row1 = List[Int]()
    row1.append(1)
    row1.append(7)
    row1.append(2)
    matrix.append(row1^)

    var row2 = List[Int]()
    row2.append(0)
    row2.append(1)
    row2.append(9)
    matrix.append(row2^)

    var metrics = compute_matrix_metrics(matrix)
    var accuracy = metrics[0]

    # Accuracy: (8+7+9) / 29 = 0.8275...
    # Total = 8+1+0+1+7+2+0+1+9 = 29, correct = 8+7+9 = 24
    assert_true(accuracy > 0.82 and accuracy < 0.84)


fn test_single_class_confusion_matrix() raises:
    """Test confusion matrix with single class."""
    var y_true = List[Int]()
    y_true.append(0)
    y_true.append(0)
    y_true.append(0)

    var y_pred = List[Int]()
    y_pred.append(0)
    y_pred.append(0)
    y_pred.append(0)

    var matrix = compute_confusion_matrix(y_true, y_pred)
    assert_equal(len(matrix), 1)
    assert_equal(matrix[0][0], 3)


fn test_visualize_simple_model() raises:
    """Test visualizing simple neural network architecture."""
    var layers = List[String]()
    layers.append("Input: (batch, 784)")
    layers.append("Linear: (batch, 128)")
    layers.append("ReLU: (batch, 128)")
    layers.append("Linear: (batch, 10)")

    var result = visualize_model_architecture("SimpleNN", layers)
    assert_true(result)


fn test_visualize_conv_model() raises:
    """Test visualizing convolutional neural network."""
    var layers = List[String]()
    layers.append("Input: (batch, 1, 28, 28)")
    layers.append("Conv2D: (batch, 32, 26, 26)")
    layers.append("ReLU: (batch, 32, 26, 26)")
    layers.append("MaxPool2D: (batch, 32, 13, 13)")
    layers.append("Flatten: (batch, 5408)")
    layers.append("Linear: (batch, 10)")

    var result = visualize_model_architecture("LeNet", layers)
    assert_true(result)


fn test_visualize_model_with_shapes() raises:
    """Test visualizing model with tensor shapes at each layer."""
    var input_shape = List[Int]()
    input_shape.append(1)
    input_shape.append(28)
    input_shape.append(28)
    input_shape.append(1)

    var layer_shapes = List[List[Int]]()

    var shape1 = List[Int]()
    shape1.append(1)
    shape1.append(24)
    shape1.append(24)
    shape1.append(32)
    layer_shapes.append(shape1^)

    var shape2 = List[Int]()
    shape2.append(1)
    shape2.append(12)
    shape2.append(12)
    shape2.append(32)
    layer_shapes.append(shape2^)

    var result = visualize_tensor_shapes(input_shape, layer_shapes)
    assert_true(result)


fn test_save_architecture_diagram() raises:
    """Test saving architecture diagram to file."""
    var layers = List[String]()
    layers.append("Input: (batch, 784)")
    layers.append("Linear: (batch, 10)")

    var result = visualize_model_architecture("Model", layers, "arch.png")
    assert_true(result)


fn test_visualize_gradient_magnitudes() raises:
    """Test visualizing gradient magnitudes by layer."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(0.005)
    gradients.append(0.001)
    gradients.append(0.0005)

    var layer_names = List[String]()
    layer_names.append("conv1")
    layer_names.append("conv2")
    layer_names.append("fc1")
    layer_names.append("fc2")

    var result = visualize_gradient_flow(gradients, layer_names)
    assert_true(result)


fn test_detect_vanishing_gradients() raises:
    """Test detecting vanishing gradient problem."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(0.0001)
    gradients.append(1e-8)  # Very small - vanishing

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_true(has_vanishing)
    assert_false(has_exploding)


fn test_detect_exploding_gradients() raises:
    """Test detecting exploding gradient problem."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(10.0)
    gradients.append(1000.0)  # Very large - exploding

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_false(has_vanishing)
    assert_true(has_exploding)


fn test_detect_both_gradient_issues() raises:
    """Test detecting both vanishing and exploding gradients."""
    var gradients = List[Float32]()
    gradients.append(1e-10)  # Vanishing
    gradients.append(0.01)  # Normal
    gradients.append(1000.0)  # Exploding

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_true(has_vanishing)
    assert_true(has_exploding)


fn test_no_gradient_issues() raises:
    """Test normal gradients without issues."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(0.005)
    gradients.append(0.001)

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_false(has_vanishing)
    assert_false(has_exploding)


fn test_plot_gradient_flow() raises:
    """Test plotting gradient flow through network."""
    var gradients = List[Float32]()
    gradients.append(0.01)
    gradients.append(0.008)
    gradients.append(0.006)

    var layer_names = List[String]()
    layer_names.append("layer1")
    layer_names.append("layer2")
    layer_names.append("layer3")

    var result = visualize_gradient_flow(gradients, layer_names)
    assert_true(result)


fn test_empty_gradients() raises:
    """Test gradient detection with empty list."""
    var gradients = List[Float32]()

    var issues = detect_gradient_issues(gradients)
    var has_vanishing = issues[0]
    var has_exploding = issues[1]

    assert_false(has_vanishing)
    assert_false(has_exploding)


fn test_visualize_image_batch() raises:
    """Test visualizing batch of images in grid."""
    var images = List[String]()
    for i in range(16):
        images.append("image_" + String(i) + ".png")

    var labels = List[String]()

    var result = show_images(images, labels, nrow=4)
    assert_true(result)


fn test_visualize_images_with_labels() raises:
    """Test visualizing images with labels."""
    var images = List[String]()
    images.append("img1.png")
    images.append("img2.png")
    images.append("img3.png")

    var labels = List[String]()
    labels.append("cat")
    labels.append("dog")
    labels.append("bird")

    var result = show_images(images, labels)
    assert_true(result)


fn test_visualize_augmented_images() raises:
    """Test visualizing original and augmented images side by side."""
    var original = List[String]()
    original.append("orig1.png")
    original.append("orig2.png")

    var augmented = List[String]()
    augmented.append("aug1.png")
    augmented.append("aug2.png")

    var result = show_augmented_images(original, augmented, nrow=2)
    assert_true(result)


fn test_visualize_feature_maps() raises:
    """Test visualizing convolutional feature maps."""
    var feature_maps = List[String]()
    feature_maps.append("fmap_0")
    feature_maps.append("fmap_1")
    feature_maps.append("fmap_2")
    feature_maps.append("fmap_3")

    var result = visualize_feature_maps(feature_maps, "conv1")
    assert_true(result)


fn test_visualize_feature_maps_no_layer_name() raises:
    """Test visualizing feature maps without layer name."""
    var feature_maps = List[String]()
    feature_maps.append("fmap_0")

    var result = visualize_feature_maps(feature_maps)
    assert_true(result)


fn test_save_figure_png() raises:
    """Test saving figure as PNG."""
    var result = save_figure("output.png", "png")
    assert_true(result)


fn test_save_figure_svg() raises:
    """Test saving figure as SVG."""
    var result = save_figure("output.svg", "svg")
    assert_true(result)


fn test_save_figure_pdf() raises:
    """Test saving figure as PDF."""
    var result = save_figure("output.pdf", "pdf")
    assert_true(result)


fn test_clear_figure() raises:
    """Test clearing figure."""
    var result = clear_figure()
    assert_true(result)


fn test_show_figure() raises:
    """Test showing figure."""
    var result = show_figure()
    assert_true(result)


fn main() raises:
    """Run all test_visualization tests."""
    print("Running test_visualization tests...")

    test_plot_data_default_init()
    print("✓ test_plot_data_default_init")

    test_plot_data_set_attributes()
    print("✓ test_plot_data_set_attributes")

    test_plot_series_default_init()
    print("✓ test_plot_series_default_init")

    test_plot_series_add_series()
    print("✓ test_plot_series_add_series")

    test_confusion_matrix_data_default_init()
    print("✓ test_confusion_matrix_data_default_init")

    test_plot_training_loss()
    print("✓ test_plot_training_loss")

    test_plot_training_and_validation_loss()
    print("✓ test_plot_training_and_validation_loss")

    test_plot_accuracy_curves()
    print("✓ test_plot_accuracy_curves")

    test_plot_loss_only_single_series()
    print("✓ test_plot_loss_only_single_series")

    test_plot_accuracy_only_single_series()
    print("✓ test_plot_accuracy_only_single_series")

    test_plot_with_save_path()
    print("✓ test_plot_with_save_path")

    test_create_confusion_matrix()
    print("✓ test_create_confusion_matrix")

    test_confusion_matrix_with_num_classes()
    print("✓ test_confusion_matrix_with_num_classes")

    test_plot_confusion_matrix()
    print("✓ test_plot_confusion_matrix")

    test_confusion_matrix_with_class_names()
    print("✓ test_confusion_matrix_with_class_names")

    test_empty_confusion_matrix()
    print("✓ test_empty_confusion_matrix")

    test_confusion_matrix_normalization()
    print("✓ test_confusion_matrix_normalization")

    test_confusion_matrix_accuracy()
    print("✓ test_confusion_matrix_accuracy")

    test_single_class_confusion_matrix()
    print("✓ test_single_class_confusion_matrix")

    test_visualize_simple_model()
    print("✓ test_visualize_simple_model")

    test_visualize_conv_model()
    print("✓ test_visualize_conv_model")

    test_visualize_model_with_shapes()
    print("✓ test_visualize_model_with_shapes")

    test_save_architecture_diagram()
    print("✓ test_save_architecture_diagram")

    test_visualize_gradient_magnitudes()
    print("✓ test_visualize_gradient_magnitudes")

    test_detect_vanishing_gradients()
    print("✓ test_detect_vanishing_gradients")

    test_detect_exploding_gradients()
    print("✓ test_detect_exploding_gradients")

    test_detect_both_gradient_issues()
    print("✓ test_detect_both_gradient_issues")

    test_no_gradient_issues()
    print("✓ test_no_gradient_issues")

    test_plot_gradient_flow()
    print("✓ test_plot_gradient_flow")

    test_empty_gradients()
    print("✓ test_empty_gradients")

    test_visualize_image_batch()
    print("✓ test_visualize_image_batch")

    test_visualize_images_with_labels()
    print("✓ test_visualize_images_with_labels")

    test_visualize_augmented_images()
    print("✓ test_visualize_augmented_images")

    test_visualize_feature_maps()
    print("✓ test_visualize_feature_maps")

    test_visualize_feature_maps_no_layer_name()
    print("✓ test_visualize_feature_maps_no_layer_name")

    test_save_figure_png()
    print("✓ test_save_figure_png")

    test_save_figure_svg()
    print("✓ test_save_figure_svg")

    test_save_figure_pdf()
    print("✓ test_save_figure_pdf")

    test_clear_figure()
    print("✓ test_clear_figure")

    test_show_figure()
    print("✓ test_show_figure")

    print("\nAll test_visualization tests passed!")
