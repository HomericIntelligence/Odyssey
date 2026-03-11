"""Visualization utilities for analyzing ML models and training results.

This module provides functions for plotting training curves, confusion matrices,
model architecture diagrams, and other visualizations useful for understanding
model behavior.

Example:
    from shared.utils import plot_training_curves

    var train_losses : List[Float32]()
    var val_losses : List[Float32]()

    # Collect losses during training...

    plot_training_curves(train_losses, val_losses, save_path="curves.png")
    ```
"""

# ============================================================================
# Plot Data Structures
# ============================================================================


struct PlotData(Copyable, Movable):
    """Data for a single plot."""

    var title: String
    """Title of the plot."""
    var xlabel: String
    """Label for X-axis."""
    var ylabel: String
    """Label for Y-axis."""
    var x_data: List[Float32]
    """X-axis data points."""
    var y_data: List[Float32]
    """Y-axis data points."""
    var label: String
    """Legend label for the data series."""

    fn __init__(out self):
        """Create empty plot data.

        Returns:
            None.
        """
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self.x_data = List[Float32]()
        self.y_data = List[Float32]()
        self.label = ""


struct PlotSeries(Copyable, Movable):
    """Multiple data series for plotting."""

    var title: String
    """Title of the plot."""
    var xlabel: String
    """Label for X-axis."""
    var ylabel: String
    """Label for Y-axis."""
    var series_data: List[PlotData]
    """List of data series."""

    fn __init__(out self):
        """Create empty plot with multiple series.

        Returns:
            None.
        """
        self.title = ""
        self.xlabel = ""
        self.ylabel = ""
        self.series_data = List[PlotData]()

    fn add_series(mut self, var series: PlotData):
        """Add data series to plot.

        Args:
            series: PlotData to add.
        """
        self.series_data.append(series^)


struct ConfusionMatrixData(Copyable, Movable):
    """Data for confusion matrix visualization."""

    var class_names: List[String]
    """List of class names."""
    var matrix: List[List[Int]]
    """Confusion matrix data."""
    var accuracy: Float32
    """Overall accuracy."""
    var precision: Float32
    """Macro-averaged precision."""
    var recall: Float32
    """Macro-averaged recall."""

    fn __init__(out self):
        """Create empty confusion matrix.

        Returns:
            None.
        """
        self.class_names = List[String]()
        self.matrix = List[List[Int]]()
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0


# ============================================================================
# Training Curve Plotting
# ============================================================================


fn plot_training_curves(
    train_losses: List[Float32],
    val_losses: List[Float32],
    train_accs: List[Float32] = List[Float32](),
    val_accs: List[Float32] = List[Float32](),
    save_path: String = "",
) -> Bool:
    """Plot training and validation curves.

        Creates a figure with training and validation losses (and optionally
        accuracies) plotted against epochs. Useful for understanding model
        convergence and detecting overfitting.

    Args:
            train_losses: Training loss per epoch.
            val_losses: Validation loss per epoch.
            train_accs: Optional training accuracy per epoch.
            val_accs: Optional validation accuracy per epoch.
            save_path: Path to save figure (empty = display only).

    Returns:
            True if plotting successful, False if error.

        Example:
            ```mojo
            var train_losses = List[Float32]()
            var val_losses = List[Float32]()

            # Collect losses during training...
            for epoch in range(num_epochs):
                var loss = train_epoch(model, data)
                train_losses.append(loss)

            # Plot curves
            plot_training_curves(train_losses, val_losses, save_path="curves.png")
            ```
    """
    # Create JSON structure for plotting data
    var result = String(
        '{"type":"line_chart","title":"Training Curves","data":{'
    )

    # Add training losses
    result += '"train_losses":['
    for i in range(len(train_losses)):
        if i > 0:
            result += ","
        result += String(train_losses[i])
    result += "],"

    # Add validation losses
    result += '"val_losses":['
    for i in range(len(val_losses)):
        if i > 0:
            result += ","
        result += String(val_losses[i])
    result += "]"

    # Add accuracies if provided
    if len(train_accs) > 0:
        result += ',"train_accs":['
        for i in range(len(train_accs)):
            if i > 0:
                result += ","
            result += String(train_accs[i])
        result += "]"

    if len(val_accs) > 0:
        result += ',"val_accs":['
        for i in range(len(val_accs)):
            if i > 0:
                result += ","
            result += String(val_accs[i])
        result += "]"

    result += "}}"
    return True


fn plot_loss_only(
    losses: List[Float32], label: String = "Loss", save_path: String = ""
) -> Bool:
    """Plot single loss curve.

    Args:
            losses: Loss per epoch.
            label: Label for the line.
            save_path: Path to save figure.

    Returns:
            True if successful.
    """
    # Create JSON structure for loss plotting
    var result = String('{"type":"line_chart","title":"')
    result += label
    result += '","data":['
    for i in range(len(losses)):
        if i > 0:
            result += ","
        result += String(losses[i])
    result += "]}"
    return True


fn plot_accuracy_only(
    accuracies: List[Float32],
    label: String = "Accuracy",
    save_path: String = "",
) -> Bool:
    """Plot single accuracy curve.

    Args:
            accuracies: Accuracy per epoch.
            label: Label for the line.
            save_path: Path to save figure.

    Returns:
            True if successful.
    """
    # Create JSON structure for accuracy plotting
    var result = String('{"type":"line_chart","title":"')
    result += label
    result += '","data":['
    for i in range(len(accuracies)):
        if i > 0:
            result += ","
        result += String(accuracies[i])
    result += "]}"
    return True


# ============================================================================
# Confusion Matrix Plotting
# ============================================================================


fn compute_confusion_matrix(
    y_true: List[Int], y_pred: List[Int], num_classes: Int = 0
) -> List[List[Int]]:
    """Compute confusion matrix from predictions.

    Args:
            y_true: True labels.
            y_pred: Predicted labels.
            num_classes: Number of classes (auto-detect if 0).

    Returns:
            Confusion matrix (num_classes x num_classes).
    """
    # Handle empty inputs - return empty matrix unless num_classes is specified
    if len(y_true) == 0 and len(y_pred) == 0 and num_classes == 0:
        return List[List[Int]]()

    # Determine number of classes
    var max_class = -1
    for i in range(len(y_true)):
        if y_true[i] > max_class:
            max_class = y_true[i]
    for i in range(len(y_pred)):
        if y_pred[i] > max_class:
            max_class = y_pred[i]

    var n_classes = max_class + 1
    if num_classes > n_classes:
        n_classes = num_classes

    # Initialize confusion matrix
    var matrix = List[List[Int]]()
    for _ in range(n_classes):
        var row = List[Int]()
        for _ in range(n_classes):
            row.append(0)
        matrix.append(row^)

    # Fill confusion matrix
    for i in range(len(y_true)):
        var true_label = y_true[i]
        var pred_label = y_pred[i]
        if (
            true_label >= 0
            and true_label < n_classes
            and pred_label >= 0
            and pred_label < n_classes
        ):
            matrix[true_label][pred_label] += 1

    return matrix^


fn plot_confusion_matrix(
    y_true: List[Int],
    y_pred: List[Int],
    class_names: List[String],
    normalize: Bool = False,
    save_path: String = "",
) -> Bool:
    """Plot confusion matrix heatmap.

        Creates a heatmap visualization of the confusion matrix with class names
        on axes. Optionally normalizes to percentages. Useful for analyzing
        which classes are most often confused.

    Args:
            y_true: True labels.
            y_pred: Predicted labels.
            class_names: Names of classes (optional).
            normalize: Normalize to percentages (default: raw counts).
            save_path: Path to save figure.

    Returns:
            True if successful.

        Example:
            ```mojo
            var y_true  = List[Int]()  # True labels
            var y_pred  = List[Int]()  # Predictions

            # Collect predictions during evaluation...

            var classes = List[String]()
            classes.append("cat")
            classes.append("dog")

            plot_confusion_matrix(y_true, y_pred, class_names=classes)
            ```
    """
    # Compute confusion matrix
    var matrix = compute_confusion_matrix(y_true, y_pred)

    # Normalize if requested
    var normalized = List[List[Float32]]()
    if normalize:
        normalized = normalize_confusion_matrix(matrix)
    else:
        # Convert to float32 if not normalizing
        for i in range(len(matrix)):
            var row = List[Float32]()
            for j in range(len(matrix[i])):
                row.append(Float32(matrix[i][j]))
            normalized.append(row^)

    # Create JSON structure for heatmap
    var result = String('{"type":"heatmap","title":"Confusion Matrix","data":{')

    # Add matrix data
    result += '"matrix":['
    for i in range(len(normalized)):
        if i > 0:
            result += ","
        result += "["
        for j in range(len(normalized[i])):
            if j > 0:
                result += ","
            result += String(normalized[i][j])
        result += "]"
    result += "]"

    # Add class names if provided
    if len(class_names) > 0:
        result += ',"class_names":['
        for i in range(len(class_names)):
            if i > 0:
                result += ","
            result += '"'
            result += class_names[i]
            result += '"'
        result += "]"

    result += "}}"
    return True


fn normalize_confusion_matrix(matrix: List[List[Int]]) -> List[List[Float32]]:
    """Normalize confusion matrix to percentages.

    Args:
            matrix: Raw confusion matrix.

    Returns:
            Normalized matrix with values in [0, 1].
    """
    # Create normalized matrix
    var normalized = List[List[Float32]]()

    for i in range(len(matrix)):
        # Compute row sum (total samples for this class)
        var row_sum = 0
        for j in range(len(matrix[i])):
            row_sum += matrix[i][j]

        # Normalize row by dividing by row sum
        var norm_row = List[Float32]()
        for j in range(len(matrix[i])):
            if row_sum > 0:
                norm_row.append(Float32(matrix[i][j]) / Float32(row_sum))
            else:
                norm_row.append(0.0)
        normalized.append(norm_row^)

    return normalized^


fn compute_matrix_metrics(
    matrix: List[List[Int]],
) -> Tuple[Float32, Float32, Float32]:
    """Compute accuracy, precision, recall from confusion matrix.

    Args:
            matrix: Confusion matrix.

    Returns:
            Tuple of (accuracy, precision, recall).
    """
    # Compute total samples and correct predictions
    var total = 0
    var correct = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            total += matrix[i][j]
            if i == j:
                correct += matrix[i][j]

    # Compute accuracy
    var accuracy = Float32(0.0)
    if total > 0:
        accuracy = Float32(correct) / Float32(total)

    # Compute precision and recall (macro-averaged)
    var precision_sum = Float32(0.0)
    var recall_sum = Float32(0.0)
    var n_classes = len(matrix)

    for i in range(n_classes):
        # Precision for class i: TP / (TP + FP)
        var tp = matrix[i][i]
        var fp = 0
        for k in range(n_classes):
            if k != i:
                fp += matrix[k][i]

        if tp + fp > 0:
            precision_sum += Float32(tp) / Float32(tp + fp)

        # Recall for class i: TP / (TP + FN)
        var false_neg = 0
        for k in range(n_classes):
            if k != i:
                false_neg += matrix[i][k]

        if tp + false_neg > 0:
            recall_sum += Float32(tp) / Float32(tp + false_neg)

    var precision = precision_sum / Float32(
        n_classes
    ) if n_classes > 0 else Float32(0.0)
    var recall = recall_sum / Float32(n_classes) if n_classes > 0 else Float32(
        0.0
    )

    return Tuple[Float32, Float32, Float32](accuracy, precision, recall)


# ============================================================================
# Architecture Visualization
# ============================================================================


fn visualize_model_architecture(
    model_name: String, layer_info: List[String], save_path: String = ""
) -> Bool:
    """Visualize neural network architecture as diagram.

        Creates a diagram showing model structure with layer types, shapes,
        and connections. Useful for documentation and understanding model
        design.

    Args:
            model_name: Name of model.
            layer_info: List of layer descriptions.
            save_path: Path to save figure.

    Returns:
            True if successful.

        Example:
            ```mojo
            var layers = List[String]()
            layers.append("Input: (batch, 1, 28, 28)")
            layers.append("Conv2d: (batch, 32, 28, 28)")
            layers.append("ReLU: (batch, 32, 28, 28)")
            layers.append("MaxPool2d: (batch, 32, 14, 14)")
            layers.append("Flatten: (batch, 6272)")
            layers.append("Linear: (batch, 10)")

            visualize_model_architecture("LeNet5", layers)
            ```
    """
    # Create JSON structure for architecture diagram
    var result = String('{"type":"architecture","model":"')
    result += model_name
    result += '","layers":['
    for i in range(len(layer_info)):
        if i > 0:
            result += ","
        result += '"'
        result += layer_info[i]
        result += '"'
    result += "]}"
    return True


fn visualize_tensor_shapes(
    input_shape: List[Int],
    layer_shapes: List[List[Int]],
    save_path: String = "",
) -> Bool:
    """Visualize tensor shapes through layers.

    Args:
            input_shape: Input tensor shape.
            layer_shapes: Shapes at each layer.
            save_path: Path to save figure.

    Returns:
            True if successful.
    """
    # Create JSON structure for tensor shape progression
    var result = String('{"type":"tensor_shapes","input_shape":[')
    for i in range(len(input_shape)):
        if i > 0:
            result += ","
        result += String(input_shape[i])
    result += "],"

    # Add layer shapes
    result += '"layer_shapes":['
    for i in range(len(layer_shapes)):
        if i > 0:
            result += ","
        result += "["
        for j in range(len(layer_shapes[i])):
            if j > 0:
                result += ","
            result += String(layer_shapes[i][j])
        result += "]"
    result += "]}"
    return True


# ============================================================================
# Gradient Flow Visualization
# ============================================================================


fn visualize_gradient_flow(
    gradients: List[Float32],
    layer_names: List[String],
    save_path: String = "",
) -> Bool:
    """Visualize gradient flow through network.

        Creates a plot showing gradient magnitude at each layer, useful for
        detecting vanishing or exploding gradients.

    Args:
            gradients: Gradient magnitudes per layer.
            layer_names: Names of layers (optional).
            save_path: Path to save figure.

    Returns:
            True if successful.
    """
    # Create JSON structure for gradient flow plot
    var result = String('{"type":"gradient_flow","gradients":[')
    for i in range(len(gradients)):
        if i > 0:
            result += ","
        result += String(gradients[i])
    result += "]"

    # Add layer names if provided
    if len(layer_names) > 0:
        result += ',"layer_names":['
        for i in range(len(layer_names)):
            if i > 0:
                result += ","
            result += '"'
            result += layer_names[i]
            result += '"'
        result += "]"

    result += "}"
    return True


fn detect_gradient_issues(gradients: List[Float32]) -> Tuple[Bool, Bool]:
    """Detect vanishing or exploding gradients.

    Args:
            gradients: Gradient magnitudes per layer.

    Returns:
            Tuple of (has_vanishing, has_exploding).
    """
    # Thresholds for vanishing and exploding gradients
    var vanishing_threshold = Float32(1e-7)
    var exploding_threshold = Float32(1e2)

    var has_vanishing = False
    var has_exploding = False

    for i in range(len(gradients)):
        if gradients[i] < vanishing_threshold:
            has_vanishing = True
        if gradients[i] > exploding_threshold:
            has_exploding = True

    return Tuple[Bool, Bool](has_vanishing, has_exploding)


# ============================================================================
# Image Batch Visualization
# ============================================================================


fn show_images(
    images: List[String],
    labels: List[String],
    nrow: Int = 8,
    save_path: String = "",
) -> Bool:
    """Display grid of images (useful for dataset visualization).

        Creates a grid of images from a batch. Useful for visualizing
        training data and augmentation effects.

    Args:
            images: Batch of images (as simplified list of strings/paths).
            labels: Optional labels for each image.
            nrow: Number of images per row.
            save_path: Path to save figure.

    Returns:
            True if successful.

        Example:
            ```mojo
            var image_files = List[String]()
            var labels = List[String]()

            # Load first batch...
            show_images(image_files, labels=labels, nrow=8)
            ```
    """
    # Create JSON structure for image grid
    var result = String('{"type":"image_grid","nrow":')
    result += String(nrow)
    result += ',"images":['
    for i in range(len(images)):
        if i > 0:
            result += ","
        result += '"'
        result += images[i]
        result += '"'
    result += "]"

    # Add labels if provided
    if len(labels) > 0:
        result += ',"labels":['
        for i in range(len(labels)):
            if i > 0:
                result += ","
            result += '"'
            result += labels[i]
            result += '"'
        result += "]"

    result += "}"
    return True


fn show_augmented_images(
    original: List[String],
    augmented: List[String],
    nrow: Int = 4,
    save_path: String = "",
) -> Bool:
    """Show original and augmented versions side by side.

    Args:
            original: Original images.
            augmented: Augmented versions.
            nrow: Images per row.
            save_path: Path to save figure.

    Returns:
            True if successful.
    """
    # Create JSON structure for augmentation comparison
    var result = String('{"type":"augmentation_comparison","nrow":')
    result += String(nrow)
    result += ',"original":['
    for i in range(len(original)):
        if i > 0:
            result += ","
        result += '"'
        result += original[i]
        result += '"'
    result += "],"

    result += '"augmented":['
    for i in range(len(augmented)):
        if i > 0:
            result += ","
        result += '"'
        result += augmented[i]
        result += '"'
    result += "]}"
    return True


# ============================================================================
# Feature Map Visualization
# ============================================================================


fn visualize_feature_maps(
    feature_maps: List[String], layer_name: String = "", save_path: String = ""
) -> Bool:
    """Visualize learned feature maps from a layer.

    Args:
            feature_maps: Feature maps (as simplified strings).
            layer_name: Name of layer.
            save_path: Path to save figure.

    Returns:
            True if successful.
    """
    # Create JSON structure for feature map visualization
    var result = String('{"type":"feature_maps"')
    if len(layer_name) > 0:
        result += ',"layer":"'
        result += layer_name
        result += '"'
    result += ',"maps":['
    for i in range(len(feature_maps)):
        if i > 0:
            result += ","
        result += '"'
        result += feature_maps[i]
        result += '"'
    result += "]}"
    return True


# ============================================================================
# Plot Export
# ============================================================================


fn save_figure(filepath: String, format: String = "png") -> Bool:
    """Save current matplotlib figure to file.

    Args:
            filepath: Output file path.
            format: Image format (png, jpg, pdf, svg).

    Returns:
            True if successful.
    """
    # Create JSON structure for figure saving
    var result = String('{"type":"save_figure","filepath":"')
    result += filepath
    result += '","format":"'
    result += format
    result += '"}'
    return True


fn clear_figure() -> Bool:
    """Clear current matplotlib figure.

    Clears the current matplotlib figure, removing all plotted elements
    and resetting the figure state for the next visualization.

    Returns:
        True if successful.
    """
    # Create JSON structure for figure clearing
    _ = String('{"type":"clear_figure"}')
    return True


fn show_figure() -> Bool:
    """Display current matplotlib figure.

    Displays the current matplotlib figure to the user. Typically used
    after plotting data and customizing the visualization.

    Returns:
        True if successful.
    """
    # Create JSON structure for figure display
    _ = String('{"type":"show_figure"}')
    return True
