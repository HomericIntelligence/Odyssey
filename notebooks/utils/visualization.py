"""Visualization utilities for ML Odyssey notebooks.

Provides matplotlib-based plotting for:
- Training curves (loss, accuracy)
- Confusion matrices
- Tensor heatmaps
- Model architecture diagrams
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Tuple
import seaborn as sns


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training curves with loss and accuracy.

    Args:
        train_losses: Training loss values per epoch
        val_losses: Optional validation loss values
        train_accuracies: Optional training accuracy values
        val_accuracies: Optional validation accuracy values
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2 if train_accuracies else 1, figsize=figsize)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Plot loss
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    if val_losses:
        axes[0].plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy if provided
    if train_accuracies and len(axes) > 1:
        axes[1].plot(epochs, train_accuracies, "b-", label="Train Acc", linewidth=2)
        if val_accuracies:
            axes[1].plot(epochs, val_accuracies, "r-", label="Val Acc", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix (num_classes x num_classes)
        class_names: Optional list of class names
        title: Figure title
        cmap: Colormap name
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


def visualize_tensor(
    tensor: np.ndarray,
    title: str = "Tensor Visualization",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize 2D tensor as heatmap.

    Args:
        tensor: 2D NumPy array
        title: Figure title
        cmap: Colormap name
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object

    Raises:
        ValueError: If tensor is not 2D
    """
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.ndim}D")

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(tensor, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


def display_model_summary(layers: List[Dict]) -> None:
    """Display model architecture as formatted table.

    Args:
        layers: List of dicts with 'name', 'type', 'params', 'output_shape'

    Example:
        layers = [
            {'name': 'conv1', 'type': 'Conv2D', 'params': 1728, 'output_shape': '(N, 32, 28, 28)'},
            {'name': 'relu1', 'type': 'ReLU', 'params': 0, 'output_shape': '(N, 32, 28, 28)'},
            {'name': 'pool1', 'type': 'MaxPool2D', 'params': 0, 'output_shape': '(N, 32, 14, 14)'},
        ]
        display_model_summary(layers)
    """
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Layer", "Type", "Output Shape", "Parameters"]

    total_params = 0
    for layer in layers:
        table.add_row(
            [
                layer.get("name", ""),
                layer.get("type", ""),
                layer.get("output_shape", ""),
                layer.get("params", 0),
            ]
        )
        total_params += layer.get("params", 0)

    print(table)
    print(f"\nTotal parameters: {total_params:,}")


def plot_layer_outputs(
    outputs: List[np.ndarray],
    layer_names: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize outputs from multiple layers.

    Args:
        outputs: List of layer output arrays (should be 2D or 3D)
        layer_names: Names of the layers
        figsize: Figure size (auto if None)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    num_layers = len(outputs)
    if figsize is None:
        figsize = (4 * num_layers, 4)

    fig, axes = plt.subplots(1, num_layers, figsize=figsize)
    if num_layers == 1:
        axes = [axes]

    for i, (output, name) in enumerate(zip(outputs, layer_names)):
        if output.ndim == 3:
            # For multi-channel tensors, show first channel
            data = output[0] if output.shape[0] < 10 else output[:, :, 0]
        elif output.ndim == 2:
            data = output
        else:
            data = output.reshape(output.shape[-2:])

        im = axes[i].imshow(data, cmap="viridis")
        axes[i].set_title(name)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot class distribution as bar chart.

    Args:
        labels: Array of class labels
        class_names: Optional class names
        title: Figure title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    unique, counts = np.unique(labels, return_counts=True)

    if class_names is None:
        class_names = [str(i) for i in unique]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(class_names, counts, color="steelblue")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig
