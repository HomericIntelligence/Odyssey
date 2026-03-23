"""Normalization operations for data preprocessing.

This module provides pure functional operations for normalizing input data
Unlike normalization.mojo which contains statistical normalization layers
(batch norm, layer norm, etc.), this module contains utility functions for
preprocessing and data transformations.

Functions:
    normalize_rgb: Normalize RGB images with per-channel mean and std deviation

Example:
    from .normalize_ops import normalize_rgb

    # Normalize RGB images with ImageNet statistics
    var mean = (Float32(0.485), Float32(0.456), Float32(0.406))
    var std = (Float32(0.229), Float32(0.224), Float32(0.225))
    var normalized = normalize_rgb(images, mean, std)
    ```
"""

from shared.tensor.any_tensor import AnyTensor, zeros


# ============================================================================
# RGB Image Normalization
# ============================================================================


fn normalize_rgb(
    images: AnyTensor,
    mean: Tuple[Float32, Float32, Float32],
    std: Tuple[Float32, Float32, Float32],
) raises -> AnyTensor:
    """Normalize RGB images with per-channel mean and standard deviation.

    Applies normalization to RGB images using the formula:
        normalized = (pixel / 255.0 - mean) / std

    This is commonly used with ImageNet statistics for transfer learning.

    Args:
        images: Input uint8 tensor of shape (N, 3, H, W).
        mean: Mean values for R, G, B channels (e.g., (0.485, 0.456, 0.406)).
        std: Std deviation values for R, G, B channels (e.g., (0.229, 0.224, 0.225)).

    Returns:
        Normalized float32 tensor of shape (N, 3, H, W).

    Raises:
        Error: If input is not 4D or doesn't have 3 channels.

    Example:
        ```mojo
        from .normalize_ops import normalize_rgb

        # ImageNet normalization
        var mean = (Float32(0.485), Float32(0.456), Float32(0.406))
        var std = (Float32(0.229), Float32(0.224), Float32(0.225))
        var normalized = normalize_rgb(images, mean, std)
        ```

    Note:
        - Input must be uint8 [0, 255].
        - Output is float32 with normalized values.
        - Common for CIFAR-10, ImageNet, and other RGB datasets.
    """
    var shape = images.shape()
    if len(shape) != 4:
        raise Error("normalize_rgb requires 4D input (N, 3, H, W)")

    var num_images = shape[0]
    var num_channels = shape[1]
    var num_rows = shape[2]
    var num_cols = shape[3]

    if num_channels != 3:
        raise Error(
            "normalize_rgb requires 3 RGB channels, got: "
            + String(num_channels)
        )

    # Create output tensor (float32)
    var normalized = zeros(shape, DType.float32)

    var src_data = images._data
    var dst_data = normalized._data.bitcast[Float32]()

    # Extract mean and std values
    var mean_r = mean[0]
    var mean_g = mean[1]
    var mean_b = mean[2]
    var std_r = std[0]
    var std_g = std[1]
    var std_b = std[2]

    # Normalize each image
    for n in range(num_images):
        for h in range(num_rows):
            for w in range(num_cols):
                # R channel (c=0)
                var idx_r = (
                    n * (num_channels * num_rows * num_cols)
                    + 0 * (num_rows * num_cols)
                    + h * num_cols
                    + w
                )
                var pixel_r = Float32(src_data[idx_r]) / 255.0
                dst_data[idx_r] = (pixel_r - mean_r) / std_r

                # G channel (c=1)
                var idx_g = (
                    n * (num_channels * num_rows * num_cols)
                    + 1 * (num_rows * num_cols)
                    + h * num_cols
                    + w
                )
                var pixel_g = Float32(src_data[idx_g]) / 255.0
                dst_data[idx_g] = (pixel_g - mean_g) / std_g

                # B channel (c=2)
                var idx_b = (
                    n * (num_channels * num_rows * num_cols)
                    + 2 * (num_rows * num_cols)
                    + h * num_cols
                    + w
                )
                var pixel_b = Float32(src_data[idx_b]) / 255.0
                dst_data[idx_b] = (pixel_b - mean_b) / std_b

    return normalized
