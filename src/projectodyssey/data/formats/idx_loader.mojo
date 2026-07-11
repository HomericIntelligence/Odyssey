"""IDX File Format Loader.

Provides functions to load data from IDX file format (used by MNIST, EMNIST, and similar datasets).

IDX File Format:
    Labels: [magic(4B)][count(4B)][label_data...]
    Images: [magic(4B)][count(4B)][rows(4B)][cols(4B)][pixel_data...]
    Images RGB: [magic(4B)][count(4B)][channels(4B)][rows(4B)][cols(4B)][pixel_data...]

All integers are big-endian (network byte order).

Magic Numbers:
    - 2049: Label files
    - 2051: Grayscale image files (1 channel)
    - 2052: RGB image files (3 channels, custom extension)

References:
    - IDX Format: http://yann.lecun.com/exdb/mnist/
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from std.memory import UnsafePointer


def read_uint32_be[
    origin: Origin
](data: UnsafePointer[UInt8, origin], offset: Int) -> Int:
    """Read 32-bit unsigned integer in big-endian format.

    Args:
            data: Pointer to byte array.
            offset: Byte offset to read from.

    Returns:
            Integer value in host byte order.
    """
    var b0 = Int(data[offset])
    var b1 = Int(data[offset + 1])
    var b2 = Int(data[offset + 2])
    var b3 = Int(data[offset + 3])

    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3


def load_idx_labels(filepath: String) raises -> AnyTensor:
    """Load labels from IDX file format.

    Args:
            filepath: Path to IDX labels file.

    Returns:
            AnyTensor of shape (num_samples,) with uint8 label values.

    Raises:
            Error: If file format is invalid or cannot be read.
    """
    # Read entire file as raw bytes (binary IDX files are not valid UTF-8)
    var content: List[UInt8]
    with open(filepath, "r") as f:
        content = f.read_bytes()

    var file_size = len(content)
    if file_size < 8:
        raise Error("IDX file too small")

    var data_bytes = content.unsafe_ptr()

    var magic = read_uint32_be(data_bytes, 0)
    if magic != 2049:  # Label file magic number
        raise Error("Invalid IDX label file magic number: " + String(magic))

    var num_items = read_uint32_be(data_bytes, 4)

    if file_size < 8 + num_items:
        raise Error("IDX file size mismatch")

    # Create output tensor
    var shape = List[Int]()
    shape.append(num_items)
    var labels = zeros(shape, DType.uint8)

    # Copy label data
    var labels_data = labels._data
    for i in range(num_items):
        labels_data[i] = data_bytes[8 + i]

    return labels^


def load_idx_images(filepath: String) raises -> AnyTensor:
    """Load grayscale images from IDX file format.

    Args:
            filepath: Path to IDX grayscale images file.

    Returns:
            AnyTensor of shape (num_samples, 1, rows, cols) with uint8 pixel values.

    Raises:
            Error: If file format is invalid or cannot be read.

    Note:
            For single-channel (grayscale) images. Shape includes channel dimension for
            consistency with multi-channel loaders.
    """
    # Read entire file as raw bytes (binary IDX files are not valid UTF-8)
    var content: List[UInt8]
    with open(filepath, "r") as f:
        content = f.read_bytes()

    var file_size = len(content)
    if file_size < 16:
        raise Error("IDX file too small")

    var data_bytes = content.unsafe_ptr()

    # Parse header
    var magic = read_uint32_be(data_bytes, 0)
    if magic != 2051:  # Image file magic number
        raise Error("Invalid IDX image file magic number: " + String(magic))

    var num_images = read_uint32_be(data_bytes, 4)
    var num_rows = read_uint32_be(data_bytes, 8)
    var num_cols = read_uint32_be(data_bytes, 12)

    var expected_size = 16 + (num_images * num_rows * num_cols)
    if file_size < expected_size:
        raise Error("IDX file size mismatch")

    # Create output tensor (num_images, 1, rows, cols) for CNN input
    var shape = List[Int]()
    shape.append(num_images)
    shape.append(1)  # Single channel (grayscale)
    shape.append(num_rows)
    shape.append(num_cols)
    var images = zeros(shape, DType.uint8)

    # Copy image data
    var images_data = images._data
    var total_pixels = num_images * num_rows * num_cols
    for i in range(total_pixels):
        images_data[i] = data_bytes[16 + i]

    return images^


def load_idx_images_rgb(filepath: String) raises -> AnyTensor:
    """Load RGB images from IDX file format.

    Args:
            filepath: Path to IDX RGB images file.

    Returns:
            AnyTensor of shape (num_samples, channels, rows, cols) with uint8 pixel values.

    Raises:
            Error: If file format is invalid or cannot be read.

    Note:
            For CIFAR-10 and similar: (N, 3, 32, 32) where 3 channels are RGB.
    """
    # Read entire file as raw bytes (binary IDX files are not valid UTF-8)
    var content: List[UInt8]
    with open(filepath, "r") as f:
        content = f.read_bytes()

    var file_size = len(content)
    if (
        file_size < 20
    ):  # Header is 20 bytes for RGB images (magic + count + channels + rows + cols)
        raise Error("IDX file too small")

    var data_bytes = content.unsafe_ptr()

    # Parse header.
    #
    # Two magic numbers carry the same 4-D `(N, C, H, W)` RGB layout:
    #   2052 — the project's custom RGB-image extension.
    #   2051 — the standard IDX3 image magic (0x00000803). The shared dataset
    #          downloader (`scripts/download_cifar10.py`) writes RGB CIFAR-10
    #          batches with this magic followed by a 5-field `N,C,H,W` header,
    #          which is byte-identical to the 2052 layout below.
    # Accept either so this loader works with downloader output unchanged.
    var magic = read_uint32_be(data_bytes, 0)
    if magic != 2052 and magic != 2051:
        raise Error("Invalid IDX RGB image file magic number: " + String(magic))

    var num_images = read_uint32_be(data_bytes, 4)
    var num_channels = read_uint32_be(data_bytes, 8)
    var num_rows = read_uint32_be(data_bytes, 12)
    var num_cols = read_uint32_be(data_bytes, 16)

    if num_channels != 3:
        raise Error("Expected 3 RGB channels, got: " + String(num_channels))

    var expected_size = 20 + (num_images * num_channels * num_rows * num_cols)
    if file_size < expected_size:
        raise Error("IDX file size mismatch")

    # Create output tensor (num_images, channels, rows, cols) for CNN input
    var shape = List[Int]()
    shape.append(num_images)
    shape.append(num_channels)  # RGB channels
    shape.append(num_rows)
    shape.append(num_cols)
    var images = zeros(shape, DType.uint8)

    # Copy image data
    var images_data = images._data
    var total_pixels = num_images * num_channels * num_rows * num_cols
    for i in range(total_pixels):
        images_data[i] = data_bytes[20 + i]

    return images^


def normalize_images(mut images: AnyTensor) raises -> AnyTensor:
    """Normalize uint8 images to float32 in range [0, 1].

    Args:
            images: Input images as uint8 AnyTensor.

    Returns:
            Normalized images as float32 AnyTensor.

    Note:
            Converts pixel values from [0, 255] to [0.0, 1.0].

    Raises:
            Error: If operation fails.
    """
    var shape = images.shape()
    var normalized = zeros(shape, DType.float32)

    var num_elements = images.numel()
    var src_data = images._data
    var dst_data = normalized._data.bitcast[Float32]()

    for i in range(num_elements):
        dst_data[i] = Float32(src_data[i]) / 255.0

    return normalized^


def one_hot_encode(labels: AnyTensor, num_classes: Int) raises -> AnyTensor:
    """Convert integer labels to one-hot encoded float32 tensor.

    Args:
            labels: Integer labels as uint8 AnyTensor, shape (num_samples,).
            num_classes: Total number of classes.

    Returns:
            One-hot encoded labels as float32 AnyTensor, shape (num_samples, num_classes).

        Example:
            ```mojo
            abels = [0, 2, 1]  # 3 samples, 3 classes
            one_hot = one_hot_encode(labels, 3)
            # Result shape: (3, 3)
            # [[1.0, 0.0, 0.0],
            #  [0.0, 0.0, 1.0],
            #  [0.0, 1.0, 0.0]]
            ```

    Raises:
            Error: If operation fails.
    """
    var num_samples = labels.shape()[0]

    # Create output tensor (num_samples, num_classes)
    var shape = List[Int]()
    shape.append(num_samples)
    shape.append(num_classes)
    var one_hot = zeros(shape, DType.float32)

    # Fill one-hot encoding
    var labels_data = labels._data
    var one_hot_data = one_hot._data.bitcast[Float32]()

    for i in range(num_samples):
        var label_idx = Int(labels_data[i])
        if label_idx < 0 or label_idx >= num_classes:
            raise Error("Label index out of range: " + String(label_idx))

        # Set the corresponding class to 1.0
        var offset = i * num_classes + label_idx
        one_hot_data[offset] = 1.0

    return one_hot^


def normalize_images_rgb(mut images: AnyTensor) raises -> AnyTensor:
    """Normalize uint8 RGB images to float32 with ImageNet normalization.

    Args:
            images: Input images as uint8 AnyTensor of shape (N, 3, H, W).

    Returns:
            Normalized images as float32 AnyTensor.

    Note:
            Applies ImageNet normalization:
            - mean=[0.485, 0.456, 0.406] for RGB channels
            - std=[0.229, 0.224, 0.225] for RGB channels
            - Converts pixel values from [0, 255] to normalized float.

    Raises:
            Error: If operation fails.
    """
    # ImageNet normalization parameters (R, G, B)
    # ImageNet normalization constants per channel
    var mean_r = Float32(0.485)
    var mean_g = Float32(0.456)
    var mean_b = Float32(0.406)
    var std_r = Float32(0.229)
    var std_g = Float32(0.224)
    var std_b = Float32(0.225)

    var shape = images.shape()
    var normalized = zeros(shape, DType.float32)

    var batch_size = shape[0]
    var channels = shape[1]
    var height = shape[2]
    var width = shape[3]

    var src_data = images._data
    var dst_data = normalized._data.bitcast[Float32]()

    for n in range(batch_size):
        for c in range(channels):
            var mean_val: Float32
            var std_val: Float32

            if c == 0:  # Red channel
                mean_val = mean_r
                std_val = std_r
            elif c == 1:  # Green channel
                mean_val = mean_g
                std_val = std_g
            else:  # Blue channel
                mean_val = mean_b
                std_val = std_b

            for h in range(height):
                for w in range(width):
                    var src_idx = ((n * channels + c) * height + h) * width + w
                    var pixel_val = Float32(src_data[src_idx]) / 255.0
                    var normalized_val = (pixel_val - mean_val) / std_val
                    dst_data[src_idx] = normalized_val

    return normalized^


def load_cifar10_batch(
    batch_dir: String, batch_name: String
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Load a single CIFAR-10 batch (images and labels) from IDX format.

    Args:
            batch_dir: Directory containing CIFAR-10 IDX files.
            batch_name: Batch name without extension (e.g., "train_batch_1", "test_batch").

    Returns:
            Tuple of (images, labels):
            - images: AnyTensor of shape (N, 3, 32, 32) normalized float32.
            - labels: AnyTensor of shape (N,) uint8.

    Raises:
            Error: If batch files cannot be read.

    Note:
            Images are loaded from IDX RGB format and normalized using ImageNet parameters.
            Labels are kept as uint8 (class indices 0-9).
    """
    var images_path = batch_dir + "/" + batch_name + "_images.idx"
    var labels_path = batch_dir + "/" + batch_name + "_labels.idx"

    # Load raw images and labels
    var images_raw = load_idx_images_rgb(images_path)
    var labels = load_idx_labels(labels_path)

    # Normalize images
    var images_normalized = normalize_images_rgb(images_raw)

    return (images_normalized^, labels^)
