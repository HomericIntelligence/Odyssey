"""CIFAR Format Binary Data Loader

Provides functionality to load CIFAR-10 and CIFAR-100 binary format datasets

CIFAR Format Overview:
    - CIFAR-10: 1 label + 3*32*32 pixel bytes per image (3073 bytes total).
    - CIFAR-100: 2 labels (coarse + fine) + 3*32*32 pixel bytes (3074 bytes total).
    - Format: [label(s)][red_pixels][green_pixels][blue_pixels].

CIFAR-10 Structure:
    - Images: 32x32 RGB (3 channels).
    - Labels: 10 classes (0-9).
    - Training: 50,000 images (5 batches of 10,000).
    - Test: 10,000 images (1 batch).

CIFAR-100 Structure:
    - Images: 32x32 RGB (3 channels).
    - Labels: 100 fine classes, 20 coarse superclasses.
    - Training: 50,000 images (1 batch).
    - Test: 10,000 images (1 batch).

References:
    - CIFAR-10/100 Homepage: https://www.cs.toronto.edu/~kriz/cifar.html.
    - Binary format details: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz.
"""

from collections import List
from memory import UnsafePointer
from shared.core import AnyTensor, zeros
from shared.data import (
    CIFAR10_IMAGE_SIZE,
    CIFAR10_CHANNELS,
    CIFAR10_BYTES_PER_IMAGE,
    CIFAR10_NUM_CLASSES,
    CIFAR100_IMAGE_SIZE,
    CIFAR100_CHANNELS,
    CIFAR100_BYTES_PER_IMAGE,
    CIFAR100_NUM_CLASSES_FINE,
    CIFAR100_NUM_CLASSES_COARSE,
)


struct CIFARLoader(Copyable, Movable):
    """Loader for CIFAR-10 and CIFAR-100 binary format files.

    Supports loading batches of images and labels from CIFAR binary format.
    Each batch file contains multiple images packed sequentially.

    Attributes:
        cifar_version: Version of CIFAR format (10 or 100).
        image_size: Size of each image (32x32 for standard CIFAR).
        channels: Number of color channels (3 for RGB).
        bytes_per_image: Total bytes per image including label(s).

    Example:
        ```mojo
        # Load CIFAR-10 batch
        var loader = CIFARLoader(10)
        var images = loader.load_images("data/data_batch_1.bin")
        var labels = loader.load_labels("data/data_batch_1.bin")
        ```
    """

    var cifar_version: Int
    """Version of CIFAR format (10 or 100)."""
    var image_size: Int
    """Size of each square image (32 for standard CIFAR)."""
    var channels: Int
    """Number of color channels (3 for RGB)."""
    var bytes_per_image: Int
    """Total bytes per image including label(s)."""

    fn __init__(out self, cifar_version: Int) raises:
        """Initialize CIFAR loader with specified version.

        Args:
            cifar_version: CIFAR version (10 or 100).

        Raises:
            Error: If cifar_version is not 10 or 100.
        """
        if cifar_version != 10 and cifar_version != 100:
            raise Error(
                "CIFAR version must be 10 or 100, got: " + String(cifar_version)
            )

        self.cifar_version = cifar_version
        self.image_size = CIFAR10_IMAGE_SIZE
        self.channels = CIFAR10_CHANNELS

        if cifar_version == 10:
            self.bytes_per_image = CIFAR10_BYTES_PER_IMAGE
        else:
            self.bytes_per_image = CIFAR100_BYTES_PER_IMAGE

    fn _validate_file_size(self, file_size: Int) raises:
        """Validate that file size is consistent with CIFAR format.

        Args:
            file_size: Size of file in bytes.

        Raises:
            Error: If file size is not a multiple of bytes_per_image.
        """
        if file_size % self.bytes_per_image != 0:
            raise Error(
                "Invalid file size "
                + String(file_size)
                + ": not a multiple of "
                + String(self.bytes_per_image)
            )

    fn _calculate_num_images(self, file_size: Int) -> Int:
        """Calculate number of images in file based on file size.

        Args:
            file_size: Size of file in bytes.

        Returns:
            Number of images in file.
        """
        return file_size // self.bytes_per_image

    fn load_labels(self, filepath: String) raises -> AnyTensor:
        """Load labels from CIFAR binary format file.

        For CIFAR-10, returns shape (num_images,) with single label per image.
        For CIFAR-100, returns shape (num_images, 2) with (coarse, fine) labels.

        Args:
            filepath: Path to CIFAR binary file.

        Returns:
            AnyTensor containing labels.

        Raises:
            Error: If file cannot be read or format is invalid.
        """
        var content: String
        with open(filepath, "r") as f:
            content = f.read()

        var file_size = len(content)
        self._validate_file_size(file_size)

        var num_images = self._calculate_num_images(file_size)
        var data_bytes = content.unsafe_ptr()

        if self.cifar_version == 10:
            # CIFAR-10: 1 label per image
            var shape = List[Int]()
            shape.append(num_images)
            var labels = zeros(shape, DType.uint8)

            var labels_data = labels._data
            for i in range(num_images):
                var offset = i * self.bytes_per_image
                labels_data[i] = data_bytes[offset]

            return labels^
        else:
            # CIFAR-100: 2 labels (coarse, fine) per image
            var shape = List[Int]()
            shape.append(num_images)
            shape.append(2)
            var labels = zeros(shape, DType.uint8)

            var labels_data = labels._data
            for i in range(num_images):
                var offset = i * self.bytes_per_image
                # Coarse label at offset
                labels_data[i * 2] = data_bytes[offset]
                # Fine label at offset + 1
                labels_data[i * 2 + 1] = data_bytes[offset + 1]

            return labels^

    fn load_images(self, filepath: String) raises -> AnyTensor:
        """Load images from CIFAR binary format file.

        Returns shape (num_images, channels, image_size, image_size) with uint8 pixel values.
        Pixel data is stored as: [red_pixels][green_pixels][blue_pixels] for each image.

        Args:
            filepath: Path to CIFAR binary file.

        Returns:
            AnyTensor of shape (num_images, 3, 32, 32) with uint8 pixel values.

        Raises:
            Error: If file cannot be read or format is invalid.
        """
        var content: String
        with open(filepath, "r") as f:
            content = f.read()

        var file_size = len(content)
        self._validate_file_size(file_size)

        var num_images = self._calculate_num_images(file_size)
        var data_bytes = content.unsafe_ptr()

        # Create tensor to hold images
        var shape = List[Int]()
        shape.append(num_images)
        shape.append(self.channels)
        shape.append(self.image_size)
        shape.append(self.image_size)
        var images = zeros(shape, DType.uint8)

        var images_data = images._data
        var pixels_per_image = self.image_size * self.image_size

        # For each image, extract pixel data
        # Offset depends on CIFAR version (1 or 2 label bytes)
        var label_bytes = 1 if self.cifar_version == 10 else 2

        for img_idx in range(num_images):
            var file_offset = img_idx * self.bytes_per_image + label_bytes
            var tensor_offset = img_idx * self.channels * pixels_per_image

            # Copy pixel data: R channel, then G channel, then B channel
            for pixel_idx in range(pixels_per_image * self.channels):
                images_data[tensor_offset + pixel_idx] = data_bytes[
                    file_offset + pixel_idx
                ]

        return images^

    fn load_batch(self, filepath: String) raises -> Tuple[AnyTensor, AnyTensor]:
        """Load a complete batch of images and labels from CIFAR file.

        Convenience function that loads both images and labels in a single call.

        Args:
            filepath: Path to CIFAR binary file.

        Returns:
            Tuple of (images, labels) where:
            - images: AnyTensor of shape (num_images, 3, 32, 32) with uint8 pixels.
            - labels: AnyTensor with shape (num_images,) for CIFAR-10 or (num_images, 2) for CIFAR-100.

        Raises:
            Error: If file cannot be read or format is invalid.
        """
        var images = self.load_images(filepath)
        var labels = self.load_labels(filepath)

        return Tuple[AnyTensor, AnyTensor](images, labels)
