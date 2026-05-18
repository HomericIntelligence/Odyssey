"""Data transformation and augmentation utilities.

This module provides transformations for preprocessing and augmenting data.

IMPORTANT LIMITATIONS:
- Image transforms assume square images (H = W)
- Default assumption: 3 channels (RGB)
- AnyTensor layout: Flattened (H, W, C) with channels-last
- For non-square or grayscale images, dimensions must be manually validated

These limitations are due to Mojo's current AnyTensor API not exposing shape metadata.
Future versions may support arbitrary image dimensions.
"""

from projectodyssey.tensor.any_tensor import AnyTensor, zeros
from std.math import sqrt, floor, ceil, sin, cos
from std.random import random_si64
from projectodyssey.data.random_transform_base import RandomTransformBase, random_float


# ============================================================================
# Helper Functions
# ============================================================================


def infer_image_dimensions(
    data: AnyTensor, channels: Int = 3
) raises -> Tuple[Int, Int, Int]:
    """Infer image dimensions from flattened tensor.

    Assumes square images: H = W = sqrt(num_elements / channels).
    Auto-detects channels if default doesn't work (tries 3, then 1).

    Args:
        data: Flattened image tensor.
        channels: Number of channels (default: 3 for RGB, auto-detects if mismatch).

    Returns:
        Tuple of (height, width, channels).

    Raises:
        Error: If dimensions don't work out to square image with any supported channel count.
    """
    var total_elements = data.num_elements()

    # Try the provided channels first
    var pixels = total_elements // channels
    var size = Int(sqrt(Float64(pixels)))

    if size * size * channels == total_elements:
        return (size, size, channels)

    # If that didn't work, try grayscale (1 channel)
    if channels != 1:
        pixels = total_elements
        size = Int(sqrt(Float64(pixels)))

        if size * size == total_elements:
            return (size, size, 1)

    # Neither worked - raise error
    raise Error("AnyTensor size doesn't match square image assumption")


# ============================================================================
# Transform Trait
# ============================================================================


trait Transform(Copyable, ImplicitlyDestructible, Movable):
    """Base interface for all transforms.

    Transforms modify data in-place or return transformed copies.
    """

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply the transform to data.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor.

        Raises:
            Error: If transform cannot be applied.
        """
        ...


# ============================================================================
# Compose Transform
# ============================================================================


struct Compose[T: Transform & Copyable & Movable](Copyable, Movable, Transform):
    """Compose multiple transforms sequentially.

    Applies transforms in order, passing output of each to the next.

    Parameters:
        T: Type of transforms in the composition (must implement Transform).
    """

    var transforms: List[Self.T]
    """List of transforms to apply in order."""

    def __init__(out self):
        """Create empty composition of transforms."""
        self.transforms = List[Self.T]()

    def __init__(out self, var transforms: List[Self.T]):
        """Create composition of transforms.

        Args:
            transforms: List of transforms to apply in order.
        """
        self.transforms = transforms^

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply all transforms sequentially.

        Args:
            data: Input tensor.

        Returns:
            Transformed tensor after all transforms.

        Raises:
            Error: If any transform cannot be applied.
        """
        var result = data
        for i in range(len(self.transforms)):
            result = self.transforms[i](result)
        return result

    def __len__(self) -> Int:
        """Return number of transforms."""
        return len(self.transforms)

    def append(mut self, var transform: Self.T):
        """Add a transform to the pipeline.

        Args:
            transform: Transform to add.
        """
        self.transforms.append(transform^)


# Type comptime for Pipeline as Compose
comptime Pipeline[T: Transform & Copyable & Movable] = Compose[T]


# ============================================================================
# AnyTensor Transforms
# ============================================================================


struct ToAnyTensor(Copyable, Movable, Transform):
    """Convert data to tensor format.

    Ensures data is in tensor format with appropriate dtype.
    """

    def __init__(out self):
        """Create ToAnyTensor converter."""
        pass

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Convert to tensor.

        Args:
            data: Input data.

        Returns:
            Data as tensor.

        Raises:
            Error: If conversion fails.
        """
        # Already a tensor, just return
        return data


struct Normalize(Copyable, Movable, Transform):
    """Normalize tensor with mean and standard deviation.

    Applies: (x - mean) / std.
    """

    var mean: Float64
    """Mean value to subtract from all elements."""
    var std: Float64
    """Standard deviation to divide all elements by."""

    def __init__(out self, mean: Float64 = 0.0, std: Float64 = 1.0):
        """Create normalize transform.

        Args:
            mean: Mean to subtract.
            std: Standard deviation to divide by.
        """
        self.mean = mean
        self.std = std

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Normalize tensor by subtracting mean and dividing by std.

        Applies the formula: (data - mean) / std to all elements.

        Args:
            data: Input tensor.

        Returns:
            Normalized tensor.

        Raises:
            Error: If std is zero.
        """
        if self.std == 0.0:
            raise Error("Cannot normalize with std=0")

        # Create a list to hold normalized values
        var normalized = List[Float32](capacity=data.num_elements())

        # Normalize each element: (x - mean) / std
        for i in range(data.num_elements()):
            var value = Float64(data[i])
            var norm_value = (value - self.mean) / self.std
            normalized.append(Float32(norm_value))

        # Create tensor from normalized values
        return AnyTensor(normalized^)


struct Reshape(Copyable, Movable, Transform):
    """Reshape tensor to target shape.

    Changes tensor dimensions while preserving total elements.
    """

    var target_shape: List[Int]
    """Target shape dimensions for the tensor."""

    def __init__(out self, var target_shape: List[Int]):
        """Create reshape transform.

        Args:
            target_shape: Target shape for tensor.
        """
        self.target_shape = target_shape^

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Reshape tensor to target shape.

        Validates that the total number of elements remains the same.

        Args:
            data: Input tensor.

        Returns:
            Reshaped tensor.

        Raises:
            Error: If target shape has different number of elements.
        """
        # Calculate total elements in target shape
        var target_elements = 1
        for dim in self.target_shape:
            target_elements *= dim

        # Validate element count matches
        if target_elements != data.num_elements():
            raise Error(
                "Cannot reshape tensor with "
                + String(data.num_elements())
                + " elements to shape with "
                + String(target_elements)
                + " elements"
            )

        # Build reshaped data as a List
        var reshaped_data = List[Float32](capacity=data.num_elements())

        # Copy all values from source tensor to the list
        for i in range(data.num_elements()):
            reshaped_data.append(Float32(data[i]))

        # Create output tensor from the list
        var reshaped = AnyTensor(reshaped_data^)

        return reshaped^


# ============================================================================
# Image Transforms
# ============================================================================


struct Resize(Copyable, Movable, Transform):
    """Resize image to target size.

    Resizes spatial dimensions of image tensors.
    """

    var size: Tuple[Int, Int]
    """Target (height, width) dimensions for resizing."""
    var interpolation: String
    """Interpolation method to use."""

    def __init__(
        out self, size: Tuple[Int, Int], interpolation: String = "bilinear"
    ):
        """Create resize transform.

        Args:
            size: Target (height, width).
            interpolation: Interpolation method.
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Resize image tensor using bilinear interpolation.

        Resizes spatial dimensions (H, W) while preserving channels (C).
        Uses bilinear interpolation for smooth resizing.
        Assumes tensor layout: flattened (H, W, C) with channels-last.

        Args:
            data: Input image tensor.

        Returns:
            Resized image tensor.

        Raises:
            Error: If operation fails.
        """
        # Get image dimensions (assumes square images with H=W, C=3 by default)
        var dims = infer_image_dimensions(data, 3)
        var old_h = dims[0]
        var old_w = dims[1]
        var channels = dims[2]

        var new_h = self.size[0]
        var new_w = self.size[1]

        # Calculate scale factors
        var scale_h = (
            Float64(old_h - 1) / Float64(new_h - 1) if new_h > 1 else 0.0
        )
        var scale_w = (
            Float64(old_w - 1) / Float64(new_w - 1) if new_w > 1 else 0.0
        )

        # Build resized data as a List
        var resized_data = List[Float32](capacity=new_h * new_w * channels)

        # For each output pixel
        for y_new in range(new_h):
            for x_new in range(new_w):
                # Map to source image coordinates (floating point)
                var y_src = Float64(y_new) * scale_h
                var x_src = Float64(x_new) * scale_w

                # Get integer and fractional parts
                var y_int = Int(y_src)
                var x_int = Int(x_src)
                var y_frac = y_src - Float64(y_int)
                var x_frac = x_src - Float64(x_int)

                # Bounds check
                if y_int < 0 or y_int >= old_h or x_int < 0 or x_int >= old_w:
                    # Fill with zeros if out of bounds
                    for _ in range(channels):
                        resized_data.append(Float32(0.0))
                    continue

                # Bilinear interpolation with 4 neighboring pixels
                # Get the 4 neighbors: (y_int, x_int), (y_int+1, x_int), etc.
                var y_int_next = min(y_int + 1, old_h - 1)
                var x_int_next = min(x_int + 1, old_w - 1)

                # For each channel, interpolate the value
                for c in range(channels):
                    # Get the 4 source values
                    var idx_00 = (y_int * old_w + x_int) * channels + c
                    var idx_01 = (y_int * old_w + x_int_next) * channels + c
                    var idx_10 = (y_int_next * old_w + x_int) * channels + c
                    var idx_11 = (
                        y_int_next * old_w + x_int_next
                    ) * channels + c

                    var v_00 = Float64(data[idx_00])
                    var v_01 = Float64(data[idx_01])
                    var v_10 = Float64(data[idx_10])
                    var v_11 = Float64(data[idx_11])

                    # Bilinear interpolation:
                    # v = (1-y_frac) * [(1-x_frac)*v_00 + x_frac*v_01] +
                    #      y_frac * [(1-x_frac)*v_10 + x_frac*v_11]
                    var v0 = (1.0 - x_frac) * v_00 + x_frac * v_01
                    var v1 = (1.0 - x_frac) * v_10 + x_frac * v_11
                    var v = (1.0 - y_frac) * v0 + y_frac * v1

                    # Append interpolated value to output list
                    resized_data.append(Float32(v))

        # Create output tensor from the list
        var resized = AnyTensor(resized_data^)

        return resized^


struct CenterCrop(Copyable, Movable, Transform):
    """Crop the center of an image.

    Extracts a center crop of specified size.
    """

    var size: Tuple[Int, Int]
    """Target (height, width) dimensions of the crop region."""

    def __init__(out self, size: Tuple[Int, Int]):
        """Create center crop transform.

        Args:
            size: Target (height, width) of crop.
        """
        self.size = size

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Center crop image to target size.

        Extracts a center rectangle from a 2D image tensor.
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

        Args:
            data: Input image tensor.

        Returns:
            Center-cropped image tensor.

        Raises:
            Error: If crop size exceeds image size.
        """
        # Determine image dimensions
        var dims = infer_image_dimensions(data, 3)
        var height = dims[0]
        var width = dims[1]
        var channels = dims[2]

        var crop_h = self.size[0]
        var crop_w = self.size[1]

        # Validate crop size doesn't exceed image size
        if crop_h > height or crop_w > width:
            raise Error("Crop size exceeds image size")

        # Calculate center position
        var offset_h = (height - crop_h) // 2
        var offset_w = (width - crop_w) // 2

        # Create cropped tensor
        var cropped = List[Float32](capacity=crop_h * crop_w * channels)

        # Extract center rectangle
        for h in range(crop_h):
            for w in range(crop_w):
                for c in range(channels):
                    var src_idx = (
                        (offset_h + h) * width + (offset_w + w)
                    ) * channels + c
                    cropped.append(Float32(data[src_idx]))

        return AnyTensor(cropped^)


struct RandomCrop(Copyable, Movable, Transform):
    """Random crop from an image.

    Extracts a random crop of specified size.
    """

    var size: Tuple[Int, Int]
    """Target (height, width) dimensions of the crop region."""
    var padding: Optional[Int]
    """Optional padding to apply before cropping."""

    def __init__(
        out self, size: Tuple[Int, Int], padding: Optional[Int] = None
    ):
        """Create random crop transform.

        Args:
            size: Target (height, width) of crop.
            padding: Optional padding before cropping.
        """
        self.size = size
        self.padding = padding

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Random crop image to target size.

        Extracts a random rectangle from a 2D image tensor.
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.
        Supports optional padding for edge handling.

        Args:
            data: Input image tensor.

        Returns:
            Randomly cropped image tensor.

        Raises:
            Error: If crop size exceeds padded image size.
        """
        # Determine image dimensions
        var dims = infer_image_dimensions(data, 3)
        var height = dims[0]
        var width = dims[1]
        var channels = dims[2]

        # Apply padding if specified (conceptually increase image size)
        var padded_height = height
        var padded_width = width
        if self.padding:
            var pad = self.padding.value()
            padded_height = height + 2 * pad
            padded_width = width + 2 * pad

        var crop_h = self.size[0]
        var crop_w = self.size[1]

        # Validate crop size
        if crop_h > padded_height or crop_w > padded_width:
            raise Error("Crop size exceeds padded image size")

        # Random top-left position within valid range
        var max_h = padded_height - crop_h
        var max_w = padded_width - crop_w
        var top = Int(random_si64(Int64(0), Int64(max_h + 1)))
        var left = Int(random_si64(Int64(0), Int64(max_w + 1)))

        # Adjust for padding offset
        var actual_top = top
        var actual_left = left
        if self.padding:
            var pad = self.padding.value()
            actual_top = top - pad
            actual_left = left - pad

        # Create cropped tensor
        var cropped = List[Float32](capacity=crop_h * crop_w * channels)

        # Extract random rectangle
        for h in range(crop_h):
            for w in range(crop_w):
                # Calculate source pixel position
                var src_h = actual_top + h
                var src_w = actual_left + w

                # For each channel
                for c in range(channels):
                    # Check if source pixel is within original image bounds
                    if (
                        src_h >= 0
                        and src_h < height
                        and src_w >= 0
                        and src_w < width
                    ):
                        # Sample from source pixel
                        var src_idx = (src_h * width + src_w) * channels + c
                        cropped.append(Float32(data[src_idx]))
                    else:
                        # Out of bounds (in padding region), fill with 0
                        cropped.append(Float32(0.0))

        return AnyTensor(cropped^)


struct RandomHorizontalFlip(Copyable, Movable, Transform):
    """Randomly flip image horizontally.

    Flips with specified probability using RandomTransformBase for probability handling.
    """

    var base: RandomTransformBase
    """Random transform base for probability handling."""

    def __init__(out self, p: Float64 = 0.5):
        """Create random horizontal flip transform.

        Args:
            p: Probability of flipping (0.0 to 1.0).
        """
        self.base = RandomTransformBase(p)

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Randomly flip image horizontally with probability p.

        Flips the image along the width dimension (reverses each row).
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

        Args:
            data: Input image tensor.

        Returns:
            Possibly flipped image tensor.

        Raises:
            Error: If operation fails.
        """
        # Check probability - don't flip if should_apply returns False
        if not self.base.should_apply():
            return data

        # Determine image dimensions
        var dims = infer_image_dimensions(data, 3)
        var height = dims[0]
        var width = dims[1]
        var channels = dims[2]
        var total_elements = data.num_elements()

        # Create flipped tensor
        var flipped = List[Float32](capacity=total_elements)

        # For each row
        for h in range(height):
            # For each column (in reverse for flip)
            for w_idx in range(width):
                # Original column index (from right to left)
                var w_orig = width - 1 - w_idx
                # Copy all channels for this pixel
                for c in range(channels):
                    var src_idx = (h * width + w_orig) * channels + c
                    flipped.append(Float32(data[src_idx]))

        return AnyTensor(flipped^)


struct RandomVerticalFlip(Copyable, Movable, Transform):
    """Randomly flip image vertically.

    Flips with specified probability using RandomTransformBase for probability handling.
    """

    var base: RandomTransformBase
    """Random transform base for probability handling."""

    def __init__(out self, p: Float64 = 0.5):
        """Create random vertical flip transform.

        Args:
            p: Probability of flipping (0.0 to 1.0).
        """
        self.base = RandomTransformBase(p)

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Randomly flip image vertically with probability p.

        Flips the image along the height dimension (reverses rows).
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

        Args:
            data: Input image tensor.

        Returns:
            Possibly flipped image tensor.

        Raises:
            Error: If operation fails.
        """
        # Check probability - don't flip if should_apply returns False
        if not self.base.should_apply():
            return data

        # Determine image dimensions
        var dims = infer_image_dimensions(data, 3)
        var height = dims[0]
        var width = dims[1]
        var channels = dims[2]
        var total_elements = data.num_elements()

        # Create flipped tensor
        var flipped = List[Float32](capacity=total_elements)

        # For each row (in reverse for vertical flip)
        for h_idx in range(height):
            # Original row index (from bottom to top)
            var h_orig = height - 1 - h_idx
            # For each column
            for w in range(width):
                # Copy all channels for this pixel
                for c in range(channels):
                    var src_idx = (h_orig * width + w) * channels + c
                    flipped.append(Float32(data[src_idx]))

        return AnyTensor(flipped^)


struct RandomRotation(Copyable, Movable, Transform):
    """Randomly rotate image.

    Rotates within specified degree range.
    """

    var degrees: Tuple[Float64, Float64]
    """Range of rotation degrees (min, max)."""
    var fill_value: Float64
    """Value to fill empty pixels after rotation."""

    def __init__(
        out self, degrees: Tuple[Float64, Float64], fill_value: Float64 = 0.0
    ):
        """Create random rotation transform.

        Args:
            degrees: Range of rotation degrees (min, max).
            fill_value: Value to fill empty pixels after rotation.
        """
        self.degrees = degrees
        self.fill_value = fill_value

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Randomly rotate image within specified degree range.

        Performs rotation around image center using nearest-neighbor sampling.
        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

        Args:
            data: Input image tensor.

        Returns:
            Rotated image tensor.

        Raises:
            Error: If operation fails.
        """
        # Generate random rotation angle in degrees range
        var angle_range = self.degrees[1] - self.degrees[0]
        var rand_val = random_float()
        var angle_deg = self.degrees[0] + (rand_val * angle_range)

        # Convert angle to radians
        var pi = 3.14159265359
        var angle_rad = angle_deg * (pi / 180.0)

        # Determine image dimensions
        var dims = infer_image_dimensions(data, 3)
        var height = dims[0]
        var width = dims[1]
        var channels = dims[2]
        var total_elements = data.num_elements()

        # Compute rotation matrix values
        var cos_angle = cos(angle_rad)
        var sin_angle = sin(angle_rad)

        # Image center
        var cx = Float64(width) / 2.0
        var cy = Float64(height) / 2.0

        # Create rotated tensor with fill_value for empty regions
        var rotated = List[Float32](capacity=total_elements)

        # For each output pixel
        for y in range(height):
            for x in range(width):
                # Convert to floating point for rotation calculation
                var x_f = Float64(x)
                var y_f = Float64(y)

                # Apply inverse rotation to find source pixel
                # x_src = (x - cx) * cos(θ) - (y - cy) * sin(θ) + cx
                # y_src = (x - cx) * sin(θ) + (y - cy) * cos(θ) + cy
                var x_src = (x_f - cx) * cos_angle - (y_f - cy) * sin_angle + cx
                var y_src = (x_f - cx) * sin_angle + (y_f - cy) * cos_angle + cy

                # Round to nearest integer for nearest-neighbor sampling
                var x_src_int = Int(x_src + 0.5)
                var y_src_int = Int(y_src + 0.5)

                # For each channel
                for c in range(channels):
                    # Check if source pixel is within bounds
                    if (
                        x_src_int >= 0
                        and x_src_int < width
                        and y_src_int >= 0
                        and y_src_int < height
                    ):
                        # Sample from source pixel
                        var src_idx = (
                            y_src_int * width + x_src_int
                        ) * channels + c
                        rotated.append(Float32(data[src_idx]))
                    else:
                        # Fill with fill_value for out-of-bounds pixels
                        rotated.append(Float32(self.fill_value))

        return AnyTensor(rotated^)


struct RandomErasing(Copyable, Movable, Transform):
    """Randomly erase rectangular regions in images (Cutout augmentation).

    Randomly selects a rectangle region and erases it by setting pixels to a fill value.
    Helps improve model robustness to occlusion.

    Uses RandomTransformBase for probability handling.

    Reference: "Random Erasing Data Augmentation" (Zhong et al., 2017).
    """

    var base: RandomTransformBase
    """Random transform base for probability handling."""
    var scale: Tuple[Float64, Float64]
    """Range of proportion of area to erase (min, max)."""
    var ratio: Tuple[Float64, Float64]
    """Range of aspect ratio of erased region (min, max)."""
    var value: Float64
    """Pixel value to fill erased region with."""

    def __init__(
        out self,
        p: Float64 = 0.5,
        scale: Tuple[Float64, Float64] = (0.02, 0.33),
        ratio: Tuple[Float64, Float64] = (0.3, 3.3),
        value: Float64 = 0.0,
    ):
        """Create random erasing transform.

        Args:
            p: Probability of applying erasing (0.0 to 1.0).
            scale: Range of proportion of erased area (min, max).
            ratio: Range of aspect ratio of erased area (min, max).
            value: Pixel value to fill erased region with.
        """
        self.base = RandomTransformBase(p)
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, data: AnyTensor) raises -> AnyTensor:
        """Apply random erasing with probability p.

        Randomly erases a rectangular region from the image by:
        1. Checking probability to decide if erasing should occur.
        2. Calculating target erased area based on scale parameter.
        3. Determining rectangle dimensions based on aspect ratio.
        4. Randomly positioning the rectangle within image bounds.
        5. Setting all pixels in rectangle to fill value.

        Assumes flattened tensor with shape (H, W, C) where H = W.
        Default assumption: C = 3 (RGB); adjust for grayscale.

        Args:
            data: Input image tensor.

        Returns:
            Image with randomly erased rectangular region (or original if not applied).

        Raises:
            Error: If operation fails.
        """
        # Step 1: Check probability - don't erase if should_apply returns False
        if not self.base.should_apply():
            return data  # Don't erase

        # Step 2: Infer image dimensions
        var dims = infer_image_dimensions(data, 3)
        var height = dims[0]
        var width = dims[1]
        var channels = dims[2]
        var total_elements = data.num_elements()
        var area = height * width

        # Step 3: Randomly select erased region size
        # Target area as fraction of image
        var scale_range = self.scale[1] - self.scale[0]
        var scale_rand = random_float()
        var target_area_fraction = self.scale[0] + (scale_rand * scale_range)
        var target_area = Float64(area) * target_area_fraction

        # Aspect ratio (width/height)
        var ratio_range = self.ratio[1] - self.ratio[0]
        var ratio_rand = random_float()
        var aspect_ratio = self.ratio[0] + (ratio_rand * ratio_range)

        # Calculate width and height from area and aspect ratio
        # area = h * w, ratio = w / h
        # => area = h * (h * ratio) = h^2 * ratio
        # => h = sqrt(area / ratio)
        # => w = sqrt(area * ratio)
        var erase_h = Int(sqrt(target_area / aspect_ratio))
        var erase_w = Int(sqrt(target_area * aspect_ratio))

        # Ensure within image bounds
        erase_h = min(erase_h, height)
        erase_w = min(erase_w, width)

        # Skip if region is too small
        if erase_h <= 0 or erase_w <= 0:
            return data

        # Step 4: Randomly select top-left position
        var max_top = height - erase_h
        var max_left = width - erase_w

        # Handle edge case where erased region equals image size
        if max_top < 0 or max_left < 0:
            return data

        var top = Int(random_si64(Int64(0), Int64(max_top + 1)))
        var left = Int(random_si64(Int64(0), Int64(max_left + 1)))

        # Step 5: Erase the rectangle
        # Copy original data and mark erased regions
        var result = List[Float32](capacity=total_elements)

        for i in range(total_elements):
            # Check if this element is in the erased rectangle
            # Calculate (row, col) from flat index
            var pixel_idx = i // channels
            var col = pixel_idx % width
            var row = pixel_idx // width

            # Check if in erased region
            if (
                row >= top
                and row < top + erase_h
                and col >= left
                and col < left + erase_w
            ):
                result.append(Float32(self.value))
            else:
                result.append(Float32(data[i]))

        return AnyTensor(result^)
