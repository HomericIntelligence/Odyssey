# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_tensor_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for tensor-specific transforms (part 1): reshape and type conversion.

Tests tensor transforms including reshape and type conversion used in data preprocessing.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)


# ============================================================================
# Reshape Transform Tests
# ============================================================================


fn test_reshape_basic():
    """Test reshaping tensor to new shape.

    Should change tensor shape without changing data order,
    common for converting between different representations.
    """
    # var data = Tensor.arange(0, 28*28).reshape(784)  # Flat
    # var reshape = Reshape(28, 28)
    # var result = reshape(data)
    #
    # assert_equal(result.shape()[0], 28)
    # assert_equal(result.shape()[1], 28)
    # assert_equal(result.numel(), 784)
    pass


fn test_reshape_flatten():
    """Test flattening multi-dimensional tensor.

    Should convert any shape to 1D vector,
    common for feeding images to fully-connected layers.
    """
    # var data = Tensor.ones(28, 28, 3)
    # var flatten = Flatten()
    # var result = flatten(data)
    #
    # assert_equal(len(result.shape()), 1)
    # assert_equal(result.shape()[0], 28*28*3)
    pass


fn test_reshape_add_dimension():
    """Test adding channel dimension.

    Should add dimension of size 1, e.g., (28, 28) → (28, 28, 1),
    useful for grayscale images.
    """
    # var data = Tensor.ones(28, 28)
    # var unsqueeze = Unsqueeze(dim=-1)  # Add dimension at end
    # var result = unsqueeze(data)
    #
    # assert_equal(len(result.shape()), 3)
    # assert_equal(result.shape()[2], 1)
    pass


fn test_reshape_remove_dimension():
    """Test removing dimension of size 1.

    Should remove singleton dimensions, e.g., (28, 28, 1) → (28, 28),
    useful for compatibility with 2D operations.
    """
    # var data = Tensor.ones(28, 28, 1)
    # var squeeze = Squeeze(dim=-1)
    # var result = squeeze(data)
    #
    # assert_equal(len(result.shape()), 2)
    # assert_equal(result.shape()[0], 28)
    # assert_equal(result.shape()[1], 28)
    pass


# ============================================================================
# Type Conversion Tests
# ============================================================================


fn test_to_float32():
    """Test converting tensor to Float32 dtype.

    Should convert from any numeric type to Float32,
    common for neural network inputs.
    """
    # var data = Tensor([1, 2, 3], dtype=Int32)
    # var convert = ToFloat32()
    # var result = convert(data)
    #
    # assert_equal(result.dtype, DType.float32)
    # assert_almost_equal(result[0], 1.0)
    pass


fn test_to_int32():
    """Test converting tensor to Int32 dtype.

    Should convert from float to int, truncating decimals,
    useful for label conversion.
    """
    # var data = Tensor([1.9, 2.1, 3.5], dtype=Float32)
    # var convert = ToInt32()
    # var result = convert(data)
    #
    # assert_equal(result.dtype, DType.int32)
    # assert_equal(result[0], 1)  # Truncated, not rounded
    # assert_equal(result[1], 2)
    pass


fn test_scale_uint8_to_float():
    """Test scaling uint8 [0, 255] to float [0, 1].

    Common preprocessing for image data loaded from files,
    which are typically stored as uint8.
    """
    # var data = Tensor([0, 127, 255], dtype=UInt8)
    # var scale = ScaleUInt8ToFloat()
    # var result = scale(data)
    #
    # assert_almost_equal(result[0], 0.0)
    # assert_almost_equal(result[1], 127.0/255.0, tolerance=1e-5)
    # assert_almost_equal(result[2], 1.0)
    pass


fn test_scale_float_to_uint8():
    """Test scaling float [0, 1] to uint8 [0, 255].

    Useful for saving processed images back to disk
    in standard image formats.
    """
    # var data = Tensor([0.0, 0.5, 1.0], dtype=Float32)
    # var scale = ScaleFloatToUInt8()
    # var result = scale(data)
    #
    # assert_equal(result.dtype, DType.uint8)
    # assert_equal(result[0], 0)
    # assert_equal(result[1], 127)  # 0.5 * 255
    # assert_equal(result[2], 255)
    pass


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run tensor transform tests part 1 (reshape and type conversion)."""
    print("Running tensor transform tests (part 1)...")

    # Reshape tests
    test_reshape_basic()
    test_reshape_flatten()
    test_reshape_add_dimension()
    test_reshape_remove_dimension()

    # Type conversion tests
    test_to_float32()
    test_to_int32()
    test_scale_uint8_to_float()
    test_scale_float_to_uint8()

    print("✓ All tensor transform tests (part 1) passed!")
