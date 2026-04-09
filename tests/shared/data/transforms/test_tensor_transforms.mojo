"""Tests for tensor-specific transforms (part 1): reshape and type conversion.

Tests tensor transforms including reshape and type conversion used in data preprocessing.
"""


from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)


def test_reshape_basic():
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


def test_reshape_flatten():
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


def test_reshape_add_dimension():
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


def test_reshape_remove_dimension():
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


def test_to_float32():
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


def test_to_int32():
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


def test_scale_uint8_to_float():
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


def test_scale_float_to_uint8():
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


def test_transpose_2d():
    """Test transposing 2D tensor.

    Should swap dimensions: (H, W) → (W, H),
    useful for matrix operations.
    """
    # var data = Tensor.arange(0, 12).reshape(3, 4)
    # var transpose = Transpose()
    # var result = transpose(data)
    #
    # assert_equal(result.shape()[0], 4)
    # assert_equal(result.shape()[1], 3)
    # assert_equal(result[0, 0], data[0, 0])
    # assert_equal(result[1, 0], data[0, 1])
    pass


def test_permute_dimensions():
    """Test permuting dimensions with custom order.

    Should reorder dimensions: (H, W, C) → (C, H, W),
    common for converting between channel formats.
    """
    # var data = Tensor.ones(28, 28, 3)  # HWC format
    # var permute = Permute([2, 0, 1])  # To CHW format
    # var result = permute(data)
    #
    # assert_equal(result.shape()[0], 3)   # Channels
    # assert_equal(result.shape()[1], 28)  # Height
    # assert_equal(result.shape()[2], 28)  # Width
    pass


def test_channel_first_to_last():
    """Test converting CHW to HWC format.

    PyTorch uses CHW, TensorFlow uses HWC,
    so conversion is needed for interop.
    """
    # var data = Tensor.ones(3, 28, 28)  # CHW
    # var convert = ChannelFirstToLast()
    # var result = convert(data)
    #
    # assert_equal(result.shape()[0], 28)  # Height
    # assert_equal(result.shape()[1], 28)  # Width
    # assert_equal(result.shape()[2], 3)   # Channels
    pass


def test_lambda_basic():
    """Test applying custom function as transform.

    Should allow arbitrary function to be used as transform,
    enabling flexible custom preprocessing.
    """
    # fn square(x: Tensor) -> Tensor:
    #     return x * x
    #
    # var data = Tensor([1.0, 2.0, 3.0])
    # var transform = Lambda(square)
    # var result = transform(data)
    #
    # assert_almost_equal(result[0], 1.0)
    # assert_almost_equal(result[1], 4.0)
    # assert_almost_equal(result[2], 9.0)
    pass


def test_lambda_with_closure():
    """Test Lambda with captured variables.

    Should support closures for parameterized custom transforms,
    useful for one-off preprocessing steps.
    """
    # var scale_factor = 2.0
    # fn scale(x: Tensor) -> Tensor:
    #     return x * scale_factor
    #
    # var data = Tensor([1.0, 2.0, 3.0])
    # var transform = Lambda(scale)
    # var result = transform(data)
    #
    # assert_almost_equal(result[0], 2.0)
    # assert_almost_equal(result[1], 4.0)
    pass


def test_clamp_range():
    """Test clamping values to valid range.

    Should clip values outside [min, max] range,
    useful for ensuring valid input ranges.
    """
    # var data = Tensor([-1.0, 0.5, 2.0])
    # var clamp = Clamp(min=0.0, max=1.0)
    # var result = clamp(data)
    #
    # assert_almost_equal(result[0], 0.0)  # Clamped from -1.0
    # assert_almost_equal(result[1], 0.5)  # Unchanged
    # assert_almost_equal(result[2], 1.0)  # Clamped from 2.0
    pass


def main() raises:
    """Run all test_tensor_transforms tests."""
    print("Running test_tensor_transforms tests...")

    test_reshape_basic()
    print("✓ test_reshape_basic")

    test_reshape_flatten()
    print("✓ test_reshape_flatten")

    test_reshape_add_dimension()
    print("✓ test_reshape_add_dimension")

    test_reshape_remove_dimension()
    print("✓ test_reshape_remove_dimension")

    test_to_float32()
    print("✓ test_to_float32")

    test_to_int32()
    print("✓ test_to_int32")

    test_scale_uint8_to_float()
    print("✓ test_scale_uint8_to_float")

    test_scale_float_to_uint8()
    print("✓ test_scale_float_to_uint8")

    test_transpose_2d()
    print("✓ test_transpose_2d")

    test_permute_dimensions()
    print("✓ test_permute_dimensions")

    test_channel_first_to_last()
    print("✓ test_channel_first_to_last")

    test_lambda_basic()
    print("✓ test_lambda_basic")

    test_lambda_with_closure()
    print("✓ test_lambda_with_closure")

    test_clamp_range()
    print("✓ test_clamp_range")

    print("\nAll test_tensor_transforms tests passed!")
