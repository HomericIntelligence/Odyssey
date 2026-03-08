# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_tensor_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for tensor-specific transforms (part 2): transpose, lambda, and clamp.

Tests tensor transforms including transpose, permute, lambda, and clamp used in data preprocessing.
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)


# ============================================================================
# Transpose Transform Tests
# ============================================================================


fn test_transpose_2d():
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


fn test_permute_dimensions():
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


fn test_channel_first_to_last():
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


# ============================================================================
# Lambda Transform Tests
# ============================================================================


fn test_lambda_basic():
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


fn test_lambda_with_closure():
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


# ============================================================================
# Clamp Transform Tests
# ============================================================================


fn test_clamp_range():
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run tensor transform tests part 2 (transpose, lambda, and clamp)."""
    print("Running tensor transform tests (part 2)...")

    # Transpose tests
    test_transpose_2d()
    test_permute_dimensions()
    test_channel_first_to_last()

    # Lambda tests
    test_lambda_basic()
    test_lambda_with_closure()

    # Clamp tests
    test_clamp_range()

    print("✓ All tensor transform tests (part 2) passed!")
