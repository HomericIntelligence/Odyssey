"""Tests for dropout backward pass via autograd module.

This module tests the dropout_backward and dropout2d_backward functions
exported from the autograd module. These tests ensure that the backward
passes correctly handle mask caching and gradient computation.

Tests cover:
- Correct mask application in backward pass
- Proper scaling factor (inverted dropout: 1/(1-p))
- Edge cases (p=0, p close to 1)
- Gradient flow through non-dropped elements only
- Both standard dropout and spatial dropout (dropout2d)
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from shared.autograd import dropout_backward, dropout2d_backward
from shared.core.dropout import dropout, dropout2d
from shared.core.extensor import ExTensor, zeros, ones, zeros_like, ones_like


# ============================================================================
# Standard Dropout Backward Tests (via Autograd Module)
# ============================================================================


fn test_dropout_backward_exported() raises:
    """Test that dropout_backward is properly exported from autograd module."""
    # This test simply verifies the function is accessible
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var x = ones(shape, DType.float32)

    # Forward pass
    var result = dropout(x, p=0.5, training=True, seed=42)
    _ = result[0]
    var mask = result[1]

    # Backward pass using autograd-exported function
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout_backward(grad_output, mask, p=0.5)

    # Verify output is a valid tensor
    assert_equal(grad_input.shape()[0], 3)
    assert_equal(grad_input.shape()[1], 3)


fn test_dropout_backward_p_zero() raises:
    """Test dropout_backward when p=0 (no dropout).

    When p=0, all elements are kept (mask is all 1s).
    Scaling factor is 1/(1-0) = 1.
    """
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)
    var x = ones(shape, DType.float32)

    # Forward pass with p=0
    var result2 = dropout(x, p=0.0, training=True, seed=42)
    var output = result2[0]
    var mask = result2[1]

    # Backward pass
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout_backward(grad_output, mask, p=0.0)

    # With p=0, all gradients should pass through unchanged
    for i in range(x.numel()):
        assert_almost_equal(
            grad_input._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


fn test_dropout_backward_p_high() raises:
    """Test dropout_backward with high dropout probability.

    When p is high (e.g., 0.9), most elements are dropped.
    Scaling factor is 1/(1-0.9) = 10.
    """
    var shape = List[Int]()
    shape.append(10)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var p = 0.9
    var result3 = dropout(x, p=p, training=True, seed=42)
    var output = result3[0]
    var mask = result3[1]

    # Backward pass
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout_backward(grad_output, mask, p=p)

    # Check scaling is correct
    var scale = Float32(1.0 / (1.0 - p))

    for i in range(x.numel()):
        var mask_val = mask._data.bitcast[Float32]()[i]
        var grad_val = grad_input._data.bitcast[Float32]()[i]

        if mask_val > 0:
            # Gradient should be scaled by 1/(1-p) = 10
            assert_almost_equal(grad_val, scale, tolerance=1e-5)
        else:
            # Dropped elements should have zero gradient
            assert_almost_equal(grad_val, Float32(0.0), tolerance=1e-5)


fn test_dropout_backward_mask_application() raises:
    """Test that dropout_backward correctly applies the mask.

    The mask is binary (1.0 for kept, 0.0 for dropped).
    Backward should zero out gradients where mask is 0.
    """
    var shape = List[Int]()
    shape.append(4)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var p = 0.5
    var result4 = dropout(x, p=p, training=True, seed=42)
    var output = result4[0]
    var mask = result4[1]

    # Create gradient tensor with all 5s
    var grad_output = ones(shape, DType.float32)
    var grad_output_ptr = grad_output._data.bitcast[Float32]()
    for i in range(grad_output.numel()):
        grad_output_ptr[i] = 5.0

    # Backward pass
    var grad_input = dropout_backward(grad_output, mask, p=p)

    var scale = Float32(1.0 / (1.0 - p))

    for i in range(x.numel()):
        var mask_val = mask._data.bitcast[Float32]()[i]
        var grad_val = grad_input._data.bitcast[Float32]()[i]

        if mask_val > 0:
            # Gradient should be 5.0 * scale
            var expected = 5.0 * scale
            assert_almost_equal(grad_val, expected, tolerance=1e-5)
        else:
            # Dropped elements should have zero gradient
            assert_almost_equal(grad_val, Float32(0.0), tolerance=1e-5)


fn test_dropout_backward_consistency() raises:
    """Test that dropout_backward is consistent across multiple calls.

    Using the same mask, multiple backward passes should give identical results.
    """
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var x = ones(shape, DType.float32)

    var p = 0.3
    var result5 = dropout(x, p=p, training=True, seed=42)
    var output = result5[0]
    var mask = result5[1]

    var grad_output = ones(shape, DType.float32)

    # Compute gradient multiple times with same mask
    var grad1 = dropout_backward(grad_output, mask, p=p)
    var grad2 = dropout_backward(grad_output, mask, p=p)

    # Results should be identical
    for i in range(x.numel()):
        assert_almost_equal(
            grad1._data.bitcast[Float32]()[i],
            grad2._data.bitcast[Float32]()[i],
            tolerance=1e-6,
        )


# ============================================================================
# Spatial Dropout (Dropout2D) Backward Tests (via Autograd Module)
# ============================================================================


fn test_dropout2d_backward_exported() raises:
    """Test that dropout2d_backward is properly exported from autograd module.
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    shape.append(4)
    var x = ones(shape, DType.float32)

    # Forward pass
    var result6 = dropout2d(x, p=0.2, training=True, seed=42)
    _ = result6[0]
    var mask = result6[1]

    # Backward pass using autograd-exported function
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout2d_backward(grad_output, mask, p=0.2)

    # Verify output is a valid tensor
    assert_equal(grad_input.shape()[0], 2)
    assert_equal(grad_input.shape()[1], 3)
    assert_equal(grad_input.shape()[2], 4)
    assert_equal(grad_input.shape()[3], 4)


fn test_dropout2d_backward_scaling() raises:
    """Test dropout2d_backward applies correct scaling factor.

    Since dropout2d uses same formula as dropout, scaling should be 1/(1-p).
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(4)
    shape.append(6)
    shape.append(6)
    var x = ones(shape, DType.float32)

    var p = 0.3
    var result7 = dropout2d(x, p=p, training=True, seed=42)
    var output = result7[0]
    var mask = result7[1]

    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout2d_backward(grad_output, mask, p=p)

    var scale = Float32(1.0 / (1.0 - p))

    # Check a few positions to verify scaling
    var num_checks = 0
    var max_checks = 10

    for i in range(x.numel()):
        if num_checks >= max_checks:
            break

        var mask_val = mask._data.bitcast[Float32]()[i]
        var grad_val = grad_input._data.bitcast[Float32]()[i]

        if mask_val > 0:
            assert_almost_equal(grad_val, scale, tolerance=1e-4)
            num_checks += 1
        else:
            assert_almost_equal(grad_val, Float32(0.0), tolerance=1e-5)


fn test_dropout2d_backward_channel_consistency() raises:
    """Test that dropout2d_backward zeros entire channels consistently.

    Where dropout2d masks out a channel, all spatial positions should be zero.
    """
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(4)
    shape.append(4)
    var x = ones(shape, DType.float32)

    var p = 0.5
    var result8 = dropout2d(x, p=p, training=True, seed=42)
    var output = result8[0]
    var mask = result8[1]

    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout2d_backward(grad_output, mask, p=p)

    # Check that channels are consistently handled
    var channels = 3
    var height = 4
    var width = 4
    var spatial_size = height * width

    for c in range(channels):
        # Get mask value for first pixel of channel
        var first_idx = c * spatial_size
        var channel_mask_val = mask._data.bitcast[Float32]()[first_idx]

        # All gradients in this channel should have same pattern
        for h in range(height):
            for w in range(width):
                var idx = c * spatial_size + h * width + w
                var grad_val = grad_input._data.bitcast[Float32]()[idx]

                if channel_mask_val > 0:
                    # Channel is kept - all should have same gradient
                    assert_true(grad_val > 0.0)
                else:
                    # Channel is dropped - all should be zero
                    assert_almost_equal(grad_val, Float32(0.0), tolerance=1e-5)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all dropout backward tests from autograd module."""
    print("Running autograd dropout_backward tests...")

    # Standard dropout backward tests
    test_dropout_backward_exported()
    print("✓ test_dropout_backward_exported")

    test_dropout_backward_p_zero()
    print("✓ test_dropout_backward_p_zero")

    test_dropout_backward_p_high()
    print("✓ test_dropout_backward_p_high")

    test_dropout_backward_mask_application()
    print("✓ test_dropout_backward_mask_application")

    test_dropout_backward_consistency()
    print("✓ test_dropout_backward_consistency")

    # Spatial dropout backward tests
    test_dropout2d_backward_exported()
    print("✓ test_dropout2d_backward_exported")

    test_dropout2d_backward_scaling()
    print("✓ test_dropout2d_backward_scaling")

    test_dropout2d_backward_channel_consistency()
    print("✓ test_dropout2d_backward_channel_consistency")

    print("\nAll autograd dropout_backward tests passed!")
