"""Tests for spatial dropout (Dropout2D) regularization (Part 2).

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_dropout.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Spatial dropout (channel-wise for CNNs)
- Training vs inference mode
- Mask generation and backward pass

All tests use pure functional API.
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_close_float,
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
)
from tests.shared.conftest import TestFixtures
from shared.tensor.any_tensor import AnyTensor, zeros, ones, zeros_like, ones_like
from shared.core.dropout import (
    dropout,
    dropout2d,
    dropout_backward,
    dropout2d_backward,
)
from shared.testing import check_gradient


# ============================================================================
# Spatial Dropout (Dropout2D) Tests
# ============================================================================


fn test_dropout2d_shapes() raises:
    """Test that dropout2d returns correct output and mask shapes."""
    var shape = List[Int]()
    shape.append(2)  # batch
    shape.append(3)  # channels
    shape.append(4)  # height
    shape.append(4)  # width
    var x = ones(shape, DType.float32)

    # Training mode
    var result10 = dropout2d(x, p=0.2, training=True, seed=42)
    var output = result10[0]

    # Check shapes match input
    assert_equal(output.shape()[0], 2)
    assert_equal(output.shape()[1], 3)
    assert_equal(output.shape()[2], 4)
    assert_equal(output.shape()[3], 4)


fn test_dropout2d_channel_level() raises:
    """Test that dropout2d drops entire channels (all spatial positions)."""
    var shape = List[Int]()
    shape.append(1)  # batch
    shape.append(4)  # channels
    shape.append(3)  # height
    shape.append(3)  # width
    var x = ones(shape, DType.float32)

    var result11 = dropout2d(x, p=0.5, training=True, seed=42)
    var output = result11[0]
    var mask = result11[1]

    # Check that entire channels are either all kept or all dropped
    var channels = 4
    var height = 3
    var width = 3
    var spatial_size = height * width

    for c in range(channels):
        # Get first pixel value in channel
        var first_idx = c * spatial_size
        var first_val = mask._data.bitcast[Float32]()[first_idx]

        # All pixels in this channel should have same mask value
        for h in range(height):
            for w in range(width):
                var idx = c * spatial_size + h * width + w
                var val = mask._data.bitcast[Float32]()[idx]
                assert_almost_equal(val, first_val, tolerance=1e-5)


fn test_dropout2d_inference_mode() raises:
    """Test that dropout2d passes input unchanged in inference mode."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    shape.append(4)
    var x = ones(shape, DType.float32)

    # Inference mode
    var result12 = dropout2d(x, p=0.5, training=False)
    var output = result12[0]
    var mask = result12[1]

    # Output should be unchanged
    var size = x.numel()
    for i in range(size):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            x._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


fn test_dropout2d_backward_shapes() raises:
    """Test that dropout2d_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(4)
    shape.append(8)
    shape.append(8)
    var x = ones(shape, DType.float32)

    # Forward pass
    var result13 = dropout2d(x, p=0.2, training=True, seed=42)
    var mask = result13[1]

    # Backward pass
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout2d_backward(grad_output, mask, p=0.2)

    # Check shape
    assert_equal(grad_input.shape()[0], 2)
    assert_equal(grad_input.shape()[1], 4)
    assert_equal(grad_input.shape()[2], 8)
    assert_equal(grad_input.shape()[3], 8)


fn test_dropout2d_backward_gradient() raises:
    """Test dropout2d_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(2)
    shape.append(4)
    shape.append(4)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values
    for i in range(x.numel()):
        x._data.bitcast[Float32]()[i] = Float32(i) * 0.1 - 0.8

    # Forward pass to create mask ONCE
    # For gradient checking, we need the function to be deterministic,
    # so we use the SAME mask for all forward passes
    var result14 = dropout2d(x, p=0.2, training=True, seed=42)
    var output = result14[0]
    var mask = result14[1]
    var grad_out = ones_like(output)
    var p = 0.2

    # Forward function wrapper - manually apply the SAME mask
    # This makes the function deterministic for gradient checking
    fn forward(x: AnyTensor) raises escaping -> AnyTensor:
        # Apply the same mask that was generated initially
        from shared.core.arithmetic import multiply
        from shared.tensor.any_tensor import full_like

        var masked = multiply(x, mask)
        var scale = 1.0 / (1.0 - p)
        var scale_tensor = full_like(x, scale)
        return multiply(masked, scale_tensor)

    # Backward function wrapper - use stored mask instead of regenerating
    fn backward(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        # Use the mask from forward pass to ensure consistency
        return dropout2d_backward(grad, mask, p=p)

    # Use numerical gradient checking (gold standard)
    # Note: Using relaxed tolerances due to Float32 precision limits
    # Dropout2d uses larger tensors, requiring more relaxed tolerances
    check_gradient(forward, backward, x, grad_out, rtol=1e-2, atol=1e-3)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run spatial dropout tests (Part 2)."""
    print("Running dropout tests (part 2)...")

    # Spatial dropout (dropout2d) tests
    test_dropout2d_shapes()
    print("✓ test_dropout2d_shapes")

    test_dropout2d_channel_level()
    print("✓ test_dropout2d_channel_level")

    test_dropout2d_inference_mode()
    print("✓ test_dropout2d_inference_mode")

    test_dropout2d_backward_shapes()
    print("✓ test_dropout2d_backward_shapes")

    test_dropout2d_backward_gradient()
    print("✓ test_dropout2d_backward_gradient")

    print("\nAll dropout tests (part 2) passed!")
