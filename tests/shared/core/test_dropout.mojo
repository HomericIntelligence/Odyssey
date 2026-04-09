"""Tests for standard dropout regularization

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_dropout.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Standard dropout (element-wise)
- Training vs inference mode
- Mask generation and backward pass
- Reproducibility with seed

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


def test_dropout_shapes() raises:
    """Test that dropout returns correct output and mask shapes."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(10)
    var x = ones(shape, DType.float32)

    # Training mode
    var result = dropout(x, p=0.5, training=True, seed=42)
    var output = result[0]
    var mask = result[1]

    # Check shapes
    assert_equal(output.shape()[0], 4)
    assert_equal(output.shape()[1], 10)
    assert_equal(mask.shape()[0], 4)
    assert_equal(mask.shape()[1], 10)


def test_dropout_inference_mode() raises:
    """Test that dropout passes input unchanged in inference mode."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(5)
    var x = ones(shape, DType.float32)

    # Inference mode
    var result2 = dropout(x, p=0.5, training=False)
    var output = result2[0]
    var mask = result2[1]

    # Output should be unchanged
    var size = x.numel()
    for i in range(size):
        assert_almost_equal(
            output._data.bitcast[Float32]()[i],
            x._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )

    # Mask should be all ones
    for i in range(size):
        assert_almost_equal(
            mask._data.bitcast[Float32]()[i], Float32(1.0), tolerance=1e-5
        )


def test_dropout_probability() raises:
    """Test that dropout approximately drops p% of elements."""
    var shape = List[Int]()
    shape.append(100)
    shape.append(100)
    var x = ones(shape, DType.float32)

    var p = 0.5
    var result3 = dropout(x, p=p, training=True, seed=42)
    var output = result3[0]
    var mask = result3[1]

    # Count dropped elements (where mask is 0)
    var total = x.numel()
    var dropped = 0

    for i in range(total):
        if mask._data.bitcast[Float32]()[i] == 0.0:
            dropped += 1

    # Should be approximately 50% dropped (within 10% tolerance for randomness)
    var drop_rate = Float64(dropped) / Float64(total)
    var expected = p
    var tolerance = 0.1  # 10% tolerance

    assert_true(drop_rate > expected - tolerance)
    assert_true(drop_rate < expected + tolerance)


def test_dropout_scaling() raises:
    """Test that kept elements are scaled by 1/(1-p)."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(10)
    var x = ones(shape, DType.float32)

    var p = 0.5
    var result4 = dropout(x, p=p, training=True, seed=42)
    var output = result4[0]
    var mask = result4[1]

    # Elements that weren't dropped should be scaled by 1/(1-p) = 2.0
    var expected_scale = Float32(1.0 / (1.0 - p))

    for i in range(x.numel()):
        var mask_val = mask._data.bitcast[Float32]()[i]
        var out_val = output._data.bitcast[Float32]()[i]

        if mask_val > 0:
            # Not dropped - should be scaled
            assert_almost_equal(out_val, expected_scale, tolerance=1e-5)
        else:
            # Dropped - should be zero
            assert_almost_equal(out_val, Float32(0.0), tolerance=1e-5)


def test_dropout_reproducibility() raises:
    """Test that dropout with same seed produces same mask."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)
    var x = ones(shape, DType.float32)

    # Same seed should produce same mask
    var result5 = dropout(x, p=0.5, training=True, seed=42)
    var output1 = result5[0]
    var mask1 = result5[1]
    var result6 = dropout(x, p=0.5, training=True, seed=42)
    var output2 = result6[0]
    var mask2 = result6[1]

    # Masks should be identical
    for i in range(x.numel()):
        assert_almost_equal(
            mask1._data.bitcast[Float32]()[i],
            mask2._data.bitcast[Float32]()[i],
            tolerance=1e-5,
        )


def test_dropout_backward_shapes() raises:
    """Test that dropout_backward returns correct gradient shape."""
    var shape = List[Int]()
    shape.append(4)
    shape.append(8)
    var x = ones(shape, DType.float32)

    # Forward pass
    var result7 = dropout(x, p=0.5, training=True, seed=42)
    var mask = result7[1]

    # Backward pass
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout_backward(grad_output, mask, p=0.5)

    # Check shape
    assert_equal(grad_input.shape()[0], 4)
    assert_equal(grad_input.shape()[1], 8)


def test_dropout_backward_gradient_flow() raises:
    """Test that dropout_backward only passes gradients through non-dropped elements.
    """
    var shape = List[Int]()
    shape.append(3)
    shape.append(3)
    var x = ones(shape, DType.float32)

    # Forward pass
    var result8 = dropout(x, p=0.5, training=True, seed=42)
    var output = result8[0]
    var mask = result8[1]

    # Backward pass with all-ones gradient
    var grad_output = ones(shape, DType.float32)
    var grad_input = dropout_backward(grad_output, mask, p=0.5)

    # Gradient should only flow where mask is non-zero
    var scale = Float32(1.0 / (1.0 - 0.5))
    for i in range(x.numel()):
        var mask_val = mask._data.bitcast[Float32]()[i]
        var grad_val = grad_input._data.bitcast[Float32]()[i]

        if mask_val > 0:
            # Gradient should be scaled
            assert_almost_equal(grad_val, scale, tolerance=1e-5)
        else:
            # Gradient should be zero
            assert_almost_equal(grad_val, Float32(0.0), tolerance=1e-5)


def test_dropout_backward_gradient() raises:
    """Test dropout_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(5)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values using safe API (avoids ASAP-destruction UAF from bitcast pointer aliasing)
    x.set(0, Float32(-0.5))
    x.set(1, Float32(0.0))
    x.set(2, Float32(0.2))
    x.set(3, Float32(0.5))
    x.set(4, Float32(1.0))

    # Forward pass to create mask ONCE
    # For gradient checking, we need the function to be deterministic,
    # so we use the SAME mask for all forward passes
    var result9 = dropout(x, p=0.3, training=True, seed=42)
    var output = result9[0]
    var mask = result9[1]
    var grad_out = ones_like(output)
    var p = 0.3

    # Forward function wrapper - manually apply the SAME mask
    # This makes the function deterministic for gradient checking
    def forward(x: AnyTensor) raises escaping -> AnyTensor:
        # Apply the same mask that was generated initially
        from shared.core.arithmetic import multiply
        from shared.tensor.any_tensor import full_like

        var masked = multiply(x, mask)
        var scale = 1.0 / (1.0 - p)
        var scale_tensor = full_like(x, scale)
        return multiply(masked, scale_tensor)

    # Backward function wrapper - use the same stored mask
    def backward(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        # Use the mask from forward pass to ensure consistency
        return dropout_backward(grad, mask, p=p)

    # Use numerical gradient checking (gold standard)
    # Note: Using relaxed tolerances due to Float32 precision limits
    check_gradient(forward, backward, x, grad_out, rtol=2e-3, atol=1e-5)


def test_dropout2d_shapes() raises:
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


def test_dropout2d_channel_level() raises:
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


def test_dropout2d_inference_mode() raises:
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


def test_dropout2d_backward_shapes() raises:
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


def test_dropout2d_backward_gradient() raises:
    """Test dropout2d_backward with numerical gradient checking."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(2)
    shape.append(4)
    shape.append(4)
    var x = zeros(shape, DType.float32)

    # Set non-uniform values using safe API (avoids ASAP-destruction UAF from bitcast pointer aliasing)
    for i in range(x.numel()):
        x.set(i, Float32(i) * Float32(0.1) - Float32(0.8))

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
    def forward(x: AnyTensor) raises escaping -> AnyTensor:
        # Apply the same mask that was generated initially
        from shared.core.arithmetic import multiply
        from shared.tensor.any_tensor import full_like

        var masked = multiply(x, mask)
        var scale = 1.0 / (1.0 - p)
        var scale_tensor = full_like(x, scale)
        return multiply(masked, scale_tensor)

    # Backward function wrapper - use stored mask instead of regenerating
    def backward(grad: AnyTensor, x: AnyTensor) raises escaping -> AnyTensor:
        # Use the mask from forward pass to ensure consistency
        return dropout2d_backward(grad, mask, p=p)

    # Use numerical gradient checking (gold standard)
    # Note: Using relaxed tolerances due to Float32 precision limits
    # Dropout2d uses larger tensors, requiring more relaxed tolerances
    check_gradient(forward, backward, x, grad_out, rtol=1e-2, atol=1e-3)


def main() raises:
    """Run all test_dropout tests."""
    print("Running test_dropout tests...")

    test_dropout_shapes()
    print("✓ test_dropout_shapes")

    test_dropout_inference_mode()
    print("✓ test_dropout_inference_mode")

    test_dropout_probability()
    print("✓ test_dropout_probability")

    test_dropout_scaling()
    print("✓ test_dropout_scaling")

    test_dropout_reproducibility()
    print("✓ test_dropout_reproducibility")

    test_dropout_backward_shapes()
    print("✓ test_dropout_backward_shapes")

    test_dropout_backward_gradient_flow()
    print("✓ test_dropout_backward_gradient_flow")

    test_dropout_backward_gradient()
    print("✓ test_dropout_backward_gradient")

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

    print("\nAll test_dropout tests passed!")
