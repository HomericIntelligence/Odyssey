"""Unit tests for DropoutLayer.

Tests cover:
- Initialization with various dropout rates
- Training vs inference modes
- Forward pass dropout application and scaling
- Backward pass gradient propagation
- Shape preservation across forward/backward passes
"""

from testing import assert_true, assert_false, assert_almost_equal
from shared.core.layers import DropoutLayer
from shared.core.any_tensor import AnyTensor, zeros, ones, full


fn test_dropout_init_valid() raises:
    """Test dropout layer initialization with valid dropout rates."""
    # Valid dropout rates: 0.0 to 0.99
    var layer1 = DropoutLayer(0.0)
    assert_true(
        layer1.dropout_rate == 0.0, "Should initialize with dropout_rate=0.0"
    )

    var layer2 = DropoutLayer(0.5)
    assert_true(
        layer2.dropout_rate == 0.5, "Should initialize with dropout_rate=0.5"
    )

    var layer3 = DropoutLayer(0.99)
    assert_true(
        layer3.dropout_rate == 0.99, "Should initialize with dropout_rate=0.99"
    )

    print("✓ test_dropout_init_valid passed")


fn test_dropout_init_invalid() raises:
    """Test dropout layer initialization with invalid dropout rates."""
    # Test invalid rates
    var error_caught = False
    try:
        var layer = DropoutLayer(-0.1)
    except e:
        error_caught = True
        assert_true(
            "dropout_rate" in String(e), "Error should mention dropout_rate"
        )

    assert_true(error_caught, "Should raise error for negative dropout_rate")

    error_caught = False
    try:
        var layer = DropoutLayer(1.0)
    except e:
        error_caught = True
        assert_true(
            "dropout_rate" in String(e), "Error should mention dropout_rate"
        )

    assert_true(error_caught, "Should raise error for dropout_rate=1.0")

    error_caught = False
    try:
        var layer = DropoutLayer(1.5)
    except e:
        error_caught = True
        assert_true(
            "dropout_rate" in String(e), "Error should mention dropout_rate"
        )

    assert_true(error_caught, "Should raise error for dropout_rate>1.0")

    print("✓ test_dropout_init_invalid passed")


fn test_dropout_training_mode() raises:
    """Test setting training mode."""
    var layer = DropoutLayer(0.5)

    # Default should be False (inference mode)
    assert_true(layer.training == False, "Default should be inference mode")

    # Enable training
    layer.set_training(True)
    assert_true(layer.training == True, "Should enable training mode")

    # Disable training
    layer.set_training(False)
    assert_true(layer.training == False, "Should disable training mode")

    print("✓ test_dropout_training_mode passed")


fn test_dropout_forward_inference_mode() raises:
    """Test dropout forward pass in inference mode (training=False)."""
    var layer = DropoutLayer(0.5)
    layer.set_training(False)

    # Create input
    var input = AnyTensor([4], DType.float32)
    var input_values: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    for i in range(4):
        input._data.bitcast[Float32]()[i] = input_values[i]

    var output = layer.forward(input)

    # In inference mode, output should be identical to input
    for i in range(4):
        var out_val = output._data.bitcast[Float32]()[i]
        assert_true(
            out_val == input_values[i],
            "Inference mode should pass input unchanged",
        )

    print("✓ test_dropout_forward_inference_mode passed")


fn test_dropout_forward_training_mode_shape() raises:
    """Test dropout forward pass preserves shape during training."""
    var layer = DropoutLayer(0.5)
    layer.set_training(True)

    # Create input with various shapes
    var input_1d = AnyTensor([10], DType.float32)
    for i in range(10):
        input_1d._data.bitcast[Float32]()[i] = Float32(i)

    var output_1d = layer.forward(input_1d)
    assert_true(
        len(output_1d._shape) == 1 and output_1d._shape[0] == 10,
        "1D output shape should be preserved",
    )

    var input_2d = AnyTensor([4, 5], DType.float32)
    for i in range(20):
        input_2d._data.bitcast[Float32]()[i] = Float32(i)

    var output_2d = layer.forward(input_2d)
    assert_true(
        len(output_2d._shape) == 2
        and output_2d._shape[0] == 4
        and output_2d._shape[1] == 5,
        "2D output shape should be preserved",
    )

    print("✓ test_dropout_forward_training_mode_shape passed")


fn test_dropout_forward_training_mode_zeros() raises:
    """Test dropout forward pass zeros some elements during training."""
    var layer = DropoutLayer(0.5)
    layer.set_training(True)

    # Create input with all ones
    var input = AnyTensor([100], DType.float32)
    for i in range(100):
        input._data.bitcast[Float32]()[i] = 1.0

    var output = layer.forward(input)

    # Count non-zero elements (should be roughly 50%)
    var non_zero_count = 0
    for i in range(100):
        var val = output._data.bitcast[Float32]()[i]
        if val != 0.0:
            non_zero_count += 1

    # With dropout=0.5, expect roughly 50% to be non-zero
    # Allow some variance: 30-70% non-zero
    assert_true(
        non_zero_count > 30 and non_zero_count < 70,
        "Dropout should zero approximately 50% of elements",
    )

    print("✓ test_dropout_forward_training_mode_zeros passed")


fn test_dropout_forward_training_mode_scale() raises:
    """Test dropout forward pass scales kept elements."""
    var layer = DropoutLayer(0.5)
    layer.set_training(True)

    # Create input with all ones
    var input = AnyTensor([100], DType.float32)
    for i in range(100):
        input._data.bitcast[Float32]()[i] = 1.0

    var output = layer.forward(input)

    # Non-zero elements should be scaled by 2.0 (1/(1-0.5) = 2)
    for i in range(100):
        var val = output._data.bitcast[Float32]()[i]
        if val != 0.0:
            assert_true(
                val == 2.0,
                "Non-zero elements should be scaled by 2.0 for dropout=0.5",
            )

    print("✓ test_dropout_forward_training_mode_scale passed")


fn test_dropout_backward_shape() raises:
    """Test dropout backward pass preserves shape."""
    var layer = DropoutLayer(0.5)
    layer.set_training(True)

    # Create input and forward pass to get mask
    var input = AnyTensor([4, 5], DType.float32)
    for i in range(20):
        input._data.bitcast[Float32]()[i] = 1.0

    var output = layer.forward(input)

    # Create gradient
    var grad_output = AnyTensor([4, 5], DType.float32)
    for i in range(20):
        grad_output._data.bitcast[Float32]()[i] = 0.1

    var grad_input = layer.backward(grad_output, layer.last_mask)

    assert_true(
        len(grad_input._shape) == 2
        and grad_input._shape[0] == 4
        and grad_input._shape[1] == 5,
        "Backward pass should preserve shape",
    )

    print("✓ test_dropout_backward_shape passed")


fn test_dropout_backward_scaling() raises:
    """Test dropout backward pass applies mask and scaling."""
    var layer = DropoutLayer(0.5)
    layer.set_training(True)

    # Create input and forward pass
    var input = AnyTensor([4], DType.float32)
    for i in range(4):
        input._data.bitcast[Float32]()[i] = 1.0

    var output = layer.forward(input)
    var mask = layer.last_mask

    # Create gradient with all ones
    var grad_output = AnyTensor([4], DType.float32)
    for i in range(4):
        grad_output._data.bitcast[Float32]()[i] = 1.0

    var grad_input = layer.backward(grad_output, mask)

    # Backward should apply same mask and scale
    # For elements where mask=1: grad = 1.0 * 2.0 = 2.0
    # For elements where mask=0: grad = 0.0 * 2.0 = 0.0
    for i in range(4):
        var grad_val = grad_input._data.bitcast[Float32]()[i]
        var mask_val = mask._data.bitcast[Float32]()[i]

        if mask_val == 1.0:
            assert_true(
                grad_val == 2.0, "Unmasked gradient should be scaled by 2.0"
            )
        else:
            assert_true(grad_val == 0.0, "Masked gradient should be 0.0")

    print("✓ test_dropout_backward_scaling passed")


fn test_dropout_parameters() raises:
    """Test that dropout has no learnable parameters."""
    var layer = DropoutLayer(0.5)
    var params = layer.parameters()
    assert_true(len(params) == 0, "Dropout should have no parameters")

    print("✓ test_dropout_parameters passed")


fn test_dropout_forward_float64() raises:
    """Test dropout forward pass with float64 dtype."""
    var layer = DropoutLayer(0.5)
    layer.set_training(True)

    var input = AnyTensor([50], DType.float64)
    for i in range(50):
        input._data.bitcast[Float64]()[i] = 1.0

    var output = layer.forward(input)

    # Count non-zero elements
    var non_zero_count = 0
    for i in range(50):
        var val = output._data.bitcast[Float64]()[i]
        if val != 0.0:
            non_zero_count += 1

    # Expect roughly 50% non-zero
    assert_true(
        non_zero_count > 15 and non_zero_count < 35,
        "Dropout should zero approximately 50% for float64",
    )

    print("✓ test_dropout_forward_float64 passed")


fn test_dropout_zero_dropout_rate() raises:
    """Test dropout with dropout_rate=0 (no dropout)."""
    var layer = DropoutLayer(0.0)
    layer.set_training(True)

    var input = AnyTensor([10], DType.float32)
    for i in range(10):
        input._data.bitcast[Float32]()[i] = 1.0

    var output = layer.forward(input)

    # With dropout_rate=0, nothing should be dropped
    # Scale factor = 1/(1-0) = 1
    for i in range(10):
        var val = output._data.bitcast[Float32]()[i]
        assert_true(
            val == 1.0, "With dropout_rate=0, output should be unchanged"
        )

    print("✓ test_dropout_zero_dropout_rate passed")


fn test_dropout_high_dropout_rate() raises:
    """Test dropout with high dropout_rate."""
    var layer = DropoutLayer(0.9)
    layer.set_training(True)

    var input = AnyTensor([100], DType.float32)
    for i in range(100):
        input._data.bitcast[Float32]()[i] = 1.0

    var output = layer.forward(input)

    # Count non-zero elements (should be roughly 10%)
    var non_zero_count = 0
    for i in range(100):
        var val = output._data.bitcast[Float32]()[i]
        if val != 0.0:
            non_zero_count += 1

    # With dropout_rate=0.9, expect roughly 10% to remain
    # Allow variance: 0-30% non-zero
    assert_true(
        non_zero_count >= 0 and non_zero_count <= 30,
        "Dropout with rate=0.9 should keep roughly 10% of elements",
    )

    # Check scaling: kept elements should be scaled by 1/(1-0.9) = 10
    for i in range(100):
        var val = output._data.bitcast[Float32]()[i]
        if val != 0.0:
            # Check that scaled value is approximately 10.0
            var diff = abs(val - 10.0)
            assert_true(
                diff < 0.01,
                "Scale factor for dropout=0.9 should be approximately 10.0",
            )

    print("✓ test_dropout_high_dropout_rate passed")


fn main():
    """Run all Dropout tests."""
    print("Running DropoutLayer tests...")
    try:
        test_dropout_init_valid()
        test_dropout_init_invalid()
        test_dropout_training_mode()
        test_dropout_forward_inference_mode()
        test_dropout_forward_training_mode_shape()
        test_dropout_forward_training_mode_zeros()
        test_dropout_forward_training_mode_scale()
        test_dropout_backward_shape()
        test_dropout_backward_scaling()
        test_dropout_parameters()
        test_dropout_forward_float64()
        test_dropout_zero_dropout_rate()
        test_dropout_high_dropout_rate()
        print("\nAll DropoutLayer tests passed!")
    except e:
        print("Test failed:", e)
